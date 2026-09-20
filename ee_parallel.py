"""Parallel drop-ins for the per-reference and bootstrap loops.

WHERE THE TIME ACTUALLY GOES. One configuration costs 8 Soar calls for Gamma, plus
8 * n_boot for the bootstrap null, plus up to 8 * max_passes for equalisation. At
n_boot=10 that is ~90-120 Soar calls, and the Part 5 scan multiplies it by the number
of configurations. All of them are independent, and all of them ran serially.

THE THREE LEVELS, and why the split matters:

    config_workers  x  ref_workers  x  n_jobs   <=  physical cores
    (scan configs)     (the 8 refs)   (sklearn kNN threads inside one Soar)

Nested parallelism oversubscribes silently: 8 configs x 8 refs x n_jobs=8 asks for
512 threads on a 112-core box and runs SLOWER than serial, because every kNN call
fights the others for cache and the BLAS threads thrash. The helpers here therefore
force the inner n_jobs down when they fan out, and `plan_workers` tells you what the
product comes to before you commit.

DETERMINISM. Everything downstream of `stats_null` is deterministic, so the parallel
result is bit-identical to the serial one -- except for sklearn's own parallel kNN
tie-breaking, which can differ in the last ulp when equidistant neighbours are
reordered. `verify_parallel_identity` checks this on real data rather than assuming.
"""
import contextlib
import io
import os
import sys
from contextlib import redirect_stdout
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def plan_workers(n_configs=1, n_refs=8, n_jobs=1, cores=None):
    """Report the thread budget and flag oversubscription before it costs you an hour."""
    cores = cores or os.cpu_count() or 1
    total = int(n_configs) * int(n_refs) * int(n_jobs)
    return dict(cores=cores, config_workers=n_configs, ref_workers=n_refs,
                inner_n_jobs=n_jobs, total_threads=total,
                oversubscribed=total > cores,
                advice=("fine" if total <= cores else
                        f"asks for {total} threads on {cores} cores -- reduce n_jobs"))


_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _ensure_child_libpath():
    """Make sure a freshly-exec'd worker can find the conda runtime libraries.

    loky does not fork -- it execs a new interpreter -- so the dynamic linker in the
    child re-resolves everything from LD_LIBRARY_PATH. A Jupyter kernel often runs
    without that variable (it got its libstdc++ another way at startup), and the child
    then dies with

        ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.29' not found

    while merely un-pickling the task, which joblib reports as the far less helpful
    "A task has failed to un-serialize".

    Editing LD_LIBRARY_PATH here does not affect THIS process -- its linker already
    ran -- but children inherit os.environ, which is exactly what we need.
    """
    libdir = os.path.join(sys.prefix, "lib")
    if not os.path.isdir(libdir):
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if libdir not in cur.split(os.pathsep):
        os.environ["LD_LIBRARY_PATH"] = (libdir + os.pathsep + cur) if cur else libdir


def pmap(fn, tasks, workers, verbose=0):
    """Run `fn` over `tasks` in parallel. joblib/loky, for three measured reasons.

    THREE FAILURES, EACH RULING OUT THE OBVIOUS CHOICE:

    1. ProcessPoolExecutor + fork. numpy initialises OpenBLAS to every core in the
       parent; a forked child inherits a thread-pool descriptor whose threads do not
       exist, and the first BLAS call hangs or aborts. Measured: serial 53 s, the
       8-way fork fan-out still unfinished at 9 min, then BrokenProcessPool.
    2. ProcessPoolExecutor + spawn. Fixes the BLAS inheritance but re-imports
       __main__ in every child. From a Jupyter kernel (or a stdin script) __main__ is
       not importable, so children re-execute or hang -- unusable where this lives.
    3. Setting OMP_NUM_THREADS in a worker initializer. Too late both ways: under
       fork BLAS is already running, under spawn the imports have already happened.

    loky handles all three: it starts cleanly without touching __main__, reuses its
    workers between calls, and `inner_max_num_threads=1` caps BLAS inside each worker
    so eight of them do not each try to own 112 cores.
    """
    from joblib import Parallel, delayed, parallel_backend
    n = max(1, int(workers))
    tasks = list(tasks)
    if n == 1:
        return [fn(t) for t in tasks]
    _ensure_child_libpath()
    try:
        with parallel_backend("loky", n_jobs=n, inner_max_num_threads=1):
            return Parallel(verbose=verbose)(delayed(fn)(t) for t in tasks)
    except Exception as e:
        # Never let a broken pool cost the caller their run: say what happened and
        # finish the work serially. Common causes are an un-picklable argument and a
        # child that cannot import the stack (see _ensure_child_libpath).
        print(f"  [pmap] parallel execution failed ({type(e).__name__}: "
              f"{str(e)[:160]}); falling back to SERIAL", flush=True)
        return [fn(t) for t in tasks]


# THE THREAD TRAP THAT MAKES NAIVE FAN-OUT SLOWER THAN SERIAL.
# `n_jobs` controls sklearn/joblib only. It does NOT touch OpenBLAS/MKL, which numpy
# initialises to ALL cores (112 here). Fork 8 workers and each still believes it owns
# the whole machine: ~900 threads fighting for 112 cores, and the "parallel" run comes
# out an order of magnitude SLOWER than serial. Measured: serial 53 s, 8-way fan-out
# still unfinished after 9 minutes.
#
# Setting OMP_NUM_THREADS in the child is TOO LATE -- BLAS is already initialised in
# the parent and inherited through fork. threadpool_limits reaches into the loaded
# libraries at runtime, which is why it is used inside every worker below.
def _limit_threads(n=1):
    """Context manager pinning BLAS/OMP inside a worker. Falls back to a no-op."""
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=n)
    except Exception:
        return contextlib.nullcontext()


def _ref_task(args):
    """One reference: Soar + clustering + repechage + the two estimators.

    Top-level (not a closure) so ProcessPoolExecutor can pickle it. EagleEye prints
    heavily; in a fan-out that interleaves into noise, so stdout is swallowed here.
    """
    (j, Xj, Y, stats_null_j, K_M, p_ext, Z, upsilon_key, n_jobs) = args
    import EagleEye
    from utils_EE import partitioning_function
    from analyze_known_dwarfs import (S_rootB_estimate_Y_overdensities,
                                      B_estimate_Y_overdensities)
    with _limit_threads(1), redirect_stdout(io.StringIO()):
        rd, _ = EagleEye.Soar(Xj, Y, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs,
                              stats_null=stats_null_j, result_dict_in={}, do_IDE=True)
        clusters = partitioning_function(Xj, Y, rd, p_ext=p_ext, Z=Z)
        book = EagleEye.Repechage(Xj, Y, rd, clusters, p_ext=p_ext)
        srb = S_rootB_estimate_Y_overdensities(rd, book)
        bb = B_estimate_Y_overdensities(rd, book)
    ups = np.asarray(rd[upsilon_key], float)
    # stats_null is ~3 MB and the parent already holds it; do not pickle it back.
    rd.pop("stats_null", None)
    return j, ups, rd, book, srb, bb


def compute_Gamma_i_multi_ref_par(X_refs, Y, stats_null_multi, K_M=300, p_ext=1e-5,
                                  n_jobs=1, Z=2.65, upsilon_key="Upsilon_i_Y",
                                  ref_workers=8):
    """Drop-in for akd.compute_Gamma_i_multi_ref, references fanned out over processes."""
    import analyze_known_dwarfs as akd
    R = len(X_refs)
    if ref_workers <= 1 or R == 1:
        return akd.compute_Gamma_i_multi_ref(
            X_refs=X_refs, Y=Y, stats_null_multi=stats_null_multi, K_M=K_M,
            p_ext=p_ext, n_jobs=n_jobs, Z=Z, upsilon_key=upsilon_key)

    tasks = [(j, X_refs[j], Y, stats_null_multi["by_ref"][j]["stats_null"],
              K_M, p_ext, Z, upsilon_key, n_jobs) for j in range(R)]
    out = [None] * R
    for j, ups, rd, book, srb, bb in pmap(_ref_task, tasks, min(int(ref_workers), R)):
        rd["stats_null"] = stats_null_multi["by_ref"][j]["stats_null"]
        out[j] = (ups, rd, book, srb, bb)

    Ups = np.zeros((R, len(Y)), float)
    rds, books, srbs, bbs = [], [], [], []
    for j, (ups, rd, book, srb, bb) in enumerate(out):
        Ups[j] = ups; rds.append(rd); books.append(book)
        srbs.append(srb); bbs.append(bb)
    return {"Gamma_i": Ups.sum(axis=0), "Upsilon_by_ref": Ups,
            "result_dicts": rds, "EE_books": books,
            "SrootB_by_ref": srbs, "B_by_ref": bbs,
            "p_ext": p_ext, "K_M": K_M, "Z": Z, "upsilon_key": upsilon_key}


def _boot_task(args):
    """One bootstrap realisation, all its references done in-process."""
    (b, seed, nY, d, nX_list, stats_null_multi, K_M, p_ext, upsilon_key, n_jobs) = args
    import analyze_known_dwarfs as akd
    # REPRODUCE THE SERIAL STREAM EXACTLY. The serial loop draws every realisation
    # from one Generator, so realisation b starts after b*(nY+sum nX)*d doubles have
    # been consumed. PCG64.advance() jumps straight there, which makes the parallel
    # bootstrap bit-identical to the serial one instead of merely equivalent in
    # distribution. That matters: at n_boot=4 an independent reseed moved Gamma*
    # from 52.4 to 61.7 and changed the cluster count from 2 to 1.
    per = (nY + int(np.sum(nX_list))) * d
    rng = np.random.Generator(np.random.PCG64(seed).advance(b * per))
    Yb = rng.random((nY, d))
    Xb = [rng.random((nX, d)) for nX in nX_list]
    with _limit_threads(1), redirect_stdout(io.StringIO()):
        g = akd.compute_Gamma_i_multi_ref(
            X_refs=Xb, Y=Yb, stats_null_multi=stats_null_multi, K_M=K_M,
            p_ext=p_ext, n_jobs=n_jobs, upsilon_key=upsilon_key)
    return b, np.asarray(g["Gamma_i"], float)


def bootstrap_gamma_null_uniform_par(X_refs, Y, stats_null_multi, K_M=300, p_ext=1e-5,
                                     n_jobs=1, upsilon_key="Upsilon_i_Y", n_boot=1,
                                     seed=42, workers=8):
    """Parallel over BOOTSTRAPS, which is the better axis: they are fully independent
    and there are usually more of them than references.

    Bit-identical to the serial version: each worker jumps its PCG64 stream to the
    offset that realisation would have reached serially (see `_boot_task`), so cached
    results and re-runs agree exactly rather than only in distribution.
    """
    import analyze_known_dwarfs as akd
    if workers <= 1 or n_boot == 1:
        return akd.bootstrap_gamma_null_uniform(
            X_refs=X_refs, Y=Y, stats_null_multi=stats_null_multi, K_M=K_M,
            p_ext=p_ext, n_jobs=n_jobs, upsilon_key=upsilon_key, n_boot=n_boot,
            seed=seed)
    nY, d = len(Y), Y.shape[1]
    nX_list = [len(x) for x in X_refs]
    tasks = [(b, seed, nY, d, nX_list, stats_null_multi, K_M, p_ext,
              upsilon_key, n_jobs) for b in range(n_boot)]
    out = np.zeros((n_boot, nY), float)
    for _b, g in pmap(_boot_task, tasks, min(int(workers), n_boot)):
        out[_b] = g
    return out


def verify_parallel_identity(X_refs, Y, stats_null_multi, **kw):
    """Run both paths on the same inputs and report the largest disagreement."""
    import analyze_known_dwarfs as akd
    kw.pop("ref_workers", None)
    with redirect_stdout(io.StringIO()):
        a = akd.compute_Gamma_i_multi_ref(X_refs=X_refs, Y=Y,
                                          stats_null_multi=stats_null_multi, **kw)
    b = compute_Gamma_i_multi_ref_par(X_refs, Y, stats_null_multi,
                                      ref_workers=len(X_refs), **kw)
    ga, gb = np.asarray(a["Gamma_i"]), np.asarray(b["Gamma_i"])
    dmax = float(np.nanmax(np.abs(ga - gb))) if ga.shape == gb.shape else np.inf
    nd = int(np.sum(ga != gb)) if ga.shape == gb.shape else -1
    return dict(identical=bool(dmax == 0.0), max_abs_diff=dmax, n_differing=nd,
                n=int(ga.size))


# ----------------------------------------------------------------------
# Silencing EagleEye
# ----------------------------------------------------------------------


@contextlib.contextmanager
def quiet_ee(enabled=True, capture=False):
    """Suppress EagleEye's per-call chatter.

    TWO CHANNELS, and missing either one leaves half the noise. EagleEye emits
    progress through `print()` AND through `IPython.display.display()`. In a notebook
    `display()` goes to the kernel's display hook, NOT to stdout, so redirecting
    stdout alone silences the prints and leaves a stream of "<IPython.core.display
    .Math object>" behind. Both modules' `display` names are therefore rebound to a
    no-op for the duration, and restored afterwards even on exception.

    `capture=True` keeps the swallowed stdout and yields the buffer, for when a run
    fails and you want the trace of what it was doing.
    """
    if not enabled:
        yield None
        return
    mods = []
    for name in ("EagleEye", "utils_EE"):
        m = __import__(name) if name not in globals() else globals()[name]
        if hasattr(m, "display"):
            mods.append((m, m.display))
    buf = io.StringIO()
    try:
        for m, _ in mods:
            m.display = lambda *a, **k: None
        with redirect_stdout(buf):
            yield buf if capture else None
    finally:
        for m, orig in mods:
            m.display = orig

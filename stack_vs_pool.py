"""Stacked Gamma vs a single pooled reference, on synthetic data matched to a real run.

THE QUESTION. Gamma = sum_j Upsilon^(j) keeps the R reference tiles separate, each
with its own p_hat_j; pooling concatenates them into one X. Which detects a FAINT
overdensity better when the references are broadly similar but individually lumpy
and of DIFFERENT SIZE?

WHY CARDINALITY MATTERS AND MUST COME FROM A REAL RUN. Equal-sized references give
every comparison the same p_hat, which removes the heterogeneity stacking is built to
absorb -- a benchmark built that way answers a question nobody has. Pass the nXs of an
actual run so the spread in p_hat_j is the one you really face.

WHAT IS MEASURED. TPR at a common false-positive rate taken from each method's OWN
background, plus AUC. Both are threshold-free comparisons of raw discriminating power,
so a difference in null calibration between the two methods cannot flatter either one.
"""
import numpy as np
import pandas as pd


def _bumpy(rng, n, d, n_bumps, bump_frac, bump_w):
    """Uniform background plus n_bumps small Gaussian lumps at random positions."""
    k = int(round(n * bump_frac))
    parts = [rng.random((max(n - k * n_bumps, 0), d))]
    for _ in range(n_bumps):
        c = rng.random(d)
        parts.append(np.clip(c + rng.normal(0, bump_w, (k, d)), 0, 1))
    out = np.vstack(parts)
    return out[:n] if len(out) > n else out


def _nulls(p_list, K_M, N, n_jobs):
    """compute_the_null depends only on (p, K_M) -- cache it across all trials."""
    from utils_EE import compute_the_null
    import robustness as rb
    cache = {}
    with rb._q():
        for p in p_list:
            key = round(float(p), 10)
            if key not in cache:
                cache[key] = compute_the_null(p=p, K_M=K_M, N=N)
    return cache


def _bootstrap_gstar(nY, nXs, nulls, p_list, K_M, p_ext, n_boot, n_jobs, seed=42):
    """Gamma* for these cardinalities. Depends only on (nY, nXs, K_M, p_ext) -- NOT on
    the injected signal -- so compute it once and reuse across the whole scan."""
    import analyze_known_dwarfs as akd
    import robustness as rb
    snm = {"p_list": list(p_list), "K_M": K_M, "N": None,
           "by_ref": {j: {"p": p_list[j], "stats_null": nulls[round(p_list[j], 10)]}
                      for j in range(len(nXs))}}
    rng = np.random.default_rng(seed)
    Yb = rng.random((nY, 4))
    Xb = [rng.random((nx, 4)) for nx in nXs]
    with rb._q():
        g = akd.bootstrap_gamma_null_uniform(X_refs=Xb, Y=Yb, stats_null_multi=snm,
                                             K_M=K_M, p_ext=p_ext, n_jobs=n_jobs,
                                             n_boot=n_boot, seed=seed)
    g = np.asarray(g, float).reshape(-1); g = g[np.isfinite(g)]
    return float(np.quantile(g, 1 - p_ext)), snm


def stacking_benchmark(nY, nXs, *, sig_ns=(8, 14, 22), sig_ws=(0.035, 0.055, 0.085),
                       n_trials=4, d=4, n_bumps=8, bump_frac=0.02, bump_w=0.05,
                       K_M=None, p_ext=1e-3, null_N=100_000, fpr=0.01, n_boot=3,
                       n_jobs=8, seed=0, verbose=True):
    """Scan signal STRENGTH and DIFFUSENESS; compare stacked Gamma against pooled X.

    REPORTS, for each cell of the grid:
      tpr_*   -- recovery of the injected stars at a common FPR taken from each
                 method's own background. This is the number that matters when
                 EagleEye is used to FLAG regions for later dynamical follow-up:
                 purity is somebody else's problem, missed objects are not.
      srb_*   -- S/sqrt(B) exactly as this notebook computes it. For the stacked side
                 that means akd.srootB_for_Gamma_clusters_from_Brefs, i.e. DBSCAN on
                 the Gamma-flagged set with an overlap-weighted background from the
                 per-reference repechage books. For the pooled side it is EagleEye's
                 own S_rootB_estimate_Y_overdensities on the single comparison.
      auc_*   -- threshold-free, and far less noisy than TPR at these counts.

    Gamma* and the per-reference nulls depend only on the cardinalities and K_M, not
    on the injected signal, so both are computed once and reused across the grid.

    Diffuseness (`sig_ws`) is the axis to watch: at fixed star count a broader anomaly
    has lower contrast, and the two methods need not degrade at the same rate.
    """
    import EagleEye
    import analyze_known_dwarfs as akd
    import robustness as rb
    from utils_EE import compute_the_null, partitioning_function
    from sklearn.metrics import roc_auc_score

    nXs = [int(x) for x in nXs]
    R = len(nXs)
    K_M = int(K_M if K_M else 0.05 * min(min(nXs), nY))
    p_list = [nY / (nY + nx) for nx in nXs]
    nXp = int(sum(nXs)); p_pool = nY / (nY + nXp)
    K_Mp = min(K_M, int(0.05 * min(nXp, nY)))
    nulls = _nulls(list(p_list), K_M, null_N, n_jobs)
    with rb._q():
        null_pool = compute_the_null(p=p_pool, K_M=K_Mp, N=null_N)
    ups_star_p = float(np.quantile(np.asarray(null_pool[p_pool], float), 1 - p_ext))
    Gstar, snm = _bootstrap_gstar(nY, nXs, nulls, p_list, K_M, p_ext, n_boot, n_jobs)
    if verbose:
        print(f"nY={nY}  nXs={nXs}")
        print(f"  p_hat spread {100*np.std(p_list)/np.mean(p_list):.1f}%   "
              f"K_M stacked={K_M} pooled={K_Mp}")
        print(f"  Gamma*={Gstar:.2f} (n_boot={n_boot}), pooled Upsilon*={ups_star_p:.2f}\n")

    rows = []
    for sig_w in sig_ws:
        for sig_n in sig_ns:
            acc = {k: [] for k in ("ts", "tp", "as_", "ap", "ss", "sp")}
            for t in range(n_trials):
                rng = np.random.default_rng(seed + 7919 * sig_n + 104729 * t
                                            + int(sig_w * 1e4))
                ctr = np.full(d, 0.5)
                Y = np.vstack([rng.random((nY - sig_n, d)),
                               np.clip(ctr + rng.normal(0, sig_w, (sig_n, d)), 0, 1)])
                truth = np.zeros(nY, bool); truth[nY - sig_n:] = True
                X = [_bumpy(rng, nx, d, n_bumps, bump_frac, bump_w) for nx in nXs]
                with rb._q():
                    out = akd.compute_Gamma_i_multi_ref(
                        X_refs=X, Y=Y, stats_null_multi=snm, K_M=K_M, p_ext=p_ext,
                        n_jobs=n_jobs, Z=2.65)
                G = np.asarray(out["Gamma_i"], float)
                with rb._q():
                    cl = akd.srootB_for_Gamma_clusters_from_Brefs(
                        EE_books=out["EE_books"], SrootB_by_ref=out["SrootB_by_ref"],
                        B_by_ref=out["B_by_ref"], Y=Y, A_Gamma_mask=(G > Gstar),
                        overlap_min=1, use_cluster_key="Repechaged",
                        weight_mode="overlap", dbscan_eps=1.0, dbscan_min_samples=5,
                        cluster_cols=tuple(range(d)))
                # the Gamma cluster overlapping the injected signal
                best = np.nan
                for gid, ix in cl["gamma_clusters"].items():
                    if truth[np.asarray(ix, int)].sum() > 0:
                        v = float(cl["cluster_stats"][int(gid)]["s_over_rootB"])
                        if np.isfinite(v):
                            best = v if not np.isfinite(best) else max(best, v)
                with rb._q():
                    rdp, _ = EagleEye.Soar(np.vstack(X), Y, K_M=K_Mp, p_ext=p_ext,
                                           n_jobs=n_jobs, stats_null=null_pool,
                                           result_dict_in={}, do_IDE=True)
                    sp = np.nan
                    try:
                        cp = partitioning_function(np.vstack(X), Y, rdp,
                                                   p_ext=p_ext, Z=2.65)
                        bk = EagleEye.Repechage(np.vstack(X), Y, rdp, cp, p_ext=p_ext)
                        sr = EagleEye.S_rootB_estimate_Y_overdensities(rdp, bk)
                        for a, v in bk["Y_OVER_clusters"].items():
                            rep = np.asarray(v.get("Repechaged", []), int)
                            if rep.size and truth[rep].sum() > 0:
                                q = float(sr["s/root(B)"].get(a, np.nan))
                                if np.isfinite(q):
                                    sp = q if not np.isfinite(sp) else max(sp, q)
                    except Exception:
                        pass
                U = np.asarray(rdp["Upsilon_i_Y"], float)
                thr = lambda v: np.quantile(v[~truth], 1 - fpr)
                acc["ts"].append(float((G[truth] > thr(G)).mean()))
                acc["tp"].append(float((U[truth] > thr(U)).mean()))
                acc["as_"].append(roc_auc_score(truth, G))
                acc["ap"].append(roc_auc_score(truth, U))
                acc["ss"].append(best); acc["sp"].append(sp)
            nm = lambda a: float(np.nanmean(a)) if np.any(np.isfinite(a)) else np.nan
            rows.append(dict(sig_w=sig_w, sig_n=sig_n,
                             tpr_stacked=np.mean(acc["ts"]),
                             tpr_pooled=np.mean(acc["tp"]),
                             tpr_gain=np.mean(acc["ts"]) - np.mean(acc["tp"]),
                             srb_stacked=nm(acc["ss"]), srb_pooled=nm(acc["sp"]),
                             auc_stacked=np.mean(acc["as_"]),
                             auc_pooled=np.mean(acc["ap"]),
                             tpr_sd_stacked=np.std(acc["ts"])))
            if verbose:
                r = rows[-1]
                print(f"  w={sig_w:.3f} N={sig_n:3d}   TPR {r['tpr_stacked']:.3f} vs "
                      f"{r['tpr_pooled']:.3f} ({r['tpr_gain']:+.3f})   "
                      f"S/rootB {r['srb_stacked']:6.2f} vs {r['srb_pooled']:6.2f}   "
                      f"AUC {r['auc_stacked']:.3f}/{r['auc_pooled']:.3f}", flush=True)
    return pd.DataFrame(rows)


def demo_realisation(nY, nXs, sig_n, *, d=4, n_bumps=8, bump_frac=0.02, bump_w=0.05,
                     sig_w=0.055, K_M=None, p_ext=1e-3, null_N=100_000, n_jobs=8,
                     seed=1, figsize=(17.5, 4.3)):
    """One realisation, drawn before and after the EagleEye analysis.

    Left two panels are the INPUT: the test tile with its injected overdensity, and
    the pooled reference showing the per-tile lumps. Right two are the OUTPUT: the
    same test tile coloured by the stacked Gamma and by the pooled Upsilon, with the
    stars each method flags ringed. Only the first two feature dimensions are shown --
    the analysis runs in all `d`.
    """
    import matplotlib.pyplot as plt
    import EagleEye
    import robustness as rb

    nXs = [int(x) for x in nXs]
    K_M = int(K_M if K_M else 0.05 * min(min(nXs), nY))
    p_list = [nY / (nY + nx) for nx in nXs]
    nXp = sum(nXs); p_pool = nY / (nY + nXp)
    K_Mp = min(K_M, int(0.05 * min(nXp, nY)))
    nulls = _nulls(list(p_list) + [p_pool], K_M, null_N, n_jobs)
    null_p = _nulls([p_pool], K_Mp, null_N, n_jobs) if K_Mp != K_M else nulls

    rng = np.random.default_rng(seed); ctr = np.full(d, 0.5)
    Y = np.vstack([rng.random((nY - sig_n, d)),
                   np.clip(ctr + rng.normal(0, sig_w, (sig_n, d)), 0, 1)])
    truth = np.zeros(nY, bool); truth[nY - sig_n:] = True
    X = [_bumpy(rng, nx, d, n_bumps, bump_frac, bump_w) for nx in nXs]

    G = np.zeros(nY)
    with rb._q():
        for j, Xj in enumerate(X):
            rd, _ = EagleEye.Soar(Xj, Y, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs,
                                  stats_null=nulls[round(p_list[j], 10)],
                                  result_dict_in={}, do_IDE=False)
            G += np.asarray(rd["Upsilon_i_Y"], float)
        rdp, _ = EagleEye.Soar(np.vstack(X), Y, K_M=K_Mp, p_ext=p_ext, n_jobs=n_jobs,
                               stats_null=null_p[round(p_pool, 10)],
                               result_dict_in={}, do_IDE=False)
    U = np.asarray(rdp["Upsilon_i_Y"], float)
    fpr = 0.01
    gA = G > np.quantile(G[~truth], 1 - fpr)
    uA = U > np.quantile(U[~truth], 1 - fpr)

    fig, ax = plt.subplots(1, 4, figsize=figsize)
    ax[0].scatter(Y[~truth, 0], Y[~truth, 1], s=2, c="0.8", lw=0, rasterized=True)
    ax[0].scatter(Y[truth, 0], Y[truth, 1], s=26, c="crimson", lw=0)
    ax[0].set_title(f"BEFORE: test tile, {sig_n} injected", fontsize=10)
    Xp = np.vstack(X)
    ax[1].scatter(Xp[:, 0], Xp[:, 1], s=1, c="0.6", lw=0, rasterized=True)
    ax[1].set_title(f"BEFORE: pooled reference ({nXp} stars,\n"
                    f"{n_bumps} lumps per tile)", fontsize=10)
    for a, stat, flag, lab in ((ax[2], G, gA, r"AFTER: stacked $\Gamma$"),
                               (ax[3], U, uA, r"AFTER: pooled $\Upsilon$")):
        s = a.scatter(Y[:, 0], Y[:, 1], s=3, c=stat, cmap="viridis", lw=0,
                      rasterized=True)
        a.scatter(Y[flag, 0], Y[flag, 1], s=48, facecolor="none", edgecolor="crimson",
                  lw=1.0)
        rec = int((flag & truth).sum())
        a.set_title(f"{lab}\n{int(flag.sum())} flagged, {rec}/{sig_n} true "
                    f"(FPR set to {fpr:.0%})", fontsize=10)
        plt.colorbar(s, ax=a, fraction=.046)
    for a in ax:
        a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_aspect("equal")
        a.set_xlabel("feature 1")
    ax[0].set_ylabel("feature 2")
    fig.tight_layout()
    return fig, dict(Gamma=G, Upsilon=U, truth=truth,
                     stacked_rec=int((gA & truth).sum()),
                     pooled_rec=int((uA & truth).sum()))


def false_alarm_test(nY, nXs, *, n_trials=20, thresholds=(2, 3, 4, 5, 7), d=4,
                     n_bumps=8, bump_frac=0.02, bump_w=0.05, K_M=None, p_ext=1e-3,
                     null_N=100_000, n_boot=3, n_jobs=8, seed=0, verbose=True):
    """NO signal injected. How often does each method still produce a cluster?

    The companion to `stacking_benchmark`: that measures recovery, this measures what
    each method invents from nothing at the SAME cardinalities and the same reference
    lumpiness. Together they give the pair you actually need -- sensitivity read
    against a false-alarm rate rather than against p_ext, which is only a per-star
    flagging level and says nothing about how often a CLUSTER appears.

    Returns (summary DataFrame, per-trial max S/sqrt(B) for each method). The
    summary is the fraction of signal-free trials producing a cluster at or above
    each S/sqrt(B) threshold -- read the quantile you can live with off it and use
    that as the minimum passable value for a real candidate.
    """
    import EagleEye
    import analyze_known_dwarfs as akd
    import robustness as rb
    from utils_EE import compute_the_null, partitioning_function

    nXs = [int(x) for x in nXs]
    K_M = int(K_M if K_M else 0.05 * min(min(nXs), nY))
    p_list = [nY / (nY + nx) for nx in nXs]
    nXp = int(sum(nXs)); p_pool = nY / (nY + nXp)
    K_Mp = min(K_M, int(0.05 * min(nXp, nY)))
    nulls = _nulls(list(p_list), K_M, null_N, n_jobs)
    with rb._q():
        null_pool = compute_the_null(p=p_pool, K_M=K_Mp, N=null_N)
    Gstar, snm = _bootstrap_gstar(nY, nXs, nulls, p_list, K_M, p_ext, n_boot, n_jobs)
    if verbose:
        print(f"false-alarm test: {n_trials} SIGNAL-FREE realisations, "
              f"nY={nY}, {len(nXs)} refs, Gamma*={Gstar:.2f}")

    ms, mp = [], []
    for t in range(n_trials):
        rng = np.random.default_rng(seed + 31337 * t)
        Y = rng.random((nY, d))
        X = [_bumpy(rng, nx, d, n_bumps, bump_frac, bump_w) for nx in nXs]
        with rb._q():
            out = akd.compute_Gamma_i_multi_ref(X_refs=X, Y=Y, stats_null_multi=snm,
                                                K_M=K_M, p_ext=p_ext, n_jobs=n_jobs,
                                                Z=2.65)
            G = np.asarray(out["Gamma_i"], float)
            cl = akd.srootB_for_Gamma_clusters_from_Brefs(
                EE_books=out["EE_books"], SrootB_by_ref=out["SrootB_by_ref"],
                B_by_ref=out["B_by_ref"], Y=Y, A_Gamma_mask=(G > Gstar),
                overlap_min=1, use_cluster_key="Repechaged", weight_mode="overlap",
                dbscan_eps=1.0, dbscan_min_samples=5, cluster_cols=tuple(range(d)))
        vs = [float(cl["cluster_stats"][int(g)]["s_over_rootB"])
              for g in cl["gamma_clusters"]]
        ms.append(max([v for v in vs if np.isfinite(v)], default=0.0))
        vp = 0.0
        with rb._q():
            try:
                Xp = np.vstack(X)
                rdp, _ = EagleEye.Soar(Xp, Y, K_M=K_Mp, p_ext=p_ext, n_jobs=n_jobs,
                                       stats_null=null_pool, result_dict_in={},
                                       do_IDE=True)
                cp = partitioning_function(Xp, Y, rdp, p_ext=p_ext, Z=2.65)
                bk = EagleEye.Repechage(Xp, Y, rdp, cp, p_ext=p_ext)
                sr = EagleEye.S_rootB_estimate_Y_overdensities(rdp, bk)["s/root(B)"]
                vp = max([float(v) for k, v in sr.items()
                          if k != "Total" and np.isfinite(float(v))], default=0.0)
            except Exception:
                pass
        mp.append(vp)
        if verbose and (t + 1) % 5 == 0:
            print(f"    {t+1}/{n_trials} done", flush=True)
    ms, mp = np.asarray(ms), np.asarray(mp)
    rows = [dict(threshold=th,
                 stacked_rate=float((ms >= th).mean()),
                 pooled_rate=float((mp >= th).mean())) for th in thresholds]
    df = pd.DataFrame(rows)
    if verbose:
        print(f"\n  max S/rootB from nothing --  stacked: median {np.median(ms):.2f}, "
              f"90% {np.quantile(ms,.9):.2f}, max {ms.max():.2f}")
        print(f"                               pooled : median {np.median(mp):.2f}, "
              f"90% {np.quantile(mp,.9):.2f}, max {mp.max():.2f}")
        print(f"\n  fraction of signal-free trials producing a cluster >= threshold:")
        print("  " + df.to_string(index=False).replace("\n", "\n  "))
        print(f"\n  -> the 90% column is the minimum passable S/rootB at this "
              f"configuration")
    return df, dict(stacked=ms, pooled=mp)


def _detect(cl_stats, clusters, truth, thresh, cluster_key="gamma"):
    """Did this method produce a cluster above `thresh` that OVERLAPS the signal?

    Overlap is required deliberately: a spurious cluster elsewhere in the tile is not
    a detection of the injected object, and counting it as one would reward whichever
    method is noisier.
    """
    for gid, ix in clusters.items():
        ix = np.asarray(ix, int)
        if ix.size and truth[ix].sum() > 0:
            v = float(cl_stats[int(gid)]["s_over_rootB"]) if cluster_key == "gamma" \
                else float(cl_stats.get(gid, np.nan))
            if np.isfinite(v) and v >= thresh:
                return True
    return False


def paired_detection_test(nY, nXs, *, srb_thresh_stacked=5.0,
                          sig_ns=(8, 14, 22), sig_ws=(0.035, 0.055, 0.085),
                          n_trials=10, n_fa_trials=20, d=4, n_bumps=8,
                          bump_frac=0.02, bump_w=0.05, K_M=None, p_ext=1e-3,
                          null_N=100_000, n_boot=3, n_jobs=8, seed=0, verbose=True):
    """How often does stacked find a real signal that pooled misses, and vice versa?

    CALIBRATION FIRST, AND WHY IT MATTERS. The two methods do not report S/sqrt(B) on
    the same scale -- on identical data the stacked value can be several times the
    pooled one -- so applying the SAME numeric threshold to both would decide the
    contest by scale, not by sensitivity. Here you fix the stacked threshold
    (`srb_thresh_stacked`, default 5) and the pooled threshold is set to whatever
    delivers the SAME false-alarm rate on signal-free data. Both methods are then
    operating at an identical error budget and the comparison is meaningful.

    Returns (DataFrame, info). The DataFrame is a 2x2 contingency per grid cell:
    both / stacked_only / pooled_only / neither, with a McNemar exact p-value on the
    discordant pairs -- the correct test for paired binary outcomes, since the two
    methods see the SAME realisations.
    """
    import EagleEye
    import analyze_known_dwarfs as akd
    import robustness as rb
    from scipy import stats as _st
    from utils_EE import compute_the_null, partitioning_function

    nXs = [int(x) for x in nXs]
    K_M = int(K_M if K_M else 0.05 * min(min(nXs), nY))
    p_list = [nY / (nY + nx) for nx in nXs]
    nXp = int(sum(nXs)); p_pool = nY / (nY + nXp)
    K_Mp = min(K_M, int(0.05 * min(nXp, nY)))
    nulls = _nulls(list(p_list), K_M, null_N, n_jobs)
    with rb._q():
        null_pool = compute_the_null(p=p_pool, K_M=K_Mp, N=null_N)
    Gstar, snm = _bootstrap_gstar(nY, nXs, nulls, p_list, K_M, p_ext, n_boot, n_jobs)

    def run_one(rng, sig_n, sig_w):
        """One realisation -> (stacked S/rootB on the signal, pooled S/rootB, truth)."""
        ctr = np.full(d, 0.5)
        if sig_n:
            Y = np.vstack([rng.random((nY - sig_n, d)),
                           np.clip(ctr + rng.normal(0, sig_w, (sig_n, d)), 0, 1)])
        else:
            Y = rng.random((nY, d))
        truth = np.zeros(nY, bool)
        if sig_n:
            truth[nY - sig_n:] = True
        X = [_bumpy(rng, nx, d, n_bumps, bump_frac, bump_w) for nx in nXs]
        with rb._q():
            out = akd.compute_Gamma_i_multi_ref(X_refs=X, Y=Y, stats_null_multi=snm,
                                                K_M=K_M, p_ext=p_ext, n_jobs=n_jobs,
                                                Z=2.65)
            G = np.asarray(out["Gamma_i"], float)
            cl = akd.srootB_for_Gamma_clusters_from_Brefs(
                EE_books=out["EE_books"], SrootB_by_ref=out["SrootB_by_ref"],
                B_by_ref=out["B_by_ref"], Y=Y, A_Gamma_mask=(G > Gstar),
                overlap_min=1, use_cluster_key="Repechaged", weight_mode="overlap",
                dbscan_eps=1.0, dbscan_min_samples=5, cluster_cols=tuple(range(d)))
        # best S/rootB among clusters overlapping the signal (or any, if no signal)
        def best(clusters, stats_lookup):
            vals = []
            for gid, ix in clusters.items():
                ix = np.asarray(ix, int)
                if sig_n and not (ix.size and truth[ix].sum() > 0):
                    continue
                v = stats_lookup(gid)
                if np.isfinite(v):
                    vals.append(v)
            return max(vals, default=0.0)
        s_val = best(cl["gamma_clusters"],
                     lambda g: float(cl["cluster_stats"][int(g)]["s_over_rootB"]))
        p_val = 0.0
        with rb._q():
            try:
                Xp = np.vstack(X)
                rdp, _ = EagleEye.Soar(Xp, Y, K_M=K_Mp, p_ext=p_ext, n_jobs=n_jobs,
                                       stats_null=null_pool, result_dict_in={},
                                       do_IDE=True)
                cp = partitioning_function(Xp, Y, rdp, p_ext=p_ext, Z=2.65)
                bk = EagleEye.Repechage(Xp, Y, rdp, cp, p_ext=p_ext)
                sr = EagleEye.S_rootB_estimate_Y_overdensities(rdp, bk)["s/root(B)"]
                vals = []
                for a, v in bk["Y_OVER_clusters"].items():
                    rep = np.asarray(v.get("Repechaged", []), int)
                    if sig_n and not (rep.size and truth[rep].sum() > 0):
                        continue
                    q = float(sr.get(a, np.nan))
                    if np.isfinite(q):
                        vals.append(q)
                p_val = max(vals, default=0.0)
            except Exception:
                pass
        return s_val, p_val

    # ---- 1) calibrate the pooled threshold to the stacked false-alarm rate ----
    fa_s, fa_p = [], []
    for t in range(n_fa_trials):
        a, b = run_one(np.random.default_rng(seed + 91193 * t), 0, 0.0)
        fa_s.append(a); fa_p.append(b)
    fa_s, fa_p = np.asarray(fa_s), np.asarray(fa_p)
    rate_s = float((fa_s >= srb_thresh_stacked).mean())
    if rate_s <= 0:
        thr_p = float(fa_p.max()) * 1.000001 + 1e-9   # match a zero false-alarm rate
    else:
        thr_p = float(np.quantile(fa_p, 1 - rate_s))
    rate_p = float((fa_p >= thr_p).mean())
    if verbose:
        print(f"calibration on {n_fa_trials} signal-free trials:")
        print(f"  stacked threshold {srb_thresh_stacked:.2f} -> false-alarm rate "
              f"{rate_s:.3f}")
        print(f"  matched pooled threshold {thr_p:.2f} -> false-alarm rate "
              f"{rate_p:.3f}")
        print(f"  (signal-free max S/rootB: stacked {fa_s.max():.2f}, "
              f"pooled {fa_p.max():.2f})\n")

    # ---- 2) paired detections on the SAME realisations ----
    rows = []
    for sig_w in sig_ws:
        for sig_n in sig_ns:
            b = so = po = ne = 0
            for t in range(n_trials):
                rng = np.random.default_rng(seed + 7919 * sig_n + 104729 * t
                                            + int(sig_w * 1e4))
                sv, pv = run_one(rng, sig_n, sig_w)
                ds, dp = sv >= srb_thresh_stacked, pv >= thr_p
                b += ds and dp; so += ds and not dp
                po += dp and not ds; ne += (not ds) and (not dp)
            n_disc = so + po
            pmc = (float(_st.binomtest(so, n_disc, 0.5).pvalue) if n_disc else np.nan)
            rows.append(dict(sig_w=sig_w, sig_n=sig_n, both=b, stacked_only=so,
                             pooled_only=po, neither=ne,
                             det_stacked=(b + so) / n_trials,
                             det_pooled=(b + po) / n_trials, mcnemar_p=pmc))
            if verbose:
                r = rows[-1]
                print(f"  w={sig_w:.3f} N={sig_n:3d}   both={b:3d} "
                      f"stacked-only={so:3d} pooled-only={po:3d} neither={ne:3d}"
                      f"   det {r['det_stacked']:.2f}/{r['det_pooled']:.2f}"
                      f"   McNemar p={pmc:.3f}" if np.isfinite(pmc) else
                      f"  w={sig_w:.3f} N={sig_n:3d}   both={b:3d} "
                      f"stacked-only={so:3d} pooled-only={po:3d} neither={ne:3d}",
                      flush=True)
    return pd.DataFrame(rows), dict(thr_stacked=srb_thresh_stacked, thr_pooled=thr_p,
                                    fa_rate=rate_s, fa_stacked=fa_s, fa_pooled=fa_p)


# ----------------------------------------------------------------------
# False alarms for the 8-fold ROBUSTNESS SCAN (persistence, not one config)
# ----------------------------------------------------------------------
def blank_field(r, *, pos_bw=0.35, pm_bw=1.0, seed=0):
    """A signal-free star table with the real field's density, gradient and PMs.

    Smoothed bootstrap: resample the real rows and jitter them by a kernel wider than
    any clump but narrower than the field's smooth structure. Below the bandwidth the
    generating density is flat BY CONSTRUCTION, so anything the scan then finds is
    manufactured -- while the large-scale gradient, the tile-to-tile density spread
    and the PM distribution all survive, which a uniform mock would destroy.
    """
    rng = np.random.default_rng(seed)
    n = len(np.asarray(r["ra"]))
    idx = rng.integers(0, n, n)
    cd = np.cos(np.deg2rad(float(np.median(np.asarray(r["dec"], float)))))
    out = {k: (np.asarray(v)[idx] if hasattr(v, "__len__") and len(v) == n else v)
           for k, v in r.items()}
    out["ra"] = np.asarray(r["ra"], float)[idx] + rng.normal(0, pos_bw, n) / cd
    out["dec"] = np.asarray(r["dec"], float)[idx] + rng.normal(0, pos_bw, n)
    out["pmra"] = np.asarray(r["pmra"], float)[idx] + rng.normal(0, pm_bw, n)
    out["pmdec"] = np.asarray(r["pmdec"], float)[idx] + rng.normal(0, pm_bw, n)
    return out


def _persistence(entries, match_deg):
    """Largest set of DIFFERENT configurations whose clusters share a sky position.

    entries: list of (config_label, ra, dec, s_over_rootB). Greedy: seed on each
    cluster in turn, count how many other configurations have one within match_deg,
    and keep the best. Returns (count, centre, [s_over_rootB of the members]).
    """
    best = (0, None, [])
    for ra0, dec0 in [(e[1], e[2]) for e in entries]:
        cd = np.cos(np.deg2rad(dec0))
        seen, vals = {}, []
        for lab, ra, dec, srb in entries:
            if np.hypot((ra - ra0) * cd, dec - dec0) <= match_deg:
                if lab not in seen or srb > seen[lab]:
                    seen[lab] = srb
        if len(seen) > best[0]:
            best = (len(seen), (ra0, dec0), sorted(seen.values(), reverse=True))
    return best


def scan_false_alarm(r, anomaly, h0, *, n_trials=10, match_deg=0.25, pos_bw=0.35,
                     pm_bw=1.0, seed=0, scan_kw=None, verbose=True):
    """How often does the FULL 8-configuration robustness scan invent a PERSISTENT
    anomaly out of nothing?

    WHY THIS AND NOT `false_alarm_test`. That one asks how often a single
    configuration produces a cluster. The scan's whole claim is that a real feature
    survives regridding, so the quantity to calibrate is how often SEVERAL
    configurations agree on the same patch of sky by chance. A per-configuration
    false-alarm rate of 10% does not imply a 10% rate for "5 of 8 configurations
    agreeing within 0.25 deg" -- it could be far lower, or, because the
    configurations share most of their stars, far higher.

    THE BOX-SIZE PROBLEM, WHICH THIS DOES NOT SOLVE FOR YOU. Shrinking the tile
    lowers nY, hence the locality bound on K_M, hence the achievable Upsilon -- so the
    same object reports a smaller S/sqrt(B) in the smaller configurations, and a
    single threshold applied across all eight is biased against them. This function
    therefore records S/sqrt(B) PER CONFIGURATION under the null. Use those
    distributions to set a per-configuration threshold (e.g. each config's own 90th
    percentile) rather than one number for all eight; `summary['per_config']` is the
    table to read that off.

    Returns (summary dict, per-trial records). Signal-free by construction, so every
    persistent group found here is a false alarm.
    """
    import robustness as rb
    scan_kw = dict(scan_kw or {})
    scan_kw.setdefault("verbose", False)
    recs = []
    for t in range(n_trials):
        rb_field = blank_field(r, pos_bw=pos_bw, pm_bw=pm_bw, seed=seed + 101 * t)
        try:
            summ, runs, cfgs = rb.scan(rb_field, anomaly, h0=h0, **scan_kw)
        except Exception as e:
            if verbose:
                print(f"  trial {t+1}: scan FAILED {type(e).__name__}: {e}", flush=True)
            continue
        entries = []
        for lab, run in runs.items():
            ti = np.asarray(run["parts"]["test_idx"], int)
            for gid, ix in (run.get("gamma_clusters") or {}).items():
                srb = float(run["cluster_stats"][int(gid)]["s_over_rootB"])
                if not np.isfinite(srb):
                    continue
                ii = ti[np.asarray(ix, int)]
                entries.append((lab,
                                float(np.mean(np.asarray(rb_field["ra"], float)[ii])),
                                float(np.mean(np.asarray(rb_field["dec"], float)[ii])),
                                srb))
        k, centre, vals = _persistence(entries, match_deg) if entries else (0, None, [])
        recs.append(dict(trial=t, n_clusters=len(entries), persistence=k,
                         centre=centre, srb_members=vals,
                         srb_min_of_group=(min(vals) if vals else np.nan),
                         per_config={lab: max([e[3] for e in entries if e[0] == lab],
                                              default=0.0) for lab in runs}))
        if verbose:
            print(f"  trial {t+1}/{n_trials}: {len(entries)} clusters over "
                  f"{len(runs)} configs, best persistence {k}"
                  + (f", min S/rootB in that group {min(vals):.2f}" if vals else ""),
                  flush=True)
    if not recs:
        return dict(n_trials=0), recs

    pers = np.array([x["persistence"] for x in recs])
    labs = sorted({l for x in recs for l in x["per_config"]})
    per_cfg = pd.DataFrame([{**{"trial": x["trial"]},
                             **{l: x["per_config"].get(l, np.nan) for l in labs}}
                            for x in recs])
    rows = []
    for k in range(1, max(int(pers.max()), 1) + 1):
        sel = [x for x in recs if x["persistence"] >= k]
        # the weakest member of the group is what a "k configs above T" cut must beat
        mins = [x["srb_min_of_group"] for x in sel
                if np.isfinite(x["srb_min_of_group"])]
        rows.append(dict(k_configs=k, rate=float((pers >= k).mean()),
                         srb_med=float(np.median(mins)) if mins else np.nan,
                         srb_q90=float(np.quantile(mins, .9)) if mins else np.nan,
                         srb_max=float(np.max(mins)) if mins else np.nan))
    summary = dict(n_trials=len(recs), match_deg=match_deg,
                   persistence=pd.DataFrame(rows), per_config=per_cfg,
                   per_config_q90={l: float(np.nanquantile(per_cfg[l], .9))
                                   for l in labs})
    if verbose:
        print(f"\n  === {len(recs)} signal-free scans, match radius {match_deg} deg ===")
        print("  " + summary["persistence"].round(3).to_string(index=False)
              .replace("\n", "\n  "))
        print("\n  per-configuration 90th percentile of S/rootB under the null")
        print("  (use these as PER-CONFIG thresholds -- a single number across all")
        print("   eight is biased against the smaller boxes):")
        for l, v in summary["per_config_q90"].items():
            print(f"    {l:<14} {v:6.2f}")
        print(f"\n  rate cannot resolve below {1/len(recs):.3f} with {len(recs)} trials")
    return summary, recs

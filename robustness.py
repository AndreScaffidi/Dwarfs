"""Automated robustness scan for an unsupervised EagleEye detection.

Given one nominal run, characterise every Gamma cluster (centroid + extent + identity),
pick a target anomaly, and re-run the pipeline on a small battery of perturbed grids
anchored on the anomaly itself. No catalogue is required at any point: the anomaly's
own centroid and 90th-percentile radius replace the catalogue position and half-light
radius, and configurations are matched by Gaia `source_id`, never by index or position.

THREE CONFOUNDS ARE PINNED, because otherwise a box-size scan is uninterpretable:

1. The metric. `make_Xrefs_and_Y_from_window_3x3` refits median/MAD per configuration.
   The positional MADs scale linearly with the tile half-width h (measured: mad_dRA/h
   = 0.29, mad_dDec/h = 0.50, constant to <3% over h = 0.83..2.17) while the PM MADs
   are pinned near 1.42 by the +/-5 mas/yr cut. So the position:PM weighting of the
   Euclidean metric changes by ~2.6x across that range and you are not measuring the
   same distance function at different box sizes. `freeze_scale=True` fits the scaling
   once on the nominal configuration and reuses it everywhere.

2. K_M. The method requires K_M <= 0.05*min(nX, nY). Shrinking the tile shrinks nY, so
   a fixed K_M silently violates the bound (at nY = 1141 the bound is 57, not 100).
   `km_mode="bounded"` clips it; `"proportional"` holds K_M/nY fixed instead.

3. DBSCAN eps. It is expressed in scaled units, so its physical radius tracks the MAD.
   Freezing the scale makes a fixed eps mean a fixed angular size automatically.
"""
from __future__ import annotations

import time
import numpy as np
import pandas as pd

import analyze_known_dwarfs as akd
from sklearn.cluster import DBSCAN


# ----------------------------------------------------------------------
# 1) Describe what was found
# ----------------------------------------------------------------------
def _centroid(ra, dec):
    dec_c = float(np.nanmean(dec))
    w = np.cos(np.deg2rad(dec_c))
    ra_c = float(np.nanmean(ra * w) / w) if w != 0 else float(np.nanmean(ra))
    return ra_c, dec_c


def characterise_anomalies(run, r, known_ids=None, pct=90.0):
    """One row per Gamma cluster: identity, centroid, extent, strength.

    `known_ids` maps a system name -> set of source_ids (e.g. built from Battaglia).
    Pass None to stay fully unsupervised; every cluster is then 'unmatched'.
    """
    parts = run["parts"]
    test_idx = np.asarray(parts["test_idx"], int)
    sid = np.asarray(r["source_id"], dtype="int64")[test_idx]
    ra = np.asarray(r["ra"], float)[test_idx]
    dec = np.asarray(r["dec"], float)[test_idx]
    pmra = np.asarray(r["pmra"], float)[test_idx]
    pmdec = np.asarray(r["pmdec"], float)[test_idx]

    rows = []
    for gid in sorted(run["gamma_clusters"]):
        ix = np.asarray(run["gamma_clusters"][gid], int)
        if ix.size == 0:
            continue
        ra_c, dec_c = _centroid(ra[ix], dec[ix])
        cosd = np.cos(np.deg2rad(dec_c))
        dra = akd.wrap_dra_deg(ra[ix], ra_c) * cosd
        ddec = dec[ix] - dec_c
        rad = np.hypot(dra, ddec)                      # sky radius, degrees
        pm_c = (float(np.nanmedian(pmra[ix])), float(np.nanmedian(pmdec[ix])))
        pmrad = np.hypot(pmra[ix] - pm_c[0], pmdec[ix] - pm_c[1])

        ids = set(int(s) for s in sid[ix])
        match, novl = "", 0
        if known_ids:
            for name, kids in known_ids.items():
                n = len(ids & kids)
                if n > novl:
                    match, novl = name, n

        cs = run["cluster_stats"][int(gid)]
        rows.append({
            "gid": int(gid), "n": int(ix.size),
            "ra_c": ra_c, "dec_c": dec_c, "pmra_c": pm_c[0], "pmdec_c": pm_c[1],
            f"r{int(pct)}_deg": float(np.percentile(rad, pct)),
            "r_max_deg": float(rad.max()),
            f"pm_r{int(pct)}": float(np.percentile(pmrad, pct)),
            "S": cs["S"], "B_weighted": cs["B_weighted"],
            "s_over_rootB": cs["s_over_rootB"],
            "matched_to": match, "n_matched": novl,
            "unmatched": novl == 0,
            "source_ids": ids,
        })
    return pd.DataFrame(rows)


def pick_target(anom_df, prefer_unmatched=True):
    """Default target: the strongest cluster that matches nothing known."""
    if anom_df.empty:
        raise ValueError("no Gamma clusters in this run -- nothing to scan")
    df = anom_df
    if prefer_unmatched and df["unmatched"].any():
        df = df[df["unmatched"]]
    return df.sort_values("s_over_rootB", ascending=False).iloc[0].to_dict()


# ----------------------------------------------------------------------
# 2) The perturbation battery
# ----------------------------------------------------------------------
def scan_configs(anomaly, h0, rho=0.6, scale_factors=(2 ** -0.5, 2 ** 0.5),
                 extent_key=None, min_margin=0.05):
    """Grids anchored on the anomaly. Offsets are RAW RA / Dec, as split_into_3x3 uses.

    rho is the fraction of the available room the anomaly is pushed toward the tile
    edge; rho=1 would place it exactly on the edge, which is where detection dies.
    """
    if extent_key is None:
        extent_key = next(k for k in anomaly if k.startswith("r") and k.endswith("_deg")
                          and k != "r_max_deg")
    rA = float(anomaly[extent_key])
    cosd = np.cos(np.deg2rad(float(anomaly["dec_c"])))

    def room(h):
        """How far the grid may slide before the anomaly reaches the tile edge."""
        return (max(h - rA / cosd - min_margin, 0.0),      # RA, raw degrees
                max(h - rA - min_margin, 0.0))             # Dec, degrees

    cfgs = [{"label": "nominal", "h": h0, "dra_raw": 0.0, "ddec": 0.0}]

    dra_room, ddec_room = room(h0)
    for sgn, nm in ((+1, "E"), (-1, "W")):
        cfgs.append({"label": f"shift{nm}", "h": h0,
                     "dra_raw": sgn * rho * dra_room, "ddec": 0.0})
    for sgn, nm in ((+1, "N"), (-1, "S")):
        cfgs.append({"label": f"shift{nm}", "h": h0,
                     "dra_raw": 0.0, "ddec": sgn * rho * ddec_room})

    for f in scale_factors:
        cfgs.append({"label": f"scale x{f:.2f}", "h": h0 * f,
                     "dra_raw": 0.0, "ddec": 0.0})

    d = rho / np.sqrt(2)
    cfgs.append({"label": "diagonal", "h": h0,
                 "dra_raw": d * dra_room, "ddec": d * ddec_room})

    for c in cfgs:
        c["dra_sky"] = c["dra_raw"] * cosd
    return cfgs


def plan_query(h0, scale_factors=(1 / 1.3, 1.3), rho=0.6, margin=0.3,
               grid_dra_raw=0.0, grid_ddec=0.0):
    """Half-widths (raw RA, Dec) about the DISCOVERY grid centre that are guaranteed
    to cover every scan grid -- whatever the anomaly turns out to be.

    Lets you make ONE query up front instead of discovering mid-scan that the outer
    reference tiles fall off the data. The bound: the anomaly must lie somewhere in
    the discovery Y tile, so it is at most h0 from the discovery centre; a scan grid
    is centred on the anomaly, spans 3h, and may be shifted by up to rho*h, with
    h = h0 * max(scale_factors). Hence

        half = h0 * (1 + s_max * (3 + rho)) + margin

    Returns (ra_half_raw, dec_half). Both are RAW degrees, matching split_into_3x3.
    """
    s_max = max(scale_factors + (1.0,))
    reach = h0 * (1.0 + s_max * (3.0 + rho)) + margin
    return reach + abs(grid_dra_raw), reach + abs(grid_ddec)


def estimate_rows(ra_half, dec_half, dec_c, density_per_sqdeg=760.0):
    """Rough row count for a query box, for sanity-checking before you launch it."""
    area = (2 * ra_half * np.cos(np.deg2rad(dec_c))) * (2 * dec_half)
    return area, int(area * density_per_sqdeg)


def required_query_halfwidths(anomaly, cfgs, margin=0.2):
    ra_half = max(3 * c["h"] + abs(c["dra_raw"]) for c in cfgs) + margin
    dec_half = max(3 * c["h"] + abs(c["ddec"]) for c in cfgs) + margin
    return ra_half, dec_half


# ----------------------------------------------------------------------
# 2b) Repairing references contaminated by a localised overdensity
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Output verbosity
# ----------------------------------------------------------------------
# 0  silent
# 1  one summary line per run              <- default
# 2  + per-reference equalisation detail
# 3  + EagleEye's own chatter (Soar / KNN / Repechage / Math objects)
VERBOSITY = 1
SHOW_CONTRIBUTIONS = True   # per-reference Li&Ma line under every run_summary


def _q(level=3):
    """Swallow EagleEye output unless VERBOSITY has reached `level`."""
    import ee_parallel as ep
    return ep.quiet_ee(enabled=VERBOSITY < level)


def run_summary(run, label="", t=None):
    """The one line worth seeing per run: geometry, cleaning, threshold, result."""
    cfg = run.get("cfg", {})
    eq = run.get("eq_summary")
    n_rep = int(eq["repaired"].sum()) if eq is not None else 0
    n_cut = int(eq["n_cut"].sum()) if eq is not None else 0
    cl = run.get("gamma_clusters", {}) or {}
    srb = max((float(run["cluster_stats"][g]["s_over_rootB"])
               for g in cl if np.isfinite(run["cluster_stats"][g]["s_over_rootB"])),
              default=float("nan"))
    parts = [f"{label:<14}" if label else "",
             f"h={run['grid']['cell_h']:.3f}",
             f"nY={run['nY']:5d}",
             f"kM={cfg.get('kM','?')}",
             f"eq {n_rep}/{len(run['nXs'])} refs ({n_cut} cut)" if eq is not None else "eq off",
             f"G*={run['Gamma_star']:.1f}",
             f"|A|={int(np.sum(run['A_Gamma_mask']))}",
             f"clusters={len(cl)}",
             (f"best S/rootB={srb:.2f}" if np.isfinite(srb) else "no cluster")]
    if t is not None:
        parts.append(f"{t:.0f}s")
    print("  " + " | ".join(x for x in parts if x), flush=True)
    # Per-reference accounting. |A_Gamma|/sqrt(B_w) hides two things that matter:
    # how many of the R references actually produced a cluster here, and that EE's
    # own z omits the control-sample variance. Both are printed per cluster.
    if SHOW_CONTRIBUTIONS and cl:
        try:
            import wake_test as _wt
            for _g in sorted(cl):
                print("      " + _wt.contribution_line(run, _g), flush=True)
        except Exception as _e:
            print(f"      (contribution report unavailable: {type(_e).__name__})")


# The X-side statistic saturates. Upsilon is -log of a binomial right-tail p-value on
# the composition of a K-neighbourhood, so with every one of the K_M-1 usable
# neighbours drawn from X it attains at most
#     Upsilon_max^- = -(K_M - 1) * ln(1 - p_hat),      p_hat = nY / (nY + nX_j)
# INDEPENDENT of how large the contaminant is. A reference carrying a big globular
# has nX_j >> nY, so p_hat -> 0, the ceiling collapses toward zero, and it can fall
# BELOW the detection threshold Upsilon*^- = quantile(stats_null[1-p_hat], 1-p_ext).
# When that happens nothing can ever be flagged and the pass returns "no clusters" --
# indistinguishable from a clean reference. Measured on M3 in a Bootes III reference
# at DEGSEP=4, K_M=50: nX_j=24040, nY=2684, Upsilon_max = Upsilon* = 5.19 exactly,
# X^+ empty, the worst contaminant in the field silently ignored.
#
# The fix is to run the EQUALISATION at the largest K_M the locality bound allows,
# K_M^eq = floor(0.05 * min(nX_j, nY)), recomputed per reference and per pass, and to
# RAISE rather than return empty when even that leaves no headroom. Equalisation is a
# cleaning step, not the measurement: the final stacked Gamma still uses the K_M the
# user set. Note K_M^eq is pinned by nY for as long as nX_j > nY, which is the entire
# time it matters -- the per-pass recomputation only bites once a reference has been
# cut below the test set.
KM_EQ_FLOOR = 25          # KSTAR_RANGE is range(20, K_M); below this there is no test


def resolve_km(kM, nXs, nY, *, floor=KM_EQ_FLOOR):
    """The SINGLE K_M for the stacked analysis. Returns (kM_eff, bound, mode).

    kM = "auto" -> the locality bound floor(0.05 * min(min_j nX_j, nY)); an int is
    taken as given (clip it separately if you want km_mode="bounded" behaviour).

    WHY ONE VALUE, AND WHY THE SMALLEST REFERENCE. Gamma_i = sum_j Upsilon_i^(j), so
    every term has to be the same statistic. Per-reference K_M would give each term a
    different null AND a different ceiling -(K_M-1)*ln(1-p_hat_j), so the sum would be
    weighted by tile size rather than by evidence. The bound is therefore taken over
    the SMALLEST reference: K_M must be local in every X_j, not on average.

    WHY 0.05*min IS THE RIGHT SORT OF BOUND. Upsilon's null assumes the X:Y ratio is
    constant inside the K-neighbourhood. It is not -- two sky tiles have a slowly
    varying ratio -- and the neighbourhood radius grows like k^(1/d), so the bigger
    K_M is the more of that variation it swallows and the more background gradient it
    reports as signal. Measured on a synthetic field with a mild ratio gradient and NO
    anomaly, false-positive rate as a multiple of the nominal p_ext:

        K_M      25     50    100    300    600
        d = 2   2.0x   6.7x  12.5x  46.0x  118x
        d = 4   1.3x   1.8x   2.0x   2.2x  6.8x

    The features here are 4D, where the bound is conservative -- which is why "auto"
    is a reasonable default rather than a reckless one. It is still a proxy, not a
    measurement: `km_gradient_check` measures the inflation on the actual field.

    RESOLVE AFTER EQUALISATION. Equalisation shrinks the references, so calling this
    on the raw nXs would bound K_M by a contaminated tile's size.
    """
    bound = int(0.05 * min(min(nXs), nY))
    if isinstance(kM, str):
        if kM != "auto":
            raise ValueError(f"kM must be an int or 'auto', got {kM!r}")
        return max(floor, bound), bound, "auto"
    return int(kM), bound, "fixed"


def _nanagg(fn, vals):
    """np.nanmax/nanmin, but quiet and NaN-valued on an empty or all-NaN input."""
    a = np.asarray(vals, float)
    a = a[np.isfinite(a)]
    return float(fn(a)) if a.size else np.nan


def _equalise_ref_task(args):
    """Equalise ONE reference: the whole pass loop, in its own process.

    Split out because the 8 references are independent and were the last
    serial bottleneck: with ref_workers>1 the caller drops the inner kNN
    n_jobs to 1, which made a SERIAL equalisation 8x slower than before the
    parallelism was added. Top-level so ProcessPoolExecutor can pickle it.
    """
    import EagleEye
    from utils_EE import compute_the_null, partitioning_function
    import ee_parallel as _ep
    (j, Xj0, Y, kM, p_ext, null_N, dpa_Z, n_jobs, srb_threshold,
     max_passes, km_eq, headroom_min, verbose) = args
    _tl = _ep._limit_threads(1)          # see the thread trap note in ee_parallel
    _tl.__enter__()
    nY = len(Y)
    Xj0 = np.asarray(Xj0)
    keep = np.ones(len(Xj0), bool)
    n_pass, cut_total, hits = 0, 0, []
    last = {}
    Xj0 = np.asarray(Xj0)
    keep = np.ones(len(Xj0), bool)
    n_pass, cut_total, hits = 0, 0, []
    last = {}                       # diagnostics of the most recent pass

    for it in range(max_passes):
        Xj = Xj0[keep]
        nXj = len(Xj)
        p = nY / (nY + nXj)

        if km_eq == "auto":
            kM_pass = int(0.05 * min(nXj, nY))
        elif km_eq is None:
            kM_pass = int(kM)
        else:
            kM_pass = int(km_eq)
        kM_pass = max(kM_pass, KM_EQ_FLOOR)

        sn = compute_the_null(p=p, K_M=kM_pass, N=null_N)

        # Can this configuration flag anything at all? Compare the X-side ceiling
        # against the X-side threshold, both at the K_M actually about to be used.
        ups_star = float(np.quantile(np.asarray(sn[1 - p], float), 1 - p_ext))
        ups_max = -(kM_pass - 1) * np.log1p(-p)
        head = ups_max - ups_star
        if head <= headroom_min:
            # sqrt(20*ln(1/p_ext)*nX) is where the small-p approximation
            # 0.05*nY * (nY/nX) > ln(1/p_ext) turns over. Indicative, not exact.
            nY_need = int(np.ceil(np.sqrt(20.0 * np.log(1.0 / p_ext) * nXj)))
            raise RuntimeError(
                f"reference {j} pass {it+1}: the X-side test is saturated -- no "
                f"K_M within the locality bound can flag anything.\n"
                f"  nX_j={nXj}  nY={nY}  p_hat={p:.4f}  p_ext={p_ext:g}\n"
                f"  K_M^eq={kM_pass} (locality bound 0.05*min(nX,nY)"
                f"={int(0.05*min(nXj, nY))}, floor {KM_EQ_FLOOR})\n"
                f"  Upsilon_max={ups_max:.2f} vs Upsilon*={ups_star:.2f}"
                f"  -> headroom {head:+.2f} (need > {headroom_min:g})\n"
                f"  The reference is too large relative to the test set. Remedies, "
                f"in order of preference: enlarge the test tile (needs nY of order "
                f"{nY_need} here), relax p_ext, or exclude this placement.")

        last = {"kM_eq": kM_pass, "p_hat": p, "ups_max": ups_max,
                "ups_star": ups_star, "headroom": head}

        with _q():
            rd, _ = EagleEye.Soar(Xj, Y, K_M=kM_pass, p_ext=p_ext, n_jobs=n_jobs,
                                  stats_null=sn, result_dict_in={}, do_IDE=True)
        try:
            with _q():
                clusters = partitioning_function(Xj, Y, rd, p_ext=p_ext, Z=dpa_Z)
                book = EagleEye.Repechage(Xj, Y, rd, clusters, p_ext=p_ext)
        except Exception as e:                 # no X-side clusters at all
            if verbose and VERBOSITY >= 2:
                print(f"    ref {j} pass {it+1}: no X-side clusters "
                      f"({type(e).__name__})  [K_M={kM_pass}, headroom {head:+.2f}]")
            break

        # Significance and purity separately and defensively: a failure of one must
        # not discard the other, and neither must abort the pass.
        def _safe(fn, key):
            try:
                return fn(rd, book)[key]
            except Exception:
                return {}
        srb = _safe(EagleEye.S_rootB_estimate_X_overdensities, "s/root(B)")
        pur = _safe(EagleEye.S_SB_estimate_X_overdensities, "Purity")

        # THE ESTIMATOR HAS A VALIDITY BOUNDARY. Eq.9 uses
        #     B_hat = lenBo * (nY - lenWo) / (nX - lenWu)
        # so once a cluster's excess mass lenWo exceeds |Y| the numerator goes
        # negative, B_hat < 0, and S/sqrt(B) is NaN. That is precisely what a large
        # globular in a reference produces -- M3 in a Bootes III reference gave
        # lenWo=20820 against nY=5008, B_hat=-28, S/sqrt(B)=NaN -- so gating on a
        # finite S/sqrt(B) alone silently skips the worst contaminant there is.
        # Treat "excess mass larger than the whole opposing sample" as significant
        # by construction. This is the estimator's own breakdown condition, not a
        # cardinality heuristic.
        _nY = len(np.asarray(rd["Upsilon_i_Y"]))

        gated = []
        for a, v in book["X_OVER_clusters"].items():
            pr = np.asarray(v.get("Pruned", []), dtype=int)
            if pr.size == 0:
                continue
            try:
                s = float(srb.get(a, np.nan))
            except (TypeError, ValueError):
                s = np.nan
            oversize = pr.size > _nY          # B_hat < 0: estimator out of range
            if (np.isfinite(s) and s > srb_threshold) or oversize:
                try:
                    pv = float(pur.get(a, np.nan))
                except (TypeError, ValueError):
                    pv = np.nan
                gated.append((a, pr, s, pv))
        if not gated:
            break

        drop_local = np.unique(np.concatenate([g[1] for g in gated]))
        orig = np.where(keep)[0][drop_local]
        keep[orig] = False
        n_pass, cut_total = it + 1, cut_total + drop_local.size
        hits += [(it + 1, a, pr.size, s, pu, kM_pass, head)
                 for a, pr, s, pu in gated]
        if verbose and VERBOSITY >= 2:
            for a, pr, s, pu in gated:
                _s = f"{s:.1f}" if np.isfinite(s) else "undefined (B_hat<=0)"
                print(f"    ref {j} pass {it+1}: cluster {a} "
                      f"S/rootB={_s} purity={pu:.3f} -> cut {pr.size} "
                      f"[K_M={kM_pass}, headroom {head:+.2f}]")

    _tl.__exit__(None, None, None)
    return j, keep, hits, last, n_pass, cut_total


def equalise_references(Y, X_refs, *, kM, p_ext, null_N=200_000, dpa_Z=2.65,
                        n_jobs=8, srb_threshold=5.0, max_passes=3,
                        km_eq="auto", headroom_min=0.5, ref_workers=1, verbose=True):
    """Cut localised overdensities out of the reference tiles, in EagleEye's own terms.

    WHY. Upsilon tests the local Y-fraction against p_hat = nY/(nY+nX_j), a GLOBAL
    ratio. A reference that is merely denser is handled correctly -- the local ratio
    scales with it and p_hat follows. But a *concentrated* clump (a globular cluster,
    say) inflates nX_j while leaving the local density at a random field position
    untouched, so p_hat and the local ratio stop corresponding and EVERY point of Y
    looks overdense. Measured on NGC 1261 in a Reticulum II reference tile: p_hat
    0.524 -> 0.389, a +4.66 pedestal added to Upsilon for all 7911 test stars.

    WHAT IS REMOVED. Per contaminated cluster, `EE_book['X_OVER_clusters'][a]['Pruned']`
    -- the IDE-removed excess mass for that cluster, i.e. X_Pruned intersected with it.
    NOT the repechage set: repechage deliberately re-admits the surrounding region and
    so contains background by construction (paper, M4). Removing only the excess leaves
    the background in place, so the repaired reference still samples B_0 there.

    STOPPING. Iterate until no X-side cluster exceeds `srb_threshold`, i.e. until
    EagleEye no longer finds anything in the reference. No cardinality target is used:
    references of differing size are exactly what the stacking handles, and a tile that
    is legitimately denser should stay denser.

    K_M DURING EQUALISATION (`km_eq`). "auto" (default) uses the locality ceiling
    floor(0.05*min(nX_j, nY)) per reference per pass; None reuses the caller's `kM`;
    an int pins it. Whatever is chosen, the pass first checks it can detect anything
    at all -- see the note above the function -- and raises if it cannot. Because the
    threshold enters that check, the equalised references depend on `p_ext`; any cache
    key over this function must include it.

    COST. One Soar + clustering + repechage per reference per pass. This duplicates the
    first Soar that compute_Gamma_i_multi_ref will do; call it before that.

    Returns (X_refs_eq, info) where info = {"summary": per-reference DataFrame,
    "detail": per-cluster DataFrame (pass, S/rootB, purity, n_cut), "removed_idx":
    list of arrays of row indices into the ORIGINAL X_j}. The caller must recompute
    p_list, the per-reference nulls and the bootstrap from the new cardinalities.
    """
    import EagleEye
    from utils_EE import compute_the_null, partitioning_function

    nY = len(Y)
    out_refs, rows, detail, removed_idx = [], [], [], []

    _args = [(j, np.asarray(X_refs[j]), Y, kM, p_ext, null_N, dpa_Z, n_jobs,
              srb_threshold, max_passes, km_eq, headroom_min, verbose)
             for j in range(len(X_refs))]
    if ref_workers > 1 and len(X_refs) > 1:
        import ee_parallel as _ep
        _res = {}
        for _j, _k, _h, _l, _np_, _ct in _ep.pmap(
                _equalise_ref_task, _args, min(int(ref_workers), len(X_refs))):
            _res[_j] = (_k, _h, _l, _np_, _ct)
        _out = [_res[j] for j in range(len(X_refs))]
    else:
        _out = [_equalise_ref_task(a)[1:] for a in _args]

    for j, (keep, hits, last, n_pass, cut_total) in enumerate(_out):
        Xj0 = np.asarray(X_refs[j])
        out_refs.append(Xj0[keep])
        removed_idx.append(np.where(~keep)[0])
        for (ip, a, npt, sv, pu, km_p, hd) in hits:
            detail.append({"ref": j, "pass": ip, "cluster": int(a), "n_cut": int(npt),
                           "s_over_rootB": float(sv), "purity": float(pu),
                           "kM_eq": int(km_p), "headroom": float(hd)})
        rows.append({"ref": j, "n_before": len(Xj0), "n_after": int(keep.sum()),
                     "n_cut": int(cut_total), "passes": n_pass,
                     "repaired": cut_total > 0,
                     "max_srb": _nanagg(np.nanmax, [h[3] for h in hits]),
                     "min_purity": _nanagg(np.nanmin, [h[4] for h in hits]),
                     "kM_eq_last": int(last.get("kM_eq", 0)) or np.nan,
                     "p_hat_last": last.get("p_hat", np.nan),
                     "headroom_last": last.get("headroom", np.nan)})
        if verbose and VERBOSITY >= 2 and cut_total:
            print(f"  ref {j}: {len(Xj0)} -> {int(keep.sum())} "
                  f"({cut_total} cut in {n_pass} pass(es))")

    rep = pd.DataFrame(rows)
    med = float(np.median(rep["n_after"]))
    rep["ratio_after"] = rep["n_after"] / med          # reported, never gated on
    det = pd.DataFrame(detail, columns=["ref", "pass", "cluster", "n_cut",
                                        "s_over_rootB", "purity", "kM_eq",
                                        "headroom"])
    if verbose:
        n = int(rep.repaired.sum())
        _k = rep["kM_eq_last"].dropna()
        _kr = ("n/a" if not len(_k) else f"{int(_k.min())}"
               if _k.min() == _k.max() else f"{int(_k.min())}-{int(_k.max())}")
        print(f"  equalisation: {n}/{len(X_refs)} reference(s) repaired, "
              f"{int(rep.n_cut.sum())} points cut in total "
              f"(K_M^eq {_kr}, min headroom {rep['headroom_last'].min():+.2f})")
    return out_refs, {"summary": rep, "detail": det, "removed_idx": removed_idx}


# ----------------------------------------------------------------------
# 3) Run one configuration, with the confounds pinned
# ----------------------------------------------------------------------
def _build_window(r, ra0, dec0, h, fixed_scale=None):
    """akd.make_Xrefs_and_Y_from_window_3x3, but able to reuse a frozen med/MAD."""
    if fixed_scale is None:
        return akd.make_Xrefs_and_Y_from_window_3x3(
            r, ra0, dec0, h, fit_scaling_on="all", do_robust_scale=True)

    parts = akd.split_into_3x3(np.asarray(r["ra"], float),
                               np.asarray(r["dec"], float), ra0, dec0, h)
    Y_raw = akd.build_features_local_center(r, parts["test_idx"], ra0, dec0, dec0)
    X_raw = []
    for rid, idx in zip(parts["ref_ids"], parts["ref_idx_list"]):
        rc, dc = akd.cell_center_from_id(ra0, dec0, rid, h)
        X_raw.append(akd.build_features_local_center(r, idx, rc, dc, dec0))
    med, mad = fixed_scale["med"], fixed_scale["mad"]
    return (akd.robust_apply(Y_raw, med, mad),
            [akd.robust_apply(x, med, mad) for x in X_raw],
            parts, dict(fixed_scale))


def run_one(r, ra0, dec0, h, *, kM, p_ext, null_N, n_boot, dpa_Z,
            dbscan_eps, dbscan_min, n_jobs, seed,
            fixed_scale=None, km_mode="bounded", equalise=True, eq_srb=5.0,
            eq_max_passes=3, eq_km="auto", eq_headroom=0.5,
            ref_workers=1, boot_workers=1, verbose=True):
    t0 = time.time()

    # COVERAGE. split_into_3x3 never complains about running off the table -- it just
    # returns thinner outer tiles. A truncated reference is the worst possible failure
    # mode: Y stars near the corresponding edge are compared against a region with no
    # reference stars at all, which reads as an overwhelming overdensity pinned to the
    # tile boundary. (Observed: a -1 deg grid shift on a +/-5 deg table clipped 26% off
    # the western references and manufactured a 390-star "cluster" at S/rootB = 239, in
    # a patch that raw star counts show is slightly UNDERdense.) scan() checks this;
    # run_one must too, or a direct call silently reproduces it.
    _need = 3.0 * h
    _dra = akd.wrap_dra_deg(np.asarray(r["ra"], float), ra0)
    _ddec = np.asarray(r["dec"], float) - dec0
    _have_ra = min(float(-_dra.min()), float(_dra.max()))
    _have_dec = min(float(-_ddec.min()), float(_ddec.max()))
    if _have_ra + 1e-9 < _need or _have_dec + 1e-9 < _need:
        raise ValueError(
            f"star table does not cover the 3x3 about ({ra0:.4f}, {dec0:.4f}).\n"
            f"  need per side: RA +/-{_need:.3f}, Dec +/-{_need:.3f} (raw deg)\n"
            f"  have per side: RA +/-{_have_ra:.3f}, Dec +/-{_have_dec:.3f}\n"
            f"  The outer reference tiles would be TRUNCATED, which produces a "
            f"spurious high-significance overdensity along the affected Y edge. "
            f"Re-query with a larger box (see plan_query / AUTO_QUERY_PAD).")

    Y, X_refs, parts, scale = _build_window(r, ra0, dec0, h, fixed_scale)
    nY, nXs = len(Y), [len(x) for x in X_refs]
    if nY < 200 or min(nXs) < 200:
        raise RuntimeError(f"cells too small: nY={nY}, min nX={min(nXs)}")

    # Cut localised overdensities out of the references before anything downstream:
    # a concentrated clump deflates p_hat and puts a pedestal under every Upsilon.
    eq_info, eq_sids, eq_cells = None, set(), []
    if equalise:
        # eq_km="auto" runs the cleaning at the locality ceiling, NOT at the kM used
        # for the measurement below: a reference swamped by a globular saturates at the
        # user's kM and the contaminant would be missed entirely. See the note above
        # equalise_references.
        # kM may be "auto", but its bound depends on the POST-equalisation sizes --
        # circular. Resolve against the raw sizes for the (rare) km_eq=None path;
        # km_eq="auto" ignores it entirely and recomputes per reference per pass.
        _kM_pre = resolve_km(kM, nXs, nY)[0]
        X_refs, eq_info = equalise_references(
            Y, X_refs, kM=_kM_pre, p_ext=p_ext, null_N=null_N, dpa_Z=dpa_Z,
            n_jobs=(1 if ref_workers > 1 else n_jobs), srb_threshold=eq_srb, max_passes=eq_max_passes,
            km_eq=eq_km, headroom_min=eq_headroom,
            ref_workers=ref_workers, verbose=verbose)
        nXs = [len(x) for x in X_refs]
        sid_all = np.asarray(r["source_id"], dtype="int64")
        for j, rem in enumerate(eq_info["removed_idx"]):
            if rem.size:
                rows_r = np.asarray(parts["ref_idx_list"][j], int)[rem]
                eq_sids.update(int(x) for x in sid_all[rows_r])
                eq_cells.append(int(parts["ref_ids"][j]))

    _t0 = t0
    kM_eff, bound, _kmode = resolve_km(kM, nXs, nY)
    if _kmode == "fixed":                       # km_mode only applies to an explicit int
        if km_mode == "bounded":
            kM_eff = min(kM_eff, max(bound, KM_EQ_FLOOR))
        elif km_mode == "proportional":
            kM_eff = max(KM_EQ_FLOOR, bound)
    if verbose and (kM_eff != kM or _kmode == "auto"):
        print(f"    K_M {kM} -> {kM_eff}  (0.05*min(nX,nY) = {bound})")

    # Same saturation check as equalisation, but on the Y side and only a warning:
    # this is the user's measurement, not a cleaning step, so do not refuse to run.
    _pmin = nY / (nY + max(nXs))
    _hi = -(kM_eff - 1) * np.log(_pmin)
    if verbose and _hi < 20.0:
        print(f"    WARNING: Y-side ceiling is only Upsilon_max={_hi:.1f} "
              f"(K_M={kM_eff}, smallest p_hat={_pmin:.4f}). References are very "
              f"unbalanced; a real anomaly may be unable to clear Gamma*.")

    p_list = [nY / (nY + nX) for nX in nXs]
    snm = akd.compute_the_null_multi_refs(p_list=p_list, K_M=kM_eff, N=null_N,
                                          n_jobs=min(8, n_jobs))
    # Fanning the 8 references over processes means each Soar should NOT also grab
    # n_jobs kNN threads -- that product oversubscribes and runs slower than serial.
    _inner = 1 if ref_workers > 1 else n_jobs
    if ref_workers > 1:
        import ee_parallel as ep
        out = ep.compute_Gamma_i_multi_ref_par(X_refs, Y, snm, K_M=kM_eff, p_ext=p_ext,
                                               n_jobs=_inner, Z=dpa_Z,
                                               ref_workers=ref_workers)
    else:
        with _q():
            out = akd.compute_Gamma_i_multi_ref(X_refs=X_refs, Y=Y,
                                                stats_null_multi=snm, K_M=kM_eff,
                                                p_ext=p_ext, n_jobs=n_jobs, Z=dpa_Z)
    if boot_workers > 1:
        import ee_parallel as ep
        gnull = ep.bootstrap_gamma_null_uniform_par(
            X_refs, Y, snm, K_M=kM_eff, p_ext=p_ext, n_jobs=1, n_boot=n_boot,
            seed=seed, workers=boot_workers).reshape(-1)
    else:
        with _q():
            gnull = akd.bootstrap_gamma_null_uniform(
                X_refs=X_refs, Y=Y, stats_null_multi=snm, K_M=kM_eff, p_ext=p_ext,
                n_jobs=n_jobs, n_boot=n_boot, seed=seed).reshape(-1)
    gnull = gnull[np.isfinite(gnull)]
    gstar = float(np.quantile(gnull, 1 - p_ext))
    mask = out["Gamma_i"] > gstar

    cl = akd.srootB_for_Gamma_clusters_from_Brefs(
        EE_books=out["EE_books"], SrootB_by_ref=out["SrootB_by_ref"],
        B_by_ref=out["B_by_ref"], Y=Y, A_Gamma_mask=mask, overlap_min=1,
        use_cluster_key="Repechaged", weight_mode="overlap",
        dbscan_eps=dbscan_eps, dbscan_min_samples=dbscan_min, cluster_cols=(0, 1, 2, 3))

    run = {"Y": Y, "parts": parts, "scale": scale, "nY": nY, "nXs": nXs,
            "Gamma_i": out["Gamma_i"], "Upsilon_by_ref": out["Upsilon_by_ref"],
            # keep the bootstrap null: it is small (n_boot x nY) and without it the
            # Gamma-distribution plots have nothing to compare the test set against
            "Gamma_null": gnull, "Gamma_null_mat": gnull.reshape(1, -1),
            "Gamma_star": gstar, "A_Gamma_mask": mask,
            "gamma_clusters": cl["gamma_clusters"], "cluster_stats": cl["cluster_stats"],
            "labels_all": cl["labels_all"], "noise_idx": cl["noise_idx"],
            "grid": {"ra0": ra0, "dec0": dec0, "cell_h": h},
            "eq_summary": (eq_info["summary"] if eq_info else None),
            "eq_detail": (eq_info["detail"] if eq_info else None),
            "eq_removed_sids": eq_sids, "eq_repaired_cells": eq_cells,
            # kM is the MEASUREMENT K_M. Equalisation ran at its own (see eq_info
            # ["summary"]["kM_eq_last"]); the two are deliberately different.
            "cfg": {"kM": kM_eff, "kM_requested": kM, "eq_km": eq_km, "p_ext": p_ext,
                    "null_N": null_N, "n_boot": n_boot, "dpa_Z": dpa_Z,
                    "dbscan_eps": dbscan_eps, "dbscan_min": dbscan_min, "seed": seed,
                    "km_mode": km_mode, "frozen_scale": fixed_scale is not None,
                    "equalise": bool(equalise), "eq_srb": float(eq_srb)},
            "wall_time_s": time.time() - t0}
    if verbose and VERBOSITY >= 1:
        run_summary(run, t=time.time() - _t0)
    return run


# ----------------------------------------------------------------------
# 4) The scan
# ----------------------------------------------------------------------
def _scan_config_task(args):
    """One scan configuration, in its own process. Top-level so it is picklable."""
    import io as _io
    from contextlib import redirect_stdout as _rs
    (i, c, r, anomaly, kw) = args
    ra0 = float(anomaly["ra_c"]) + c["dra_raw"]
    dec0 = float(anomaly["dec_c"]) + c["ddec"]
    buf = _io.StringIO()
    try:
        with _rs(buf):
            run = run_one(r, ra0, dec0, c["h"], **kw)
        return i, c["label"], run, None, buf.getvalue()
    except Exception as e:
        return i, c["label"], None, f"{type(e).__name__}: {e}", buf.getvalue()


def scan(r, anomaly, h0, *, kM, p_ext, null_N=200_000, n_boot=10, dpa_Z=2.65,
         dbscan_eps=1.0, dbscan_min=5, n_jobs=8, seed=42,
         rho=0.6, scale_factors=(2 ** -0.5, 2 ** 0.5),
         freeze_scale=True, km_mode="bounded", equalise=True, eq_srb=5.0,
         eq_max_passes=3, eq_km="auto", eq_headroom=0.5,
         config_workers=1, ref_workers=1, boot_workers=1, verbose=True):
    """Run the battery. Returns (summary DataFrame, {label: run}, configs).

    PARALLELISM. Three nestable levels, and the product must stay under the core count:

        config_workers x ref_workers x n_jobs  <=  physical cores

    `config_workers` fans the grid placements over processes, `ref_workers` fans the
    8 references inside one placement, `n_jobs` is sklearn's kNN threading inside one
    Soar. Two safe recipes on a 112-core box:

        config_workers=8, ref_workers=1, n_jobs=8   -> 64 threads, simple and robust
        config_workers=1, ref_workers=8, n_jobs=8   -> 64 threads, per-config progress

    config_workers>1 AND ref_workers>1 nests process pools. It works under fork and
    gives the full 8x8, but a failure inside a child pool is harder to diagnose and
    the memory footprint multiplies -- prefer one level unless you need the 64x.

    NOTE `freeze_scale=True` pins the metric from the NOMINAL configuration, so that
    one must run before the others. With config_workers>1 the nominal is computed
    first, serially, and the rest are then fanned out.
    """
    A0 = set(int(s) for s in anomaly["source_ids"])
    cfgs = scan_configs(anomaly, h0, rho=rho, scale_factors=scale_factors)
    cosd = np.cos(np.deg2rad(float(anomaly["dec_c"])))

    # Does the supplied table actually cover every grid? Overrunning the data does
    # not raise anywhere downstream -- split_into_3x3 just returns thinner outer
    # tiles -- so an uncovered scan silently compares against truncated references.
    need_ra, need_dec = required_query_halfwidths(anomaly, cfgs)
    _dra = akd.wrap_dra_deg(np.asarray(r["ra"], float), float(anomaly["ra_c"]))
    _ddec = np.asarray(r["dec"], float) - float(anomaly["dec_c"])
    # PER SIDE, not max(|.|). The table is centred on whatever the discovery run used,
    # but the scan is anchored on the anomaly, so coverage is generally asymmetric: a
    # generous north side would otherwise mask a short south side, and the southern
    # reference tiles would come back nearly empty -- making all of Y look anomalous.
    ra_lo, ra_hi = float(-_dra.min()), float(_dra.max())
    dec_lo, dec_hi = float(-_ddec.min()), float(_ddec.max())
    have_ra, have_dec = min(ra_lo, ra_hi), min(dec_lo, dec_hi)
    if have_ra + 1e-9 < need_ra or have_dec + 1e-9 < need_dec:
        raise ValueError(
            f"star table does not cover the scan.\n"
            f"  need about the anomaly ({anomaly['ra_c']:.4f}, {anomaly['dec_c']:.4f}): "
            f"RA +/-{need_ra:.2f}, Dec +/-{need_dec:.2f}\n"
            f"  have  RA -{ra_lo:.2f}/+{ra_hi:.2f}   Dec -{dec_lo:.2f}/+{dec_hi:.2f}\n"
            f"  short by  RA {max(0, need_ra-ra_lo):.2f}/{max(0, need_ra-ra_hi):.2f}  "
            f"Dec {max(0, need_dec-dec_lo):.2f}/{max(0, need_dec-dec_hi):.2f}\n"
            f"Re-query a box CENTRED ON THE ANOMALY with those half-widths, or reduce "
            f"scale_factors/rho. Truncated reference tiles make all of Y look anomalous.")
    if verbose:
        print(f"coverage OK: need RA +/-{need_ra:.2f} Dec +/-{need_dec:.2f}; "
              f"have RA -{ra_lo:.2f}/+{ra_hi:.2f} Dec -{dec_lo:.2f}/+{dec_hi:.2f}")

    frozen = None
    runs, rows = {}, []

    _base = dict(kM=kM, p_ext=p_ext, null_N=null_N, n_boot=n_boot, dpa_Z=dpa_Z,
                 dbscan_eps=dbscan_eps, dbscan_min=dbscan_min, n_jobs=n_jobs,
                 seed=seed, km_mode=km_mode, equalise=equalise, eq_srb=eq_srb,
                 eq_max_passes=eq_max_passes, eq_km=eq_km, eq_headroom=eq_headroom,
                 ref_workers=ref_workers, boot_workers=boot_workers)

    if verbose:
        import ee_parallel as _ep
        print("  " + str(_ep.plan_workers(config_workers, max(ref_workers, 1),
                                          1 if ref_workers > 1 else n_jobs)), flush=True)

    order = list(enumerate(cfgs))
    done = {}
    if config_workers > 1 and len(cfgs) > 1:
        # freeze_scale pins the metric to the NOMINAL grid. That USED to mean running
        # the nominal configuration to completion first, which serialised ~half the
        # scan and threw away most of the benefit of fanning out. But the frozen scale
        # is only the med/MAD of the nominal window -- it needs the window BUILD, not
        # the pipeline. Take it directly (a second or two) and let all 8 go at once.
        if freeze_scale:
            i0, c0 = order[0]
            ra0 = float(anomaly["ra_c"]) + c0["dra_raw"]
            dec0 = float(anomaly["dec_c"]) + c0["ddec"]
            _, _, _, frozen = akd.make_Xrefs_and_Y_from_window_3x3(
                r, ra0, dec0, c0["h"], fit_scaling_on="all", do_robust_scale=True)
            if verbose:
                print(f"  frozen metric from '{c0['label']}': "
                      f"mad={np.round(np.asarray(frozen['mad'], float), 4)}", flush=True)
        import ee_parallel as _ep
        tasks = [(i, c, r, anomaly, dict(_base, fixed_scale=frozen, verbose=False))
                 for i, c in order]
        if tasks:
            for i, lab, run, err, _log in _ep.pmap(
                    _scan_config_task, tasks, min(int(config_workers), len(tasks)),
                    verbose=10 if verbose else 0):
                done[i] = (lab, run, err)
                if verbose:
                    print(f"  {lab:12s} " + (f"FAILED: {err}" if err else
                          f"nY={run['nY']} clusters={len(run['gamma_clusters'])}"),
                          flush=True)
    else:
        for i, c in order:
            if verbose:
                print(f"\n[{i+1}/{len(cfgs)}] {c['label']:12s} h={c['h']:.4f} "
                      f"dRA_raw={c['dra_raw']:+.3f} dDec={c['ddec']:+.3f}", flush=True)
            _, lab, run, err, _log = _scan_config_task(
                (i, c, r, anomaly, dict(_base, fixed_scale=frozen, verbose=verbose)))
            if _log and verbose:
                print(_log, end="")
            done[i] = (lab, run, err)
            if run is not None and freeze_scale and frozen is None:
                frozen = run["scale"]

    for i, c in enumerate(cfgs):
        if i not in done:
            continue
        lab, run, err = done[i]
        if err is not None:
            print(f"    {c['label']}: FAILED: {err}")
            rows.append({**{k: c[k] for k in ("label", "h", "dra_raw", "ddec")},
                         "status": f"failed: {err.split(':')[0]}"})
            continue

        runs[c["label"]] = run
        ti = np.asarray(run["parts"]["test_idx"], int)
        y_sid = np.asarray(r["source_id"], dtype="int64")[ti]
        in_Y = len(A0 & set(int(s) for s in y_sid))

        best = {"rec": 0.0, "jac": 0.0, "srb": np.nan, "n": 0, "gid": None}
        for gid, ix in run["gamma_clusters"].items():
            ids = set(int(s) for s in y_sid[np.asarray(ix, int)])
            inter = len(A0 & ids)
            if inter > best["n"]:
                best = {"rec": inter / max(len(A0), 1),
                        "jac": inter / max(len(A0 | ids), 1),
                        "srb": run["cluster_stats"][int(gid)]["s_over_rootB"],
                        "n": inter, "gid": int(gid)}

        gi = run["Gamma_i"]
        yi = np.where(np.isin(y_sid, list(A0)))[0]
        rows.append({
            **{k: c[k] for k in ("label", "h", "dra_raw", "ddec")},
            "dra_sky": c["dra_sky"], "status": "ok",
            "nY": run["nY"], "kM": run["cfg"]["kM"], "Gamma_star": run["Gamma_star"],
            "n_eq_refs": len(run.get("eq_repaired_cells", [])),
            "n_eq_cut": len(run.get("eq_removed_sids", set())),
            "n_A0_in_Y": in_Y, "n_clusters": len(run["gamma_clusters"]),
            "recovered": best["n"] > 0, "n_overlap": best["n"],
            "recovery": best["rec"], "jaccard": best["jac"],
            "s_over_rootB": best["srb"],
            "frac_A0_above_Gstar": (float((gi[yi] > run["Gamma_star"]).mean())
                                    if yi.size else np.nan),
        })
        if verbose:
            print(f"    |Y|={run['nY']}  A0 in Y={in_Y}/{len(A0)}  "
                  f"recovered={best['n']}  jaccard={best['jac']:.2f}  "
                  f"S/sqrt(B)={best['srb']:.1f}" if best["n"] else
                  f"    |Y|={run['nY']}  A0 in Y={in_Y}/{len(A0)}  NOT RECOVERED")

    return pd.DataFrame(rows), runs, cfgs


def verdict(summary, min_frac=0.75, min_jaccard=0.5, max_srb_ratio=3.0):
    """Robust if it survives most perturbations with a stable membership and strength."""
    ok = summary[summary.status == "ok"]
    pert = ok[ok.label != "nominal"]
    if pert.empty:
        return {"verdict": "inconclusive", "reason": "no perturbed config completed"}

    frac = float(pert.recovered.mean())
    jac = float(pert.loc[pert.recovered, "jaccard"].median()) if pert.recovered.any() else 0.0
    srb = pert.loc[pert.recovered, "s_over_rootB"].dropna()
    ratio = float(srb.max() / srb.min()) if len(srb) > 1 and srb.min() > 0 else np.nan

    passes = (frac >= min_frac and jac >= min_jaccard
              and (not np.isfinite(ratio) or ratio <= max_srb_ratio))
    return {"verdict": "ROBUST" if passes else "FRAGILE",
            "recovered_fraction": frac, "median_jaccard": jac,
            "s_over_rootB_ratio": ratio,
            "n_configs": int(len(ok)), "n_perturbed": int(len(pert)),
            "n_failed": int((summary.status != "ok").sum())}


# ----------------------------------------------------------------------
# 5) Visualisation
# ----------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


def _cfg_colours(cfgs):
    cm = plt.get_cmap("turbo")
    n = max(1, len(cfgs) - 1)
    return {c["label"]: cm(i / n) for i, c in enumerate(cfgs)}


def _yspace(run, r):
    ti = np.asarray(run["parts"]["test_idx"], int)
    return (ti,
            np.asarray(r["ra"], float)[ti], np.asarray(r["dec"], float)[ti],
            np.asarray(r["pmra"], float)[ti], np.asarray(r["pmdec"], float)[ti],
            np.asarray(r["source_id"], dtype="int64")[ti])


def plot_scan_regions(r, anomaly, cfgs, n_bg=40000, figsize=(9.5, 8.5), ax=None):
    """Every scan ROI drawn over the field, with the anomaly marked."""
    rng = np.random.default_rng(0)
    ra = np.asarray(r["ra"], float); dec = np.asarray(r["dec"], float)
    sub = rng.choice(ra.size, min(n_bg, ra.size), replace=False)
    col = _cfg_colours(cfgs)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.scatter(ra[sub], dec[sub], s=0.6, alpha=0.10, color="0.7",
               rasterized=True, zorder=0)
    for c in cfgs:
        ra0 = float(anomaly["ra_c"]) + c["dra_raw"]
        dec0 = float(anomaly["dec_c"]) + c["ddec"]
        h = c["h"]
        ax.add_patch(plt.Rectangle((ra0 - h, dec0 - h), 2*h, 2*h, fill=False,
                                   ec=col[c["label"]], lw=1.8, zorder=4,
                                   label=f"{c['label']}  h={h:.3f}"))
        ax.add_patch(plt.Rectangle((ra0 - 3*h, dec0 - 3*h), 6*h, 6*h, fill=False,
                                   ec=col[c["label"]], lw=0.7, ls=":", alpha=0.5, zorder=3))
    ax.plot(anomaly["ra_c"], anomaly["dec_c"], "*", ms=20, color="red",
            mec="k", mew=0.8, zorder=8, label="anomaly centroid")
    ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]")
    ax.set_title("scan ROIs (solid = $\\mathcal{Y}$, dotted = full 3x3)")
    ax.invert_xaxis(); ax.set_aspect("equal")
    ax.legend(fontsize=7.5, loc="best")
    ax.figure.tight_layout()
    return ax


def plot_scan_panels(r, anomaly, runs, cfgs, ncols=4, figsize=(19, 17)):
    """Per configuration: sky and PM, side by side. 8 configs -> 4x4."""
    A0 = set(int(s) for s in anomaly["source_ids"])
    labels = [c["label"] for c in cfgs if c["label"] in runs]
    nrows = int(np.ceil(2 * len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axf = np.atleast_1d(axes).ravel()
    cmap = plt.get_cmap("tab10")

    for i, lab in enumerate(labels):
        run = runs[lab]
        ti, ra, dec, pmra, pmdec, sid = _yspace(run, r)
        isA0 = np.isin(sid, list(A0))
        g = run["grid"]
        for j, (xk, yk, xl, yl, sp) in enumerate((
                (ra, dec, "RA [deg]", "Dec [deg]", "sky"),
                (pmra, pmdec, r"$\mu_{\alpha^*}$", r"$\mu_\delta$", "pm"))):
            ax = axf[2*i + j]
            ax.scatter(xk, yk, s=1.2, alpha=0.15, color="0.6", rasterized=True, zorder=1)
            for k, gid in enumerate(sorted(run["gamma_clusters"])):
                ix = np.asarray(run["gamma_clusters"][gid], int)
                ax.scatter(xk[ix], yk[ix], s=14, color=cmap(k % 10), alpha=0.9,
                           edgecolors="k", linewidths=0.2, zorder=4)
            ax.scatter(xk[isA0], yk[isA0], s=90, marker="*", facecolors="none",
                       edgecolors="red", linewidths=1.4, zorder=6)
            if sp == "sky":
                for d in (-g["cell_h"], g["cell_h"]):
                    ax.axvline(g["ra0"] + d, color="0.3", ls="--", lw=0.9)
                    ax.axhline(g["dec0"] + d, color="0.3", ls="--", lw=0.9)
                ax.invert_xaxis()
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel(xl, fontsize=9); ax.set_ylabel(yl, fontsize=9)
            ax.tick_params(labelsize=8)
            ax.set_title(f"{lab} — {sp}" + (f"  |Y|={run['nY']}" if sp == "sky" else
                                            f"  {len(run['gamma_clusters'])} clus"),
                         fontsize=9.5)
    for k in range(2 * len(labels), axf.size):
        axf[k].axis("off")
    fig.legend(handles=[Line2D([], [], ls="", marker="*", mfc="none", mec="red",
                               ms=12, label="target anomaly $A_0$"),
                        Line2D([], [], ls="", marker="o", color=cmap(0), ms=7,
                               label=r"$\Gamma$ cluster")],
               loc="upper center", ncol=2, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return fig


def plot_scan_gamma(r, anomaly, runs, cfgs, ncols=4, bins=60, figsize=(19, 8)):
    """Null vs test-set Gamma per configuration, with the A0 stars marked."""
    A0 = set(int(s) for s in anomaly["source_ids"])
    labels = [c["label"] for c in cfgs if c["label"] in runs]
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axf = np.atleast_1d(axes).ravel()
    for i, lab in enumerate(labels):
        run = runs[lab]
        ax = axf[i]
        gi = run["Gamma_i"]
        gn = run.get("Gamma_null", None)
        _, _, _, _, _, sid = _yspace(run, r)
        yi = np.where(np.isin(sid, list(A0)))[0]
        lo, hi = float(min(gi.min(), 0)), float(gi.max())
        edges = np.linspace(lo, hi, bins)
        if gn is not None and len(gn):
            ax.hist(gn, bins=edges, density=True, color="0.55", alpha=0.75, label="null")
        ax.hist(gi, bins=edges, density=True, histtype="step", lw=1.6,
                color="tab:blue", label=r"test $\Gamma_i$")
        ax.axvline(run["Gamma_star"], color="crimson", ls="--", lw=1.8,
                   label=rf"$\Gamma^\star$={run['Gamma_star']:.0f}")
        if yi.size:
            ax.plot(gi[yi], np.full(yi.size, ax.get_ylim()[1] * 0.5), "*",
                    color="red", mec="k", mew=0.4, ms=11, label="$A_0$")
        ax.set_yscale("log"); ax.set_title(lab, fontsize=10)
        ax.set_xlabel(r"$\Gamma_i$", fontsize=9); ax.tick_params(labelsize=8)
        if i == 0:
            ax.legend(fontsize=7.5)
    for k in range(len(labels), axf.size):
        axf[k].axis("off")
    fig.suptitle(r"$\Gamma$ distributions across the scan", y=1.01)
    fig.tight_layout()
    return fig


def _contour(ax, x, y, colour, label, levels=(0.5,), grid=120, pad_bw=3.5):
    """KDE contour, degrading gracefully when there are too few points.

    The grid is padded well beyond the data. A Gaussian KDE has support outside the
    point cloud, so the enclosed-mass level generally lies ~1 bandwidth past the
    outermost star: evaluating only on the bounding box clips the level set and the
    contour renders as an open arc instead of a closed loop. Padding also fixes the
    level itself, since the enclosed-mass fraction is normalised over the grid.
    """
    if x.size >= 5 and np.ptp(x) > 0 and np.ptp(y) > 0:
        try:
            k = gaussian_kde(np.vstack([x, y]))
            bw = k.factor
            padx = max(pad_bw * bw * np.std(x), 0.25 * np.ptp(x))
            pady = max(pad_bw * bw * np.std(y), 0.25 * np.ptp(y))
            xg = np.linspace(x.min() - padx, x.max() + padx, grid)
            yg = np.linspace(y.min() - pady, y.max() + pady, grid)
            X, Y = np.meshgrid(xg, yg)
            Z = k(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
            Zs = np.sort(Z.ravel())[::-1]
            cs = Zs[np.searchsorted(np.cumsum(Zs) / Zs.sum(), levels)]
            ax.contour(X, Y, Z, levels=np.sort(cs), colors=[colour], linewidths=2.0)
            ax.plot([], [], color=colour, lw=2.0, label=label)
            return
        except Exception:
            pass
    ax.plot(x, y, "o", ms=5, mfc="none", mec=colour, mew=1.6, label=label)


def plot_scan_overlay(r, anomaly, runs, cfgs, figsize=(15.5, 7.0), levels=(0.5,),
                      sky_half=2.0, pm_half=None, n_bg=120000, show_A0=False,
                      show_roi=True, bg_kw=None, stat="both", show_nrefs=True):
    """Contours of the recovered anomaly from every configuration, overlaid.

    The whole field within `sky_half` degrees (on the sky) of the anomaly is drawn in
    grey underneath, so the contours are read against the real stellar background
    rather than against the handful of points that define them.

    sky_half : half-width of the displayed sky region, in degrees ON THE SKY
    stat : which local significance goes in each contour's legend entry.
        "pipeline"  |A_Gamma| / sqrt(B_w), the overlap-weighted estimator. Note it
                    does NOT subtract the background, so it has a null floor of
                    sqrt(B) and reads high by roughly a factor of two.
        "lima"      Z_LiMa pooled over the references that produced a cluster,
                    computed on EagleEye's own repechage sets (ON = Repechaged,
                    OFF = the injected Background sample, alpha = the Eq.9
                    cardinality ratio). Background-subtracted and it accounts for
                    the control-sample variance.
        "both"      both, for comparison (default).
        "none"      neither; just the counts.
    show_nrefs : append "k/R refs" -- how many references produced an overlapping
        EE cluster at all. A contour held up by 2 of 8 references is a different
        claim from one held up by all 8, and that is invisible in either statistic.
               (the RA axis is widened by 1/cos(dec) internally).
    pm_half  : half-width of the PM panel; None uses the full range of the drawn stars.
    show_A0  : also mark the individual target stars.
    show_roi : outline each configuration's Y tile in its own contour colour.
    """
    A0 = set(int(s) for s in anomaly["source_ids"])
    col = _cfg_colours(cfgs)
    ra_c, dec_c = float(anomaly["ra_c"]), float(anomaly["dec_c"])
    cosd = np.cos(np.deg2rad(dec_c))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=figsize)

    # ---- background: every star in the window, grey ----
    ra_all = np.asarray(r["ra"], float)
    dec_all = np.asarray(r["dec"], float)
    dra = akd.wrap_dra_deg(ra_all, ra_c) * cosd
    win = (np.abs(dra) <= sky_half) & (np.abs(dec_all - dec_c) <= sky_half)
    idx = np.where(win)[0]
    if idx.size > n_bg:
        idx = np.random.default_rng(0).choice(idx, n_bg, replace=False)
    kw = dict(s=0.7, alpha=0.13, color="0.55", rasterized=True, zorder=0)
    kw.update(bg_kw or {})
    a1.scatter(ra_all[idx], dec_all[idx], **kw)
    a2.scatter(np.asarray(r["pmra"], float)[idx],
               np.asarray(r["pmdec"], float)[idx], **kw)

    # ---- one contour per configuration, plus its ROI in the same colour ----
    for c in cfgs:
        lab = c["label"]
        if lab not in runs:
            continue
        run = runs[lab]

        if show_roi:
            # drawn before the recovery test, so a config that failed to recover the
            # anomaly still shows the ROI that produced that non-detection
            g = run["grid"]
            h = g["cell_h"]
            a1.add_patch(plt.Rectangle((g["ra0"] - h, g["dec0"] - h), 2*h, 2*h,
                                       fill=False, ec=col[lab], lw=1.3, ls="--",
                                       alpha=0.75, zorder=2))

        _, ra, dec, pmra, pmdec, sid = _yspace(run, r)
        best, nbest, best_gid = None, 0, None
        for gid, ix in run["gamma_clusters"].items():
            ix = np.asarray(ix, int)
            n = len(A0 & set(int(s) for s in sid[ix]))
            if n > nbest:
                best, nbest, best_gid = ix, n, int(gid)
        if best is None:
            continue

        # local significance of the cluster this configuration actually recovered
        _srb = run.get("cluster_stats", {}).get(best_gid, {}).get("s_over_rootB", None)
        try:
            _srb = float(_srb)
        except (TypeError, ValueError):
            _srb = np.nan
        # Per-reference accounting, only computed if it will be shown.
        _nref, _zlm = None, np.nan
        if stat in ("lima", "both") or show_nrefs:
            try:
                import wake_test as _wt
                _c = _wt.combined_significance(run, best_gid)
                _nref, _zlm = _c["n_refs"], _c["Z_pooled"]
            except Exception:
                pass

        _bits = [f"{lab}  n={best.size}, {nbest}/{len(A0)} of $A_0$"]
        if stat in ("pipeline", "both"):
            _bits.append(rf"$S/\sqrt{{\hat B}}$={_srb:.1f}" if np.isfinite(_srb)
                         else r"$S/\sqrt{\hat B}$=--")
        if stat in ("lima", "both"):
            _bits.append(rf"$Z_{{\rm LiMa}}$={_zlm:.1f}" if np.isfinite(_zlm)
                         else r"$Z_{\rm LiMa}$=--")
        if show_nrefs and _nref is not None:
            _bits.append(f"{_nref}/{len(run['nXs'])} refs")
        _lab = ", ".join(_bits)

        _contour(a1, ra[best], dec[best], col[lab], _lab, levels)
        _contour(a2, pmra[best], pmdec[best], col[lab], _lab, levels)

    if show_A0:
        run0 = runs[cfgs[0]["label"]]
        _, ra, dec, pmra, pmdec, sid = _yspace(run0, r)
        m = np.isin(sid, list(A0))
        a1.plot(ra[m], dec[m], "*", color="k", ms=13, zorder=9, label="$A_0$ stars")
        a2.plot(pmra[m], pmdec[m], "*", color="k", ms=13, zorder=9, label="$A_0$ stars")

    a1.set_xlim(ra_c + sky_half / cosd, ra_c - sky_half / cosd)     # RA inverted
    a1.set_ylim(dec_c - sky_half, dec_c + sky_half)
    a1.set_xlabel("RA [deg]"); a1.set_ylabel("Dec [deg]")
    # equal SKY distances, not equal RA/Dec degrees: one RA degree subtends only
    # cos(dec) on the sky, so the axis ratio must be 1/cos(dec).
    a1.set_aspect(1.0 / cosd, adjustable="box")
    a1.set_title(f"sky  ({2*sky_half:.1f}$^\\circ$ across)")

    if pm_half is not None:
        a2.set_xlim(-pm_half, pm_half); a2.set_ylim(-pm_half, pm_half)
    a2.set_xlabel(r"$\mu_{\alpha^*}$ [mas yr$^{-1}$]")
    a2.set_ylabel(r"$\mu_\delta$ [mas yr$^{-1}$]")
    a2.set_aspect("equal", adjustable="box"); a2.set_title("proper motion")

    a1.legend(fontsize=8, loc="upper right", framealpha=0.9)
    a2.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.suptitle("recovered anomaly across every scan configuration", y=1.00)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Pooled-reference cross-check (single X, vanilla EagleEye)
# ----------------------------------------------------------------------
def refs_after_equalisation(run, r):
    """Rebuild the POST-equalisation X_refs from a stored run, without recomputing.

    `run` keeps `eq_removed_sids`, so the cleaned references can be reconstructed
    exactly by rebuilding each tile's features on the stored grid and metric and
    dropping those source_ids. Cheap, and avoids re-running equalisation (~1 min)
    or having to remember keep_heavy=True.
    """
    parts, scale, g = run["parts"], run["scale"], run["grid"]
    sid = np.asarray(r["source_id"], dtype="int64")
    removed = run.get("eq_removed_sids") or set()
    rem = np.fromiter(removed, dtype="int64") if removed else None
    out, rows = [], []
    for rid, idx in zip(parts["ref_ids"], parts["ref_idx_list"]):
        idx = np.asarray(idx, int)
        if rem is not None and rem.size:
            idx = idx[~np.isin(sid[idx], rem)]
        rc, dc = akd.cell_center_from_id(g["ra0"], g["dec0"], rid, g["cell_h"])
        raw = akd.build_features_local_center(r, idx, rc, dc, g["dec0"])
        out.append(akd.robust_apply(raw, scale["med"], scale["mad"]))
        rows.append(idx)
    return out, rows


def pooled_reference_run(run, r, *, kM=None, p_ext=None, null_N=200_000, dpa_Z=2.65,
                         n_jobs=8, verbose=True):
    """Stack the 8 cleaned references into ONE X and run vanilla EagleEye against Y.

    WHAT THIS APPROXIMATES, AND WHAT IT COSTS. The multi-reference estimator keeps the
    tiles separate precisely so each gets its own p_hat_j = nY/(nY+nX_j); Gamma then
    sums evidence over eight quasi-independent backgrounds. Pooling replaces that with
    a single p_hat = nY/(nY + sum_j nX_j) ~ 1/9, which is a DIFFERENT statistic, not a
    shortcut to the same one. Three consequences worth watching:

      * The X side saturates. Upsilon can never exceed -(K_M-1)*ln(1-p_hat), and at
        p_hat ~ 0.11 that ceiling is roughly 8x lower than at the balanced p_hat ~ 0.5
        of a single tile. `check` below reports the headroom against Upsilon*.
      * Per-tile structure is averaged away. A gradient across the grid, or one
        contaminated reference, is diluted into the pool instead of showing up as a
        disagreement between references -- which is what makes the stacked version
        diagnostic in the first place.
      * The gain is a much larger background sample, so B_hat and the repechage are
        better determined, and a real compact overdensity is measured against ~8x more
        comparison stars.

    Use it as a cross-check on the stacked result, not as a replacement.

    Returns a dict with the Soar result, the repechage book, S/sqrt(B), and `check`.
    """
    import EagleEye
    from utils_EE import compute_the_null, partitioning_function

    p_ext = float(run["cfg"]["p_ext"] if p_ext is None else p_ext)
    Y = np.asarray(run["Y"])
    X_refs, _ = refs_after_equalisation(run, r)
    Xp = np.vstack(X_refs)
    nY, nX = len(Y), len(Xp)
    p = nY / (nY + nX)

    bound = int(0.05 * min(nX, nY))
    kM = int(bound if kM is None else (bound if kM == "auto" else kM))
    with _q():
        sn = compute_the_null(p=p, K_M=kM, N=null_N)
    ups_star = float(np.quantile(np.asarray(sn[p], float), 1 - p_ext))
    ups_max = -(kM - 1) * np.log1p(-p)
    check = dict(nY=nY, nX=nX, p_hat=p, kM=kM, locality_bound=bound,
                 ups_star=ups_star, ups_max=ups_max, headroom=ups_max - ups_star)
    if verbose:
        print(f"  pooled X: nY={nY}  nX={nX} (8 refs)  p_hat={p:.4f}  K_M={kM} "
              f"(bound {bound})")
        print(f"  Y-side ceiling Upsilon_max={ups_max:.2f} vs Upsilon*={ups_star:.2f}"
              f"  -> headroom {check['headroom']:+.2f}"
              + ("   <-- SATURATED, nothing can be flagged"
                 if check["headroom"] <= 0 else ""))
    with _q():
        rd, _ = EagleEye.Soar(Xp, Y, K_M=kM, p_ext=p_ext, n_jobs=n_jobs,
                              stats_null=sn, result_dict_in={}, do_IDE=True)
        try:
            clusters = partitioning_function(Xp, Y, rd, p_ext=p_ext, Z=dpa_Z)
            book = EagleEye.Repechage(Xp, Y, rd, clusters, p_ext=p_ext)
            srb = EagleEye.S_rootB_estimate_Y_overdensities(rd, book)
        except Exception as e:
            book, srb, clusters = None, None, None
            if verbose:
                print(f"  no Y-side clusters ({type(e).__name__})")
    ups = np.asarray(rd["Upsilon_i_Y"], float)
    flagged = np.where(ups > rd["Upsilon_star_plus"][p_ext])[0]
    if verbose:
        print(f"  |Y^+| = {flagged.size}"
              + (f"   clusters = {len(book['Y_OVER_clusters'])}" if book else ""))
        if book:
            for a, v in sorted(book["Y_OVER_clusters"].items()):
                try:
                    _s = float(srb["s/root(B)"][a])
                except Exception:
                    _s = np.nan
                # Eq.9 goes out of range when the pruned mass exceeds the opposing
                # sample, giving B_hat <= 0 and a NEGATIVE "S/sqrt(B)". Pooling makes
                # nX ~ 8 nY, so this is COMMON here -- print it as undefined rather
                # than as a number someone might quote.
                _t = (f"{_s:.2f}" if np.isfinite(_s) and _s > 0
                      else "undefined (B_hat<=0, estimator out of range)")
                print(f"    cluster {a}: repechaged={len(v.get('Repechaged', []))} "
                      f"pruned={len(v.get('Pruned', []))}  S/rootB={_t}")
    return dict(rd=rd, book=book, srb=srb, clusters=clusters, X_pooled=Xp,
                Upsilon=ups, ups_star=float(rd["Upsilon_star_plus"][p_ext]),
                flagged=flagged, stats_null=sn, check=check, p_ext=p_ext)


def plot_pooled(res, run, r, *, field_cat=None, figsize=(15.5, 4.8)):
    """Sky, proper motion and the Upsilon distribution for a pooled-reference run."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ti = np.asarray(run["parts"]["test_idx"], int)
    g = run["grid"]
    ra = np.asarray(r["ra"], float); dec = np.asarray(r["dec"], float)
    pmra = np.asarray(r["pmra"], float); pmdec = np.asarray(r["pmdec"], float)
    ux = lambda a: g["ra0"] + akd.wrap_dra_deg(np.asarray(a, float), g["ra0"])

    book = res["book"]
    rep = np.array(sorted({int(i) for v in (book or {}).get("Y_OVER_clusters", {}).values()
                           for i in v.get("Repechaged", [])}), dtype=int)
    pru = np.array(sorted({int(i) for v in (book or {}).get("Y_OVER_clusters", {}).values()
                           for i in v.get("Pruned", [])}), dtype=int)
    fl = np.asarray(res["flagged"], int)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (xk, yk, xl, yl) in zip(axes[:2], (
            ("ra", "dec", "RA [deg]", "Dec [deg]"),
            ("pmra", "pmdec", r"$\mu_{\alpha^*}$ [mas/yr]", r"$\mu_\delta$ [mas/yr]"))):
        X = ux(ra[ti]) if xk == "ra" else pmra[ti]
        Yv = dec[ti] if xk == "ra" else pmdec[ti]
        ax.scatter(X, Yv, s=1.5, c="0.85", lw=0, rasterized=True, zorder=0)
        if fl.size:
            ax.scatter(X[fl], Yv[fl], s=16, c="tab:blue", lw=0, zorder=2)
        if rep.size:
            ax.scatter(X[rep], Yv[rep], s=44, facecolor="none", edgecolor="crimson",
                       lw=1.2, zorder=3)
        if pru.size:
            ax.scatter(X[pru], Yv[pru], s=70, marker="x", c="darkorange", lw=1.2, zorder=4)
        ax.set_xlabel(xl); ax.set_ylabel(yl)
        if xk == "ra":
            ax.invert_xaxis()
            ax.set_aspect(1.0 / np.cos(np.deg2rad(g["dec0"])))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v % 360:.1f}"))
    axes[0].set_title("pooled reference: sky", fontsize=11)
    axes[1].set_title("proper motion", fontsize=11)
    axes[0].legend(handles=[
        Line2D([], [], ls="", marker="o", ms=4, c="tab:blue", label=r"$\Upsilon>\Upsilon^*$"),
        Line2D([], [], ls="", marker="o", ms=7, mfc="none", mec="crimson",
               label="repechaged"),
        Line2D([], [], ls="", marker="x", ms=7, c="darkorange", label="pruned (excess)")],
        fontsize=8, loc="best")

    ax = axes[2]
    u = res["Upsilon"]
    sn = np.asarray(res["stats_null"][res["check"]["p_hat"]], float)
    bins = np.linspace(0, max(float(np.nanmax(u)), res["ups_star"]) * 1.15, 60)
    ax.hist(sn, bins=bins, density=True, histtype="step", color="0.4",
            label=f"null (p={res['check']['p_hat']:.3f})")
    ax.hist(u, bins=bins, density=True, histtype="stepfilled", alpha=.45,
            color="tab:blue", label=r"observed $\Upsilon_i$")
    ax.axvline(res["ups_star"], color="crimson", ls="--", lw=1.4,
               label=fr"$\Upsilon^*$={res['ups_star']:.2f}")
    ax.axvline(res["check"]["ups_max"], color="k", ls=":", lw=1.2,
               label=fr"ceiling {res['check']['ups_max']:.1f}")
    ax.set_yscale("log"); ax.set_xlabel(r"$\Upsilon$"); ax.set_ylabel("density")
    ax.set_title(f"pooled: $\\hat p$={res['check']['p_hat']:.3f}, "
                 f"$K_M$={res['check']['kM']}", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, axes


def plot_pooled_vs_stacked(res, run, r, *, use="repechaged", figsize=(13.5, 6.2),
                           zoom_pad=0.35, verbose=True):
    """Overlay the pooled-reference result on the stacked-Gamma result.

    Colours by AGREEMENT rather than by method, because that is the question:

        both        a star flagged by the stacked Gamma AND by the pooled run
        stacked only   Gamma > Gamma* but the pooled run missed it
        pooled only    the pooled run found it but Gamma did not

    `use` picks what counts as "found" on the pooled side: "repechaged" (the cluster
    plus its re-admitted surroundings, the closest analogue of a Gamma cluster),
    "flagged" (Upsilon > Upsilon*, the strictest), or "pruned" (the IDE excess mass
    alone). Pruned points are drawn as crosses in every case.

    Disagreement is expected and is the point: the two use different p_hat, different
    nulls and different ceilings. Stars in "both" are the ones no reasonable choice of
    reference construction removes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    ti = np.asarray(run["parts"]["test_idx"], int)
    g = run["grid"]
    ra, dec = np.asarray(r["ra"], float), np.asarray(r["dec"], float)
    pmra, pmdec = np.asarray(r["pmra"], float), np.asarray(r["pmdec"], float)
    ux = lambda a: g["ra0"] + akd.wrap_dra_deg(np.asarray(a, float), g["ra0"])

    book = res.get("book") or {}
    cl = book.get("Y_OVER_clusters", {})
    rep = {int(i) for v in cl.values() for i in v.get("Repechaged", [])}
    pru = {int(i) for v in cl.values() for i in v.get("Pruned", [])}
    pool = {"repechaged": rep, "pruned": pru,
            "flagged": set(int(i) for i in res["flagged"])}[use]
    stack = set(np.where(np.asarray(run["A_Gamma_mask"], bool))[0].tolist())

    both = np.array(sorted(stack & pool), int)
    s_only = np.array(sorted(stack - pool), int)
    p_only = np.array(sorted(pool - stack), int)
    pru_i = np.array(sorted(pru), int)

    if verbose:
        n = lambda a: len(a)
        print(f"  stacked (Gamma>Gamma*) : {len(stack)}")
        print(f"  pooled  ({use:<11})   : {len(pool)}")
        print(f"  both                   : {n(both)}"
              f"   ({100*n(both)/max(len(stack | pool),1):.0f}% of the union)")
        print(f"  stacked only           : {n(s_only)}")
        print(f"  pooled only            : {n(p_only)}")

    sel = np.concatenate([x for x in (both, s_only, p_only) if x.size]) \
        if (both.size or s_only.size or p_only.size) else np.array([], int)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, sky in zip(axes, (True, False)):
        X = ux(ra[ti]) if sky else pmra[ti]
        Yv = dec[ti] if sky else pmdec[ti]
        ax.scatter(X, Yv, s=1.5, c="0.87", lw=0, rasterized=True, zorder=0)
        for idx, col, lab, z in ((s_only, "tab:blue", "stacked only", 2),
                                 (p_only, "tab:orange", "pooled only", 2),
                                 (both, "tab:green", "both", 3)):
            if idx.size:
                ax.scatter(X[idx], Yv[idx], s=34, c=col, lw=0, zorder=z, label=lab)
        if pru_i.size:
            ax.scatter(X[pru_i], Yv[pru_i], s=95, marker="x", c="k", lw=1.0,
                       zorder=4, label="pooled pruned (excess)")
        if sky:
            ax.invert_xaxis()
            ax.set_aspect(1.0 / np.cos(np.deg2rad(g["dec0"])))
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v % 360:.1f}"))
            ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]")
            if sel.size:      # zoom on the union, else the tile dwarfs the points
                cd = np.cos(np.deg2rad(g["dec0"]))
                x0, y0 = X[sel].mean(), Yv[sel].mean()
                h = max(np.abs(X[sel]-x0).max()*cd, np.abs(Yv[sel]-y0).max()) + zoom_pad
                ax.set_xlim(x0 + h/cd, x0 - h/cd); ax.set_ylim(y0 - h, y0 + h)
            ax.set_title("sky (zoomed on the union)", fontsize=11)
        else:
            ax.set_xlabel(r"$\mu_{\alpha^*}$ [mas/yr]")
            ax.set_ylabel(r"$\mu_\delta$ [mas/yr]")
            ax.set_title("proper motion", fontsize=11)
    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle(f"pooled ({use}) vs stacked $\\Gamma>\\Gamma^*$   —   "
                 f"both {both.size}, stacked-only {s_only.size}, "
                 f"pooled-only {p_only.size}", fontsize=11)
    fig.tight_layout()
    return fig, axes, dict(both=both, stacked_only=s_only, pooled_only=p_only)


# ----------------------------------------------------------------------
# How much independent evidence do the references actually provide?
# ----------------------------------------------------------------------
def reference_independence(run, *, plot=True, ax=None, verbose=True):
    """Effective number of independent references, and the per-tile Upsilon pedestal.

    WHY NOT MEASURE THIS ON THE BOOTSTRAP NULL. `bootstrap_gamma_null_uniform` draws
    synthetic UNIFORM data, so a shape fitted to it describes the method (the
    max-over-k selection, the shared-Y correlation) and is essentially the same
    number every run. It cannot tell you whether YOUR references differ. This uses
    the real background stars instead.

    TWO NUMBERS WORTH READING:

    * R_eff -- Gamma = sum_j Upsilon_j is Fisher's combination, whose gain assumes R
      independent tests. The eight comparisons share the Y sample, so its Poisson
      noise is common-mode and the Upsilons are positively correlated. With mean
      pairwise correlation c, R_eff = R / (1 + (R-1)c): the number of references you
      are effectively stacking. Fisher's gain is sqrt(R_eff), not sqrt(R).

      NOT the same as `k_fit` from `overlay_gamma_null_curves`, which is a fitted
      Gamma SHAPE (mean^2/var of the null). The identity is
          mean^2/var = R_eff * (m^2/v)
      for marginal Upsilon mean m and variance v, so the two coincide only when
      Upsilon is Exp(1) (m=v=1). Measured here m^2/v = 1.74, so they differ by that
      factor even on identical data -- R_eff isolates the correlation, k_fit absorbs
      the non-exponential marginal shape on top of it.

    * the pedestal -- mean background Upsilon per reference. A tile whose p_hat has
      been deflated by a concentrated clump lifts Upsilon for EVERY test star, so it
      sits high here. That is the NGC 1261 / M3 signature, and a flat profile is
      independent confirmation that equalisation worked.

    Background stars only (A_Gamma excluded), or a real anomaly would inflate both.
    """
    U = np.asarray(run.get("Upsilon_by_ref"), float)
    if U.ndim != 2 and run.get("result_dicts"):
        # keep_heavy runs carry the per-reference result dicts; rebuild from those
        # rather than making the caller re-run the whole pipeline.
        U = np.vstack([np.asarray(rd["Upsilon_i_Y"], float)
                       for rd in run["result_dicts"]])
    if U.ndim != 2:
        raise ValueError(
            "run has no 'Upsilon_by_ref'. Runs made before this key was added do not "
            "store it, and it cannot be reconstructed from Gamma_i alone (the sum "
            "loses the per-reference split). Re-run the pipeline -- run_pipeline and "
            "rb.run_one both keep it now -- or pass keep_heavy=True so result_dicts "
            "are available to rebuild it from.")
    R = U.shape[0]
    bg = ~np.asarray(run["A_Gamma_mask"], bool)
    Ub = U[:, bg]
    C = np.corrcoef(Ub)
    off = C[np.triu_indices(R, 1)]
    c = float(off.mean())
    R_corr = R / (1.0 + (R - 1) * c)
    vr = float(Ub.sum(0).var() / Ub.var(1).sum())      # 1 if independent, R if identical
    R_var = R / vr if vr > 0 else np.nan
    ped = Ub.mean(1)
    z = (ped - ped.mean()) / (ped.std() if ped.std() > 0 else 1.0)

    out = dict(R=R, corr=C, c_mean=c, c_min=float(off.min()), c_max=float(off.max()),
               R_eff=float(R_corr), R_eff_alt=float(R_var), var_ratio=vr,
               pedestal=ped, pedestal_z=z, n_bg=int(bg.sum()))
    if verbose:
        print(f"  references: R = {R}, background stars = {out['n_bg']}")
        print(f"  cross-reference correlation: mean {c:+.3f} "
              f"(min {out['c_min']:+.3f}, max {out['c_max']:+.3f})")
        print(f"  R_eff = {R_corr:.2f} of {R} independent references"
              f"   -> Fisher gain sqrt(R_eff) = {np.sqrt(R_corr):.2f}x, "
              f"not sqrt({R}) = {np.sqrt(R):.2f}x")
        if R_corr < 0.35 * R:
            print(f"    references are largely redundant; the stack is worth much "
                  f"less than {R} independent tests")
        print(f"  mean background Upsilon per reference: {np.round(ped, 2)}")
        print(f"    spread {100*ped.std()/ped.mean():.1f}%"
              + (f"   <-- ref {int(np.argmax(np.abs(z)))} is {z[np.argmax(np.abs(z))]:+.1f} "
                 f"sd out: check for a clump deflating its p_hat"
                 if np.abs(z).max() > 2.0 else "   (flat: no pedestal)"))
    if plot:
        import matplotlib.pyplot as plt
        if ax is None:
            _, ax = plt.subplots(1, 2, figsize=(10.4, 4.2))
        im = ax[0].imshow(C, vmin=0, vmax=1, cmap="viridis")
        ax[0].set_xticks(range(R)); ax[0].set_yticks(range(R))
        ax[0].set_xlabel("reference"); ax[0].set_ylabel("reference")
        ax[0].set_title(f"corr($\\Upsilon^{{(j)}}$)  mean {c:+.2f}  "
                        f"$R_{{\\rm eff}}$={R_corr:.1f}/{R}", fontsize=10)
        plt.colorbar(im, ax=ax[0], fraction=.046)
        ax[1].bar(range(R), ped, color=["crimson" if abs(v) > 2 else "steelblue"
                                        for v in z])
        ax[1].axhline(ped.mean(), color="k", ls="--", lw=1,
                      label=f"mean {ped.mean():.2f}")
        ax[1].set_xlabel("reference"); ax[1].set_ylabel(r"mean background $\Upsilon$")
        ax[1].set_title("pedestal per reference (flat = clean)", fontsize=10)
        ax[1].legend(fontsize=8)
        plt.tight_layout()
    return out


def overlay_gamma_null_curves(axes, run, *, R=None, verbose=True):
    """Draw the analytic Fisher nulls over the Gamma density and survival panels.

    Gamma = sum_j Upsilon^(j) is Fisher's combination. For R INDEPENDENT references
    each -ln p is Exp(1) and the sum is exactly Gamma(R, 1) = 0.5*chi^2_{2R}. The
    references here are correlated (shared Y sample), and for R Exp(1) summands with
    mean pairwise correlation c the sum has mean R and variance R(1+(R-1)c). Matching
    a Gamma(k, theta) to those two moments gives

        theta = 1 + (R-1)c        k = R / (1 + (R-1)c)

    so Gamma(k_fit, mean/k_fit) IS Fisher's null generalised to correlated
    references -- the correlation absorbed into an effective number of degrees of
    freedom. That is the curve worth looking at; Gamma(R,1) is drawn beside it to
    show how far the independence assumption is from the truth.

    `k_fit` IS NOT `R_eff` from `reference_independence`. k_fit = mean^2/var of the
    null is a fitted Gamma shape; R_eff = R/(1+(R-1)c) is an effective reference
    count. They are related by mean^2/var = R_eff * (m^2/v) and coincide only if the
    marginal Upsilon is Exp(1). It is not (m^2/v = 1.74 here), and k_fit is measured
    on the SYNTHETIC bootstrap null while R_eff is measured on your real background --
    so expect them to differ, and quote R_eff when asking what stacking buys.

    USE IT AS A DIAGNOSTIC, NOT A CALIBRATION. Measured on this pipeline, the fit
    tracks the null to ~1e-2 and then runs high: +9% at p_ext=1e-3, +15% at 1e-4,
    because the true law is a sum of correlated MAX-of-exponentials and the
    max-over-k selection reshapes the tail. Substituting it for the empirical Gamma*
    trades 2.3x better run-to-run stability for 7x more bias, in the conservative
    direction (too high -> real anomalies discarded). The printed ratio column is the
    useful output: it says how non-Gamma your tail is, and hence how large n_boot has
    to be for the empirical quantile to resolve it.
    """
    from scipy import stats
    Gn = np.asarray(run["Gamma_null_mat"], float).reshape(-1)
    Gn = Gn[np.isfinite(Gn)]
    R = int(R if R is not None else len(run["nXs"]))
    p_ext = float(run["cfg"]["p_ext"])
    k_fit = float(Gn.mean() ** 2 / Gn.var())
    th = float(Gn.mean() / k_fit)
    c_imp = (R - k_fit) / (k_fit * (R - 1)) if k_fit < R else np.nan

    ax0, ax1 = axes[0], axes[1]
    xs = np.linspace(max(Gn.min(), 1e-6), ax0.get_xlim()[1], 500)
    ax0.plot(xs, stats.gamma.pdf(xs, k_fit, scale=th), color="seagreen", lw=2.0, ls=":",
             label=rf"$\Gamma(k_{{\rm fit}}={k_fit:.1f},\,{th:.1f})$ Fisher + corr.")
    ax0.plot(xs, stats.gamma.pdf(xs, R, scale=1.0), color="darkorange", lw=1.6, ls="--",
             label=rf"$\Gamma({R},1)$ Fisher, independent")
    ax0.legend(fontsize=7)
    ax1.plot(xs, stats.gamma.sf(xs, k_fit, scale=th), color="seagreen", lw=2.0, ls=":",
             label=rf"$\Gamma(k_{{\rm fit}}={k_fit:.1f})$")
    ax1.plot(xs, stats.gamma.sf(xs, R, scale=1.0), color="darkorange", lw=1.6, ls="--",
             label=rf"$\Gamma({R},1)$")
    ax1.legend(fontsize=7)

    if verbose:
        print(f"  k_fit (Gamma shape of the null) = {k_fit:.2f}, scale {th:.2f}"
              f"   [NOT R_eff -- see reference_independence]"
              + (f"   -> implied correlation c ~ {c_imp:.2f}"
                 if np.isfinite(c_imp) else "   (k_fit > R: selection dominates)"))
        print(f"  {'p_ext':>8}{'empirical':>11}{'Gamma(k_fit)':>14}{'ratio':>8}")
        for pe in (1e-1, 1e-2, 1e-3, 1e-4):
            e = float(np.quantile(Gn, 1 - pe))
            g = float(stats.gamma.ppf(1 - pe, k_fit, scale=th))
            flag = "  <- your p_ext" if abs(pe - p_ext) < 1e-12 else ""
            print(f"  {pe:8.0e}{e:11.2f}{g:14.2f}{g/e:8.3f}{flag}")
        print("  ratio drifting above 1 = tail heavier than Gamma; read Gamma* off")
        print("  the empirical null, and keep n_boot large enough to resolve it.")
    return dict(k_fit=k_fit, scale=th, c_implied=c_imp)

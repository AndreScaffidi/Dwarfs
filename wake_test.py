"""Bound system, or wake / debris?  Post-hoc tests on the Gamma clusters.

THE QUESTION. A dark perturber ploughing through the halo raises a Chandrasekhar
wake: an overdensity of ORDINARY HALO FIELD STARS trailing it. A bound dwarf is a
self-gravitating population. Both are sky overdensities; they differ in everything
else, and two of those differences are measurable with the data already in hand.

    property            bound dwarf at d           halo wake
    ---------------------------------------------------------------------
    velocity dispersion 3-10 km/s                  ~100 km/s (it IS the halo)
    -> proper motion    sigma_v / (4.74 d)         same formula, 10-30x larger
    line-of-sight depth one distance modulus       kpc-scale, mixed distances
    -> CMD              narrow locus               indistinguishable from field

WHAT IS AND IS NOT CIRCULAR. The Gamma clusters were found by DBSCAN in a scaled
space that INCLUDES pmra/pmdec, so they are proper-motion-tight by construction and
"the PM dispersion is small" proves nothing on its own. Two things rescue the PM
test: (1) the selection is only as tight as DBSCAN's eps, which `pm_selection_width`
converts back to mas/yr so you can see how much spread COULD have been selected --
if that width comfortably exceeds the wake prediction and the measured dispersion
does not, the test is informative; (2) the comparison against the measurement error
is a physical statement, not a selection artefact.

The CMD test is clean: colour and magnitude never enter the feature space at all.

WHAT THIS CANNOT DO. It cannot detect the perturber. See the lensing arithmetic --
for a lens at 30 kpc, Sigma_crit ~ 5.5e7 Msun/pc^2 and kappa ~ 1e-6, some 3-4 orders
below the shear floor. This only classifies the STELLAR feature.
"""
import numpy as np
import pandas as pd

K_PM = 4.74057                      # km/s per (mas/yr * kpc)

# DR3 median PM uncertainty vs G. Used ONLY if the archive cannot be reached; the
# real per-star pmra_error/pmdec_error are far better and are fetched by default.
_G_GRID   = np.array([13.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 20.7])
_PMERR_GRID = np.array([0.02, 0.025, 0.035, 0.055, 0.10, 0.25, 0.65, 1.20])


def approx_pm_error(g):
    """Fallback sigma_pm [mas/yr] from G. Approximate -- flagged wherever it is used.

    np.interp propagates NaN, and a single star with no photometry then poisons any
    sum over the array (a mean PM error came back NaN for 8961 field stars because of
    it). Missing G is treated as the faint end, the conservative choice.
    """
    g = np.asarray(g, float)
    g = np.where(np.isfinite(g), g, _G_GRID[-1])
    return np.interp(g, _G_GRID, _PMERR_GRID)


def fetch_pm_errors(source_ids, cache_dir, gaia=None, verbose=True):
    """Per-star pmra_error / pmdec_error from the archive, cached by id-set hash.

    Returns a DataFrame (source_id, pmra_error, pmdec_error) or None if unavailable;
    the caller then falls back to approx_pm_error.
    """
    import hashlib
    from pathlib import Path
    ids = sorted(int(s) for s in source_ids)
    if not ids:
        return None
    key = hashlib.sha1(",".join(map(str, ids)).encode()).hexdigest()[:12]
    pkl = Path(cache_dir) / f"pmerr_{len(ids)}_{key}.pkl"
    if pkl.exists():
        return pd.read_pickle(pkl)
    if gaia is None:
        try:
            from astroquery.gaia import Gaia as gaia
        except Exception:
            return None
    q = ("SELECT source_id, pmra_error, pmdec_error, phot_g_mean_mag "
         "FROM gaiadr3.gaia_source WHERE source_id IN ("
         + ",".join(map(str, ids)) + ")")
    try:
        t = gaia.launch_job_async(q).get_results().to_pandas()
    except Exception as e:
        if verbose:
            print(f"    (PM errors unavailable: {type(e).__name__}; "
                  f"falling back to the G-based approximation)")
        return None
    t["source_id"] = t["source_id"].astype("int64")
    t.to_pickle(pkl)
    return t


def pm_selection_width(run, eps=1.0):
    """How much PM spread DBSCAN could have admitted, in mas/yr.

    Features are scaled by the field MAD, and DBSCAN runs at radius `eps` in those
    units, so eps * mad_pm is the physical scale of the selection. Quote it next to
    any PM dispersion: a dispersion far below this is a real result, one close to it
    is the selection talking.
    """
    mad = np.asarray(run["scale"]["mad"], float)
    return float(eps * mad[2]), float(eps * mad[3])


def _sigma_int(x, err, conf=0.95):
    """Intrinsic dispersion of x after removing measurement scatter `err`.

    Returns (sigma_int, sigma_int_upper). s^2 is chi2-distributed, so the upper limit
    on the TOTAL dispersion is s*sqrt((n-1)/chi2_{1-conf, n-1}); the measurement term
    is subtracted from both. sigma_int is NaN when the variance excess is negative
    (i.e. consistent with pure measurement error) -- read the upper limit instead.
    """
    from scipy.stats import chi2
    x = np.asarray(x, float); n = x.size
    if n < 3:
        return np.nan, np.nan
    s2 = float(np.var(x, ddof=1))
    e2 = float(np.mean(np.asarray(err, float) ** 2))
    s2_hi = s2 * (n - 1) / chi2.ppf(1 - conf, n - 1)
    lo = np.sqrt(s2 - e2) if s2 > e2 else np.nan
    hi = np.sqrt(max(s2_hi - e2, 0.0))
    return lo, hi


def cmd_coherence(col_mem, g_mem, col_fld, g_fld, n_mc=4000, dg=0.25, rng=None):
    """Are the members' COLOURS tighter than field stars of the same magnitudes?

    Statistic: median pairwise |delta(BP-RP)|. The null draws, for each member, a
    random field star within dg of its G -- so the comparison is matched in magnitude
    and the test is purely about colour coherence. This matters: faint stars have
    broader colours, and a real dwarf's members sit near the detection limit, so an
    unmatched null would find "coherence" that is only a magnitude selection.

    A bound population at ONE distance modulus lies on a narrow CMD locus. A wake is
    halo field stars at mixed distances, so its colours ARE the field's.

    Returns dict(T_obs, T_null_med, p, n_used). p = P(T_null <= T_obs): SMALL p means
    more coherent than the field, i.e. bound-like.
    """
    rng = np.random.default_rng(0 if rng is None else rng)
    m = np.isfinite(col_mem) & np.isfinite(g_mem)
    col_mem, g_mem = np.asarray(col_mem)[m], np.asarray(g_mem)[m]
    f = np.isfinite(col_fld) & np.isfinite(g_fld)
    col_fld, g_fld = np.asarray(col_fld)[f], np.asarray(g_fld)[f]
    n = col_mem.size
    if n < 4 or col_fld.size < 100:
        return dict(T_obs=np.nan, T_null_med=np.nan, p=np.nan, n_used=int(n))

    def _T(c):
        return float(np.median(np.abs(c[:, None] - c[None, :])[np.triu_indices(len(c), 1)]))

    T_obs = _T(col_mem)
    order = np.argsort(g_fld)
    gs, cs = g_fld[order], col_fld[order]
    pools = []
    for g in g_mem:                       # field stars within dg of this member's G
        lo, hi = np.searchsorted(gs, [g - dg, g + dg])
        pools.append(cs[lo:hi] if hi - lo >= 5 else cs)
    Tn = np.empty(n_mc)
    for b in range(n_mc):
        Tn[b] = _T(np.array([p[rng.integers(len(p))] for p in pools]))
    return dict(T_obs=T_obs, T_null_med=float(np.median(Tn)),
                p=float(np.mean(Tn <= T_obs)), n_used=int(n))


def classify_cluster(run, r, gid, dist_kpc, *, cache_dir, eps=1.0,
                     sigma_v_bound=5.0, sigma_v_wake=100.0, n_mc=4000, verbose=True):
    """Run both tests on one Gamma cluster. Returns a dict of numbers."""
    test_idx = np.asarray(run["parts"]["test_idx"], int)
    rows = test_idx[np.asarray(run["gamma_clusters"][gid], int)]
    sid = np.asarray(r["source_id"], dtype="int64")[rows]
    pmra = np.asarray(r["pmra"], float)[rows]
    pmdec = np.asarray(r["pmdec"], float)[rows]
    gmag = np.asarray(r["phot_g_mean_mag"], float)[rows]
    col = (np.asarray(r["bp_rp"], float)[rows] if "bp_rp" in r
           else np.full(len(rows), np.nan))

    tab = fetch_pm_errors(sid, cache_dir, verbose=verbose)
    if tab is not None:
        emap = {int(a): (float(b), float(c)) for a, b, c in
                zip(tab["source_id"], tab["pmra_error"], tab["pmdec_error"])}
        e_ra = np.array([emap.get(int(s), (np.nan, np.nan))[0] for s in sid])
        e_de = np.array([emap.get(int(s), (np.nan, np.nan))[1] for s in sid])
        bad = ~np.isfinite(e_ra) | ~np.isfinite(e_de)
        if bad.any():
            e_ra[bad] = e_de[bad] = approx_pm_error(gmag[bad])
        err_src = "Gaia per-star"
    else:
        e_ra = e_de = approx_pm_error(gmag)
        err_src = "G-based approximation"

    lo_ra, hi_ra = _sigma_int(pmra, e_ra)
    lo_de, hi_de = _sigma_int(pmdec, e_de)
    s_int = np.nanmean([lo_ra, lo_de]) if np.isfinite([lo_ra, lo_de]).any() else np.nan
    s_hi = float(np.mean([hi_ra, hi_de]))
    w_ra, w_de = pm_selection_width(run, eps)

    # field CMD: every star in the 3x3 footprint, minus this cluster
    inside = np.asarray(run["parts"]["inside"], bool)
    fld = np.where(inside)[0]
    fld = np.setdiff1d(fld, rows)
    cmd = cmd_coherence(col, gmag,
                        (np.asarray(r["bp_rp"], float)[fld] if "bp_rp" in r
                         else np.full(fld.size, np.nan)),
                        np.asarray(r["phot_g_mean_mag"], float)[fld], n_mc=n_mc)

    return dict(
        gid=int(gid), n=len(rows), source_ids=sid,
        ra=np.asarray(r["ra"], float)[rows], dec=np.asarray(r["dec"], float)[rows],
        pmra=pmra, pmdec=pmdec, gmag=gmag, colour=col,
        pm_err_source=err_src, pm_err_med=float(np.median(np.r_[e_ra, e_de])),
        sigma_pm_obs=float(np.mean([np.std(pmra, ddof=1), np.std(pmdec, ddof=1)])),
        sigma_pm_int=float(s_int), sigma_pm_int_hi=s_hi,
        sigma_v=float(s_int) * K_PM * dist_kpc if np.isfinite(s_int) else np.nan,
        sigma_v_hi=s_hi * K_PM * dist_kpc,
        sel_width_mas=float(np.mean([w_ra, w_de])),
        sel_width_kms=float(np.mean([w_ra, w_de])) * K_PM * dist_kpc,
        pred_bound_mas=sigma_v_bound / (K_PM * dist_kpc),
        pred_wake_mas=sigma_v_wake / (K_PM * dist_kpc),
        median_G=float(np.median(gmag)), **{f"cmd_{k}": v for k, v in cmd.items()})


def report(res, dist_kpc):
    """One cluster, printed."""
    print(f"\n  cluster {res['gid']}:  n = {res['n']}   median G = {res['median_G']:.2f}")
    print(f"    PM errors from      : {res['pm_err_source']} "
          f"(median {res['pm_err_med']:.3f} mas/yr)")
    print(f"    observed sigma_PM   : {res['sigma_pm_obs']:.3f} mas/yr")
    si = res["sigma_pm_int"]
    if np.isfinite(si):
        print(f"    intrinsic sigma_PM  : {si:.3f} mas/yr "
              f"-> sigma_v = {res['sigma_v']:.1f} km/s  "
              f"(95% UL {res['sigma_v_hi']:.1f})")
    elif res["sigma_v_hi"] > 1.0:
        print(f"    intrinsic sigma_PM  : consistent with ZERO "
              f"-> sigma_v < {res['sigma_v_hi']:.1f} km/s (95%)")
    else:
        # s^2 below e^2 even at the 95% bound: the measurement error swamps any
        # intrinsic dispersion, so no useful limit exists. Printing "< 0.0" was wrong.
        print(f"    intrinsic sigma_PM  : unresolved -- observed scatter "
              f"({res['sigma_pm_obs']:.3f}) is BELOW the median measurement error "
              f"({res['pm_err_med']:.3f} mas/yr), so no upper limit is meaningful")
    print(f"    predicted            : bound {res['pred_bound_mas']:.3f} mas/yr | "
          f"wake {res['pred_wake_mas']:.3f} mas/yr    "
          f"[DBSCAN could admit {res['sel_width_mas']:.3f} = "
          f"{res['sel_width_kms']:.0f} km/s]")
    p = res["cmd_p"]
    if np.isfinite(p):
        print(f"    CMD colour spread   : {res['cmd_T_obs']:.4f} mag vs "
              f"{res['cmd_T_null_med']:.4f} for G-matched field   p = {p:.4f}"
              f"   ({res['cmd_n_used']} stars with BP/RP)")
    else:
        print(f"    CMD colour spread   : not testable "
              f"({res['cmd_n_used']} stars with BP/RP)")
    return _verdict(res, dist_kpc)


def _verdict(res, dist_kpc):
    """Plain-language reading. Deliberately conservative: the PM arm is only quoted
    when the selection width leaves room for it to mean anything."""
    bits = []
    wake, sel = res["pred_wake_mas"], res["sel_width_mas"]
    hi = res["sigma_pm_int_hi"]
    # THE SELECTION TELL. For a population with no intrinsic dispersion the observed
    # scatter should EQUAL the measurement error, not fall below it. When it does, the
    # members were chosen for proximity in exactly this coordinate and the "coldness"
    # is DBSCAN talking, whatever the nominal eps allowed. Discount the PM arm.
    selected = res["sigma_pm_obs"] < 0.85 * res["pm_err_med"]
    informative = (sel > 1.5 * wake) and not selected
    if selected:
        bits.append("PM: compromised (observed scatter is below the measurement "
                    "error -- the members were selected for PM proximity)")
    if informative and np.isfinite(hi):
        if hi < 0.5 * wake:
            bits.append("PM: too cold for a halo wake")
        elif hi > wake:
            bits.append("PM: consistent with halo kinematics")
    elif np.isfinite(hi):
        bits.append("PM: uninformative (DBSCAN selection is tighter than the "
                    "wake prediction, so coldness is circular)")
    p = res["cmd_p"]
    if np.isfinite(p):
        bits.append("CMD: single-population locus" if p < 0.05 else
                    "CMD: indistinguishable from the field" if p > 0.2 else
                    "CMD: marginal")
    return "; ".join(bits) if bits else "inconclusive"


# ----------------------------------------------------------------------
# Is it an object at all?  Two geometry checks that must pass FIRST.
# ----------------------------------------------------------------------
def tile_gradient(run):
    """Large-scale density gradient across the 3x3, measured from the tile counts.

    The antipodal pairing of the 3x3 cancels a LINEAR gradient in the mean, but
    Gamma = sum of a non-negative convex statistic rectifies rather than cancels the
    residual, so a strong gradient leaves a one-sided excess along whichever Y edge
    faces the sparser references. Anything above ~1.3 here deserves suspicion; the
    remedy is a smaller grid, not a better threshold.
    """
    import analyze_known_dwarfs as akd
    g, parts, h = run["grid"], run["parts"], run["grid"]["cell_h"]
    cells = []
    for rid, idx in zip(parts["ref_ids"], parts["ref_idx_list"]):
        rc, dc = akd.cell_center_from_id(g["ra0"], g["dec0"], rid, h)
        cells.append((int(rid),
                      float(akd.wrap_dra_deg(np.array([rc]), g["ra0"])[0]),
                      float(dc - g["dec0"]), len(idx)))
    n = np.array([c[3] for c in cells], float)
    dra = np.array([c[1] for c in cells]); ddec = np.array([c[2] for c in cells])
    W, E = n[dra < -0.5].mean(), n[dra > 0.5].mean()
    S, N = n[ddec < -0.5].mean(), n[ddec > 0.5].mean()
    return dict(cells=cells, nY=int(run["nY"]),
                ew_ratio=float(W / E), ns_ratio=float(S / N),
                max_ratio=float(n.max() / n.min()))


def edge_alignment(run, r, gid):
    """Does the cluster hug a tile boundary and span the tile?  The artefact signature.

    A gradient residual appears as a THIN STRIP flush against one Y edge, running the
    full extent of the perpendicular axis. A bound object is compact in both axes and
    has no reason to touch a boundary of an arbitrary grid. Reported as fractions of
    the tile half-width h, so 1.0 means "at the edge".
    """
    import analyze_known_dwarfs as akd
    g = run["grid"]; h = g["cell_h"]
    rows = np.asarray(run["parts"]["test_idx"], int)[
        np.asarray(run["gamma_clusters"][gid], int)]
    dra = akd.wrap_dra_deg(np.asarray(r["ra"], float)[rows], g["ra0"])
    ddec = np.asarray(r["dec"], float)[rows] - g["dec0"]
    span_ra = (dra.max() - dra.min()) / (2 * h)      # 1.0 = spans the whole tile
    span_dec = (ddec.max() - ddec.min()) / (2 * h)
    edge = max(abs(np.mean(dra)), abs(np.mean(ddec))) / h
    aspect = max(span_ra, span_dec) / max(min(span_ra, span_dec), 1e-6)
    suspect = (edge > 0.6 and max(span_ra, span_dec) > 0.8 and aspect > 3.0)
    return dict(span_ra=float(span_ra), span_dec=float(span_dec),
                edge=float(edge), aspect=float(aspect), suspect=bool(suspect),
                mean_dra=float(np.mean(dra)), mean_ddec=float(np.mean(ddec)))


def report_geometry(run, r, gid):
    grad = tile_gradient(run)
    ea = edge_alignment(run, r, gid)
    print(f"    grid gradient       : E-W {grad['ew_ratio']:.2f}x  "
          f"N-S {grad['ns_ratio']:.2f}x  (max tile ratio {grad['max_ratio']:.2f}x)"
          + ("   <-- STRONG" if grad["max_ratio"] > 1.3 else ""))
    print(f"    tile geometry       : spans {ea['span_ra']:.2f} x {ea['span_dec']:.2f} "
          f"of the tile, centroid {ea['edge']:.2f} h from centre, aspect "
          f"{ea['aspect']:.1f}:1" + ("   <-- EDGE-ALIGNED STRIP" if ea["suspect"] else ""))
    if ea["suspect"]:
        print("      => consistent with a gradient residual on the tile boundary, NOT")
        print("         an object. Re-run at a shifted grid: a real feature stays at")
        print("         fixed sky position, an artefact tracks the tile edge.")
    return grad, ea


def scan_persistence(scan_runs, r, cfgs=None, verbose=True):
    """Does the anomaly persist at a fixed SKY position, or at the tile EDGE?

    Persistence across grid perturbations is necessary but NOT sufficient. Three
    things survive every shift, and they are told apart by WHERE the flagged set
    lands relative to each configuration's own tile:

      real compact object  -> absolute sky centroid stable; tile-relative position
                              wanders (the grid moves under a fixed object)
      real large-scale     -> absolute centroid WANDERS, tile-relative position
      gradient                pinned to whichever edge faces the sparser references
      truncated reference  -> same as the gradient case, but the cause is missing
                              data rather than sky structure; check coverage first

    The summary line compares the scatter of the absolute centroids against the
    scatter of the tile-relative centroids: whichever is SMALLER is what the anomaly
    is actually locked to.
    """
    import analyze_known_dwarfs as akd
    rows = []
    for lab, run in scan_runs.items():
        if not run.get("gamma_clusters"):
            continue
        gid = max(run["gamma_clusters"],
                  key=lambda k: len(run["gamma_clusters"][k]))
        g, hh = run["grid"], run["grid"]["cell_h"]
        idx = np.asarray(run["parts"]["test_idx"], int)[
            np.asarray(run["gamma_clusters"][gid], int)]
        ra = np.asarray(r["ra"], float)[idx]
        dec = np.asarray(r["dec"], float)[idx]
        cd = np.cos(np.deg2rad(np.mean(dec)))
        dra = akd.wrap_dra_deg(ra, g["ra0"]); ddec = dec - g["dec0"]
        rows.append(dict(
            config=lab, n=len(idx), h=hh,
            ra_abs=float(np.mean(ra)), dec_abs=float(np.mean(dec)),
            dra_tile=float(np.mean(dra) / hh), ddec_tile=float(np.mean(ddec) / hh),
            edge=float(max(abs(np.mean(dra)), abs(np.mean(ddec))) / hh),
            r90_deg=float(np.percentile(
                np.hypot((ra - np.mean(ra)) * cd, dec - np.mean(dec)), 90))))
    df = pd.DataFrame(rows)
    if not len(df):
        print("no clusters in any configuration"); return df
    cd0 = np.cos(np.deg2rad(df["dec_abs"].mean()))
    sky_scatter = float(np.hypot(df["ra_abs"].std() * cd0, df["dec_abs"].std()))
    tile_scatter = float(np.hypot(df["dra_tile"].std(), df["ddec_tile"].std())
                         * df["h"].mean())
    if verbose:
        print(df.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
        print(f"\n  scatter of the ABSOLUTE sky centroid : {sky_scatter:.3f} deg")
        print(f"  scatter of the TILE-RELATIVE centroid: {tile_scatter:.3f} deg")
        print(f"  mean distance from tile centre       : {df['edge'].mean():.2f} h")
        if sky_scatter < 0.5 * tile_scatter:
            print("  => LOCKED TO THE SKY. Consistent with a real feature; the grid "
                  "moves under it.")
        elif tile_scatter < 0.5 * sky_scatter:
            print("  => LOCKED TO THE TILE. The flagged set follows the grid, which "
                  "no astrophysical object does.\n"
                  "     Check reference coverage first, then the tile gradient "
                  "(tile_gradient), then shrink the grid.")
        else:
            print("  => ambiguous; neither frame dominates.")
    return df


def pm_scramble_null(r, ra0, dec0, h, *, n_trials=20, scramble_seed=0,
                     verbose=True, **run_kw):
    """Trials-corrected significance for a Gamma cluster, by destroying the signal.

    NEITHER S/sqrt(B) NOR "the p_ext it survives to" IS A CALIBRATED SIGNIFICANCE.
    S/sqrt(B) is a local estimate that ignores how many independent places in the tile
    could have thrown up a clump; the p_ext readout is worse, because Gamma* moves only
    ~10-20% per DECADE of p_ext, so a 20% change in Gamma shifts the apparent p_ext by
    a decade or more. What you actually want is: how often does a search of this same
    field produce a clump this good when there is nothing there?

    THE NULL. Permute (pmra, pmdec) as PAIRS among the stars in the window. This keeps
      - every star's position, so the spatial density field is untouched, gradients and
        all;
      - the joint PM distribution, since the pairs are only relabelled;
    and destroys the position-PM correlation that a bound system has and a chance
    alignment does not. Any cluster found afterwards is spurious by construction, so
    max S/sqrt(B) over trials is drawn from the null of the FULL search.

    CAVEAT. A global permutation also flattens any real large-scale PM gradient across
    the window (Galactic rotation, LMC bulk motion), which makes the null slightly
    easier to beat and the resulting p slightly optimistic. Keep the window modest.

    `scramble_seed` drives the permutation. It is deliberately NOT called `seed`:
    run_one takes its own `seed` for the Gamma bootstrap, and callers pass that
    through run_kw, so a shared name collides ("multiple values for 'seed'").

    Returns (vals, trials): vals[i] is trial i's max S/sqrt(B); trials[i] is a list
    of (ra, dec, S/sqrt(B)) for every cluster in that trial, for position-matched
    tests via `scramble_pvalue_at`.
    """
    import robustness as rb
    run_kw.setdefault("seed", 42)
    rng = np.random.default_rng(scramble_seed)
    trials = []
    n = len(np.asarray(r["ra"]))
    vals = []
    for it in range(n_trials):
        perm = rng.permutation(n)
        rs = dict(r)
        rs["pmra"] = np.asarray(r["pmra"], float)[perm]
        rs["pmdec"] = np.asarray(r["pmdec"], float)[perm]
        try:
            run = rb.run_one(rs, ra0, dec0, h, verbose=False, **run_kw)
        except Exception as e:
            # A failure here is almost always a bad run_kw, which would otherwise
            # empty the whole array and look like "the null is weak".
            print(f"  trial {it+1}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        # Record every cluster's POSITION as well as its strength. Without positions
        # the null can only answer "max S/rootB anywhere in the tile", which is the
        # wrong question whenever the candidate's location was specified in advance
        # by an independent method -- then the trials factor is 1, not the number of
        # resolution elements in the tile (~60 here), and the tile-wide null is
        # roughly 60x too conservative.
        rows = []
        for gid, idx in run["gamma_clusters"].items():
            srb = float(run["cluster_stats"][gid]["s_over_rootB"])
            if not np.isfinite(srb):
                continue
            ii = np.asarray(run["parts"]["test_idx"], int)[np.asarray(idx, int)]
            rows.append((float(np.mean(np.asarray(rs["ra"], float)[ii])),
                         float(np.mean(np.asarray(rs["dec"], float)[ii])), srb))
        trials.append(rows)
        best = max((x[2] for x in rows), default=0.0)
        vals.append(best)
        if verbose:
            print(f"  trial {it+1}/{n_trials}: {len(rows)} cluster(s), "
                  f"max S/rootB = {best:.2f}", flush=True)
    return np.asarray(vals), trials


def scramble_pvalue(observed_srb, null_vals):
    """p = P(max S/rootB >= observed) under the scrambled null, with the usual +1."""
    v = np.asarray(null_vals, float)
    return float((np.sum(v >= observed_srb) + 1) / (v.size + 1))


def scramble_pvalue_at(observed_srb, trials, ra0, dec0, radius_deg=0.2):
    """Position-matched p-value: how often does the scrambled field put a cluster THIS
    strong within `radius_deg` of a PRE-SPECIFIED location?

    Use this, not `scramble_pvalue`, whenever the position was fixed in advance by an
    independent method (a published candidate list, another survey's detection). The
    tile-wide null asks "anywhere", which for a 1.33 deg tile and a 10 arcmin cluster
    over-corrects by a factor of order 60.

    Use `scramble_pvalue` instead for a blind search, where "anywhere" IS the question.
    """
    cd = np.cos(np.deg2rad(dec0))
    hits = 0
    for rows in trials:
        for ra, dec, srb in rows:
            if srb >= observed_srb and \
               np.hypot((ra - ra0) * cd, dec - dec0) <= radius_deg:
                hits += 1
                break
    return float((hits + 1) / (len(trials) + 1))


# ----------------------------------------------------------------------
# Empirical noise floor from BLANK SKY -- no manipulation of the data at all
# ----------------------------------------------------------------------
def offpos_null(r, h, *, n_positions=24, min_sep_h=2.0, avoid=None,
                avoid_deg=None, margin=0.05, nY_ref=None, nY_tol=0.35,
                scramble_seed=0, verbose=True, **run_kw):
    """Slide the WHOLE 3x3 to blank sky and run the identical pipeline.

    WHY THIS BEATS SCRAMBLING. A permutation null has to destroy something, and it
    always destroys more than intended: permuting PM also erases the field's real
    position-PM structure (solar reflex, Galactic rotation), while LEAVING the
    candidate's positional clump intact. Both biases are real and they point in
    opposite directions, so the resulting p is hard to defend. An off-position test
    manipulates nothing -- it is the same pipeline, same geometry, same sky, just
    somewhere there is no candidate. The classic off-source measurement.

    WHY NOT USE THE REFERENCE TILES AS PSEUDO-TESTS. Tempting and much cheaper, but
    the tiles are not interchangeable: Y sits at the centre of a symmetric 3x3 whose
    four antipodal reference pairs cancel linear gradients, while a corner tile has
    all its companions on one side and no cancellation. A corner-tile A/A test is
    systematically harsher than the real measurement, so its noise floor is too high.
    Moving the whole grid keeps the geometry identical.

    INDEPENDENCE. Centres are spaced at least `min_sep_h` * h apart, so at the default
    2.0 the TEST tiles are disjoint and the trials are quasi-independent (reference
    tiles still overlap between neighbouring positions). Contrast the Part 5 scan,
    whose configurations all sit on the anomaly and share nearly all their stars --
    which is why persistence there is a consistency check, not extra evidence.

    `avoid` is a list of (ra, dec) to stay clear of. TWO TRAPS:

    1. `avoid_deg` defaults to 3*h, NOT the Y-tile half-width. An object only h from
       the centre is outside Y but sits in a REFERENCE tile, where it deflates p_hat
       and puts a pedestal under every Upsilon -- the NGC 1261 failure mode. Keeping
       it out of Y is not enough; keep it out of the whole grid.
    2. DW20 CONTAINS NO GLOBULAR CLUSTERS -- it is a dwarf/candidate catalogue. A
       globular in the field (NGC 1261, M3, ...) never appears in `field_cat`, so the
       "blank" position beside it is not blank. Add them explicitly: Vasiliev &
       Baumgardt 2021 (VizieR J/MNRAS/505/5978) or Harris (1996, 2010 ed.) for
       globulars, Cantat-Gaudin+2020 (J/A+A/633/A99) for open clusters.

    `nY_ref` matches blank positions to the target in DENSITY: any whose test tile
    differs fractionally by more than `nY_tol` is skipped. Surface density varies
    strongly with Galactic latitude -- a floor averaged over positions twice as dense
    as the candidate's is not a like-for-like calibration.

    Returns (vals, trials, centres): vals[i] = max S/sqrt(B) at position i,
    trials[i] = [(ra, dec, S/sqrt(B)), ...], centres[i] = (ra0, dec0).
    """
    import robustness as rb
    rng = np.random.default_rng(scramble_seed)
    ra = np.asarray(r["ra"], float); dec = np.asarray(r["dec"], float)
    need = 3.0 * h
    # centre-space that keeps a full 3x3 inside the table, in RAW degrees
    cdec = np.cos(np.deg2rad(np.median(dec)))
    lo_d, hi_d = dec.min() + need + margin, dec.max() - need - margin
    lo_r, hi_r = ra.min() + need / cdec + margin, ra.max() - need / cdec - margin
    if hi_d <= lo_d or hi_r <= lo_r:
        raise ValueError(f"table too small for off-position tests at h={h}: "
                         f"need +/-{need:.2f} deg around each centre")
    av = [] if avoid is None else [(float(a), float(b)) for a, b in avoid]
    if avoid_deg is None:
        avoid_deg = 3.0 * h        # the FULL 3x3 half-width, references included

    centres, tries = [], 0
    while len(centres) < n_positions and tries < 200 * n_positions:
        tries += 1
        c = (float(rng.uniform(lo_r, hi_r)), float(rng.uniform(lo_d, hi_d)))
        cd = np.cos(np.deg2rad(c[1]))
        if any(np.hypot((c[0]-a)*cd, c[1]-b) < avoid_deg for a, b in av):
            continue
        if any(np.hypot((c[0]-a)*cd, c[1]-b) < min_sep_h*h for a, b in centres):
            continue
        centres.append(c)
    if verbose:
        print(f"{len(centres)} blank positions (requested {n_positions}), "
              f"spaced >= {min_sep_h*h:.2f} deg, avoiding {len(av)} known object(s)")

    vals, trials, kept = [], [], []
    for i, (ra0, dec0) in enumerate(centres):
        try:
            run = rb.run_one(r, ra0, dec0, h, verbose=False, **run_kw)
        except Exception as e:
            print(f"  pos {i+1}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        if nY_ref is not None and abs(run["nY"] - nY_ref) / nY_ref > nY_tol:
            if verbose:
                print(f"  pos {i+1}: nY={run['nY']} differs from target {nY_ref} "
                      f"by >{100*nY_tol:.0f}% -- skipped (density mismatch)", flush=True)
            continue
        rows = []
        for gid, idx in run["gamma_clusters"].items():
            srb = float(run["cluster_stats"][gid]["s_over_rootB"])
            if not np.isfinite(srb):
                continue
            ii = np.asarray(run["parts"]["test_idx"], int)[np.asarray(idx, int)]
            rows.append((float(np.mean(ra[ii])), float(np.mean(dec[ii])), srb))
        best = max((x[2] for x in rows), default=0.0)
        vals.append(best); trials.append(rows); kept.append((ra0, dec0))
        if verbose:
            print(f"  pos {i+1}/{len(centres)} ({ra0:7.3f},{dec0:7.3f}): "
                  f"nY={run['nY']:5d} {len(rows)} cluster(s), max S/rootB={best:.2f}",
                  flush=True)
    return np.asarray(vals), trials, kept


def passable_srb(vals, q=(0.5, 0.9, 0.95, 0.99)):
    """Empirical minimum-passable S/sqrt(B), read off the blank-sky distribution.

    This is the number to quote as a detection threshold: "a cluster must exceed
    S/sqrt(B) = X, because blank sky in this same field exceeds it only q% of the
    time at this configuration". It is specific to the field, the tile size, p_ext,
    K_M and every other setting -- recompute it whenever any of those change.
    """
    v = np.asarray(vals, float)
    out = {f"q{int(100*x)}": float(np.quantile(v, x)) for x in q}
    out["n"] = int(v.size); out["max"] = float(v.max()) if v.size else np.nan
    out["frac_no_cluster"] = float(np.mean(v == 0)) if v.size else np.nan
    return out


# ----------------------------------------------------------------------
# Method-artifact null: a field that is SMOOTH BY CONSTRUCTION
# ----------------------------------------------------------------------
def smooth_mock_null(r, ra0, dec0, h, *, n_trials=12, pos_bw=0.35, pm_bw=1.0,
                     scramble_seed=0, verbose=True, **run_kw):
    """Does EagleEye INVENT clusters on a field that provably has none?

    SCOPE. This isolates artefacts of the METHOD -- tile-edge effects, rectification
    of a residual background gradient by Gamma = sum_j Upsilon_j (Upsilon is
    non-negative and convex, so the antipodal pairs cancel only the LINEAR part),
    K_M non-locality, DBSCAN's scale coupling, estimator pathologies. It deliberately
    does NOT ask whether a clump in the real data is a bound dwarf, a stream, a
    co-moving group or a chance alignment: those are all real features OF THE SKY and
    belong to dynamical modelling, not to the flagging method. Use `offpos_null` or
    `pm_scramble_null` for those questions; use this one to answer "could the pipeline
    have manufactured this from nothing?".

    CONSTRUCTION. Smoothed bootstrap: resample the 4D vectors with replacement and add
    Gaussian jitter with bandwidth (`pos_bw` deg, `pm_bw` mas/yr). The generating
    density is then smooth on scales below the bandwidth BY CONSTRUCTION, while
    structure far above it -- the field gradient, the broad PM distribution, the
    density difference between tiles -- is preserved. Any cluster returned is
    therefore method-induced, with no appeal to what is or is not in the sky.

    CHOOSING THE BANDWIDTH -- the one judgement call. It must sit in the gap between
    the clump scale you want erased and the field structure you want kept:

        clump r90 ~ 10-20 arcmin  <<  pos_bw ~ 0.35 deg  <<  gradient scale ~ degrees
        clump sigma_pm ~ 0.4      <<  pm_bw ~ 1.0        <<  PM range +/- 5 mas/yr

    Too small and real clumps survive, so the null is contaminated and too harsh; too
    large and the gradient is washed out, so the null is too kind and the artefacts
    you are hunting disappear with it. Report the bandwidth with the result, and check
    both ends: a null whose answer is unchanged across a factor ~2 in bandwidth is
    trustworthy, one that swings is not.

    Returns (vals, trials) as for `pm_scramble_null`.
    """
    import robustness as rb
    rng = np.random.default_rng(scramble_seed)
    n = len(np.asarray(r["ra"]))
    cd = np.cos(np.deg2rad(float(dec0)))
    vals, trials = [], []
    if verbose:
        print(f"smoothed bootstrap: pos_bw={pos_bw} deg, pm_bw={pm_bw} mas/yr, "
              f"{n} stars", flush=True)
    for it in range(n_trials):
        idx = rng.integers(0, n, n)                       # resample with replacement
        rs = dict(r)
        # jitter RA on the sky, then convert back to a true RA offset
        rs["ra"] = (np.asarray(r["ra"], float)[idx]
                    + rng.normal(0, pos_bw, n) / cd)
        rs["dec"] = np.asarray(r["dec"], float)[idx] + rng.normal(0, pos_bw, n)
        rs["pmra"] = np.asarray(r["pmra"], float)[idx] + rng.normal(0, pm_bw, n)
        rs["pmdec"] = np.asarray(r["pmdec"], float)[idx] + rng.normal(0, pm_bw, n)
        try:
            run = rb.run_one(rs, ra0, dec0, h, verbose=False, **run_kw)
        except Exception as e:
            print(f"  trial {it+1}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        rows = []
        for gid, ix in run["gamma_clusters"].items():
            srb = float(run["cluster_stats"][gid]["s_over_rootB"])
            if not np.isfinite(srb):
                continue
            ii = np.asarray(run["parts"]["test_idx"], int)[np.asarray(ix, int)]
            rows.append((float(np.mean(rs["ra"][ii])), float(np.mean(rs["dec"][ii])), srb))
        best = max((x[2] for x in rows), default=0.0)
        vals.append(best); trials.append(rows)
        if verbose:
            print(f"  trial {it+1}/{n_trials}: nY={run['nY']:5d} {len(rows)} cluster(s),"
                  f" max S/rootB = {best:.2f}", flush=True)
    return np.asarray(vals), trials


# ----------------------------------------------------------------------
# Is the flagged group gravitationally plausible at all?
# ----------------------------------------------------------------------
G_PC = 4.30091e-3          # pc (km/s)^2 / Msun


def boundedness(res, dist_kpc, *, mass_per_star=0.6, lum_frac=0.02, verbose=True):
    """Back-of-envelope dynamical check on a Gamma cluster. ONE-SIDED by construction.

    WHY IT IS ONE-SIDED, AND WHY THAT MATTERS. Both things we cannot measure bias the
    answer the SAME way -- toward "bound":
      * projected separation R <= true 3D separation r, and binding goes as 1/r, so
        using R OVERSTATES how tightly the stars are held;
      * proper motion gives 2 of 3 velocity components, so the measured velocity
        spread UNDERSTATES the true spread.
    The mass this returns is therefore the SMALLEST mass that could bind the group.
    If even that is implausible, the group is definitively not a bound system. If it
    is plausible, nothing is proven -- the test simply fails to exclude.

    M_1/2 = 4 sigma_1D^2 r_1/2 / G  (Wolf et al. 2010), with r_1/2 = (4/3) R_h and
    sigma_1D taken per-component from the proper motions after removing measurement
    scatter in quadrature.

    READ THE POWER LINE FIRST. A UFD at tens of kpc has sigma_v of a few km/s, which
    at 30+ kpc is sigma_PM ~ 0.02-0.03 mas/yr -- an order of magnitude below Gaia's
    per-star error at G ~ 20. The dispersion is then unresolved no matter how real the
    object is, and the test has no power in either direction. That is a statement
    about Gaia, not about the candidate, and it is the usual reason UFD confirmation
    needs spectroscopy (velocity precision of order 2-3 km/s per star).
    """
    d = float(dist_kpc)
    ra, dec = np.asarray(res["ra"], float), np.asarray(res["dec"], float)
    cd = np.cos(np.deg2rad(np.median(dec)))
    rr = np.hypot((ra - np.median(ra)) * cd, dec - np.median(dec))
    R_h_pc = float(np.percentile(rr, 50)) * np.pi / 180.0 * d * 1e3   # projected
    r_half = (4.0 / 3.0) * R_h_pc                                     # deprojected

    n = int(res["n"])
    sig_obs = float(res["sigma_pm_obs"])          # mas/yr, mean of the two components
    err = float(res["pm_err_med"])
    sig_int = float(res["sigma_pm_int"])          # NaN when unresolved
    sig_hi = float(res["sigma_pm_int_hi"])        # 95% upper limit

    def _M(sig_mas):
        s1d = K_PM * sig_mas * d                  # km/s, per component
        return 4.0 * s1d ** 2 * r_half / G_PC, s1d

    M_obs, v_obs = _M(sig_obs)
    M_hi, v_hi = _M(sig_hi) if np.isfinite(sig_hi) else (np.nan, np.nan)
    M_lo, v_lo = _M(sig_int) if np.isfinite(sig_int) else (np.nan, np.nan)

    M_star = n * mass_per_star / max(lum_frac, 1e-6)   # crude: Gaia sees the bright tip
    v_esc = np.sqrt(2 * G_PC * M_star / max(r_half, 1e-6))

    # power: what a genuine UFD would show, against what Gaia can resolve
    sig_ufd_kms = np.sqrt(G_PC * 1e6 / (4 * max(r_half, 1e-6)))   # M_1/2 = 1e6 Msun
    sig_ufd_mas = sig_ufd_kms / (K_PM * d)
    sig_floor = err / np.sqrt(2 * max(n - 1, 1))       # error on a dispersion estimate

    out = dict(d_kpc=d, R_h_pc=R_h_pc, r_half_pc=r_half, n=n,
               sigma_pm_obs=sig_obs, pm_err=err, sigma_v_obs=v_obs,
               M_from_observed=M_obs, M_upper=M_hi, M_lower=M_lo,
               M_star_est=M_star, v_esc_stars=v_esc,
               sigma_ufd_mas=sig_ufd_mas, sigma_floor_mas=sig_floor,
               resolvable=bool(sig_ufd_mas > 2 * sig_floor))
    if verbose:
        print(f"    distance assumed    : {d:.1f} kpc")
        print(f"    R_h (projected)     : {R_h_pc:.0f} pc  -> r_1/2 = {r_half:.0f} pc")
        print(f"    sigma_PM observed   : {sig_obs:.3f} mas/yr "
              f"= {v_obs:.1f} km/s per component")
        print(f"    M_1/2 needed to bind: {M_obs:.2e} Msun (from the OBSERVED spread, "
              f"which is error-dominated)")
        if np.isfinite(M_hi) and M_hi > 0:
            print(f"      95% upper limit   : {M_hi:.2e} Msun")
        else:
            print(f"      95% upper limit   : unconstrained (observed spread is below "
                  f"the measurement error, so no limit exists)")
        print(f"    stellar mass (crude): {M_star:.1e} Msun "
              f"-> v_esc = {v_esc:.3f} km/s from stars alone")
        print(f"    POWER: a UFD (M_1/2=1e6) would show sigma_PM = "
              f"{sig_ufd_mas:.4f} mas/yr;")
        print(f"           Gaia resolves no better than {sig_floor:.4f} mas/yr here "
              f"-> {'TESTABLE' if out['resolvable'] else 'NO POWER'}")
        if not out["resolvable"]:
            print("           => the dispersion is unresolved whatever the truth is.")
            print("              Neither bound nor unbound can be concluded from PM;")
            print("              this needs spectroscopy (~2-3 km/s per star).")
    return out


def jacobi_radius_pc(M_sat, dist_kpc, v_circ=220.0):
    """Tidal (Jacobi) radius of a satellite of mass M_sat at galactocentric dist_kpc.

    r_J = D * (M_sat / (3 M_MW(<D)))^(1/3), with M_MW(<D) = v_c^2 D / G for a flat
    rotation curve. This is the radius beyond which the Milky Way's tide strips
    material, so it is a hard CEILING on the extent of anything gravitationally bound
    -- and unlike the internal velocity dispersion it needs nothing Gaia cannot
    measure. Turned around: an observed extent implies a MINIMUM bound mass.
    """
    D_pc = float(dist_kpc) * 1e3
    M_mw = v_circ ** 2 * D_pc / G_PC
    return D_pc * (np.asarray(M_sat, float) / (3.0 * M_mw)) ** (1.0 / 3.0)


def min_bound_mass(r_pc, dist_kpc, v_circ=220.0):
    """Invert jacobi_radius_pc: the smallest mass whose tidal radius reaches r_pc."""
    D_pc = float(dist_kpc) * 1e3
    M_mw = v_circ ** 2 * D_pc / G_PC
    return 3.0 * M_mw * (np.asarray(r_pc, float) / D_pc) ** 3


def plot_grav_radii(runs, r, dist_kpc, *, ra_c=None, dec_c=None,
                    masses=(1e5, 3e5, 1e6, 3e6, 1e7), v_circ=220.0, n_bg=40000,
                    levels=(0.5,), zoom_deg=None, ax=None, title=None, seed=0):
    """Stacked contours of the persistent cluster, with tidal radii drawn over them.

    `runs` is either the scan's {label: run} dict or a single run dict. For each one
    the cluster nearest the common centroid is contoured, so the spread of contours
    shows how the SAME feature is delineated under different grid placements.

    The circles are Jacobi radii for a range of assumed bound masses at `dist_kpc`.
    Read them as a ceiling: a bound object of mass M cannot be more extended than its
    circle, so the smallest circle that still encloses the contours sets the MINIMUM
    mass consistent with the group being one bound system. This uses only projected
    positions and an assumed distance -- no internal velocity dispersion, which at
    tens of kpc Gaia cannot resolve anyway (see `boundedness`).

    Caveat worth keeping in view: projected radius is a LOWER bound on true 3D extent,
    so the implied minimum mass is itself a lower bound.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import gaussian_kde, poisson
    import analyze_known_dwarfs as akd

    if isinstance(runs, dict) and "gamma_clusters" in runs:
        runs = {"run": runs}
    ra_all = np.asarray(r["ra"], float); dec_all = np.asarray(r["dec"], float)

    # members of the cluster nearest the centroid, per configuration
    picked = {}
    for lab, run in runs.items():
        if not run.get("gamma_clusters"):
            continue
        ti = np.asarray(run["parts"]["test_idx"], int)
        best, bd = None, np.inf
        for gid, ix in run["gamma_clusters"].items():
            ii = ti[np.asarray(ix, int)]
            c = (float(np.mean(ra_all[ii])), float(np.mean(dec_all[ii])))
            if ra_c is None:
                best, bd = (gid, ii), -1; break
            cd = np.cos(np.deg2rad(dec_c))
            dd = np.hypot((c[0] - ra_c) * cd, c[1] - dec_c)
            if dd < bd:
                best, bd = (gid, ii), dd
        if best is not None:
            picked[lab] = best[1]
    if not picked:
        raise ValueError("no clusters in any run")

    allii = np.unique(np.concatenate(list(picked.values())))
    if ra_c is None:
        ra_c, dec_c = float(np.median(ra_all[allii])), float(np.median(dec_all[allii]))
    cd = np.cos(np.deg2rad(dec_c))

    rJ = jacobi_radius_pc(np.asarray(masses, float), dist_kpc, v_circ)
    rJ_deg = rJ / (dist_kpc * 1e3) * 180.0 / np.pi
    # Frame on the CLUSTER, not on the largest circle -- otherwise a 1e8 Msun ring
    # sets the scale and the thing being measured is a few pixels across. Circles
    # bigger than the view simply clip, and are reported in the legend instead.
    r90_0 = float(np.percentile(np.hypot(
        (ra_all[allii] - ra_c) * cd, dec_all[allii] - dec_c), 90))
    half = float(zoom_deg) if zoom_deg else max(3.2 * r90_0, 1.15 * rJ_deg.min())

    if ax is None:
        _, ax = plt.subplots(figsize=(8.4, 8.0))
    x_all = ra_c + akd.wrap_dra_deg(ra_all, ra_c)
    m = (np.abs((x_all - ra_c) * cd) < half * 1.1) & (np.abs(dec_all - dec_c) < half * 1.1)
    xi = np.where(m)[0]
    if xi.size > n_bg:
        xi = np.random.default_rng(seed).choice(xi, n_bg, replace=False)
    ax.scatter(x_all[xi], dec_all[xi], s=1.2, c="0.82", lw=0, rasterized=True, zorder=0)

    cmap = plt.get_cmap("turbo")
    cols = [cmap(v) for v in np.linspace(0.06, 0.94, len(picked))]
    handles = []
    for (lab, ii), col in zip(picked.items(), cols):
        X, Y = x_all[ii], dec_all[ii]
        ax.scatter(X, Y, s=14, facecolor="none", edgecolor=col, lw=0.9, zorder=3)
        if len(ii) >= 8:
            try:
                k = gaussian_kde(np.vstack([X * cd, Y]))
                gx = np.linspace(X.min() - .05, X.max() + .05, 90)
                gy = np.linspace(Y.min() - .05, Y.max() + .05, 90)
                GX, GY = np.meshgrid(gx, gy)
                Z = k(np.vstack([GX.ravel() * cd, GY.ravel()])).reshape(GX.shape)
                Zs = np.sort(Z.ravel())[::-1]; cs = np.cumsum(Zs); cs /= cs[-1]
                lv = sorted({float(Zs[np.searchsorted(cs, L)]) for L in levels})
                ax.contour(GX, GY, Z, levels=lv, colors=[col], linewidths=1.6, zorder=4)
            except Exception:
                pass
        handles.append(Line2D([], [], color=col, lw=1.6, label=f"{lab} (n={len(ii)})"))

    th = np.linspace(0, 2 * np.pi, 400)
    off = []
    for M, rd in zip(masses, rJ_deg):
        ax.plot(ra_c + rd * np.cos(th) / cd, dec_c + rd * np.sin(th),
                ls="--", lw=1.2, color="k", alpha=.55, zorder=5)
        if rd < half * 0.97:
            ax.text(ra_c, dec_c + rd, f"$M=${M:.0e}   $r_J$={rd*60:.0f}'".replace("e+0", "e"),
                    fontsize=8, va="bottom", ha="center", color="k", alpha=.85, zorder=6)
        else:
            off.append(f"{M:.0e} ({rd*60:.0f}')".replace("e+0", "e"))

    r90 = r90_0
    Mmin = min_bound_mass(r90 / 180.0 * np.pi * dist_kpc * 1e3, dist_kpc, v_circ)
    ax.plot(ra_c + r90 * np.cos(th) / cd, dec_c + r90 * np.sin(th),
            color="crimson", lw=1.8, zorder=6)
    handles.append(Line2D([], [], color="crimson", lw=1.8,
                          label=f"cluster $r_{{90}}$ = {r90*60:.1f}'"))
    handles.append(Line2D([], [], color="k", ls="--", lw=1.2, alpha=.6,
                          label="tidal radius $r_J$"))
    if off:
        handles.append(Line2D([], [], color="none",
                              label="off view: " + ", ".join(off)))

    ax.set_xlim(ra_c + half / cd, ra_c - half / cd)     # RA increases left
    ax.set_ylim(dec_c - half, dec_c + half)
    ax.set_aspect(1.0 / cd)
    ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]")
    ax.set_title(title or (f"Tidal radii at d = {dist_kpc:.0f} kpc   |   "
                           f"min bound mass $\\gtrsim$ {Mmin:.1e} $M_\\odot$"),
                 fontsize=11)
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=.92)
    return ax, dict(r90_deg=r90, M_min=float(Mmin),
                    rJ_deg=dict(zip(masses, rJ_deg)), centroid=(ra_c, dec_c))


# ----------------------------------------------------------------------
# Distance-modulus scan against an empirical CMD ridgeline
# ----------------------------------------------------------------------
def build_ridgeline(g_tmpl, col_tmpl, mu_tmpl, *, deg=3, clip=2.5, n_iter=3):
    """Empirical isochrone: BP-RP as a smooth function of absolute magnitude M_G.

    An EMPIRICAL template beats a theoretical isochrone here. It is in the same
    photometric system, carries the same reddening (both objects are in the same
    field), and needs no assumption about age, [Fe/H] or the bandpass calibration --
    all of which are degenerate with distance over the short magnitude range Gaia
    reaches. The cost is coverage: the template only spans the M_G range the template
    object actually populates.

    Returns (ridge(M_G) -> colour, (M_min, M_max)).
    """
    M = np.asarray(g_tmpl, float) - float(mu_tmpl)
    c = np.asarray(col_tmpl, float)
    ok = np.isfinite(M) & np.isfinite(c)
    M, c = M[ok], c[ok]
    keep = np.ones(M.size, bool)
    for _ in range(n_iter):                       # sigma-clip against RGB outliers
        p = np.polyfit(M[keep], c[keep], deg)
        resid = c - np.polyval(p, M)
        s = 1.4826 * np.median(np.abs(resid[keep] - np.median(resid[keep])))
        keep = np.abs(resid) < clip * max(s, 1e-3)
        if keep.sum() < deg + 3:
            break
    p = np.polyfit(M[keep], c[keep], deg)
    return (lambda x: np.polyval(p, np.asarray(x, float))), (float(M.min()), float(M.max()))


def isochrone_scan(g_mem, col_mem, ridge, M_range, *, mu_grid,
                   g_field, col_field, n_null=300, dg=0.25, min_frac=0.6,
                   rng=None, verbose=True):
    """Which distance modulus, if any, makes these stars look like ONE population?

    At each trial mu a star of apparent G is predicted to have colour
    ridge(G - mu); the statistic is the median |observed - predicted|, so SMALL is a
    good fit. The scan is only meaningful where the null is: at every mu the same
    statistic is computed for magnitude-matched random field stars, because faint
    stars have broader colours and an unmatched comparison would reward the candidate
    for being faint rather than for being coherent.

    A mu is marked untestable when fewer than `min_frac` of the stars land inside the
    template's M_G coverage -- do not read a "good fit" off an extrapolated ridgeline.

    Returns a DataFrame: mu, distance, T_obs, T_null_med, p, n_used.
    """
    rng = np.random.default_rng(0 if rng is None else rng)
    g_mem = np.asarray(g_mem, float); col_mem = np.asarray(col_mem, float)
    ok = np.isfinite(g_mem) & np.isfinite(col_mem)
    g_mem, col_mem = g_mem[ok], col_mem[ok]
    gf = np.asarray(g_field, float); cf = np.asarray(col_field, float)
    f = np.isfinite(gf) & np.isfinite(cf)
    gf, cf = gf[f], cf[f]
    order = np.argsort(gf); gs, cs = gf[order], cf[order]

    pools = []
    for g in g_mem:
        lo, hi = np.searchsorted(gs, [g - dg, g + dg])
        pools.append(cs[lo:hi] if hi - lo >= 10 else cs)

    rows = []
    for mu in np.asarray(mu_grid, float):
        M = g_mem - mu
        inside = (M >= M_range[0]) & (M <= M_range[1])
        if inside.mean() < min_frac:
            rows.append(dict(mu=mu, dist_kpc=10 ** (mu / 5 - 2), T_obs=np.nan,
                             T_null_med=np.nan, p=np.nan, n_used=int(inside.sum()),
                             note="outside template"))
            continue
        pred = ridge(M[inside])
        T = float(np.median(np.abs(col_mem[inside] - pred)))
        Tn = np.empty(n_null)
        idx = np.where(inside)[0]
        for b in range(n_null):
            draw = np.array([pools[i][rng.integers(len(pools[i]))] for i in idx])
            Tn[b] = float(np.median(np.abs(draw - pred)))
        rows.append(dict(mu=mu, dist_kpc=10 ** (mu / 5 - 2), T_obs=T,
                         T_null_med=float(np.median(Tn)),
                         p=float(np.mean(Tn <= T)), n_used=int(inside.sum()), note=""))
    df = pd.DataFrame(rows)
    if verbose:
        good = df.dropna(subset=["T_obs"])
        if len(good):
            b = good.loc[good["p"].idxmin()]
            print(f"  best mu = {b['mu']:.2f}  (d = {b['dist_kpc']:.1f} kpc)  "
                  f"T_obs={b['T_obs']:.3f} vs field {b['T_null_med']:.3f}  p={b['p']:.4f}"
                  f"   [{int(b['n_used'])} stars]")
            print(f"  testable mu range {good['mu'].min():.2f}-{good['mu'].max():.2f} "
                  f"= {good['dist_kpc'].min():.0f}-{good['dist_kpc'].max():.0f} kpc "
                  f"(template coverage)")
        else:
            print("  no testable mu -- the template does not cover these magnitudes")
    return df


def plot_isochrone_scan(df, g_mem, col_mem, ridge, M_range, g_field, col_field,
                        *, mu_truth=None, label="candidate", axes=None):
    """CMD with the best-fit ridgeline, plus p versus distance modulus."""
    import matplotlib.pyplot as plt
    good = df.dropna(subset=["T_obs"])
    if not len(good):
        raise ValueError("no testable mu")
    best = good.loc[good["p"].idxmin()]
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12.4, 5.4))
    ax, ax2 = axes

    gf, cf = np.asarray(g_field, float), np.asarray(col_field, float)
    m = np.isfinite(gf) & np.isfinite(cf)
    ax.scatter(cf[m], gf[m], s=1.0, c="0.85", lw=0, rasterized=True, zorder=0)
    ax.scatter(col_mem, g_mem, s=34, facecolor="none", edgecolor="crimson",
               lw=1.3, zorder=3, label=f"{label} (n={len(g_mem)})")
    MM = np.linspace(M_range[0], M_range[1], 200)
    for mu, st, a in ((best["mu"], "-", 1.0),
                      (best["mu"] - 1.0, ":", .55), (best["mu"] + 1.0, ":", .55)):
        ax.plot(ridge(MM), MM + mu, st, color="navy", lw=1.8, alpha=a, zorder=4,
                label=(f"ridgeline @ $\\mu$={mu:.2f} ({10**(mu/5-2):.0f} kpc)"
                       if st == "-" else None))
    if mu_truth is not None:
        ax.plot(ridge(MM), MM + mu_truth, "--", color="green", lw=1.4, zorder=4,
                label=f"truth $\\mu$={mu_truth:.2f}")
    ax.set_xlim(-0.4, 2.4); ax.set_ylim(21.6, 15.0)
    ax.set_xlabel("BP $-$ RP"); ax.set_ylabel("G")
    ax.set_title("CMD and best-fit ridgeline", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")

    ax2.plot(good["dist_kpc"], good["p"], "o-", color="crimson", ms=4)
    ax2.axhline(0.05, ls="--", c="k", lw=1, alpha=.6)
    ax2.text(good["dist_kpc"].min(), 0.052, " p = 0.05", fontsize=8, va="bottom")
    ax2.set_xscale("log"); ax2.set_xlabel("distance [kpc]")
    ax2.set_ylabel("p  (fit better than G-matched field)")
    ax2.set_ylim(-0.02, max(0.55, float(good["p"].max()) * 1.1))
    ax2.set_title("distance-modulus scan   (low p = coherent population)", fontsize=11)
    ax2.grid(alpha=.25)
    return axes, best


def cluster_rows(run, tab, *, gid=None, ra_c=None, dec_c=None):
    """Row indices into `tab` for one Gamma cluster of `run`.

    `gid` picks it explicitly; otherwise (ra_c, dec_c) selects the nearest cluster,
    and failing that the largest. Cluster ids are NOT stable across scan
    configurations -- DBSCAN relabels every run -- so select by POSITION when
    comparing the same feature across configs, never by gid.
    """
    cl = run.get("gamma_clusters") or {}
    if not cl:
        raise ValueError("run has no gamma_clusters")
    ti = np.asarray(run["parts"]["test_idx"], int)
    if gid is not None:
        return ti[np.asarray(cl[gid], int)], gid
    if ra_c is not None and dec_c is not None:
        ra = np.asarray(tab["ra"], float); dec = np.asarray(tab["dec"], float)
        cd = np.cos(np.deg2rad(float(dec_c)))
        best, bd = None, np.inf
        for k, ix in cl.items():
            ii = ti[np.asarray(ix, int)]
            d = np.hypot((np.mean(ra[ii]) - ra_c) * cd, np.mean(dec[ii]) - dec_c)
            if d < bd:
                best, bd = k, d
        return ti[np.asarray(cl[best], int)], best
    k = max(cl, key=lambda x: len(cl[x]))
    return ti[np.asarray(cl[k], int)], k


# ----------------------------------------------------------------------
# Templates from systems outside the queried field
# ----------------------------------------------------------------------
def _dw20_distance(dwarfs, name):
    """DW20 heliocentric distance for a Battaglia-style key ('BootesIII').

    The two catalogues spell names differently -- Battaglia concatenates
    ('BootesIII'), DW20 spaces them ('Bootes III') -- so compare with all
    non-alphanumerics stripped.
    """
    key = "".join(ch for ch in str(name).lower() if ch.isalnum())
    for nm, d in zip(np.asarray(dwarfs["Name"]).astype(str),
                     np.asarray(dwarfs["D"], float)):
        if "".join(ch for ch in nm.lower() if ch.isalnum()) == key:
            return float(d), nm
    return None, None


def template_photometry(bat, name, r=None, *, pmemb_min=0.5, cache_dir=".",
                        gaia=None, chunk=4000, verbose=True):
    """G and BP-RP for a Battaglia system's members, fetching what is not local.

    A CMD template does not have to come from the field you queried -- Bootes III can
    calibrate a ridgeline used on a Reticulum II field, since both are old, metal-poor
    and observed by the same instrument. But its members are then absent from `r`, and
    the ridgeline build silently gets zero stars.

    This takes whatever is already in `r` (free, and guarantees the same columns) and
    queries Gaia by source_id for the remainder. Members are tens to hundreds of stars,
    so that is one small query; results are cached on the id-set hash.

    Returns (g, bp_rp, info). On a failed query it returns whatever was local, with
    `info["missing"] > 0` -- check it before trusting a sparse template.
    """
    import hashlib
    from pathlib import Path
    ids = np.fromiter(
        {int(x) for x in bat[(bat["Galaxy"] == name)
                             & (bat["Pmemb"] > pmemb_min)]["GaiaEDR3"]},
        dtype="int64")
    ids.sort()
    info = dict(name=name, n_members=int(ids.size), n_local=0, n_fetched=0, missing=0)
    if ids.size == 0:
        raise ValueError(f"no Battaglia members for {name!r} with Pmemb>{pmemb_min}")

    g = np.full(ids.size, np.nan)
    c = np.full(ids.size, np.nan)
    if r is not None and "source_id" in r:
        sid = np.asarray(r["source_id"], dtype="int64")
        order = np.argsort(sid)
        pos = np.searchsorted(sid[order], ids)
        pos[pos >= sid.size] = 0
        hit = sid[order][pos] == ids
        if hit.any():
            rows = order[pos[hit]]
            g[hit] = np.asarray(r["phot_g_mean_mag"], float)[rows]
            c[hit] = np.asarray(r["bp_rp"], float)[rows]
        info["n_local"] = int(np.isfinite(g).sum())

    need = ids[~np.isfinite(g) | ~np.isfinite(c)]
    if need.size:
        key = hashlib.sha1(",".join(map(str, need)).encode()).hexdigest()[:12]
        pkl = Path(cache_dir) / f"tmplphot_{name}_{need.size}_{key}.pkl"
        tab = None
        if pkl.exists():
            tab = pd.read_pickle(pkl)
        else:
            if gaia is None:
                try:
                    from astroquery.gaia import Gaia as gaia
                except Exception:
                    gaia = None
            if gaia is not None:
                parts = []
                for k in range(0, need.size, chunk):
                    lst = ",".join(map(str, need[k:k + chunk]))
                    q = ("SELECT source_id, phot_g_mean_mag, bp_rp "
                         "FROM gaiadr3.gaia_source WHERE source_id IN (" + lst + ")")
                    try:
                        parts.append(gaia.launch_job_async(q).get_results().to_pandas())
                    except Exception as e:
                        if verbose:
                            print(f"  Gaia query failed ({type(e).__name__}); "
                                  f"using only the {info['n_local']} local members")
                        parts = []
                        break
                if parts:
                    tab = pd.concat(parts, ignore_index=True)
                    tab["source_id"] = tab["source_id"].astype("int64")
                    tab.to_pickle(pkl)
        if tab is not None and len(tab):
            m = {int(a): (float(b), float(cc)) for a, b, cc in
                 zip(tab["source_id"], tab["phot_g_mean_mag"], tab["bp_rp"])}
            for i, s in enumerate(ids):
                if not (np.isfinite(g[i]) and np.isfinite(c[i])) and int(s) in m:
                    g[i], c[i] = m[int(s)]
            info["n_fetched"] = int(np.isfinite(g).sum()) - info["n_local"]

    ok = np.isfinite(g) & np.isfinite(c)
    info["missing"] = int((~ok).sum())
    if verbose:
        print(f"  template {name}: {info['n_members']} members "
              f"({info['n_local']} local, {info['n_fetched']} fetched, "
              f"{info['missing']} unavailable)")
    return g[ok], c[ok], info


# ==========================================================================
# Chance-cluster rate: how often does a smooth field make a clump like this?
# ==========================================================================
def chance_cluster_rate(Y, members=None, *, centre=None, m_list=(10, 20, 30),
                        bw=0.8, exclude_members=True, verbose=True):
    """Back-of-the-envelope probability that a clump this tight arose by chance.

    The question: under an inhomogeneous Poisson null with the SAME smooth 4D
    density as the data (so the field's real PM concentration is built in), how
    many m-star clumps this compact do we expect anywhere in the tile?

    Method, per m:
      r_m     = radius of the m-th nearest member to the clump centre
      lambda  = local smooth density at the centre, from a KDE whose bandwidth is
                deliberately much larger than the clump (the clump is also dropped
                from the fit) -- so lambda is the BACKGROUND, not the clump
      mu      = lambda * V_4(r_m),  V_4(R) = pi^2 R^4 / 2
      p_star  = P(Poisson(mu) >= m - 1)        per candidate centre
      E_stars = nY * p_star                    expected stars with an m-NN ball this tight
      E_clump = E_stars / m                    ... counted once per clump instead of m times

    Also returns the purely EMPIRICAL rank: how many stars in the actual tile have
    an m-NN ball at least as tight. The two disagreeing is informative -- it means
    the Poisson/KDE model does not describe the field's small-scale structure.

    Y        : (nY, d) array in the SAME scaled metric EagleEye used. "Compact" is
               metric-dependent, so this must be run["Y"], not raw RA/Dec/PM.
    members  : indices of the clump (centroid taken from them), or pass `centre`.
    """
    from scipy.stats import gaussian_kde, poisson
    from scipy.special import gamma as gamma_fn
    from sklearn.neighbors import KDTree

    Y = np.asarray(Y, float)
    nY, d = Y.shape
    if centre is None:
        if members is None:
            raise ValueError("pass either members or centre")
        centre = Y[np.asarray(members, int)].mean(axis=0)
    centre = np.asarray(centre, float)

    keep = np.ones(nY, bool)
    if exclude_members and members is not None:
        keep[np.asarray(members, int)] = False
    kde = gaussian_kde(Y[keep].T, bw_method=bw)
    lam = float(kde(centre[:, None])[0]) * int(keep.sum())

    # unit-ball volume in d dims; d=4 -> pi^2/2
    V_unit = np.pi ** (d / 2.0) / gamma_fn(d / 2.0 + 1.0)

    dist_c = np.sort(np.linalg.norm(Y - centre, axis=1))
    tree = KDTree(Y)

    rows = []
    for m in m_list:
        if m > nY:
            continue
        r_m = float(dist_c[m - 1])
        mu = lam * V_unit * r_m ** d
        p_star = float(poisson.sf(m - 2, mu))          # P(>= m-1)
        e_stars = nY * p_star
        nn, _ = tree.query(Y, k=m)
        n_tighter = int((nn[:, -1] <= r_m).sum())
        rows.append({"m": int(m), "r_m": r_m, "mu": mu, "p_per_centre": p_star,
                     "E_stars": e_stars, "E_clumps": e_stars / m,
                     "n_tighter_observed": n_tighter,
                     "frac_tighter_observed": n_tighter / nY})
    df = pd.DataFrame(rows)

    if verbose:
        print(f"chance-cluster rate   nY={nY}  d={d}  lambda_local={lam:.3f} "
              f"(KDE bw={bw}, clump {'excluded' if exclude_members else 'included'})")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print("\n  E_clumps is the expected NUMBER of chance clumps this tight in this")
        print("  tile, already look-elsewhere corrected. >1 means unremarkable.")
        print("  n_tighter_observed counts them in the REAL data (clump included), so")
        print("  it is >= 1 by construction; a value of 1 means it is the tightest.")
        print("  If E_clumps << 1 but n_tighter_observed is large, the Poisson model")
        print("  is wrong for this field -- trust the empirical count, not mu.")
    return {"lambda_local": lam, "centre": centre, "table": df}


def chance_rate_vs_mock(run, gid=0, *, m_list=(10, 20, 30), n_mock=20, bw=0.8,
                        pos_bw=0.35, pm_bw=1.0, seed=0, verbose=True):
    """The same statistic, calibrated on smooth mocks instead of a Poisson formula.

    For each mock (KDE resample of Y: same smooth structure, no small-scale
    clustering) we record the TIGHTEST m-NN radius anywhere. The observed clump's
    r_m is then placed in that distribution -- a genuine p-value that needs no
    Poisson assumption and no aperture choice.
    """
    from scipy.stats import gaussian_kde, poisson
    from sklearn.neighbors import KDTree

    Y = np.asarray(run["Y"], float)
    nY, d = Y.shape
    mem = np.asarray(run["gamma_clusters"][gid], int)
    centre = Y[mem].mean(axis=0)
    dist_c = np.sort(np.linalg.norm(Y - centre, axis=1))
    obs = {m: float(dist_c[m - 1]) for m in m_list if m <= nY}

    rng = np.random.default_rng(seed)
    kde = gaussian_kde(Y.T, bw_method=bw)
    null = {m: [] for m in obs}
    for b in range(int(n_mock)):
        Yb = kde.resample(nY, seed=int(rng.integers(1 << 31))).T
        tree = KDTree(Yb)
        for m in obs:
            nn, _ = tree.query(Yb, k=m)
            null[m].append(float(nn[:, -1].min()))

    rows = []
    for m in obs:
        v = np.asarray(null[m])
        n_le = int((v <= obs[m]).sum())
        rows.append({"m": m, "r_m_obs": obs[m], "mock_min_median": float(np.median(v)),
                     "mock_min_best": float(v.min()),
                     "n_mocks_tighter": n_le, "n_mock": int(n_mock),
                     "p_value": (n_le + 1) / (int(n_mock) + 1)})
    df = pd.DataFrame(rows)
    if verbose:
        print(f"tightest m-NN ball in {n_mock} smooth mocks vs the observed clump "
              f"(gid={gid}, |clump|={mem.size})")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print("\n  p_value is (1 + #mocks at least as tight) / (1 + n_mock): the")
        print("  smallest attainable value is 1/(n_mock+1), so raise n_mock if you")
        print("  hit the floor. No Poisson assumption, no aperture choice.")
    return {"observed": obs, "null": null, "table": df}


def pm_sky_chance_clumps(run, gid=0, *, dist_kpc=30.0, fracs=(0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0),
                         exclude_members=True, verbose=True):
    """How many RA/Dec clumps arise by chance PURELY from clustering in PM?

    Separable null: proper motions are drawn from the field's OWN pm distribution, and positions
    are independent and uniform over the patch. Then for a cell of PM radius
    dpm and sky radius theta,

        mu       = n * f_mu(dpm) * (pi theta^2 / A)          expected background
        f_mu     = fraction of FIELD stars inside the PM cell   <- measured
        E_clumps = (n / m) * P(Poisson(mu) >= m)             look-elsewhere corrected

    Unlike a ball in the joint MAD-scaled metric, this is a PRODUCT cell: "same
    proper motion AND same patch of sky". That is the physically meaningful
    question, and it is much more conservative -- a joint-metric ball selects the
    most-clustered subset in a coupled metric and can look many orders of magnitude
    more significant for the same stars. Cross-check the two; if they disagree
    wildly, the clump is elongated in the joint metric and the product cell is the
    number to trust.

    The aperture is scanned over `fracs`, so the best E_clumps carries a trials
    penalty of ~len(fracs) -- reported as E_clumps_scan.
    """
    from scipy.stats import poisson

    Y = np.asarray(run["Y"], float)
    n = len(Y)
    phys = Y * run["scale"]["mad"] + run["scale"]["med"]   # dRAcosd, dDec, pmra, pmdec
    mem = np.asarray(run["gamma_clusters"][gid], int)
    keep = np.ones(n, bool)
    if exclude_members:
        keep[mem] = False

    cm = phys[mem]
    pm_c, sky_c = cm[:, 2:].mean(0), cm[:, :2].mean(0)
    A = float(np.prod(phys[:, :2].max(0) - phys[:, :2].min(0)))
    d_pm = np.linalg.norm(cm[:, 2:] - pm_c, axis=1)
    d_sky = np.linalg.norm(cm[:, :2] - sky_c, axis=1)
    pc_per_deg = np.deg2rad(1.0) * float(dist_kpc) * 1000.0

    rows = []
    for f in fracs:
        r_pm, th = float(np.quantile(d_pm, f)), float(np.quantile(d_sky, f))
        m = int(((d_pm <= r_pm) & (d_sky <= th)).sum())
        if m < 4:
            continue
        f_mu = float((np.linalg.norm(phys[keep, 2:] - pm_c, axis=1) <= r_pm).mean())
        mu = int(keep.sum()) * f_mu * (np.pi * th ** 2 / A)
        P = float(poisson.sf(m - 1, mu))
        rows.append({"frac": f, "m": m, "r_pm": r_pm, "theta_deg": th,
                     "r_pc": th * pc_per_deg, "f_mu": f_mu, "mu": mu,
                     "P_local": P, "E_clumps": (n / m) * P})
    df = pd.DataFrame(rows)
    df["E_clumps_scan"] = df["E_clumps"] * len(df)
    best = df.loc[df["E_clumps"].idxmin()]

    if verbose:
        print(f"separable PM x sky chance rate   n={n}  A={A:.2f} deg^2  "
              f"d={dist_kpc} kpc (1 deg = {pc_per_deg:.0f} pc)")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print(f"\n  best aperture: m={int(best['m'])}  theta={best['theta_deg']:.3f} deg "
              f"({best['r_pc']:.0f} pc)  mu={best['mu']:.2f}")
        print(f"  E_clumps = {best['E_clumps']:.3g}, or {best['E_clumps_scan']:.3g} "
              f"after the {len(df)}-aperture scan penalty.")
        print("  >~1 means a clump this good is EXPECTED in this patch by chance.")
        print("\n  f_mu is the whole point: it is the fraction of the field sharing the")
        print("  clump's proper motion, measured from the data. A peaked PM distribution")
        print("  makes f_mu large, which makes sky clumping cheap -- exactly the effect")
        print("  you were worried about, quantified without any Galactic model.")
    return {"table": df, "best": best, "A_deg2": A, "n": n,
            "pm_centre": pm_c, "sky_centre": sky_c}


def chance_vs_patch_richness(run, gid=0, *, frac=0.6, n_grid=(200, 500, 1000, 2000,
                             4000, 8000), dist_kpc=30.0, verbose=True):
    """Does a SPARSER patch make chance clumps easier? Two answers, both true.

    (a) At FIXED angular scale, E_clumps falls steeply with n -- a sparse patch has
        fewer background stars to fake a clump with.
    (b) But at FIXED member count m, the angular scale at which the clump stops
        being remarkable (mu = m) GROWS as n falls, like 1/sqrt(n). Since EagleEye's
        kNN balls are adaptive, a sparse patch pushes the method to physically huge
        apertures -- and a "clump" 500 pc across is not a bound system at all.

    (b) is the real content of the worry: sparse patches do not produce more chance
    clumps, they produce chance clumps that are too big to be UFDs.
    """
    from scipy.stats import poisson

    Y = np.asarray(run["Y"], float)
    phys = Y * run["scale"]["mad"] + run["scale"]["med"]
    mem = np.asarray(run["gamma_clusters"][gid], int)
    keep = np.ones(len(Y), bool); keep[mem] = False
    cm = phys[mem]
    pm_c, sky_c = cm[:, 2:].mean(0), cm[:, :2].mean(0)
    A = float(np.prod(phys[:, :2].max(0) - phys[:, :2].min(0)))
    d_pm = np.linalg.norm(cm[:, 2:] - pm_c, axis=1)
    d_sky = np.linalg.norm(cm[:, :2] - sky_c, axis=1)
    r_pm, th = float(np.quantile(d_pm, frac)), float(np.quantile(d_sky, frac))
    m = int(((d_pm <= r_pm) & (d_sky <= th)).sum())
    f_mu = float((np.linalg.norm(phys[keep, 2:] - pm_c, axis=1) <= r_pm).mean())
    pc_per_deg = np.deg2rad(1.0) * float(dist_kpc) * 1000.0

    rows = []
    for nn in n_grid:
        mu = nn * f_mu * (np.pi * th ** 2 / A)
        P = float(poisson.sf(m - 1, mu))
        th_eq = float(np.sqrt(m * A / (nn * f_mu * np.pi)))   # theta where mu = m
        rows.append({"n": nn, "mu": mu, "P_local": P, "E_clumps": (nn / m) * P,
                     "theta_unremarkable_deg": th_eq,
                     "r_unremarkable_pc": th_eq * pc_per_deg})
    df = pd.DataFrame(rows)
    if verbose:
        print(f"fixed clump: m={m}, theta={th:.3f} deg, f_mu={f_mu:.3f}, A={A:.2f} deg^2")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print("\n  theta_unremarkable is where mu = m: a clump of this many stars inside")
        print("  that radius is exactly what the background gives you. Compare against")
        print("  UFD half-light radii (~20-300 pc) to see whether the method is even")
        print("  operating in a regime where a bound system could be distinguished.")
    return df


def pm_sky_concentration_test(run, gid=0, *, q=0.6, n_null=500, seed=0, verbose=True):
    """Does the PM-selected subsample CONCENTRATE on the sky, or merely populate it?

    `pm_sky_chance_clumps` is a COUNTING statistic: it asks only whether >= m stars
    fall inside the cell and is blind to their arrangement within it. This measures
    concentration directly.

    Take the k stars within Delta_mu of the clump's mean proper motion. Let c be the
    largest number of them inside any sky disc of radius theta. Standardise by the
    single-disc expectation nu_k = k pi theta^2 / A:

        z = (c - nu_k) / sqrt(nu_k)

    z MUST be calibrated empirically: c is a MAXIMUM over disc placements, so E[z] > 0
    even for perfectly uniform points (~2.2 at these cardinalities). Three nulls are
    returned, and they should agree:
      "pm_cells"  other PM cells of the same radius drawn from the field
      "uniform"   k points thrown uniformly on the patch
      "pm_shuffled" real sky positions, proper motions permuted
    """
    from scipy.stats import poisson

    from sklearn.neighbors import KDTree

    Y = np.asarray(run["Y"], float); n = len(Y)
    phys = Y * run["scale"]["mad"] + run["scale"]["med"]
    sky, pm = phys[:, :2], phys[:, 2:]
    lo, hi = sky.min(0), sky.max(0)
    A = float(np.prod(hi - lo))
    mem = np.asarray(run["gamma_clusters"][gid], int)
    cm = phys[mem]
    pm_c, sky_c = cm[:, 2:].mean(0), cm[:, :2].mean(0)
    r_pm = float(np.quantile(np.linalg.norm(cm[:, 2:] - pm_c, axis=1), q))
    th   = float(np.quantile(np.linalg.norm(cm[:, :2] - sky_c, axis=1), q))

    def _z(P):
        k = len(P)
        if k < 2:
            return k, np.nan
        c = int(max(len(x) for x in KDTree(P).query_radius(P, th)))
        nu_k = k * np.pi * th ** 2 / A
        return c, (c - nu_k) / np.sqrt(nu_k)

    sel = np.linalg.norm(pm - pm_c, axis=1) <= r_pm
    k_obs = int(sel.sum())
    c_obs, z_obs = _z(sky[sel])

    rng = np.random.default_rng(seed)
    null = {"pm_cells": [], "uniform": [], "pm_shuffled": []}
    for i in rng.permutation(n):
        if len(null["pm_cells"]) >= n_null:
            break
        if np.linalg.norm(pm[i] - pm_c) < 1.5 * r_pm:
            continue
        s = np.linalg.norm(pm - pm[i], axis=1) <= r_pm
        if s.sum() < 25:
            continue
        null["pm_cells"].append(_z(sky[s])[1])
    for _ in range(int(n_null)):
        null["uniform"].append(_z(rng.uniform(lo, hi, size=(k_obs, 2)))[1])
        pmS = pm[rng.permutation(n)]
        s = np.linalg.norm(pmS - pm[rng.integers(n)], axis=1) <= r_pm
        if s.sum() >= 25:
            null["pm_shuffled"].append(_z(sky[s])[1])

    rows = []
    for name, v in null.items():
        v = np.asarray([x for x in v if np.isfinite(x)])
        rows.append({"null": name, "trials": v.size, "median_z": float(np.median(v)),
                     "q90": float(np.quantile(v, .9)), "max_z": float(v.max()),
                     "p_value": float((np.sum(v >= z_obs) + 1) / (v.size + 1))})
    df = pd.DataFrame(rows)

    if verbose:
        nu_k = k_obs * np.pi * th ** 2 / A
        print(f"sky concentration of the PM-selected subsample   "
              f"(Delta_mu={r_pm:.3f} mas/yr, theta={th:.3f} deg)")
        print(f"  observed: k={k_obs}, max-disc count c={c_obs}, nu_k={nu_k:.2f}, "
              f"z={z_obs:.2f}")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print(f"\n  Poisson would give median z = 0 and p = {poisson.sf(c_obs-1, nu_k):.2e};")
        print("  the measured median z ~ 2.2 is the look-elsewhere offset of maximising")
        print("  over disc placements, NOT intrinsic clustering -- which is why the")
        print("  uniform control must be run before any of this is interpreted.")
    return {"z_obs": z_obs, "c_obs": c_obs, "k": k_obs, "r_pm": r_pm, "theta": th,
            "null": null, "table": df}


# ==========================================================================
# Metallicity: fetch, coverage audit, and a coherence test
# ==========================================================================
_FEH_SOURCES = {
    # name        : (what it is,                              practical limit)
    "gspphot":     ("Gaia DR3 GSP-Phot mh_gspphot",           "G < 19, prior-driven"),
    "gspspec":     ("Gaia DR3 GSP-Spec mh_gspspec",           "G_RVS < 12"),
    "chiti2021":   ("SMSS DR2 photometric [Fe/H] (VizieR J/ApJS/254/31)", "g < 16"),
    "smss_dr4":    ("SMSS DR4 u,v,g,i photometry (VizieR II/379/smssdr4)", "v < ~18.5"),
}


def fetch_metallicity(source_ids, ra=None, dec=None, *, source="gspphot",
                      cache_dir=".", chunk=2000, verbose=True):
    """[Fe/H] (or the photometry to derive it) for a list of Gaia source_ids.

    source="gspphot"   Gaia DR3 astrophysical_parameters. The only option with any
                       reach past G~18, but GSP-Phot is prior-driven and its quoted
                       errors are badly optimistic for metal-poor stars.
    source="gspspec"   RVS spectroscopy. G_RVS < 12; useless for halo dwarfs.
    source="chiti2021" SMSS DR2 photometric metallicities, joined on Gaia id.
                       Built for BRIGHT extremely-metal-poor candidates: g < 16.
    source="smss_dr4"  raw SMSS DR4 u/v/g/i, matched on sky within 1 arcsec. The v
                       band carries the Ca II H&K signal; check it is not all NaN
                       before doing anything else (see metallicity_coverage).

    Returns a DataFrame indexed by source_id. Cached on the id-list hash.
    """
    import hashlib, pickle as _pkl
    from pathlib import Path
    sids = np.asarray(source_ids, dtype="int64")
    key = hashlib.md5((source + ",".join(map(str, np.sort(sids)))).encode()).hexdigest()[:12]
    cf = Path(cache_dir) / f"feh_{source}_{sids.size}_{key}.pkl"
    if cf.exists():
        if verbose:
            print(f"  [Fe/H] <- cache {cf.name}")
        return _pkl.load(open(cf, "rb"))

    if source in ("gspphot", "gspspec"):
        from astroquery.gaia import Gaia
        col = "mh_gspphot" if source == "gspphot" else "mh_gspspec"
        extra = (", mh_gspphot_lower, mh_gspphot_upper, teff_gspphot, logg_gspphot"
                 if source == "gspphot" else "")
        out = []
        for i in range(0, sids.size, chunk):
            ids = ",".join(map(str, sids[i:i + chunk]))
            q = (f"SELECT source_id, {col}{extra} FROM gaiadr3.astrophysical_parameters "
                 f"WHERE source_id IN ({ids})")
            out.append(Gaia.launch_job_async(q).get_results().to_pandas())
        df = pd.concat(out, ignore_index=True)
        df = df.rename(columns={col: "feh"})
        if source == "gspphot" and "mh_gspphot_lower" in df:
            df["e_feh"] = (df["mh_gspphot_upper"] - df["mh_gspphot_lower"]) / 2.0
    else:
        from astroquery.vizier import Vizier
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        if ra is None or dec is None:
            raise ValueError(f"source={source!r} is matched on sky: pass ra, dec")
        if source == "chiti2021":
            vz = Vizier(columns=["**"], row_limit=-1)
            t = vz.query_region(SkyCoord(np.mean(ra), np.mean(dec), unit="deg"),
                                radius=2.0 * u.deg, catalog="J/ApJS/254/31")
            df = t[0].to_pandas() if len(t) else pd.DataFrame()
            if len(df):
                df["source_id"] = pd.to_numeric(df["Gaia"], errors="coerce")
                df = df.rename(columns={"[Fe/H]": "feh", "e_[Fe/H]": "e_feh"})
                df = df[df["source_id"].isin(sids)]
        else:
            vz = Vizier(columns=["RAICRS", "DEICRS", "uPSF", "vPSF", "e_vPSF",
                                 "gPSF", "iPSF"], row_limit=-1)
            t = vz.query_region(SkyCoord(np.asarray(ra), np.asarray(dec), unit="deg"),
                                radius=1.0 * u.arcsec, catalog="II/379/smssdr4")
            df = t[0].to_pandas() if len(t) else pd.DataFrame()
    if "source_id" in df:
        df = df.set_index("source_id")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    _pkl.dump(df, open(cf, "wb"))
    if verbose:
        n = int(np.isfinite(pd.to_numeric(df.get("feh", pd.Series(dtype=float)),
                                          errors="coerce")).sum()) if "feh" in df else 0
        print(f"  [Fe/H] {source}: {len(df)} rows, {n} finite -> {cf.name}")
    return df


def metallicity_coverage(run, r, gid=0, *, cache_dir=".", verbose=True):
    """Audit which metallicity sources actually reach this candidate's stars.

    Run this BEFORE designing any metallicity test. For a typical UFD candidate at
    G ~ 20 the answer is usually "none of them", and it is much cheaper to learn
    that here than after building the analysis.
    """
    ti = np.asarray(run["parts"]["test_idx"], int)
    mem = np.asarray(run["gamma_clusters"][gid], int)
    sid = np.asarray(r["source_id"], dtype="int64")[ti][mem]
    G = np.asarray(r["phot_g_mean_mag"], float)[ti][mem]
    ra = np.asarray(r["ra"], float)[ti][mem]
    dec = np.asarray(r["dec"], float)[ti][mem]
    rows = []
    for src in ("gspphot", "chiti2021", "smss_dr4"):
        try:
            df = fetch_metallicity(sid, ra, dec, source=src, cache_dir=cache_dir,
                                   verbose=False)
            if "feh" in df:
                n = int(np.isfinite(pd.to_numeric(df["feh"], errors="coerce")).sum())
            elif "vPSF" in df:
                n = int(np.isfinite(pd.to_numeric(df["vPSF"], errors="coerce")).sum())
            else:
                n = 0
        except Exception as e:
            n = -1
            if verbose:
                print(f"  {src}: FAILED {type(e).__name__}: {str(e)[:60]}")
        rows.append({"source": src, "what": _FEH_SOURCES[src][0],
                     "limit": _FEH_SOURCES[src][1], "n_usable": n, "n_members": mem.size})
    df = pd.DataFrame(rows)
    if verbose:
        print(f"metallicity coverage for {mem.size} members "
              f"(G: {G.min():.1f}-{G.max():.1f}, median {np.median(G):.1f})")
        print(df.to_string(index=False))
    return df


def metallicity_coherence(feh_mem, feh_field, *, e_mem=None, n_null=5000, seed=0,
                          verbose=True):
    """Do the members share a metallicity, and is it metal-poor for the field?

    Two one-sided tests against random field subsamples of the same size:
      dispersion  a bound system has a small intrinsic [Fe/H] spread
      location    a UFD sits metal-poor relative to the halo field

    Distance-independent, unlike an isochrone fit -- which is the reason to prefer
    it once the data exist.
    """
    fm = np.asarray(feh_mem, float); fm = fm[np.isfinite(fm)]
    ff = np.asarray(feh_field, float); ff = ff[np.isfinite(ff)]
    m = fm.size
    if m < 3 or ff.size < 10 * m:
        raise ValueError(f"not enough data: {m} members, {ff.size} field")
    sd_obs, mu_obs = float(np.std(fm, ddof=1)), float(np.mean(fm))
    sig_int = np.nan
    if e_mem is not None:
        e = np.asarray(e_mem, float)[np.isfinite(np.asarray(feh_mem, float))]
        sig_int = float(np.sqrt(max(sd_obs ** 2 - np.nanmean(e ** 2), 0.0)))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, ff.size, size=(int(n_null), m))
    sd_n = ff[idx].std(axis=1, ddof=1); mu_n = ff[idx].mean(axis=1)
    p_disp = float((np.sum(sd_n <= sd_obs) + 1) / (n_null + 1))
    p_loc  = float((np.sum(mu_n <= mu_obs) + 1) / (n_null + 1))
    if verbose:
        print(f"metallicity coherence: {m} members vs {ff.size} field stars")
        print(f"  members: mean [Fe/H] = {mu_obs:+.2f}, dispersion = {sd_obs:.2f} dex"
              + (f", intrinsic = {sig_int:.2f} dex" if np.isfinite(sig_int) else ""))
        print(f"  field  : mean [Fe/H] = {np.mean(ff):+.2f}, dispersion = {np.std(ff):.2f} dex")
        print(f"  p(dispersion this tight) = {p_disp:.4f}")
        print(f"  p(mean this metal-poor)  = {p_loc:.4f}")
    return {"n": m, "mean": mu_obs, "sd": sd_obs, "sigma_int": sig_int,
            "p_dispersion": p_disp, "p_location": p_loc,
            "null_sd": sd_n, "null_mean": mu_n}


# ==========================================================================
# Local significance: on/off counting instead of overlap-weighted repechage
# ==========================================================================
def li_ma(n_on, n_off, alpha):
    """Li & Ma (1983) Eq. 17 significance for on/off counting with exposure ratio
    alpha = t_on/t_off.  Correct for small counts, where S/sqrt(B) is not, and
    finite where B -> 0.  Signed by the sign of the excess."""
    n_on, n_off, alpha = float(n_on), float(n_off), float(alpha)
    if n_on <= 0 and n_off <= 0:
        return 0.0
    tot = n_on + n_off
    t1 = n_on * np.log(((1 + alpha) / alpha) * (n_on / tot)) if n_on > 0 else 0.0
    t2 = n_off * np.log((1 + alpha) * (n_off / tot)) if n_off > 0 else 0.0
    s = np.sqrt(max(2.0 * (t1 + t2), 0.0))
    return float(np.sign(n_on - alpha * n_off) * s)


def local_significance_onoff(run, r, gid=0, *, fracs=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
                             use_all_Y=True, verbose=True):
    """Local S and significance from direct on/off counting in the stacked geometry.

    The estimator currently in the pipeline routes a counting problem through
    EagleEye's repechage bookkeeping: it averages the per-cluster Eq.9 backgrounds
    B_hat_{j,c} with weights o_{j,c} (the overlap with A_Gamma). Three things go
    wrong with that:

      1. o_{j,c} is EXTENSIVE (a count) but is used to weight an average of
         extensive quantities -- B_hat_{j,c} refers to all of cluster (j,c), not to
         the part inside A_alpha, so large EE clusters contribute a background that
         mostly describes sky outside the Gamma cluster.
      2. the z > 0 filter discards contributions whose B_hat is large, which is a
         selection on the very quantity being estimated.
      3. S = |A_alpha| is the FLAGGED count, so S/sqrt(B) has a null floor of
         sqrt(B) rather than zero.

    None of that bookkeeping is needed. The 8 references ARE the off-region. Define
    a ball around the Gamma-cluster centroid in the shared scaled metric, count

        N_on  = Y points inside it,
        N_off = sum_j X^(j) points inside the same ball (same relative coordinates),
        alpha = n_Y / sum_j n_X^(j),
        B_hat = alpha * N_off,   S = N_on - B_hat,

    and report Li & Ma. Because the references are built on the same local frame,
    the ball transfers to each tile without any remapping.

    use_all_Y=True counts EVERY Y point in the ball, not only the flagged ones,
    which removes the post-selection bias in S. The region is still defined from
    the flagged points, so mild circularity remains -- scan `fracs` and read the
    trend, not a single row.
    """
    import robustness as rb
    Y = np.asarray(run["Y"], float)
    X_refs, _ = rb.refs_after_equalisation(run, r)
    nY, nXs = len(Y), [len(x) for x in X_refs]
    alpha = nY / float(sum(nXs))
    mem = np.asarray(run["gamma_clusters"][gid], int)
    c = Y[mem].mean(axis=0)
    d_mem = np.linalg.norm(Y[mem] - c, axis=1)
    dY = np.linalg.norm(Y - c, axis=1)
    dX = [np.linalg.norm(Xj - c, axis=1) for Xj in X_refs]

    rows = []
    for f in fracs:
        R = float(np.quantile(d_mem, f))
        n_on = int((dY <= R).sum()) if use_all_Y else int((d_mem <= R).sum())
        per = [int((dj <= R).sum()) for dj in dX]
        n_off = int(sum(per))
        B = alpha * n_off
        rows.append({"frac": f, "R": R, "N_on": n_on, "N_off": n_off,
                     "B_hat": B, "S": n_on - B,
                     "S_over_rootB": (n_on - B) / np.sqrt(B) if B > 0 else np.nan,
                     "Z_LiMa": li_ma(n_on, n_off, alpha),
                     "n_refs_nonzero": int(np.sum(np.asarray(per) > 0))})
    df = pd.DataFrame(rows)
    if verbose:
        print(f"on/off local significance   nY={nY}  sum nX={sum(nXs)}  alpha={alpha:.4f}")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print("\n  S_over_rootB here is the EXCESS over sqrt(B) -- it is zero under the")
        print("  null, unlike |A_Gamma|/sqrt(B). Z_LiMa is the likelihood-ratio version")
        print("  and is the number to quote: it is calibrated at small counts and stays")
        print("  finite as B -> 0.")
    return {"table": df, "alpha": alpha, "nY": nY, "nXs": nXs, "centroid": c}


def compare_local_estimators(run, r, gid=0, *, frac=0.8, verbose=True):
    """Side-by-side: the pipeline's overlap-weighted estimator, a footprint-corrected
    version of it, and on/off counting. Needs keep_heavy for the first two."""
    out = {}
    cs = run.get("cluster_stats", {}).get(int(gid))
    if cs is not None:
        out["pipeline"] = {"S": cs["S"], "B": cs["B_weighted"],
                           "stat": cs["s_over_rootB"], "form": "|A_Gamma|/sqrt(B_w)"}
        contrib = cs.get("contributors") or []
        if contrib and run.get("EE_books") is not None:
            num = den = 0.0
            for e in contrib:
                book = run["EE_books"][e["ref"]]["Y_OVER_clusters"][e["cid"]]
                M = len(np.asarray(book.get("Repechaged", []), int))
                if M:
                    num += e["Bhat"] * e["overlap"] / M; den += 1.0
            if den:
                Bf = num / den
                out["footprint"] = {"S": cs["S"], "B": Bf,
                                    "stat": (cs["S"] - Bf) / np.sqrt(Bf) if Bf > 0 else np.nan,
                                    "form": "(S - B_apportioned)/sqrt(B)"}
    oo = local_significance_onoff(run, r, gid, fracs=(frac,), verbose=False)
    row = oo["table"].iloc[0]
    out["onoff"] = {"S": row["S"], "B": row["B_hat"], "stat": row["Z_LiMa"],
                    "form": "Li & Ma on/off"}
    if verbose:
        print(f"{'estimator':<12} {'S':>8} {'B':>9} {'statistic':>10}   form")
        for k, v in out.items():
            print(f"{k:<12} {v['S']:>8.1f} {v['B']:>9.2f} {v['stat']:>10.2f}   {v['form']}")
    return out


def local_significance_native(run, gid=0, *, min_overlap=1, combine="scatter",
                              verbose=True):
    """Local significance from EagleEye's OWN per-reference estimator. No apertures.

    EagleEye already computes the right thing. S_rootB_estimate_Y_overdensities
    returns, per overdensity cluster c of reference j,

        z_{j,c} = ( |Repechaged_{j,c}| - B_hat_{j,c} ) / sqrt( B_hat_{j,c} )

    which IS background-subtracted. The pipeline's
    srootB_for_Gamma_clusters_from_Brefs receives these values in SrootB_by_ref and
    uses them only as a `z > 0` boolean gate, then rebuilds |A_Gamma|/sqrt(B_w)
    -- discarding the subtraction and introducing the sqrt(B) null floor.

    Here the overlap o_{j,c} is used ONLY to identify which EE cluster corresponds
    to the Gamma cluster (largest overlap wins), never to weight a background. That
    removes the footprint mismatch: each reference contributes its own internally
    consistent (S, B) pair.

    combine="scatter"  Z = mean(z) / (sd(z)/sqrt(n)). Lets the observed
                       reference-to-reference dispersion set the error. Use this:
                       if the z_j disagree by much more than unit variance the
                       model is wrong and the assumed-variance version is a fiction.
    combine="reff"     Z = mean(z) * sqrt(R_eff), with R_eff from the Upsilon
                       correlation matrix. Assumes each z_j has unit variance under
                       the null, which the scatter usually contradicts.

    n_refs is as important as Z: it is how many of the R references independently
    produced a cluster here at all.
    """
    cs = run.get("cluster_stats", {}).get(int(gid))
    if cs is None:
        raise KeyError(f"no cluster_stats for gid={gid}")
    best = {}
    for e in cs.get("contributors") or []:
        if e["overlap"] < int(min_overlap):
            continue
        if e["ref"] not in best or e["overlap"] > best[e["ref"]]["overlap"]:
            best[e["ref"]] = e
    R = int(np.asarray(run["Upsilon_by_ref"]).shape[0])
    if not best:
        return {"Z": np.nan, "n_refs": 0, "R": R, "z": np.array([])}
    refs = sorted(best)
    z = np.array([best[j]["z"] for j in refs], float)
    n = z.size
    Cm = np.corrcoef(np.asarray(run["Upsilon_by_ref"], float))
    rho = float((Cm.sum() - R) / (R * (R - 1)))
    R_eff = R / (1.0 + (R - 1) * rho)
    sd = float(np.std(z, ddof=1)) if n > 1 else np.nan
    Z_scatter = float(z.mean() / (sd / np.sqrt(n))) if (n > 1 and sd > 0) else np.nan
    Z_reff = float(z.mean() * np.sqrt(min(R_eff, n)))
    out = {"Z": Z_scatter if combine == "scatter" else Z_reff,
           "Z_scatter": Z_scatter, "Z_reff": Z_reff, "z": z, "refs": refs,
           "n_refs": n, "R": R, "rho": rho, "R_eff": R_eff,
           "mean_z": float(z.mean()), "sd_z": sd,
           "overlaps": [best[j]["overlap"] for j in refs],
           "Bhat": [best[j]["Bhat"] for j in refs]}
    if verbose:
        print(f"native EE local significance, cluster {gid} (|A_alpha|={cs['S']})")
        print(f"  contributing references: {n} of {R}   refs {refs}")
        print(f"  per-reference z: {np.round(z,2)}")
        print(f"  overlaps       : {out['overlaps']}")
        print(f"  mean z = {z.mean():.2f}, sd = {sd:.2f}"
              f"   (sd ~ 1 expected if the z_j are consistent)")
        print(f"  rho = {rho:.3f} -> R_eff = {R_eff:.2f}")
        print(f"  Z (scatter-based) = {Z_scatter:.2f}      <- recommended")
        print(f"  Z (R_eff-based)   = {Z_reff:.2f}")
        if n > 1 and sd > 2.0:
            print(f"  WARNING: sd(z)={sd:.1f} >> 1. The references disagree far more than")
            print( "  sampling allows -- one reference is probably carrying the result.")
        if n < R:
            print(f"  NOTE: {R-n} reference(s) produced no matching cluster (no overlap,")
            print( "  or dropped by EE's lenSo<5 rule or the z>0 gate). That absence is")
            print( "  persistence information and should be reported alongside Z.")
    return out


# ==========================================================================
# Per-reference contribution accounting (Li & Ma on the native repechage sets)
# ==========================================================================
def reference_contributions(run, gid=0, *, min_overlap=1, min_overlap_frac=0.0):
    """One row per reference that produced an EE cluster overlapping Gamma-cluster gid.

    Eq.9 IS an on/off measurement: ON = |Repechaged|, OFF = |Background| (the
    injected control sample), alpha = (nY-|W_o|)/(nX-|W_u|). So Li & Ma applies
    directly to EagleEye's own sets -- no aperture, no ball.

    EE's own z = (N_on - alpha N_off)/sqrt(alpha N_off) treats B_hat as exact. It is
    not: N_off is a Poisson count too, so Var = B(1+alpha). At the alpha ~ 1 that
    equalisation enforces, EE's z is overstated by ~1/sqrt(2).

    Exact N_on/N_off come from EE_books when present (keep_heavy); otherwise they are
    recovered from the stored (Bhat, z) with alpha ~ nY/nX_j, which is accurate to a
    fraction of a count. `exact` records which was used.
    """
    cs = (run.get("cluster_stats") or {}).get(int(gid))
    if cs is None:
        return pd.DataFrame()
    nY, nXs = run["nY"], run["nXs"]
    books = run.get("EE_books")
    best = {}
    for e in (cs.get("contributors") or []):
        if e["overlap"] < int(min_overlap):
            continue
        if e["ref"] not in best or e["overlap"] > best[e["ref"]]["overlap"]:
            best[e["ref"]] = e
    S_alpha = int(cs["S"])
    rows = []
    for j in sorted(best):
        e = best[j]; B = float(e["Bhat"]); z = float(e["z"])
        exact = False
        if books is not None:
            try:
                oc = books[j]["Y_OVER_clusters"][e["cid"]]
                n_on = len(np.asarray(oc["Repechaged"], int))
                n_off = len(np.asarray(oc["Background"], int))
                alpha = B / n_off if n_off else np.nan
                exact = True
            except Exception:
                exact = False
        if not exact:
            alpha = nY / float(nXs[j])
            n_on = int(round(z * np.sqrt(B) + B)); n_off = int(round(B / alpha))
        # Overlap FRACTION, not just the count. A reference whose EE cluster has
        # 101 members and shares 3 of them with a 22-point Gamma cluster is
        # describing a different object (Reticulum II, typically) and should not
        # lend it its significance. ov_frac normalises by the smaller of the two.
        ov_frac = e["overlap"] / max(1, min(S_alpha, int(n_on)))
        if ov_frac < float(min_overlap_frac):
            continue
        rows.append({"ref": j, "cid": int(e["cid"]), "overlap": int(e["overlap"]),
                     "ov_frac": float(ov_frac),
                     "N_on": int(n_on), "N_off": int(n_off), "alpha": float(alpha),
                     "B_hat": B, "z_EE": z, "Z_LiMa": li_ma(n_on, n_off, alpha),
                     "exact": exact})
    return pd.DataFrame(rows)


def combined_significance(run, gid=0, *, min_overlap=1, min_overlap_frac=0.0):
    """Combine the per-reference Li & Ma values three ways. Reports n_refs, which
    matters as much as Z: a Z built from 3 references is not a Z built from 8."""
    df = reference_contributions(run, gid, min_overlap=min_overlap,
                                 min_overlap_frac=min_overlap_frac)
    R = int(np.asarray(run["Upsilon_by_ref"]).shape[0])
    out = {"gid": int(gid), "R": R, "n_refs": len(df), "refs": list(df["ref"]) if len(df) else [],
           "S": (run["cluster_stats"][int(gid)]["S"] if gid in run.get("cluster_stats", {}) else np.nan),
           "srb_pipeline": (run["cluster_stats"][int(gid)]["s_over_rootB"]
                            if gid in run.get("cluster_stats", {}) else np.nan)}
    if not len(df):
        out.update(Z_scatter=np.nan, Z_reff=np.nan, Z_pooled=np.nan,
                   mean_Z=np.nan, sd_Z=np.nan, R_eff=np.nan, rho=np.nan)
        return out
    Z = df["Z_LiMa"].to_numpy(float); n = Z.size
    U = np.asarray(run["Upsilon_by_ref"], float); Cm = np.corrcoef(U)
    rho = float((Cm.sum() - R) / (R * (R - 1)))
    R_eff = R / (1.0 + (R - 1) * rho)
    sd = float(np.std(Z, ddof=1)) if n > 1 else np.nan
    non = float(df["N_on"].mean())                       # same Y points in each ref
    noff = float(df["N_off"].sum())
    a_pool = 1.0 / float(np.sum(1.0 / df["alpha"].to_numpy(float)))
    out.update(mean_Z=float(Z.mean()), sd_Z=sd, rho=rho, R_eff=R_eff,
               # n=2 gives a scatter estimate on one degree of freedom: it swings
               # over orders of magnitude and is not usable. Require 3.
               Z_scatter=(float(Z.mean() / (sd / np.sqrt(n))) if n >= 3 and sd > 0 else np.nan),
               Z_reff=float(Z.mean() * np.sqrt(min(R_eff, n))),
               Z_pooled=li_ma(round(non), round(noff), a_pool),
               n_indep=float(n / (1 + (n - 1) * rho)))
    return out


def contribution_line(run, gid=0):
    """One compact line for run_summary."""
    c = combined_significance(run, gid)
    if not c["n_refs"]:
        return f"cl{gid}: no reference clusters"
    return (f"cl{gid}: refs {c['n_refs']}/{c['R']} {c['refs']} | "
            f"Z_LiMa pooled={c['Z_pooled']:.2f} scatter={c['Z_scatter']:.2f} "
            f"| EE srb={c['srb_pipeline']:.2f}")


def contribution_report(run, gid=None, *, verbose=True):
    """Full per-reference table plus the combined numbers, for every Gamma cluster."""
    gids = sorted(run.get("gamma_clusters", {})) if gid is None else [int(gid)]
    tabs, summ = {}, []
    for g in gids:
        df = reference_contributions(run, g)
        c = combined_significance(run, g)
        tabs[g] = df; summ.append(c)
        if verbose:
            print(f"\n--- Gamma cluster {g}:  |A|={c['S']}  "
                  f"pipeline S/sqrt(B)={c['srb_pipeline']:.2f} ---")
            if not len(df):
                print("  no reference produced an overlapping EE cluster"); continue
            print(df.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
            print(f"  contributing references : {c['n_refs']} of {c['R']}  {c['refs']}")
            print(f"  mean Z_LiMa {c['mean_Z']:.2f}  sd {c['sd_Z']:.2f}"
                  f"   (sd ~ 1 if the references agree)")
            print(f"  rho={c['rho']:.3f}  R_eff={c['R_eff']:.2f}  "
                  f"effective independent refs here = {c['n_indep']:.1f}")
            print(f"  Z_pooled  = {c['Z_pooled']:.2f}   (pool the OFF counts; "
                  f"optimistic, refs are correlated)")
            print(f"  Z_scatter = {c['Z_scatter']:.2f}   (let the reference spread set "
                  f"the error; recommended)")
    return {"tables": tabs, "summary": pd.DataFrame(summ)}


def scan_contributions(scan_runs, cfgs=None, *, verbose=True):
    """The big table: every configuration x Gamma cluster x reference.

    Use it to see whether the SAME references support the feature as the tile moves,
    or whether a different subset carries it each time -- a persistent centroid
    driven by a rotating cast of references is a very different claim from one
    supported by the same references throughout.
    """
    # rb.scan returns a DICT {label: run}; accept a list too.
    if isinstance(scan_runs, dict):
        labs = ([c["label"] for c in cfgs if c["label"] in scan_runs]
                if cfgs is not None else list(scan_runs))
        pairs = [(l, scan_runs[l]) for l in labs]
    else:
        labs = ([c["label"] for c in cfgs] if cfgs is not None
                else [f"cfg{i}" for i in range(len(scan_runs))])
        pairs = list(zip(labs, scan_runs))
    rows, summ = [], []
    for lab, run in pairs:
        if run is None:
            continue
        for g in sorted(run.get("gamma_clusters", {})):
            df = reference_contributions(run, g)
            c = combined_significance(run, g)
            c["config"] = lab; summ.append(c)
            for _, rw in df.iterrows():
                d = rw.to_dict(); d["config"] = lab; d["gid"] = g; rows.append(d)
    per_ref = pd.DataFrame(rows)
    per_cl = pd.DataFrame(summ)
    if len(per_cl):
        cols = ["config", "gid", "S", "n_refs", "R", "refs", "Z_pooled",
                "Z_scatter", "srb_pipeline"]
        per_cl = per_cl[[c for c in cols if c in per_cl.columns]]
    if verbose and len(per_cl):
        print("per-configuration summary")
        print(per_cl.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
        if len(per_ref):
            tally = per_ref.groupby("ref").size()
            print(f"\nhow often each reference contributes, across all configs/clusters:")
            print("  " + "  ".join(f"ref{k}:{v}" for k, v in tally.items()))
    return {"per_reference": per_ref, "per_cluster": per_cl}


# ==========================================================================
# Show WHICH reference tiles support a cluster, on the Part-3 plots
# ==========================================================================
def match_cluster_to_target(run, r, target=None, *, gid=None):
    """Which Gamma cluster of `run` is the anomaly? By source_id overlap with the
    target's member set if available, else by sky centroid. Returns a gid."""
    cl = run.get("gamma_clusters") or {}
    if gid is not None:
        return int(gid)
    if not cl:
        return None
    if len(cl) == 1:
        return int(next(iter(cl)))
    if target is None:
        # largest cluster, as a last resort
        return int(max(cl, key=lambda g: np.size(cl[g])))
    import robustness as rb
    _, ra, dec, _, _, sid = rb._yspace(run, r)
    A0 = target.get("source_ids")
    if A0:
        A0 = set(int(s) for s in A0)
        best, nbest = None, -1
        for g, ix in cl.items():
            ix = np.asarray(ix, int)
            n = len(A0 & set(int(s) for s in sid[ix]))
            if n > nbest:
                best, nbest = int(g), n
        if nbest > 0:
            return best
    # fall back to the nearest centroid on the sky
    tr, td = float(target["ra_c"]), float(target["dec_c"])
    def _d(g):
        ix = np.asarray(cl[g], int)
        return np.hypot((np.mean(ra[ix]) - tr) * np.cos(np.deg2rad(td)),
                        np.mean(dec[ix]) - td)
    return int(min(cl, key=_d))


def annotate_contributing_refs(ax, run, r=None, target=None, *, gid=None,
                               space="sky", show_labels=True, alpha_fill=0.10,
                               colour="tab:red", fontsize=8, min_overlap_frac=0.0,
                               verbose=True):
    """Outline the reference TILES that produced an EE cluster overlapping the
    anomaly, annotated with both local estimates.

    Drop one line after any plot_field call:
        ax = plot_field(run_s, SCAN_R, neigh_s, field_cat_s, space="sky")
        wt.annotate_contributing_refs(ax, run_s, SCAN_R, TARGET)

    On the sky panel the contributing tiles are shaded and labelled

        R<j>  z=<EE>  Z=<LiMa>

    where z is EagleEye's own (N_on - B_hat)/sqrt(B_hat) and Z is Li & Ma on the
    same repechage sets. Non-contributing tiles are left plain: their silence is the
    point. In PM space the tiles have no meaning (all references overlap there), so
    only the text box is drawn.
    """
    import matplotlib.pyplot as plt
    import analyze_known_dwarfs as akd
    g = match_cluster_to_target(run, r, target, gid=gid)
    if g is None:
        if verbose:
            print("no Gamma clusters in this run")
        return None
    df = reference_contributions(run, g, min_overlap_frac=min_overlap_frac)
    R = len(run["nXs"])
    if not len(df):
        if verbose:
            print(f"cluster {g}: no reference produced an overlapping EE cluster")
        return {"gid": g, "table": df}

    grid = run["grid"]; h = grid["cell_h"]
    ref_ids = list(run["parts"]["ref_ids"])
    if space == "sky":
        for _, row in df.iterrows():
            j = int(row["ref"])
            cid = ref_ids[j]
            rc, dc = akd.cell_center_from_id(grid["ra0"], grid["dec0"], cid, h)
            rc = grid["ra0"] + akd.wrap_dra_deg(np.asarray([rc], float), grid["ra0"])[0]
            ax.add_patch(plt.Rectangle((rc - h, dc - h), 2 * h, 2 * h, fill=True,
                                       fc=colour, ec=colour, lw=2.0, alpha=alpha_fill,
                                       zorder=1.5))
            ax.add_patch(plt.Rectangle((rc - h, dc - h), 2 * h, 2 * h, fill=False,
                                       ec=colour, lw=2.0, zorder=4))
            if show_labels:
                ax.text(rc, dc + 0.86 * h,
                        f"R{j}  z={row['z_EE']:.1f}  Z={row['Z_LiMa']:.1f}",
                        ha="center", va="top", fontsize=fontsize, color=colour,
                        weight="bold", zorder=10,
                        bbox=dict(fc="w", ec=colour, lw=0.8, alpha=0.85, pad=1.4))

    c = combined_significance(run, g, min_overlap_frac=min_overlap_frac)
    txt = (f"cluster {g}: {c['n_refs']}/{R} refs {c['refs']}\n"
           f"EE $S/\\sqrt{{\\hat B}}$ = {c['srb_pipeline']:.2f}\n"
           f"$Z_{{\\rm LiMa}}$ pooled = {c['Z_pooled']:.2f}"
           + (f", scatter = {c['Z_scatter']:.2f}" if np.isfinite(c["Z_scatter"]) else ""))
    ax.text(0.985, 0.015, txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=fontsize + 0.5, zorder=11,
            bbox=dict(fc="w", ec=colour, lw=1.0, alpha=0.9, pad=3.0))

    if verbose:
        print(f"cluster {g} (|A|={c['S']}): {c['n_refs']}/{R} references contribute")
        print(df[["ref", "cid", "overlap", "ov_frac", "N_on", "N_off", "alpha",
                  "B_hat", "z_EE", "Z_LiMa"]].to_string(index=False,
                  float_format=lambda v: f"{v:.3g}"))
        print(f"  tiles highlighted: {[ref_ids[int(j)] for j in df['ref']]}"
              f"   (of {ref_ids})")
    return {"gid": g, "table": df, "combined": c}


def label_reference_tiles(ax, run, gid=None, r=None, target=None, *, stat="z_EE",
                          corner="lower left", pad=0.06, fontsize=8,
                          colour="tab:red", dim="0.45", box=True, outline=True,
                          verbose=False):
    """Write EagleEye's per-reference S/sqrt(B) into the corner of each reference tile.

    One number per tile of the 3x3, in the same spirit as the equalised-reference
    outline that plot_field already draws. References that produced an EE cluster
    overlapping the anomaly are drawn in `colour` and outlined; the rest are dimmed,
    and show their whole-reference 'Total' estimate when EE_books are available
    (keep_heavy=True), otherwise a dash.

    stat : "z_EE"   EagleEye's own (N_on - B_hat)/sqrt(B_hat) for that reference
           "lima"   Li & Ma on the same repechage sets
           "both"   "z / Z"

    Call it straight after plot_field, which returns its ax:
        ax = plot_field(run_s, SCAN_R, neigh_s, field_cat_s, space="sky")
        wt.label_reference_tiles(ax, run_s, r=SCAN_R, target=TARGET)
    """
    import analyze_known_dwarfs as akd
    import matplotlib.pyplot as plt

    g = match_cluster_to_target(run, r, target, gid=gid)
    df = reference_contributions(run, g) if g is not None else pd.DataFrame()
    got = {int(row["ref"]): row for _, row in df.iterrows()} if len(df) else {}

    grid = run["grid"]; h = grid["cell_h"]
    ref_ids = list(run["parts"]["ref_ids"])
    srb_by_ref = run.get("SrootB_by_ref")

    def _fmt(j):
        if j in got:
            row = got[j]
            if stat == "z_EE":   return f"{row['z_EE']:.1f}", True
            if stat == "lima":   return f"{row['Z_LiMa']:.1f}", True
            return f"{row['z_EE']:.1f} / {row['Z_LiMa']:.1f}", True
        if srb_by_ref is not None:
            try:
                tot = float(srb_by_ref[j]["s/root(B)"]["Total"])
                return f"({tot:.1f})", False          # whole-reference, not this cluster
            except Exception:
                pass
        return "--", False

    for j, cid in enumerate(ref_ids):
        rc, dc = akd.cell_center_from_id(grid["ra0"], grid["dec0"], cid, h)
        rc = grid["ra0"] + akd.wrap_dra_deg(np.asarray([rc], float), grid["ra0"])[0]
        txt, contrib = _fmt(j)
        col = colour if contrib else dim
        if outline and contrib:
            ax.add_patch(plt.Rectangle((rc - h, dc - h), 2 * h, 2 * h, fill=False,
                                       ec=colour, lw=1.8, zorder=4))
        # RA axis is inverted, so "left" on screen is the LARGER RA
        if "left" in corner:
            x, ha = rc + h * (1 - pad), "left"
        else:
            x, ha = rc - h * (1 - pad), "right"
        y, va = ((dc - h * (1 - pad), "bottom") if "lower" in corner
                 else (dc + h * (1 - pad), "top"))
        ax.text(x, y, f"R{j} {txt}", ha=ha, va=va, fontsize=fontsize, color=col,
                weight=("bold" if contrib else "normal"), zorder=10,
                bbox=(dict(fc="w", ec=col, lw=0.7, alpha=0.85, pad=1.2) if box else None))
        if verbose:
            print(f"  tile cell{cid} (ref {j}): {txt}"
                  + ("  <- contributes" if contrib else ""))
    return {"gid": g, "table": df}

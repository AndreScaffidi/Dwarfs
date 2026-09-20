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
    from scipy.stats import gaussian_kde
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

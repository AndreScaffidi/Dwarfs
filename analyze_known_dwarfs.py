#!/usr/bin/env python3
"""
Run the Moriond Tucana II multi-reference EagleEye analysis for every
class=4 (kinematically confirmed) dwarf in the Drlica-Wagner+ 2020 catalog.

For each dwarf:
  1. Gaia DR3 cone-equivalent square query around the dwarf center.
  2. Split into 3x3 grid -> Y (centre) and 8 X_refs.
  3. Build 4D features [dRAcos(dec0), dDec, pmra, pmdec], MAD-scaled.
  4. Compute per-ref nulls, run EagleEye.Soar on each (Xj, Y) pair, get Upsilon_j.
  5. Aggregate Gamma_i = sum_j Upsilon_j, repechage, S/sqrt(B) per ref.
  6. Bootstrap empirical Gamma null on Uniform([0,1]^d), threshold Gamma_i.
  7. DBSCAN-cluster A_Gamma, compute weighted Bhat per Gamma cluster,
     hence S/sqrt(Bhat) per cluster.
  8. If a Battaglia+2022 member table exists for the galaxy, crossmatch by
     DR3 source_id and compute per-cluster TPR / overlap.
  9. Save: per-dwarf pickle + sky + PM plots; one summary CSV at the end.

Usage:
  python analyze_known_dwarfs.py                       # all class=4
  python analyze_known_dwarfs.py --names "Tucana II"   # one
  python analyze_known_dwarfs.py --skip-existing       # resume
"""

import argparse
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ---- silence noisy matplotlib backend on a headless cluster
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.stats import gaussian_kde  # used by plot_Gamma_clusters_kde_overlay

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree

from astropy.io import ascii
from astroquery.gaia import Gaia
from astroquery.vizier import Vizier

# ---- EagleEye module (matches notebook layout)
EE_PATH = "/data/ascaffid/LHC_Olympics/EagleEye/eagleeye"
if EE_PATH not in sys.path:
    sys.path.append(EE_PATH)

import EagleEye  # noqa: E402
from utils_EE import partitioning_function  # noqa: E402
from EagleEye import (  # noqa: E402
    S_rootB_estimate_Y_overdensities,
    B_estimate_Y_overdensities,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==========================================================================
# Defaults (copied from the Tucana notebook)
# ==========================================================================
DEFAULT_DWARFS_TABLE = "/data/ascaffid/Dwarfs/gaia_ee/apjab7eb9t2_mrt.txt"
DEFAULT_OUT_DIR      = "/data/ascaffid/Dwarfs/gaia_ee/known_dwarfs_runs"
DEFAULT_DEGSEP       = 5.0      # 3x3 window halfwidth (deg) => centre cell 2*degsep/3
DEFAULT_KM           = 300
DEFAULT_PEXT         = 1e-5
DEFAULT_N_BOOT       = 1        # bootstraps for empirical Gamma null
DEFAULT_NULL_N       = 200_000  # MC null sample size per reference
DEFAULT_Z            = 2.65     # DPA cluster split parameter
DEFAULT_DBSCAN_EPS   = 1.0
DEFAULT_DBSCAN_MIN   = 5
DEFAULT_N_JOBS       = 8
BATTAGLIA_CATALOG    = "J/A+A/657/A54"  # Battaglia+2022 DR3 dwarf members

# ==========================================================================
# 3x3 windowing + feature build (from notebook cell 30)
# ==========================================================================
def wrap_dra_deg(ra, ra0):
    return (ra - ra0 + 180.0) % 360.0 - 180.0

def ra_condition(ra0, degsep):
    ra_min = (ra0 - degsep) % 360.0
    ra_max = (ra0 + degsep) % 360.0
    if ra_min <= ra_max:
        return f"(ra BETWEEN {ra_min} AND {ra_max})"
    return f"(ra >= {ra_min} OR ra <= {ra_max})"

def split_into_3x3(ra, dec, ra0, dec0, cell_halfwidth_deg):
    h = float(cell_halfwidth_deg)
    dra  = wrap_dra_deg(np.asarray(ra, float), ra0)
    ddec = np.asarray(dec, float) - dec0
    ix = np.floor((dra + 3*h) / (2*h)).astype(int)
    iy = np.floor((ddec + 3*h) / (2*h)).astype(int)
    inside = (ix >= 0) & (ix < 3) & (iy >= 0) & (iy < 3)
    cell_id = np.full(len(dra), -1, dtype=int)
    cell_id[inside] = iy[inside] * 3 + ix[inside]
    test_id = 4
    ref_ids = [0, 1, 2, 3, 5, 6, 7, 8]
    return {
        "cell_id": cell_id,
        "inside": inside,
        "test_idx": np.where(cell_id == test_id)[0],
        "ref_idx_list": [np.where(cell_id == rid)[0] for rid in ref_ids],
        "ref_ids": ref_ids,
    }

def robust_fit_mad(Z, eps=1e-12):
    med = np.median(Z, axis=0)
    mad = np.median(np.abs(Z - med), axis=0)
    mad = np.where(mad < eps, 1.0, mad)
    return med, mad

def robust_apply(Z, med, mad):
    return (Z - med) / mad

def build_features_local_center(r, idx, ra_c, dec_c, dec0_for_cos):
    ra    = np.asarray(r["ra"][idx], dtype=float)
    dec   = np.asarray(r["dec"][idx], dtype=float)
    pmra  = np.asarray(r["pmra"][idx], dtype=float)
    pmdec = np.asarray(r["pmdec"][idx], dtype=float)
    dra  = wrap_dra_deg(ra, ra_c) * np.cos(np.deg2rad(dec0_for_cos))  # const cos across tiles
    ddec = dec - dec_c
    return np.column_stack([dra, ddec, pmra, pmdec]).astype(float)

def cell_center_from_id(ra0, dec0, cell_id, cell_halfwidth_deg):
    h = float(cell_halfwidth_deg)
    iy = cell_id // 3
    ix = cell_id % 3
    off = np.array([-2*h, 0.0, 2*h])
    return ra0 + off[ix], dec0 + off[iy]

def make_Xrefs_and_Y_from_window_3x3(
    r, ra0, dec0, cell_halfwidth_deg,
    fit_scaling_on="all", do_robust_scale=True,
):
    ra  = np.asarray(r["ra"], float)
    dec = np.asarray(r["dec"], float)
    parts = split_into_3x3(ra, dec, ra0, dec0, cell_halfwidth_deg)

    Y_raw = build_features_local_center(r, parts["test_idx"], ra0, dec0, dec0)
    X_refs_raw = []
    for rid, idx in zip(parts["ref_ids"], parts["ref_idx_list"]):
        ra_c, dec_c = cell_center_from_id(ra0, dec0, rid, cell_halfwidth_deg)
        X_refs_raw.append(build_features_local_center(r, idx, ra_c, dec_c, dec0))

    scale = {"med": None, "mad": None}
    if do_robust_scale:
        if fit_scaling_on == "test":
            Z_fit = Y_raw
        elif fit_scaling_on == "all":
            Z_fit = np.vstack([Y_raw] + X_refs_raw)
        else:
            raise ValueError("fit_scaling_on must be 'test' or 'all'.")
        med, mad = robust_fit_mad(Z_fit)
        scale = {"med": med, "mad": mad}
        Y      = robust_apply(Y_raw, med, mad)
        X_refs = [robust_apply(Xj, med, mad) for Xj in X_refs_raw]
    else:
        Y, X_refs = Y_raw, X_refs_raw
    return Y, X_refs, parts, scale

# ==========================================================================
# Null + Soar + Gamma aggregation (from notebook cells 33, 37)
# ==========================================================================
def _compute_the_null_worker(args):
    j, p_j, K_M, N = args
    from utils_EE import compute_the_null as _cn
    return j, float(p_j), _cn(p=p_j, K_M=K_M, N=N)

def compute_the_null_multi_refs(p_list, K_M, N=200_000, n_jobs=8):
    p_list = list(map(float, p_list))
    tasks  = [(j, p, int(K_M), int(N)) for j, p in enumerate(p_list)]
    by_ref = {}
    with ProcessPoolExecutor(max_workers=min(int(n_jobs), len(tasks) or 1)) as ex:
        futures = [ex.submit(_compute_the_null_worker, t) for t in tasks]
        for fut in as_completed(futures):
            j, p_j, sn = fut.result()
            by_ref[j] = {"p": p_j, "stats_null": sn}
    return {"p_list": p_list, "K_M": K_M, "N": N, "by_ref": by_ref}

def compute_Gamma_i_multi_ref(
    X_refs, Y, stats_null_multi,
    K_M=300, p_ext=1e-5, n_jobs=16, Z=2.65, upsilon_key="Upsilon_i_Y",
):
    R = len(X_refs)
    nY = len(Y)
    Upsilon_by_ref = np.zeros((R, nY), float)
    result_dicts, EE_books, SrootB_by_ref, B_by_ref = [], [], [], []

    for j, Xj in enumerate(X_refs):
        stats_null_j = stats_null_multi["by_ref"][j]["stats_null"]
        rd, _ = EagleEye.Soar(
            Xj, Y, K_M=K_M, p_ext=p_ext, n_jobs=n_jobs,
            stats_null=stats_null_j, result_dict_in={}, do_IDE=True,
        )
        ups = np.asarray(rd[upsilon_key], float)
        Upsilon_by_ref[j] = ups
        result_dicts.append(rd)

        clusters = partitioning_function(Xj, Y, rd, p_ext=p_ext, Z=Z)
        EE_book = EagleEye.Repechage(Xj, Y, rd, clusters, p_ext=p_ext)
        EE_books.append(EE_book)
        SrootB_by_ref.append(S_rootB_estimate_Y_overdensities(rd, EE_book))
        B_by_ref.append(B_estimate_Y_overdensities(rd, EE_book))

    return {
        "Gamma_i": Upsilon_by_ref.sum(axis=0),
        "Upsilon_by_ref": Upsilon_by_ref,
        "result_dicts": result_dicts,
        "EE_books": EE_books,
        "SrootB_by_ref": SrootB_by_ref,
        "B_by_ref": B_by_ref,
        "p_ext": p_ext, "K_M": K_M, "Z": Z, "upsilon_key": upsilon_key,
    }

# ==========================================================================
# Bootstrap empirical Gamma null (cell 38)
# ==========================================================================
def bootstrap_gamma_null_uniform(
    X_refs, Y, stats_null_multi,
    K_M=300, p_ext=1e-5, n_jobs=16, upsilon_key="Upsilon_i_Y",
    n_boot=1, seed=42,
):
    rng = np.random.default_rng(seed)
    nY = len(Y); d = Y.shape[1]
    nX_list = [Xj.shape[0] for Xj in X_refs]
    Gamma_null = np.zeros((n_boot, nY), float)
    for b in range(n_boot):
        Yb = rng.random((nY, d)).astype(float)
        Xb_refs = [rng.random((nX, d)).astype(float) for nX in nX_list]
        Gb = compute_Gamma_i_multi_ref(
            X_refs=Xb_refs, Y=Yb, stats_null_multi=stats_null_multi,
            K_M=K_M, p_ext=p_ext, n_jobs=n_jobs, upsilon_key=upsilon_key,
        )
        Gamma_null[b] = Gb["Gamma_i"]
        print(f"    bootstrap {b+1}/{n_boot} done", flush=True)
    return Gamma_null

# ==========================================================================
# DBSCAN of A_Gamma + weighted B per Gamma cluster (cell 48)
# ==========================================================================
def cluster_A_Gamma_DBSCAN_no_noise(Y, A_Gamma_mask, eps=1.0, min_samples=5,
                                    use_cols=(0,1,2,3)):
    A_Gamma_mask = np.asarray(A_Gamma_mask, bool)
    nY = Y.shape[0]
    idx = np.where(A_Gamma_mask)[0]
    labels_all = np.full(nY, -2, dtype=int)
    if idx.size == 0:
        return labels_all, {}, np.array([], dtype=int)
    Z = np.asarray(Y[:, use_cols], float)
    lab = DBSCAN(eps=float(eps), min_samples=int(min_samples)).fit_predict(Z[idx])
    labels_all[idx] = lab
    noise_idx = idx[lab == -1]
    clusters = {int(k): idx[lab == k] for k in np.unique(lab) if k != -1}
    return labels_all, clusters, noise_idx

def srootB_for_Gamma_clusters_from_Brefs(
    EE_books, SrootB_by_ref, B_by_ref, Y, A_Gamma_mask,
    overlap_min=1, use_cluster_key="Repechaged",
    weight_mode="overlap", dbscan_eps=1.0, dbscan_min_samples=5,
    cluster_cols=(0,1,2,3),
):
    A_Gamma_mask = np.asarray(A_Gamma_mask, bool)
    labels_all, gamma_clusters, noise_idx = cluster_A_Gamma_DBSCAN_no_noise(
        Y, A_Gamma_mask, eps=dbscan_eps, min_samples=dbscan_min_samples,
        use_cols=cluster_cols,
    )
    R = len(EE_books)

    def _EE_members(cdict):
        if use_cluster_key == "Repechaged":
            return np.asarray(cdict.get("Repechaged", []), dtype=int)
        return np.unique(np.concatenate([
            np.asarray(cdict.get("Repechaged", []), dtype=int),
            np.asarray(cdict.get("Pruned", []), dtype=int),
        ]))

    cluster_stats = {}
    for gid, A_idx in gamma_clusters.items():
        A_idx = np.asarray(A_idx, dtype=int)
        S = int(A_idx.size)
        B_num = 0.0; W_num = 0.0; contributors = []
        for j in range(R):
            z_by_cluster = SrootB_by_ref[j].get("s/root(B)", {})
            B_by_cluster = B_by_ref[j].get("B", {})
            over_clusters = EE_books[j].get("Y_OVER_clusters", {})
            if not isinstance(over_clusters, dict) or not over_clusters:
                continue
            for cid in sorted(over_clusters.keys()):
                members = _EE_members(over_clusters[cid])
                if members.size == 0:
                    continue
                ov = int(np.intersect1d(members, A_idx, assume_unique=False).size)
                if ov < int(overlap_min):
                    continue
                z = z_by_cluster.get(cid, np.nan)
                try: z = float(z)
                except Exception: z = np.nan
                if not np.isfinite(z) or z <= 0:
                    continue
                Bc = B_by_cluster.get(cid, np.nan)
                try: Bc = float(Bc)
                except Exception: Bc = np.nan
                if not np.isfinite(Bc) or Bc <= 0:
                    continue
                w = float(ov) if weight_mode == "overlap" else 1.0
                B_num += w * Bc; W_num += w
                contributors.append(
                    {"ref": j, "cid": int(cid), "overlap": ov,
                     "w": w, "Bhat": Bc, "z": z}
                )
        B_w = (B_num / W_num) if W_num > 0 else np.nan
        srb = (S / np.sqrt(B_w)) if (np.isfinite(B_w) and B_w > 0) else np.nan
        cluster_stats[int(gid)] = {
            "S": S, "B_weighted": B_w, "s_over_rootB": srb,
            "contributors": contributors, "W_num": W_num, "A_idx": A_idx,
        }
    return {
        "gamma_clusters": gamma_clusters, "noise_idx": noise_idx,
        "cluster_stats": cluster_stats, "labels_all": labels_all,
        "dbscan": {"eps": dbscan_eps, "min_samples": dbscan_min_samples,
                   "cols": cluster_cols},
    }

# ==========================================================================
# Battaglia overlay + DR3 crossmatch (cells 46, 47, 51)
# ==========================================================================
_BATTAGLIA_CACHE = {"table": None}

def get_battaglia_table():
    """Fetch the full Battaglia+2022 (J/A+A/657/A54) DR3 member catalog once."""
    if _BATTAGLIA_CACHE["table"] is not None:
        return _BATTAGLIA_CACHE["table"]
    v = Vizier(columns=["**"], row_limit=-1)
    result = v.get_catalogs(BATTAGLIA_CATALOG)
    _BATTAGLIA_CACHE["table"] = result[0]
    return _BATTAGLIA_CACHE["table"]

def battaglia_name(dwarf_name):
    """Translate DES name 'Tucana II' -> Battaglia 'TucanaII'. Returns None if NA."""
    # Battaglia uses concatenated CamelCase with roman numerals.
    return dwarf_name.replace(" ", "")

def fetch_dwarf_members(dwarf_name, p_memb_min=0.5):
    """Return a pandas DataFrame of likely DR3 members for this dwarf, or None."""
    try:
        bat = get_battaglia_table()
    except Exception as e:
        print(f"  Battaglia fetch failed: {e}")
        return None
    galaxy_key = battaglia_name(dwarf_name)
    sub = bat[(bat["Galaxy"] == galaxy_key) & (bat["Pmemb"] > p_memb_min)]
    if len(sub) == 0:
        # try a more permissive match (some names differ, e.g. "BootesIII" vs "Bootes3")
        norm = lambda s: "".join(c for c in str(s).lower() if c.isalnum())
        target = norm(dwarf_name)
        avail = {norm(g): g for g in np.unique(bat["Galaxy"])}
        if target in avail:
            sub = bat[(bat["Galaxy"] == avail[target]) & (bat["Pmemb"] > p_memb_min)]
    if len(sub) == 0:
        return None
    return sub.to_pandas()

def crossmatch_members_to_dr3(members_df, r, tol_deg=1e-4):
    """Attach DR3 (ra, dec, pmra, pmdec, source_id) to members_df via KDTree match in r."""
    df = members_df.copy()
    # find RA/Dec columns in the Battaglia table
    rename_map = {}
    for c in ["RA_ICRS", "_RA.icrs", "RAJ2000", "RAdeg", "ra"]:
        if c in df.columns:
            rename_map[c] = "ra_deg"; break
    for c in ["DE_ICRS", "_DE.icrs", "DEJ2000", "DEdeg", "dec"]:
        if c in df.columns:
            rename_map[c] = "dec_deg"; break
    df = df.rename(columns=rename_map)
    if not {"ra_deg", "dec_deg"}.issubset(df.columns):
        print("  could not find RA/Dec columns in Battaglia table; got:", df.columns.tolist()[:10])
        return None

    r_ra    = np.asarray(r["ra"], float)
    r_dec   = np.asarray(r["dec"], float)
    r_pmra  = np.asarray(r["pmra"], float)
    r_pmdec = np.asarray(r["pmdec"], float)
    r_sid   = np.asarray(r["source_id"])

    tree = KDTree(np.column_stack([r_ra, r_dec]))
    q_ra  = df["ra_deg"].to_numpy(float)
    q_dec = df["dec_deg"].to_numpy(float)
    dist, idx = tree.query(np.column_stack([q_ra, q_dec]), k=1)
    dist = dist.ravel(); idx = idx.ravel()
    matched = dist < tol_deg

    df["source_id_dr3"] = pd.NA
    df["ra_dr3"] = np.nan; df["dec_dr3"] = np.nan
    df["pmra_dr3"] = np.nan; df["pmdec_dr3"] = np.nan
    df.loc[matched, "source_id_dr3"] = r_sid[idx[matched]]
    df.loc[matched, "ra_dr3"]    = r_ra[idx[matched]]
    df.loc[matched, "dec_dr3"]   = r_dec[idx[matched]]
    df.loc[matched, "pmra_dr3"]  = r_pmra[idx[matched]]
    df.loc[matched, "pmdec_dr3"] = r_pmdec[idx[matched]]
    return df[matched].copy()

def gamma_cluster_overlap_by_source_id(r, parts, gamma_clusters, overlay_df,
                                       overlay_id_col="source_id_dr3"):
    test_idx = np.asarray(parts["test_idx"], dtype=int)
    y_sid = np.asarray(r["source_id"][test_idx])
    member_ids = pd.Series(overlay_df[overlay_id_col]).dropna().astype("int64").to_numpy()
    member_set = set(member_ids.tolist())
    out_clusters = {}
    union = set()
    for gid, pts in gamma_clusters.items():
        pts = np.asarray(pts, dtype=int)
        if pts.size == 0:
            out_clusters[int(gid)] = {"n_cluster": 0, "n_overlap": 0,
                                      "tpr_members": 0.0,
                                      "matched_source_ids": np.array([], dtype=np.int64)}
            continue
        cid_set = set(y_sid[pts].astype("int64").tolist())
        matched = np.array(sorted(cid_set & member_set), dtype=np.int64)
        union.update(cid_set)
        out_clusters[int(gid)] = {
            "n_cluster": int(len(cid_set)),
            "n_overlap": int(len(matched)),
            "tpr_members": float(len(matched) / len(member_ids)) if len(member_ids) else np.nan,
            "matched_source_ids": matched,
        }
    union_matched = np.array(sorted(union & member_set), dtype=np.int64)
    return {
        "n_members_total": int(len(member_ids)),
        "n_overlap_total": int(len(union_matched)),
        "TPR_total": float(len(union_matched) / len(member_ids)) if len(member_ids) else np.nan,
        "matched_source_ids_total": union_matched,
        "cluster_results": out_clusters,
    }

# ==========================================================================
# Plotting (sky + PM, with optional Battaglia overlay)
# ==========================================================================
def plot_dwarf_result(dwarf_name, r, parts, gamma_clusters, cluster_stats,
                      overlay_df, center_ra, center_dec, cell_halfwidth_deg,
                      out_path_sky, out_path_pm):
    test_idx = np.asarray(parts["test_idx"], dtype=int)
    inside_idx = np.where(np.asarray(parts["inside"], bool))[0]
    ra_bg    = np.asarray(r["ra"][inside_idx], float)
    dec_bg   = np.asarray(r["dec"][inside_idx], float)
    pmra_bg  = np.asarray(r["pmra"][inside_idx], float)
    pmdec_bg = np.asarray(r["pmdec"][inside_idx], float)
    raY    = np.asarray(r["ra"][test_idx], float)
    decY   = np.asarray(r["dec"][test_idx], float)
    pmraY  = np.asarray(r["pmra"][test_idx], float)
    pmdecY = np.asarray(r["pmdec"][test_idx], float)

    labels = sorted(gamma_clusters.keys())
    cmap = cm.get_cmap("tab10", max(1, len(labels)))

    # --- Sky ---
    plt.figure(figsize=(8, 7))
    plt.scatter(ra_bg, dec_bg, s=2, alpha=0.25, color="lightgray", rasterized=True)
    for d in [-cell_halfwidth_deg, cell_halfwidth_deg]:
        plt.axvline(center_ra + d, color="red", ls="--", lw=1.2, alpha=0.7)
        plt.axhline(center_dec + d, color="red", ls="--", lw=1.2, alpha=0.7)
    for k, gid in enumerate(labels):
        pts = np.asarray(gamma_clusters[gid], dtype=int)
        if pts.size == 0: continue
        srb = cluster_stats[int(gid)]["s_over_rootB"]
        plt.scatter(raY[pts], decY[pts], s=22, alpha=0.9, color=cmap(k),
                    rasterized=True,
                    label=rf"$\alpha={gid}$ n={pts.size}, $S/\sqrt{{\hat B}}$={srb:.1f}")
    if overlay_df is not None and len(overlay_df) > 0:
        plt.scatter(overlay_df["ra_dr3"].to_numpy(float),
                    overlay_df["dec_dr3"].to_numpy(float),
                    s=55, facecolors="none", edgecolors="red", linewidths=1.5,
                    alpha=0.8, label=f"{dwarf_name} members")
    plt.xlabel("RA [deg]"); plt.ylabel("Dec [deg]")
    plt.title(f"{dwarf_name}: Gamma clusters (sky)")
    plt.gca().invert_xaxis(); plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(fontsize=8, frameon=True, framealpha=0.9)
    plt.tight_layout(); plt.savefig(out_path_sky, dpi=120); plt.close()

    # --- PM ---
    plt.figure(figsize=(8, 7))
    plt.scatter(pmra_bg, pmdec_bg, s=2, alpha=0.2, color="lightgray", rasterized=True)
    for k, gid in enumerate(labels):
        pts = np.asarray(gamma_clusters[gid], dtype=int)
        if pts.size == 0: continue
        srb = cluster_stats[int(gid)]["s_over_rootB"]
        plt.scatter(pmraY[pts], pmdecY[pts], s=22, alpha=0.9, color=cmap(k),
                    rasterized=True,
                    label=rf"$\alpha={gid}$ n={pts.size}, $S/\sqrt{{\hat B}}$={srb:.1f}")
    if (overlay_df is not None and len(overlay_df) > 0
        and {"pmra_dr3","pmdec_dr3"}.issubset(overlay_df.columns)):
        plt.scatter(overlay_df["pmra_dr3"].to_numpy(float),
                    overlay_df["pmdec_dr3"].to_numpy(float),
                    s=55, facecolors="none", edgecolors="red", linewidths=1.5,
                    alpha=0.8, label=f"{dwarf_name} members")
    plt.xlabel("pmra [mas/yr]"); plt.ylabel("pmdec [mas/yr]")
    plt.title(f"{dwarf_name}: Gamma clusters (PM)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(fontsize=8, frameon=True, framealpha=0.9)
    plt.tight_layout(); plt.savefig(out_path_pm, dpi=120); plt.close()


def plot_Gamma_clusters_kde_overlay(
    r, parts, gamma_clusters, cluster_stats, overlay_df,
    out_path,
    plot_space="pos",            # "pos" or "pm"
    overlay_label="Battaglia et al. members",
    center_ra=None, center_dec=None, degsep=None,
    center_pmra=None, center_pmdec=None, dpm=None,
    figsize=(8, 7),
    s_bg=0.5, alpha_bg=0.5,
    gamma_as_points=True, gamma_as_kde=True,
    s_gamma=20, alpha_gamma=0.85,
    gamma_kde_levels=(0.3, 0.6, 0.95),
    gamma_min_pts_kde=5,
    overlay_as_points=True, overlay_as_kde=True,
    s_overlay=10, alpha_overlay=0.95,
    overlay_kde_levels=(0.3, 0.6, 0.95),
    overlay_min_pts_kde=5,
    kde_grid=200,
    title="",
    print_overlap=False,
    gamma_color="#e07b00",       # amber (paper style)
    overlay_color="#1f5fbf",     # deep blue, readable on white
):
    """Light-theme replica of the Tucana notebook's KDE overlay plot.
    Writes a PNG to `out_path`; never calls plt.show().
    """
    if plot_space not in {"pm", "pos"}:
        raise ValueError("plot_space must be 'pm' or 'pos'")

    test_idx = np.asarray(parts["test_idx"], dtype=int)

    if plot_space == "pm":
        x_all = np.asarray(r["pmra"], float)
        y_all = np.asarray(r["pmdec"], float)
        xY    = np.asarray(r["pmra"][test_idx], float)
        yY    = np.asarray(r["pmdec"][test_idx], float)
        xlabel = r"$\mu_{\alpha *}\,[\mathrm{mas\,yr^{-1}}]$"
        ylabel = r"$\mu_{\delta}\,[\mathrm{mas\,yr^{-1}}]$"
        invert_x = False
        if overlay_df is not None and {"pmra_dr3", "pmdec_dr3"}.issubset(overlay_df.columns):
            x_mem = overlay_df["pmra_dr3"].to_numpy(float)
            y_mem = overlay_df["pmdec_dr3"].to_numpy(float)
            ok = np.isfinite(x_mem) & np.isfinite(y_mem)
            x_mem, y_mem = x_mem[ok], y_mem[ok]
        else:
            x_mem = y_mem = None
    else:  # pos
        x_all = np.asarray(r["ra"], float)
        y_all = np.asarray(r["dec"], float)
        xY    = np.asarray(r["ra"][test_idx], float)
        yY    = np.asarray(r["dec"][test_idx], float)
        xlabel = r"RA $[\mathrm{deg}]$"
        ylabel = r"Dec $[\mathrm{deg}]$"
        invert_x = True
        if overlay_df is not None and {"ra_dr3", "dec_dr3"}.issubset(overlay_df.columns):
            x_mem = overlay_df["ra_dr3"].to_numpy(float)
            y_mem = overlay_df["dec_dr3"].to_numpy(float)
            ok = np.isfinite(x_mem) & np.isfinite(y_mem)
            x_mem, y_mem = x_mem[ok], y_mem[ok]
        else:
            x_mem = y_mem = None

    ok = np.isfinite(x_all) & np.isfinite(y_all)
    x_all, y_all = x_all[ok], y_all[ok]

    xmin, xmax = float(np.min(x_all)), float(np.max(x_all))
    ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
    xx, yy = np.meshgrid(np.linspace(xmin, xmax, kde_grid),
                         np.linspace(ymin, ymax, kde_grid))
    grid = np.vstack([xx.ravel(), yy.ravel()])

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("white")

    # background field
    ax.scatter(x_all, y_all, s=s_bg, c="0.55", alpha=alpha_bg,
               rasterized=True, linewidths=0)
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)

    # guide lines for the test-cell boundary
    if plot_space == "pos" and (center_ra is not None and center_dec is not None
                                and degsep is not None):
        h = degsep / 3.0
        for d in (-h, h):
            ax.axvline(center_ra + d, color="0.3", ls="--", lw=1.0, alpha=0.7)
            ax.axhline(center_dec + d, color="0.3", ls="--", lw=1.0, alpha=0.7)
    if plot_space == "pm" and (center_pmra is not None and center_pmdec is not None
                               and dpm is not None):
        for d in (-dpm/3, dpm/3):
            ax.axvline(center_pmra + d, color="0.3", ls="--", lw=1.0, alpha=0.7)
            ax.axhline(center_pmdec + d, color="0.3", ls="--", lw=1.0, alpha=0.7)

    # overlay first (so Gamma renders on top)
    if x_mem is not None and len(x_mem) > 0:
        if overlay_as_points:
            ax.scatter(x_mem, y_mem, s=s_overlay,
                       facecolors="none", edgecolors=overlay_color,
                       linewidths=1.4, alpha=alpha_overlay)
        if overlay_as_kde and len(x_mem) >= overlay_min_pts_kde:
            kde = gaussian_kde(np.vstack([x_mem, y_mem]))
            zz = kde(grid).reshape(xx.shape)
            zmax = float(np.nanmax(zz))
            levels = [lev * zmax for lev in overlay_kde_levels]
            ax.contour(xx, yy, zz, levels=levels, colors=[overlay_color]*len(levels),
                       linewidths=2.0, alpha=0.95)
        ax.plot([], [], color=overlay_color, lw=2.0, label=overlay_label)

    # Gamma clusters
    gamma_union = []
    for gid in sorted(gamma_clusters.keys()):
        pts = np.asarray(gamma_clusters[gid], dtype=int)
        if pts.size == 0:
            continue
        x_c, y_c = xY[pts], yY[pts]
        ok = np.isfinite(x_c) & np.isfinite(y_c)
        x_c, y_c = x_c[ok], y_c[ok]
        if x_c.size == 0:
            continue
        gamma_union.append(np.column_stack([x_c, y_c]))
        srb = cluster_stats[int(gid)]["s_over_rootB"]
        if gamma_as_points:
            ax.scatter(x_c, y_c, s=s_gamma, alpha=alpha_gamma,
                       facecolors="none", edgecolors=gamma_color, linewidths=1.2,
                       marker="o")
        if gamma_as_kde and x_c.size >= gamma_min_pts_kde:
            kde = gaussian_kde(np.vstack([x_c, y_c]))
            zz = kde(grid).reshape(xx.shape)
            zmax = float(np.nanmax(zz))
            levels = [lev * zmax for lev in gamma_kde_levels]
            ax.contour(xx, yy, zz, levels=levels, colors=[gamma_color]*len(levels),
                       linewidths=2.0, alpha=0.95)
        srb_str = f"{srb:.2f}" if (srb is not None and np.isfinite(srb)) else "n/a"
        ax.plot([], [], color=gamma_color, lw=2.0,
                label=rf"EagleEye $\Gamma_{{{gid}}}$: $S/\sqrt{{\hat B}}={srb_str}$")

    if print_overlap and x_mem is not None and len(gamma_union) > 0:
        gU = np.vstack(gamma_union)
        tol = 0.05 if plot_space == "pm" else 1e-4
        matched = 0
        for px, py in zip(x_mem, y_mem):
            if np.any((gU[:, 0]-px)**2 + (gU[:, 1]-py)**2 <= tol**2):
                matched += 1
        print(f"    overlap ({plot_space}): {matched}/{len(x_mem)}")

    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    if title:
        ax.set_title(title, fontsize=16)
    ax.tick_params(labelsize=12)
    if invert_x:
        ax.invert_xaxis()
    ax.set_aspect("equal", adjustable="box")
    leg = ax.legend(frameon=True, fancybox=True, framealpha=0.95,
                    loc="best", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()

# ==========================================================================
# Per-dwarf driver
# ==========================================================================
def run_for_dwarf(name, center_ra, center_dec, args, out_dir):
    print(f"\n========== {name} (RA={center_ra:.4f}, Dec={center_dec:.4f}) ==========",
          flush=True)
    t0 = time.time()
    safe = name.replace(' ', '_')
    dwarf_dir = out_dir / safe
    dwarf_dir.mkdir(parents=True, exist_ok=True)
    pkl_path     = dwarf_dir / f"{safe}.pkl"
    sky_path     = dwarf_dir / f"{safe}_sky.png"
    pm_path      = dwarf_dir / f"{safe}_pm.png"
    sky_kde_path = dwarf_dir / f"{safe}_sky_kde.png"
    pm_kde_path  = dwarf_dir / f"{safe}_pm_kde.png"

    # ---- 1) Gaia query ----
    ra_cut  = ra_condition(center_ra, args.degsep)
    dec_cut = f"(dec BETWEEN {center_dec - args.degsep} AND {center_dec + args.degsep})"
    query = f"""
    SELECT source_id, ra, dec, pmra, pmdec, phot_g_mean_mag, parallax,
           parallax_error, ruwe
    FROM gaiadr3.gaia_source
    WHERE {ra_cut}
      AND {dec_cut}
      AND pmra BETWEEN -5 AND 5
      AND pmdec BETWEEN -5 AND 5
      AND (parallax - 3*parallax_error) < 0.0
    """
    print(f"  Gaia query...", flush=True)
    r = None
    for attempt in range(1, args.gaia_retries + 1):
        try:
            job = Gaia.launch_job_async(query)
            r = job.get_results()
            break
        except Exception as e:
            wait = min(60, 5 * attempt)
            print(f"  Gaia attempt {attempt}/{args.gaia_retries} failed: "
                  f"{type(e).__name__}: {e}  (retry in {wait}s)", flush=True)
            if attempt == args.gaia_retries:
                return {"name": name, "status": f"gaia_failed: {e}"}
            time.sleep(wait)
    print(f"  Gaia: {len(r)} rows", flush=True)
    if len(r) < 1000:
        print(f"  too few stars ({len(r)}), skipping")
        return {"name": name, "status": "too_few_stars", "n_gaia": int(len(r))}

    # ---- 2) Y, X_refs ----
    cell_h = args.degsep / 3.0
    Y, X_refs, parts, scale = make_Xrefs_and_Y_from_window_3x3(
        r, center_ra, center_dec, cell_h,
        fit_scaling_on="all", do_robust_scale=True,
    )
    nY = len(Y); nXs = [len(Xj) for Xj in X_refs]
    print(f"  |Y|={nY}, |X_j|={nXs}", flush=True)
    if nY < 1000 or min(nXs) < 1000:
        print("  empty/small cells, skipping")
        return {"name": name, "status": "empty_cells", "nY": int(nY), "nXs": nXs}

    # ---- 3) Per-ref null distributions ----
    p_list = [nY / (nY + nX) for nX in nXs]
    print(f"  computing nulls (N={args.null_N})...", flush=True)
    stats_null_multi = compute_the_null_multi_refs(
        p_list=p_list, K_M=args.kM, N=args.null_N, n_jobs=min(8, args.n_jobs),
    )

    # ---- 4) Soar + Gamma_i ----
    print("  running Soar on 8 references...", flush=True)
    out = compute_Gamma_i_multi_ref(
        X_refs=X_refs, Y=Y, stats_null_multi=stats_null_multi,
        K_M=args.kM, p_ext=args.p_ext, n_jobs=args.n_jobs, Z=args.dpa_Z,
    )
    Gamma_i = out["Gamma_i"]
    EE_books = out["EE_books"]
    SrootB_by_ref = out["SrootB_by_ref"]
    B_by_ref = out["B_by_ref"]

    # ---- 5) Empirical Gamma null via uniform bootstrap ----
    print(f"  bootstrap Gamma null (n_boot={args.n_boot})...", flush=True)
    Gamma_null_mat = bootstrap_gamma_null_uniform(
        X_refs=X_refs, Y=Y, stats_null_multi=stats_null_multi,
        K_M=args.kM, p_ext=args.p_ext, n_jobs=args.n_jobs,
        n_boot=args.n_boot, seed=args.seed,
    )
    Gamma_null_1d = Gamma_null_mat.reshape(-1)
    Gamma_null_1d = Gamma_null_1d[np.isfinite(Gamma_null_1d)]
    Gamma_star_emp = float(np.quantile(Gamma_null_1d, 1 - args.p_ext))
    print(f"  Gamma* (emp) = {Gamma_star_emp:.3f}", flush=True)

    # ---- 6) DBSCAN + weighted S/sqrt(B) per Gamma cluster ----
    A_Gamma_mask = Gamma_i > Gamma_star_emp
    print(f"  |A_Gamma| = {A_Gamma_mask.sum()}", flush=True)
    cluster_out = srootB_for_Gamma_clusters_from_Brefs(
        EE_books=EE_books, SrootB_by_ref=SrootB_by_ref, B_by_ref=B_by_ref,
        Y=Y, A_Gamma_mask=A_Gamma_mask, overlap_min=1,
        use_cluster_key="Repechaged", weight_mode="overlap",
        dbscan_eps=args.dbscan_eps, dbscan_min_samples=args.dbscan_min,
        cluster_cols=(0,1,2,3),
    )
    n_clusters = len(cluster_out["gamma_clusters"])
    print(f"  n_Gamma_clusters = {n_clusters}", flush=True)

    # ---- 7) Battaglia overlay + DR3 crossmatch ----
    overlay_df = None; overlap = None
    members_raw = fetch_dwarf_members(name, p_memb_min=0.5)
    if members_raw is not None and len(members_raw) > 0:
        overlay_df = crossmatch_members_to_dr3(members_raw, r)
        if overlay_df is not None and len(overlay_df) > 0:
            print(f"  Battaglia: matched {len(overlay_df)} DR3 members", flush=True)
            overlap = gamma_cluster_overlap_by_source_id(
                r, parts, cluster_out["gamma_clusters"], overlay_df,
            )
            print(f"  TPR_total = {overlap['TPR_total']:.3f} "
                  f"({overlap['n_overlap_total']}/{overlap['n_members_total']})",
                  flush=True)
        else:
            overlay_df = None
            print("  Battaglia members present but no DR3 crossmatch", flush=True)
    else:
        print("  no Battaglia entry for this dwarf", flush=True)

    # ---- 8) Plots ----
    try:
        plot_dwarf_result(
            name, r, parts, cluster_out["gamma_clusters"], cluster_out["cluster_stats"],
            overlay_df, center_ra, center_dec, cell_h, sky_path, pm_path,
        )
    except Exception as e:
        print(f"  plotting failed (basic): {e}", flush=True)

    try:
        plot_Gamma_clusters_kde_overlay(
            r, parts, cluster_out["gamma_clusters"], cluster_out["cluster_stats"],
            overlay_df, out_path=sky_kde_path,
            plot_space="pos",
            overlay_label=f"{name} members [Battaglia+22]",
            center_ra=center_ra, center_dec=center_dec, degsep=args.degsep,
            title=name, print_overlap=(overlap is not None),
        )
        plot_Gamma_clusters_kde_overlay(
            r, parts, cluster_out["gamma_clusters"], cluster_out["cluster_stats"],
            overlay_df, out_path=pm_kde_path,
            plot_space="pm",
            overlay_label=f"{name} members [Battaglia+22]",
            title=name, print_overlap=(overlap is not None),
        )
    except Exception as e:
        print(f"  plotting failed (kde): {e}", flush=True)

    # ---- 9) Build summary row ----
    rows = []
    for gid in sorted(cluster_out["gamma_clusters"].keys()):
        cs = cluster_out["cluster_stats"][gid]
        cres = (overlap["cluster_results"].get(gid, {}) if overlap else {})
        rows.append({
            "name": name, "gid": gid,
            "S": cs["S"], "B_weighted": cs["B_weighted"],
            "s_over_rootB": cs["s_over_rootB"],
            "n_overlap": cres.get("n_overlap", np.nan),
            "tpr_members": cres.get("tpr_members", np.nan),
        })

    # ---- 10) Save per-dwarf pickle ----
    # Persist enough state to replay the plots without re-querying Gaia.
    r_dict = {col: np.asarray(r[col]) for col in
              ("source_id", "ra", "dec", "pmra", "pmdec")}
    payload = {
        "name": name, "center_ra": center_ra, "center_dec": center_dec,
        "degsep": args.degsep, "kM": args.kM, "p_ext": args.p_ext,
        "n_boot": args.n_boot, "n_jobs": args.n_jobs,
        "nY": int(nY), "nXs": list(map(int, nXs)),
        "p_list": p_list, "scale": scale,
        "Gamma_i": Gamma_i, "Gamma_star_emp": Gamma_star_emp,
        "Gamma_null_mat": Gamma_null_mat,
        "gamma_clusters": cluster_out["gamma_clusters"],
        "cluster_stats": cluster_out["cluster_stats"],
        "labels_all": cluster_out["labels_all"],
        "n_overlay_members": (0 if overlay_df is None else int(len(overlay_df))),
        "TPR_total": (overlap["TPR_total"] if overlap else np.nan),
        "n_overlap_total": (overlap["n_overlap_total"] if overlap else 0),
        "rows": rows,
        "wall_time_s": time.time() - t0,
        "status": "ok",
        # --- plot-replay inputs ---
        "r_dict": r_dict,
        "parts": parts,
        "overlay_df": overlay_df,
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"  saved {pkl_path}  ({payload['wall_time_s']:.1f}s)", flush=True)
    return payload

# ==========================================================================
# Plot replay (no Gaia / no Soar / no bootstrap)
# ==========================================================================
def replay_plots_from_pickle(pkl_path, args, out_dir):
    """Re-generate the four PNGs for a dwarf from its saved pickle.
    Requires the pickle to contain 'r_dict', 'parts' (added in v2 payloads)."""
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    name = payload.get("name", pkl_path.stem.replace("_", " "))
    print(f"\n----- [replay] {name} -----", flush=True)

    if payload.get("status") != "ok":
        print(f"  skipping: status='{payload.get('status')}'")
        return payload
    if "r_dict" not in payload or "parts" not in payload:
        print("  skipping: pickle predates plot-replay support; "
              "re-run without --plots-only to regenerate")
        return payload

    r          = payload["r_dict"]    # dict-of-arrays, indexable like astropy Table
    parts      = payload["parts"]
    overlay_df = payload.get("overlay_df", None)
    overlap_ok = payload.get("n_overlay_members", 0) > 0
    center_ra  = payload["center_ra"]
    center_dec = payload["center_dec"]
    cell_h     = payload["degsep"] / 3.0

    safe = name.replace(" ", "_")
    # Plots go alongside the pickle (i.e. into the same per-dwarf subdir).
    dwarf_dir    = pkl_path.parent
    sky_path     = dwarf_dir / f"{safe}_sky.png"
    pm_path      = dwarf_dir / f"{safe}_pm.png"
    sky_kde_path = dwarf_dir / f"{safe}_sky_kde.png"
    pm_kde_path  = dwarf_dir / f"{safe}_pm_kde.png"

    try:
        plot_dwarf_result(
            name, r, parts,
            payload["gamma_clusters"], payload["cluster_stats"],
            overlay_df, center_ra, center_dec, cell_h, sky_path, pm_path,
        )
        print(f"  wrote {sky_path.name}, {pm_path.name}")
    except Exception as e:
        print(f"  basic plot failed: {e}")

    try:
        plot_Gamma_clusters_kde_overlay(
            r, parts, payload["gamma_clusters"], payload["cluster_stats"],
            overlay_df, out_path=sky_kde_path,
            plot_space="pos",
            overlay_label=f"{name} members [Battaglia+22]",
            center_ra=center_ra, center_dec=center_dec, degsep=payload["degsep"],
            title=name, print_overlap=overlap_ok,
        )
        plot_Gamma_clusters_kde_overlay(
            r, parts, payload["gamma_clusters"], payload["cluster_stats"],
            overlay_df, out_path=pm_kde_path,
            plot_space="pm",
            overlay_label=f"{name} members [Battaglia+22]",
            title=name, print_overlap=overlap_ok,
        )
        print(f"  wrote {sky_kde_path.name}, {pm_kde_path.name}")
    except Exception as e:
        print(f"  kde plot failed: {e}")

    return payload

# ==========================================================================
# Main
# ==========================================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dwarfs-table", default=DEFAULT_DWARFS_TABLE)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--names", nargs="+", default=None,
                    help="restrict to these dwarf names (else all class=4)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip dwarfs whose .pkl already exists")
    ap.add_argument("--degsep", type=float, default=DEFAULT_DEGSEP)
    ap.add_argument("--kM",     type=int,   default=DEFAULT_KM)
    ap.add_argument("--p_ext",  type=float, default=DEFAULT_PEXT)
    ap.add_argument("--n-boot", type=int,   default=DEFAULT_N_BOOT)
    ap.add_argument("--null-N", dest="null_N", type=int, default=DEFAULT_NULL_N)
    ap.add_argument("--dpa-Z",  type=float, default=DEFAULT_Z)
    ap.add_argument("--dbscan-eps", type=float, default=DEFAULT_DBSCAN_EPS)
    ap.add_argument("--dbscan-min", type=int,   default=DEFAULT_DBSCAN_MIN)
    ap.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    ap.add_argument("--seed",   type=int, default=42)
    ap.add_argument("--gaia-retries", type=int, default=4,
                    help="retry transient Gaia 500/timeout errors this many times")
    ap.add_argument("--plots-only", action="store_true",
                    help="skip Gaia/Soar/bootstrap; regenerate plots from "
                         "existing pickles in --out-dir only")
    return ap.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dwarfs = ascii.read(args.dwarfs_table)
    c4 = dwarfs[dwarfs["Class"] == 4]
    if args.names:
        c4 = c4[np.isin(np.asarray(c4["Name"]).astype(str), args.names)]

    print(f"running on {len(c4)} dwarfs"
          + (" (PLOTS-ONLY MODE)" if args.plots_only else "")
          + ":")
    for row in c4:
        print(f"  - {row['Name']}")

    summary_rows = []
    for row in c4:
        name = str(row["Name"])
        ra   = float(row["RAdeg"])
        dec  = float(row["DEdeg"])
        safe = name.replace(' ', '_')
        pkl  = out_dir / safe / f"{safe}.pkl"
        # Backward-compat: pick up older flat-layout pickles if subdir empty
        if not pkl.exists():
            legacy = out_dir / f"{safe}.pkl"
            if legacy.exists():
                pkl = legacy

        if args.plots_only:
            if not pkl.exists():
                print(f"\n[plots-only] {name}: no pickle at {pkl} — skipping")
                continue
            try:
                payload = replay_plots_from_pickle(pkl, args, out_dir)
            except Exception as e:
                import traceback
                traceback.print_exc()
                payload = {"name": name, "status": f"replay_error: {e}"}
        elif args.skip_existing and pkl.exists():
            print(f"\n[skip] {name} (pkl exists)")
            with open(pkl, "rb") as f:
                payload = pickle.load(f)
        else:
            try:
                payload = run_for_dwarf(name, ra, dec, args, out_dir)
            except Exception as e:
                import traceback
                traceback.print_exc()
                payload = {"name": name, "status": f"error: {e}"}
        summary_rows.append({
            "name": name, "RA": ra, "Dec": dec,
            "status": payload.get("status", "?"),
            "nY": payload.get("nY", np.nan),
            "n_clusters": len(payload.get("gamma_clusters", {})) if "gamma_clusters" in payload else 0,
            "Gamma_star_emp": payload.get("Gamma_star_emp", np.nan),
            "TPR_total": payload.get("TPR_total", np.nan),
            "n_overlap_total": payload.get("n_overlap_total", 0),
            "best_s_over_rootB": max(
                (cs["s_over_rootB"] for cs in payload.get("cluster_stats", {}).values()
                 if cs["s_over_rootB"] is not None and np.isfinite(cs["s_over_rootB"])),
                default=np.nan,
            ),
            "wall_time_s": payload.get("wall_time_s", np.nan),
        })
        # write summary incrementally so a crash doesn't lose progress
        # (skip in --plots-only to avoid clobbering an existing summary
        #  with a subset of dwarfs)
        if not args.plots_only:
            pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)

    if not args.plots_only:
        print(f"\nWrote summary to {out_dir/'summary.csv'}")
    else:
        print(f"\nReplayed plots for {len(summary_rows)} dwarfs")

if __name__ == "__main__":
    main()

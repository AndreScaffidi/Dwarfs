"""Monkey-patch for EagleEye.IDE_step_optimized: ragged neighbour-cache crash.

THE BUG
-------
`IDE` seeds its neighbour cache from a model fitted with `n_neighbors = K_M*10`:

    precomputed_neighbors = Knn_model.kneighbors(Y[putative_indices, :])[1]
    neighbors_dict = {idx: row for idx, row in zip(putative_indices, precomputed_neighbors)}

so every cached row starts `K_M*10` long. Later, when the banned set has consumed
enough neighbours that a row has fewer than `K_M` valid entries, `IDE_step_optimized`
mutates the shared model and writes the *shorter* rows straight back into that cache:

    Knn_model.n_neighbors = int((X.shape[0] + Y.shape[0]) * 0.05)
    new_rows = Knn_model.kneighbors(Y[indices_to_update, :])[1]
    ...
    neighbors_dict[putative_indices[row_idx]] = new_row      # different length!

The cache is now ragged, and the next iteration dies on

    new_indices = np.array([neighbors_dict[i] for i in putative_indices])
    ValueError: setting an array element with a sequence.
                The requested array has an inhomogeneous shape ...

The recompute branch only fires while pruning a large, rich anomaly, which is why
most fields are unaffected and a handful (Fornax, Sculptor, Horologium I, Bootes III,
and Reticulum II at several grid placements) crash.

THE FIX
-------
Truncate the cached rows to their common width before stacking. Rows are ordered by
distance, so this keeps the nearest neighbours -- and it is a strict no-op whenever
the widths already agree, which is every case that works today.

Deliberately NOT changed: `Knn_model.n_neighbors` is still mutated and not restored.
That is a separate latent issue; leaving it alone keeps this patch behaviour-preserving.

USAGE
-----
    import ee_patch; ee_patch.apply()      # returns the original function
    ee_patch.revert()
"""
import numpy as np
import EagleEye
from EagleEye import PValueCalculator, collect_unique_

_ORIGINAL = None


def IDE_step_optimized_fixed(Knn_model, X, Y, putative_indices, banned_set, K_M, p,
                             Upsilon_i, Upsilon_star_plus, neighbors_dict, nX):
    """Byte-for-byte the upstream function except for the three marked lines."""
    # ---------------- PATCH ----------------
    # was: new_indices = np.array([neighbors_dict[i] for i in putative_indices])
    _rows = [neighbors_dict[i] for i in putative_indices]
    _w = min(len(r) for r in _rows)
    new_indices = np.array([r[:_w] for r in _rows])
    # -------------- END PATCH --------------

    is_overdensity = new_indices[0, 0] >= nX
    if is_overdensity:
        banned_set = {x + nX for x in banned_set}

    banned_array = np.array(list(banned_set))

    mask = ~np.isin(new_indices, banned_array)
    valid_counts = mask.sum(axis=1)

    num_points = new_indices.shape[0]
    filtered_indices = np.empty((num_points, K_M), dtype=int)

    rows_ok = np.where(valid_counts >= K_M)[0]
    if rows_ok.size > 0:
        valid_rows = [new_indices[i, :][mask[i]][:K_M] for i in rows_ok]
        for idx, row in zip(rows_ok, valid_rows):
            filtered_indices[idx, :] = row

    rows_to_update = np.where(valid_counts < K_M)[0]
    if rows_to_update.size > 0:
        indices_to_update = [putative_indices[i] for i in rows_to_update]
        Knn_model.n_neighbors = int((X.shape[0] + Y.shape[0]) * 0.05)
        new_rows = Knn_model.kneighbors(Y[indices_to_update, :])[1]

        for idx, row_idx in enumerate(rows_to_update):
            new_row = new_rows[idx]
            neighbors_dict[putative_indices[row_idx]] = new_row
            row_mask = ~np.isin(new_row, banned_array)
            row_filtered = new_row[row_mask]
            if row_filtered.size < K_M:
                pad_value = row_filtered[-1] if row_filtered.size > 0 else 0
                padded = np.pad(row_filtered, (0, K_M - row_filtered.size),
                                mode='constant', constant_values=pad_value)
                filtered_indices[row_idx, :] = padded
            else:
                filtered_indices[row_idx, :] = row_filtered[:K_M]

    if is_overdensity:
        binary_loc = (filtered_indices > nX).astype(int)
    else:
        binary_loc = (~(filtered_indices > nX)).astype(int)

    KSTAR_RANGE = range(20, K_M)
    p_val_info = PValueCalculator(binary_loc, KSTAR_RANGE, p=p)

    max_p_val = p_val_info.min_pval_plus.max()
    index_max = np.where(p_val_info.min_pval_plus == max_p_val)[0]

    if is_overdensity:
        unique_elements = collect_unique_(filtered_indices[index_max, :], nX, True)
        unique_elements_l = [z - nX for z in unique_elements
                             if Upsilon_i[z - nX] > Upsilon_star_plus]
    else:
        unique_elements = collect_unique_(filtered_indices[index_max, :], nX, False)
        unique_elements_l = [z for z in unique_elements
                             if Upsilon_i[z] > Upsilon_star_plus]

    updated_putative = [elem for elem, stat in
                        zip(putative_indices, p_val_info.min_pval_plus)
                        if stat > Upsilon_star_plus]

    return unique_elements_l, max_p_val, updated_putative


def apply():
    global _ORIGINAL
    if _ORIGINAL is None:
        _ORIGINAL = EagleEye.IDE_step_optimized
    EagleEye.IDE_step_optimized = IDE_step_optimized_fixed
    return _ORIGINAL


def revert():
    global _ORIGINAL
    if _ORIGINAL is not None:
        EagleEye.IDE_step_optimized = _ORIGINAL

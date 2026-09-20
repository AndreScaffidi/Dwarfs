# EagleEye Multi-Reference Persistent Anomaly Detection  
## Project Summary & Technical Notes

---

# 1. Scientific Goal

Develop a **multi-reference anomaly detection framework** for stellar overdensity searches in Gaia data using EagleEye.

Main application:
- Discovery / recovery of ultra-faint dwarf galaxies and stellar clusters
- Examples explored:
  - Candidate 1 (wavelet-paper candidate)
  - Tucana II
  - Carina

Core idea:
- Compare a **test region** $begin:math:text$ \\mathcal\{Y\} $end:math:text$ against multiple local reference regions $begin:math:text$ \\mathcal\{X\}\^\{\(j\)\} $end:math:text$
- Detect overdense structures persistent across references
- Aggregate evidence into a combined statistic:
$begin:math:display$
\\Gamma\_i \= \\sum\_\{j\=1\}\^\{R\} \\Upsilon\_i\^\{\(j\)\}
$end:math:display$

where:
- $begin:math:text$ \\Upsilon\_i\^\{\(j\)\} $end:math:text$ is the EagleEye local anomaly score for point $begin:math:text$i$end:math:text$ relative to reference $begin:math:text$j$end:math:text$
- $begin:math:text$R\=8$end:math:text$ local reference tiles in the current implementation

---

# 2. Gaia Data Pipeline

## Query Strategy

Data retrieved from:
$begin:math:display$
\\texttt\{gaiadr3\.gaia\\\_source\}
$end:math:display$

Features:
- RA
- Dec
- PMRA
- PMDEC
- Parallax
- RUWE
- Photometry

Typical cuts:
```sql
pmra BETWEEN -5 AND 5
pmdec BETWEEN -5 AND 5
(parallax - 3*parallax_error) < 0
```

---

# 3. Spatial Windowing

The sky region is partitioned into a $begin:math:text$3\\times3$end:math:text$ grid:

- Central tile:
$begin:math:display$
\\mathcal\{Y\}
$end:math:display$
(test set)

- Surrounding 8 tiles:
$begin:math:display$
\\mathcal\{X\}\^\{\(j\)\}\, \\quad j\=1\,\\dots\,8
$end:math:display$
(reference sets)

Each tile is represented in local coordinates:
$begin:math:display$
\[\\Delta \\mathrm\{RA\}\\cos\\delta\_0\,\\ \\Delta\\mathrm\{Dec\}\,\\ \\mu\_\\alpha\,\\ \\mu\_\\delta\]
$end:math:display$

with wrapped RA:
$begin:math:display$
\\Delta\\mathrm\{RA\}
\=
\(\\mathrm\{RA\}\-\\mathrm\{RA\}\_0\)\_\{\\mathrm\{wrapped\}\}
$end:math:display$

---

# 4. Feature Scaling

Robust scaling applied globally:
$begin:math:display$
Z\_\{\\mathrm\{scaled\}\}
\=
\\frac\{Z\-\\mathrm\{median\}\(Z\)\}\{\\mathrm\{MAD\}\(Z\)\}
$end:math:display$

Scaling fit using:
- either all tiles
- or test tile only

Current preferred:
```python
fit_scaling_on="all"
```

---

# 5. EagleEye Detection

For each reference:
$begin:math:display$
\\mathcal\{X\}\^\{\(j\)\} \\rightarrow \\mathcal\{Y\}
$end:math:display$

Run:
```python
EagleEye.Soar(...)
```

Outputs:
- $begin:math:text$ \\Upsilon\_i\^\{\(j\)\} $end:math:text$
- IDE masks
- Repechage bookkeeping
- local anomaly clusters

---

# 6. Multi-Reference Statistic

Per-point combined score:
$begin:math:display$
\\Gamma\_i
\=
\\sum\_\{j\=1\}\^\{8\}
\\Upsilon\_i\^\{\(j\)\}
$end:math:display$

Persistent anomalies correspond to large:
$begin:math:display$
\\Gamma\_i
$end:math:display$

---

# 7. Null Construction

## Original Single-Reference Null

EagleEye null:
$begin:math:display$
\\mathcal\{U\}\[0\,1\]\^d
\\ \\text\{vs\}\\
\\mathcal\{U\}\[0\,1\]\^d
$end:math:display$

used to generate:
$begin:math:display$
\\Upsilon\^\\star
$end:math:display$

---

## Multi-Reference Empirical Null

Constructed via Monte Carlo bootstrap:

For each bootstrap:
- generate:
$begin:math:display$
Y\_b \\sim \\mathcal\{U\}\[0\,1\]\^d
$end:math:display$
and
$begin:math:display$
X\_b\^\{\(j\)\} \\sim \\mathcal\{U\}\[0\,1\]\^d
$end:math:display$

- rerun full multi-reference pipeline
- compute:
$begin:math:display$
\\Gamma\_i\^\{\(b\)\}
$end:math:display$

Final empirical null:
$begin:math:display$
\\Gamma\_\{\\mathrm\{null\}\}
\=
\\\{
\\Gamma\_i\^\{\(b\)\}
\\\}
$end:math:display$

Threshold:
$begin:math:display$
\\Gamma\^\\star
\=
Q\_\{1\-\\alpha\}\(\\Gamma\_\{\\mathrm\{null\}\}\)
$end:math:display$

This is effectively a:
- Monte Carlo null
- bootstrap-style empirical null
- full-pipeline simulation null

---

# 8. Persistence & Pruning

Construct:
$begin:math:display$
\\widetilde\{\\Gamma\}\_i
\=
\\sum\_j
\\widetilde\{\\Upsilon\}\_i\^\{\(j\)\}
$end:math:display$

where pruned/IDE-removed contributions are zeroed.

Persistence score:
$begin:math:display$
P\_i
\=
\\sum\_j
\\mathbf\{1\}
\\left\(
\\widetilde\{\\Upsilon\}\_i\^\{\(j\)\} \> 0
\\right\)
$end:math:display$

Persistent candidates:
```python
persistence >= threshold
```

---

# 9. Gamma Clustering

Thresholded anomaly region:
$begin:math:display$
A\_\\Gamma
\=
\\\{i\:\\Gamma\_i\>\\Gamma\^\\star\\\}
$end:math:display$

DBSCAN clustering applied:
```python
DBSCAN(eps, min_samples)
```

Produces:
- Gamma clusters
- local persistent anomaly structures

Noise removal disabled in latest implementation.

---

# 10. Local Significance Estimation

For each Gamma cluster:
$begin:math:display$
S \= \|A\_\\Gamma\^\{\(\\alpha\)\}\|
$end:math:display$

Background estimate obtained from overlapping repechage clusters.

Weighted background:
$begin:math:display$
\\widehat\{B\}
\=
\\frac\{
\\sum\_j
o\_\\alpha\^\{\(j\)\}
\\widehat\{B\}\_\\alpha\^\{\(j\)\}
\}\{
\\sum\_j
o\_\\alpha\^\{\(j\)\}
\}
$end:math:display$

where:
- $begin:math:text$o\_\\alpha\^\{\(j\)\}$end:math:text$ = overlap count with reference cluster $begin:math:text$j$end:math:text$

Final local significance:
$begin:math:display$
\\frac\{S\}\{\\sqrt\{\\widehat\{B\}\}\}
$end:math:display$

Used as:
- approximate local detection significance
- ranking statistic for Gamma clusters

---

# 11. Repechage Integration

Repechage clusters:
```python
EE_book["Y_OVER_clusters"]
```

Used to:
- estimate local backgrounds
- compute overlap persistence
- build weighted significance estimates

Only clusters satisfying:
$begin:math:display$
z\_\\alpha \> 0
$end:math:display$
are retained in weighted background estimation.

---

# 12. Validation Against Known Systems

## Candidate 1
- Successfully rediscovered using:
  - multiple data-driven references
  - no Poisson background assumption
- Compared against original wavelet detection

Important claim:
> Recovery achieved using fully empirical multi-reference backgrounds rather than idealized Poisson noise.

---

## Tucana II
Crossmatched DR3 members using:
```python
source_id
```

Validation products:
- Sky overlays
- PM-space overlays
- KDE contour comparisons
- TPR / overlap metrics

---

# 13. Visualization System

Developed publication-ready plotting:
- LaTeX rendering
- RA/Dec plots
- PMRA/PMDEC plots
- KDE contours
- stitched full-grid views
- Gamma cluster annotations

Features:
- Gamma cluster labels
- local $begin:math:text$S\/\\sqrt\{B\}$end:math:text$
- candidate overlays
- grid boundaries

---

# 14. DBSCAN Comparison Baseline

Single-sample DBSCAN benchmark implemented:
- directly on $begin:math:text$Y$end:math:text$
- no reference subtraction

Purpose:
- compare EagleEye vs naive clustering
- demonstrate value of multi-reference subtraction

---

# 15. Current Conceptual Interpretation

EagleEye persistent anomalies:
- not merely overdensities
- but overdensities stable under:
$begin:math:display$
\\mathcal\{X\}\^\{\(j\)\}
\\rightarrow
\\mathcal\{Y\}
$end:math:display$
background substitutions.

The method effectively performs:
- local background marginalization
- systematic robustness testing
- multi-reference anomaly consensus detection

---

# 16. Main Open Questions

## Statistical
- Exact interpretation of:
$begin:math:display$
S\/\\sqrt\{B\}
$end:math:display$
under correlated references

- Optimal global thresholding for:
$begin:math:display$
\\Gamma\_i
$end:math:display$

- Extreme-value corrections

---

## Astrophysical
- completeness vs wavelet methods
- false-positive suppression
- crowded-field behavior
- sensitivity to diffuse systems

---

# 17. Key Achievements So Far

✅ Multi-reference EagleEye operational  
✅ Persistent anomaly formalism developed  
✅ Empirical Gamma null implemented  
✅ Candidate 1 recovered  
✅ Tucana II recovered  
✅ Local significance estimator implemented  
✅ Publication-quality visualization pipeline complete  
✅ Full DR3 coordinate reconciliation solved  
✅ Weighted repechage background framework operational
---
---

# PART II — Methodology and Validation Update (Sept 2026)

*Everything below post-dates the original notes above. Sections 1–17 describe the
framework; this part records the corrections, the calibration strategy, and the
tests actually run. Where a number is quoted it was measured, not estimated.*

---

# 18. $K_M$: the locality bound and `"auto"`

## 18.1 The bound

EagleEye's binomial model assumes the $k$-NN neighbourhood is *local* — i.e. the
local density is approximately constant over the ball. The paper (M&M, after
Eq. 2) sets

$$
K_M \;\le\; 0.05 \,\min\!\left(n_X,\, n_Y\right)
$$

Note **min**, not the union. The expression `0.05*(nX+nY)` does appear inside
`IDE_step_optimized`, but it governs the neighbour *re-fetch* buffer, a different
quantity. Using the union as the bound over-runs locality by up to 2×.

Lower edge: `KSTAR_RANGE = range(20, K_M)`, so $K_M \lesssim 25$ leaves
essentially no test to perform. Encoded as `robustness.KM_EQ_FLOOR = 25`.

## 18.2 Why larger $K_M$ is (mostly) better

$\Upsilon_i$ is a max over $k \in [20, K_M)$ of $-\log$(binomial right-tail).
Its ceiling is

$$
\Upsilon_{\max} = -(K_M - 1)\ln(1 - \hat p), \qquad \hat p = \frac{n_Y}{n_X + n_Y}
$$

which is **independent of the contaminant size**. So a small $K_M$ caps the
achievable significance regardless of how strong the anomaly is, and a
*diffuse* anomaly — one whose excess only accumulates over many neighbours —
needs a large $K_M$ to be seen at all. The cost of large $K_M$ is loss of
locality (background gradients leak in) and compute.

Critically, $\Upsilon_{\max}$ **collapses when $\hat p \to 0$**, i.e. when the
reference is much larger than the test set. This is the mechanism behind the
equalisation requirement in §19.

## 18.3 `resolve_km` and `kM="auto"`

`robustness.resolve_km(kM, nXs, nY)` returns `(kM_eff, bound, mode)`:

```python
bound = int(0.05 * min(min(nXs), nY))
if kM == "auto":
    return max(KM_EQ_FLOOR, bound), bound, "auto"
return int(kM), bound, "fixed"
```

`"auto"` = **the largest $K_M$ the binomial/locality assumption permits**, given
the realised cardinalities. Available in two places:

| Setting | Scope |
|---|---|
| `CONFIG["KM"] = "auto"` | final stacked $\Gamma$ analysis |
| `EQ_KM = "auto"` | per-reference, per-pass during equalisation |

They resolve independently — the equalisation passes each recompute the bound
from the *current* cardinalities, while the final $\Gamma$ uses one $K_M$
shared across all 8 sub-analyses (this is required: $\Upsilon^{(j)}$ must be
commensurable before summing).

---

# 19. Reference equalisation

`equalise_references(Y, X_refs, kM=EQ_KM, headroom_min=EQ_HEADROOM, ...)`

References are iteratively pruned so that no $\mathcal{X}^{(j)}$ is so much
larger than $\mathcal{Y}$ that $\hat p$ collapses and $\Upsilon$ saturates.
Each pass:

1. recompute $K_M$ from the current $(n_X^{(j)}, n_Y)$ under the locality bound;
2. check **headroom** $= \Upsilon_{\text{obs,max}} / \Upsilon_{\max}$;
3. raise if headroom $>$ `1 - headroom_min` (saturation — the statistic can no
   longer distinguish "strong" from "stronger").

Parallel over references via `ref_workers` (`_equalise_ref_task`).

---

# 20. $\Gamma$ is Fisher's method — and what that buys

Since $\Upsilon = -\log p$,

$$
\Gamma_i = \sum_{j=1}^{R} \Upsilon_i^{(j)} = -\log \prod_j p_i^{(j)}
$$

which is exactly Fisher's combined-probability statistic. For *independent*
uniform $p$-values,

$$
2\Gamma \sim \chi^2_{2R}, \qquad \text{equivalently} \qquad \Gamma \sim \mathrm{Gamma}(R, 1)
$$

The references are **not** independent (they share the real background), so the
empirical bootstrap null (§7) remains the operational threshold. But the
analytic family gives two useful diagnostics:

## 20.1 $R_{\rm eff}$ vs $k_{\rm fit}$ — two different numbers

These were previously both called `k_eff` and conflated. They are now named
distinctly:

| Name | Definition | Source |
|---|---|---|
| `R_eff` | $R / \left[1 + (R-1)\,c\right]$, $c$ = mean pairwise correlation of $\Upsilon^{(j)}$ | `reference_independence()` |
| `k_fit` | $\mathrm{mean}^2/\mathrm{var}$ of the empirical $\Gamma$ null | `overlay_gamma_null_curves()` |

They are related by the identity

$$
\frac{\mathrm{mean}^2}{\mathrm{var}} = R_{\rm eff} \times \frac{m^2}{v}
$$

where $m, v$ are the mean and variance of a single $\Upsilon$. They coincide
**only** if $\Upsilon \sim \mathrm{Exp}(1)$ exactly. They do not, so the two
numbers differ — legitimately.

`overlay_gamma_null_curves(axes, run)` draws both $\mathrm{Gamma}(k_{\rm fit})$
and $\mathrm{Gamma}(R, 1)$ over the empirical null. The gap between them is the
visual measure of how much the reference correlation costs.

---

# 21. **Bug fix in `EagleEye.py` — Eq. 9 background estimator**

## 21.1 The bug

The paper's injection-based background estimator (§M4, Eq. 9) is

$$
\hat{B} = |Y_{\rm inj}| \,\frac{n_Y - |\hat{Y}_+|}{n_X - |\hat{X}_+|},
\qquad
\Lambda = \frac{|Y_{\rm anom}| - \hat{B}}{\sqrt{\hat{B}}}
$$

The code had $n_1$ and $n_2$ **swapped** in the ratio:

```python
B_hat = lenBo * (n2 - lenWo) / (n1 - lenWu)    # wrong
B_hat = lenBo * (n1 - lenWo) / (n2 - lenWu)    # correct
```

The X-side functions relabel $(Y_{\rm anom}, \hat Y_+, n_Y, Y_{\rm inj})
\leftrightarrow (X_{\rm anom}, \hat X_+, n_X, X_{\rm inj})$ — that interchange
is deliberate and correct (EagleEye is symmetric in which set is "reference").
The ratio was nevertheless inverted **in both**.

## 21.2 Exactly what changed

11 lines, one substring each, in
`/data/ascaffid/LHC_Olympics/EagleEye/eagleeye/EagleEye.py`
(backup: `EagleEye.py.bak_20260910_141411`):

| Function | Lines |
|---|---|
| `S_SB_estimate_Y_overdensities` | 677 |
| `S_rootB_estimate_Y_overdensities` | 704, 715 |
| `B_estimate_Y_overdensities` | 740, 750 |
| `S_rootB_estimate_X_overdensities` | 775, 776, 786, 787 |
| `S_SB_estimate_X_overdensities` | 814, 825 |

(Lines 666, 703, 714 had already been fixed by hand.)

## 21.3 Impact

- **Balanced runs ($n_X \approx n_Y$): ~4 %.** This is why it survived — and why
  all the balanced stacked results above remain essentially valid.
- **Unbalanced runs: catastrophic.** The equalisation gate produced *negative*
  $\hat B$; the pooled-reference run produced $\hat B$ **66× too large**.
- A *partial* fix (numerator only) is worse than either: mixing the two $\hat B$
  conventions gave 0.132 where the correct answer is 1.071.

The `oversize` branch previously added to `equalise_references` existed only to
paper over this and is now **dead code**.

**Still open:** $\hat B = 0$ (when `lenBo == 0`) divides by zero instead of
returning NaN. The old inflated $\hat B$ hid this path; the corrected smaller
values make it reachable. Worth reporting the inversion upstream — it is
invisible at $n_X \approx n_Y$ but bites precisely in the unbalanced regime the
paper claims robustness in.

---

# 22. Stacking vs pooling

Two ways to use $R$ references:

- **Stack**: run EE once per reference, sum $\Upsilon^{(j)} \to \Gamma$ (Fisher).
- **Pool**: concatenate $\mathcal{X}_{\rm pooled} = \bigcup_j \mathcal{X}^{(j)}$
  and run the vanilla EE suite (pruning + repêchage) once.

`robustness.pooled_reference_run()` builds the pooled set from the
*post-equalisation* references (`refs_after_equalisation`), recomputes the
empirical null natively, and runs the full suite. `plot_pooled_vs_stacked()`
overlays the pruned+repêchaged pooled points on the stacked $\Gamma > \Gamma^\star$
contours.

## 22.1 Measured, not argued

| Quantity | Value |
|---|---|
| Information per Y-neighbour, stacked / pooled | **1.02** (not 10–18×) |
| $\Gamma / \Upsilon_{\rm pool}$ (total) | **8.13** $\approx R$ |
| $z$-advantage, stacked / pooled | **1.45×** $\approx \sqrt{R_{\rm eff}} = 1.43$ |

The earlier analytic claim that pooling discards 10–18× the information was
**wrong**: it held the neighbourhood volume fixed. $k$-NN is adaptive — a denser
pooled reference shrinks the ball and compensates almost exactly.

## 22.2 The crossover (the actual result)

`stack_vs_pool.stacking_benchmark(nY, nXs, sig_ns, sig_ws, ...)` runs the
comparison on synthetic fields with **the real cardinalities of the current
Gaia tile** and lumpy (locally perturbed, unequal-size) references. Scored on
**TPR**, not purity — EagleEye here is a flagging stage; purity is the job of
the later supervised dynamical study.

| Anomaly | Stacked TPR | Pooled TPR |
|---|---|---|
| Compact ($w=0.035$, $N=10$) | 0.25 | **0.95** |
| Diffuse ($w=0.085$, $N=20$) | **0.425** | 0.225 |

**Compact anomalies favour pooling; diffuse anomalies favour stacking.** The
mechanism: stacking is a persistence measure and suppresses variance on a signal
present in every comparison, which is what a diffuse excess looks like; but a
$\hat p$ pedestal from reference lumpiness is *additive and undiluted* under
stacking, whereas pooling averages it away.

Consistent with the contamination test: under M3 contamination, stacked FPR was
**75×** vs pooled **37×** — stacking is the *worse* of the two there.

## 22.3 Reported $S/\sqrt{B}$ is not sensitivity

Same benchmark, compact case: stacked **reported** $S/\sqrt{B} = 8.52$ while
recovering only 25 %; pooled reported 1.78 while recovering 95 %. The reported
significance and the actual detection rate are close to *anti*-correlated here.
This is the single strongest argument for calibrating the flagging threshold on
measured false-alarm rates rather than on $S/\sqrt{B}$ read off a run.

## 22.4 Related tooling

- `paired_detection_test(..., srb_thresh_stacked=5.0)` — McNemar's exact test:
  *given* stacked is calibrated to $S/\sqrt{B} > 5$, how often does it see
  something pooled cannot?
- `demo_realisation()` — 4-panel before/after view of the tile.
- `false_alarm_test(..., thresholds=(2,3,4,5,7))` — FA rate vs threshold.

---

# 23. Null tests

Scrambling alone was (correctly) judged too weak. Four independent nulls now
exist, in increasing order of realism:

## 23.1 PM scramble — `wake_test.pm_scramble_null`

Shuffle proper motions within the tile, destroying PM–position coherence while
preserving both marginals. `scramble_pvalue_at(observed, trials, ra, dec,
radius_deg)` gives a *position-matched* $p$-value.

Result for the Reticulum II-field candidate: observed $S/\sqrt{B} = 7.05$ vs a
maximum of 3.54 over 12 blank trials, $p < 0.08$.

**Interpretation limit:** this null only answers "could EagleEye have
manufactured this from the marginals alone?" It cannot distinguish a real chance
density from a real system.

## 23.2 Off-position — `wake_test.offpos_null`

Run the identical pipeline at $N$ positions elsewhere on the sky and read off
the empirical distribution of $S/\sqrt{B}$. `passable_srb(vals, q=(.5,.9,.95,.99))`
converts this to a **minimum passable $S/\sqrt{B}$**. This is the empirical
alternative the scramble null cannot provide.

Hardening applied after a first pass was found to be unsound:
- `avoid_deg` defaults to **$3h$**, not $0.8°$ — a $3\times3$ grid spans $\pm 3h$,
  so a small exclusion kept known objects out of $\mathcal{Y}$ but left them
  sitting in the *references*.
- `EXTRA_AVOID` for objects absent from DW20 (DW20 has no globulars).
- `nY_ref` / `nY_tol=0.35` density matching, so off-positions are compared at
  comparable star counts.

## 23.3 Smooth mock — `wake_test.smooth_mock_null`

KDE-resample the field (`pos_bw=0.35`, `pm_bw=1.0`) to build a mock with the
same smooth large-scale structure but no small-scale clustering, then run the
pipeline. Tests whether the smooth gradient alone suffices.

## 23.4 Scan-level persistence false alarm — `stack_vs_pool.scan_false_alarm`

**The one that matters for flagging policy.** A marginal $S/\sqrt{B} \sim 2.5$–3
cluster means little in isolation; the same cluster reappearing *at the same sky
position* across all 8 tile perturbations is a different claim, and carries
statistical power the single-run $S/\sqrt{B}$ does not express.

`scan_false_alarm(r, anomaly, h0, n_trials, match_deg=0.25, scan_kw=...)` runs
the full 8-configuration robustness scan on blank/mock fields and measures how
often a cluster persists across configurations within `match_deg`. Output is the
**post-robustness calibration of $S/\sqrt{B}$** — the number to flag on.

Important caveat built into the design: **shrinking the box shrinks significance**
by construction (that is how EagleEye works), so the threshold must be scaled
down correspondingly per configuration rather than held fixed.

Parallelism (`SFA_SCAN_KW`): `config_workers=8, ref_workers=1, boot_workers=1`.
joblib disables nested `Parallel`, so `config_workers=8` + `ref_workers=8`
silently degrades to config-level only; and a configuration (~112 s) amortises
loky's startup cost where a single `Soar` (0.7 s) does not. Still serial *across
trials* — saturating the 112 cores would need separate processes with different
`seed`. `n_jobs=8` is fine at `config_workers=8` (64 threads) but must drop to 1
if `config_workers` exceeds ~14.

---

# 24. Robustness scan and the threshold question

`robustness.scan(r, anomaly, h0, ...)` perturbs the tile grid
(8 configurations: offsets × scale factors) and measures whether a candidate
survives. `verdict(summary, min_frac=0.75, min_jaccard=0.5, max_srb_ratio=3.0)`.

`freeze_scale` now takes the MAD metric from a bare window build rather than a
full nominal run (verified bit-identical) — a substantial speedup.

## 24.1 Why small tiles need a looser `p_ext`

Observed: at $3\times3$ with $h$ reduced, the candidate vanishes unless
`p_ext = 1e-2`; at `1e-3` it is recovered but at $S/\sqrt{B} \approx 4$.

This is **not** a bug and not $K_M$ hitting its floor. Two effects compound:

1. Smaller tiles → smaller $n_Y, n_X$ → smaller locality bound → smaller
   $K_M^{\rm auto}$ → lower $\Upsilon_{\max}$ ceiling (§18.2).
2. `p_ext` is the per-point pruning threshold; with fewer points the
   multiple-comparison burden drops, so the *same* physical excess sits at a
   less extreme quantile.

A **diffuse** overdensity is exactly the case that needs the looser threshold —
it never produces a single extreme point, only many mildly elevated ones.
Consistent with the stacking/diffuseness result in §22.2.

## 24.2 The Reticulum II contamination effect

Observed: when Reticulum II falls inside the ROI, the *candidate's* significance
and flagged-point count both jump. The likely mechanism is not that Ret II leaks
into $\mathcal{Y}$, but that at some grid offsets **Ret II sits in a reference
tile** — inflating that reference's local density, depressing $\hat p$ there,
and skewing the stacked $\Gamma$. This is a reference-contamination
sensitivity, and is one more reason the off-position null (§23.2) must exclude
known systems from references, not just from the test set.

---

# 25. Candidate characterisation: the Reticulum II-field anomaly

27 stars at $(\alpha, \delta) = (52.881, -54.936)$.

| Test | Result | Reading |
|---|---|---|
| Morphology | compact, round, not edge-aligned | not a tiling artefact |
| PM scramble null | $p < 0.08$ (7.05 vs max 3.54 / 12) | not manufactured from marginals |
| Tile-size stability | $\times 0.32$ (Ret II: $\times 1.08$) | **fails** — but confounded by 2.2× larger extent |
| CMD coherence | $p = 0.345$; injection test shows 14/15 power | no isochrone coherence, and the test *has* power |
| Raw star counts | $1.03\times$, $+0.4\sigma$ (Ret II: $4.41\times$, $+11.4\sigma$) | flat — no photometric overdensity |
| Mean PM vs Ret II | $16.9\sigma$ (468 km/s) | **not associated** with Ret II |
| Minimum bound mass | $\gtrsim 1.8\times10^5\,M_\odot$ | see §26 |

Net: a genuine PM-space clustering with **no photometric counterpart**. The
outstanding concern is whether 10–20 points (tile-dependent) are chance
alignments within a strongly clustered PM distribution.

## 25.1 Weak lensing / stellar wake

Assessed and rejected as impractical: a dark perturber at Ret II's distance
produces a shear far below any achievable measurement with available data, and
the two features have incompatible proper motions (16.9$\sigma$), so a
common-origin wake interpretation is not supported.

`wake_test.classify_cluster` / `report` / `_verdict` implement the
bound-system vs wake vs artefact decision, including a **selection tell**:
$\sigma_{\rm obs} < 0.85\,\sigma_{\rm error}$ indicates the "cluster" is
narrower than the measurement error, i.e. a selection-function artefact rather
than a physical structure.

`report_geometry` / `edge_alignment` / `tile_gradient` test for tiling
artefacts; `scan_persistence` separates **sky-locked** from **tile-locked**
behaviour — the key discriminant, since an artefact should follow the tile, not
the sky.

---

# 26. Gravitational and photometric diagnostics

## 26.1 Boundedness

`wake_test.boundedness(res, dist_kpc)`, `jacobi_radius_pc(M_sat, dist_kpc)`,
`min_bound_mass(r_pc, dist_kpc)`, `plot_grav_radii(...)` — the last overlays the
gravitationally allowed radius on the stacked contours of the persistent cluster.

**Result: boundedness has no discriminating power at 30 kpc.** Proven by running
it on Reticulum II, a known bound system: it fails identically, both "requiring"
$\sim 5\times10^8\,M_\odot$. The test is dominated by the fact that RA/Dec are
angles, not distances — the line-of-sight extent is unconstrained. Reported here
so it is not re-run expecting an answer.

## 26.2 Isochrone fitting

`build_ridgeline(g_tmpl, col_tmpl, mu_tmpl, deg=3, clip=2.5, n_iter=3)` →
`isochrone_scan(g_mem, col_mem, ridge, M_range, mu_grid)` →
`plot_isochrone_scan(...)`.

Empirical ridgelines are built from a **template dwarf**, fetched from Gaia on
demand if not already cached (`template_photometry(bat, name, r, ...)`), using
Battaglia+2022 membership probabilities (`pmemb_min=0.5`).

Distance moduli come from a DW20 lookup (`_dw20_distance`), **not** hard-coded:
`TEMPLATE_DIST = 25.0` kpc for Bootes III was wrong — DW20 gives **47 kpc**
($\mu = 18.36$, not 17.0).

## 26.3 Metallicity as a 5th feature

Gaia DR3 *does* carry metallicity, via `gaiadr3.astrophysical_parameters`
(`mh_gspphot`, `mh_gspspec`) — not in `gaia_source` directly. Usable in
principle as a 5th EagleEye dimension, with the caveat that GSP-Phot
metallicities are unreliable at the faint magnitudes UFD members occupy, so the
feature would be mostly noise exactly where it is needed. Not currently enabled.

---

# 27. Parallelisation

`ee_parallel.py`:

| Function | Role |
|---|---|
| `plan_workers(n_configs, n_refs, n_jobs, cores)` | worker budgeting |
| `pmap(fn, tasks, workers)` | loky map with serial fallback |
| `compute_Gamma_i_multi_ref_par` | 8 references on separate cores |
| `bootstrap_gamma_null_uniform_par` | parallel bootstrap null |
| `verify_parallel_identity` | asserts bit-equivalence to serial |
| `quiet_ee(enabled, capture)` | suppresses EagleEye chatter |

## 27.1 Correctness constraints learned the hard way

- **fork breaks OpenBLAS; spawn re-imports `__main__` (unusable in Jupyter).**
  Only **loky with `inner_max_num_threads=1`** works.
- **joblib disables nested `Parallel`** — `config_workers` × `ref_workers` does
  not multiply; the inner level silently degrades.
- **Reseeding the bootstrap changes the answer.** Naive per-worker reseeding
  moved $\Gamma^\star$ from 52.38 → 61.66, turning 2 clusters into 1. Fixed with
  `PCG64(seed).advance(b * per)` so the parallel stream reproduces the serial one
  exactly.
- **`_ensure_child_libpath()`**: the Jupyter kernel has conda's `libstdc++`
  mapped but `LD_LIBRARY_PATH` absent from `os.environ`; loky execs a fresh
  interpreter which then fails to import pandas (`GLIBCXX_3.4.29`). Reproduced,
  fixed, and backed by a serial fallback.

## 27.2 Measured speedup

At `ref_workers=8, boot_workers=20`: **1.67×** ($n_{\rm boot}=4$),
**3.47×** ($n_{\rm boot}=20$). $\Gamma^\star$ identical;
$\max|\Delta\Gamma_i| = 1.4\times10^{-14}$.

## 27.3 Output control

`robustness.VERBOSITY`: 0 silent / 1 per-run summary / 2 + equalisation detail /
3 + full EagleEye chatter. `_q(level)` context manager; `run_summary(run, label, t)`
prints the one-line per-run digest.

---

# 28. Data-integrity guards

A run on Reticulum II was invalidated by calling `run_one` directly on a
$\pm5°$ table with a $(-1,-1)$ grid offset, bypassing `AUTO_QUERY_PAD`. The
western references silently lost **26 % of their RA extent**, manufacturing a
390-star "cluster" at $S/\sqrt{B} = 239$ and a fictitious 45 % Magellanic
gradient.

Two fixes:
1. **Hard coverage guard in `run_one`** — raises if the supplied table does not
   cover $\pm 3h$ around the window centre.
2. `plan_query` / `required_query_halfwidths` / `estimate_rows` to size the
   query before it is issued.

**Rule: never call `run_one` on a table you did not size with `plan_query`.**

---

# 29. Current status of the flagging strategy

The working position, in order of authority:

1. **$\Gamma$ persistence across references** flags candidate regions.
2. **The 8-fold robustness scan** tests sky-locked vs tile-locked behaviour
   (`scan_persistence`) — a tile-locked feature is rejected.
3. **The threshold to flag on is calibrated from `scan_false_alarm`**, not from
   a run's reported $S/\sqrt{B}$ (§22.3 shows the latter is untrustworthy).
4. **Pooling is the cross-check, not the alternative** — run both; disagreement
   is informative because the crossover in §22.2 tells you the anomaly's
   diffuseness.
5. Anything surviving 1–4 goes to supervised dynamical modelling. EagleEye's job
   ends at flagging; purity is not its metric.

On the interpretation of survivors: only artefacts *invoked by EagleEye itself*
are being excluded here. A chance density genuinely present in the data
($H_{\rm chance}$) is physically interesting and warrants dynamical follow-up;
a systematic in the Gaia data ($H_{\rm systematic}$) is *also* physically
interesting. Neither is a reason to discard a candidate at this stage.

---

# 30. Open items

**Code**
- `EagleEye.py`: $\hat B = 0 \Rightarrow$ `ZeroDivisionError`; return NaN instead.
- Dead `oversize` branch in `equalise_references` — remove.
- `plot_scan_overlay` has unguarded RA-wrap exposure.
- `compute_the_null` is unseeded.
- **Nothing committed.** `EagleEye.py` carries the estimator fix, the IDE
  ragged-cache fix, and ~121 lines of pre-existing uncommitted changes. Commit
  before anything else touches that file.

**Measurements not yet made**
- Scan-level persistence false-alarm rate (staged, parallelised, no number yet).
- Whether the stacking/pooling diffuseness crossover survives $>2$ trials/cell.
- Extended scramble null at higher trial count.
- `offpos_null` rerun under the hardened avoid logic.
- Manual second scan with hand-picked patches.
- `reference_checks.py` Tiers 0–3.
- Re-run the four crashed catalogue dwarfs.

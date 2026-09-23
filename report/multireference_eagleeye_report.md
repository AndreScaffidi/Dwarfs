# Multi-Reference EagleEye for Stellar Overdensity Detection in *Gaia*
## Method definition, reference handling, and thresholding

**Internal technical report — for project collaborators. Not a submission draft.**

*Draft of 2026-09-20. Section 8 (Calibration) is deliberately incomplete and is marked as such.*

---

## 0. Purpose and scope

This report documents how we apply EagleEye (Springer et al. 2025) to *Gaia* DR3 phase-space data
to search for localised stellar overdensities, and how every threshold in that pipeline is set.
It is written so that a collaborator can reproduce a run from the configuration table in §2.4 and
audit any flagged star back to the threshold that admitted it.

Three things are established here. First, the multi-reference construction: we compare a test
field against eight empirical neighbouring sky tiles rather than against a background model, and
combine the evidence into a single per-star statistic (§3). Second, reference handling: a
concentrated overdensity sitting in one of those reference tiles corrupts the statistic for
*every* test star, and we describe the mechanism, its detection, and its repair (§4). Third,
thresholding: the complete chain from raw *Gaia* rows to a flagged cluster, with each threshold
stated and justified (§5). We then show the pipeline recovering two known dwarf galaxies (§6) and
characterise one unidentified overdensity found alongside Reticulum II (§7).

One thing is **not** established. We do not yet have a calibrated false-alarm rate, and therefore
we do not have a defensible detection threshold. Section 8 lists the machinery that exists for
this and the measurements that have not been made. Until those exist, the per-cluster
significance we report should be used for *ranking* candidates, not for deciding whether a
candidate is real. This is stated in §5.4 where the statistic is defined, and again in §8.3.

Results in this report come from three runs, deliberately chosen to exercise different parts of
the pipeline. They were made at **three different configurations**, and are not directly
comparable with one another; §2.4 gives the parameters and says where the differences bite.

---

## 1. EagleEye in brief

This section is a summary of the parent method, included so that §3 onwards can be read without
reference to the EagleEye paper. Readers familiar with the method should skip to §1.4.

### 1.1 The coin-flip statistic

Given a reference sample $X$ and a test sample $Y$, EagleEye scores each point by the composition
of its neighbourhood in the pooled set $U = X \cup Y$. For a test point $Y_i$, the membership of
its $k$-th nearest neighbour is encoded as $b_i^k = 1$ if that neighbour lies in $Y$ and $0$ if
it lies in $X$. Under the null hypothesis that the two samples are drawn from the same underlying
density, the cumulative count $B(i,K) = \sum_{k \le K} b_i^k$ behaves as a sequence of coin flips,

$$B(i,K) \sim \mathrm{Binomial}(K, \hat p), \qquad \hat p = \frac{n_Y}{n_X + n_Y}.$$

The exact null is hypergeometric, because neighbours are drawn without replacement; the binomial
is an accurate approximation in the regime $K \ll n_X + n_Y$ that the method operates in.

The per-point anomaly score is the right-tail $p$-value of the observed count, maximised over
neighbourhood rank:

$$\Upsilon_i = \max_{1 \le K \le K_M} \left[ -\log \mathrm{pval}\left(B_{\rm obs}(i,K)\right) \right].$$

Maximising over $K$ is what gives the statistic sensitivity across a range of anomaly scales
without committing to one. Points with $\Upsilon_i \ge \Upsilon^*_+$ are flagged, where the
threshold is calibrated by Monte Carlo (§5.1).

The parameter $K_M$ is bounded above by the requirement that the neighbourhood be *local* — that
the $X{:}Y$ ratio be approximately constant across it. The paper sets

$$K_M \le 0.05 \min(n_X, n_Y).$$

The $\min$ matters, and is re-derived operationally in §4.4; an expression of the form
$0.05(n_X + n_Y)$ appears elsewhere in the implementation but governs a different quantity, the
neighbour re-fetch buffer.

### 1.2 IDE and repêchage

The flagged set $Y_+$ contains a halo of points whose neighbourhoods merely *overlap* a genuine
anomaly. Two refinement stages separate signal from halo.

**Iterative density equalisation (IDE)** repeatedly removes the highest-scoring remaining point
together with its nearest neighbours, and rescores, until nothing exceeds the threshold. The
removed set $\hat Y_+$ is a representation of the local density *excess* — the mass that has to
be taken out before the two samples become locally indistinguishable. It is smaller than $Y_+$
and is the quantity used wherever we need "the excess itself" rather than "everything the excess
influenced".

**Repêchage** clusters the flagged set using Density Peaks Advanced (d'Errico et al. 2021), as
implemented in `dadapy` (Glielmo et al. 2022), then, within each cluster
$\alpha$, sets a cluster-specific threshold at a low quantile ($q = 10^{-2}$) of the $\Upsilon$
values of that cluster's IDE members. Points exceeding it form the final anomaly set
$Y^{\rm anom}_\alpha$.

The distinction matters later. Repêchage **deliberately re-admits the surrounding region** and so
contains background by construction. That is what makes the repêchage set usable as a local
background estimate (§5.4) and it is why reference cleaning removes the *pruned* set and not the
repêchaged one (§4.3).

### 1.3 The injection background estimator

To estimate how much of an anomaly set is irreducible background, EagleEye moves each reference
point into the test set one at a time, rescores it, and counts how many are themselves flagged.
That injected count $Y^{\rm inj}_\alpha$, rescaled by the relative sizes of the two background
components, estimates the background in the anomaly region:

$$\hat B_\alpha = \left|Y^{\rm inj}_\alpha\right| \frac{n_Y - |\hat Y_+|}{n_X - |\hat X_+|},
\qquad
\Lambda_\alpha = \frac{|Y^{\rm anom}_\alpha| - \hat B_\alpha}{\sqrt{\hat B_\alpha}}.$$

$\Lambda_\alpha$ is interpretable as an approximate $z$-score only in the large-background regime;
outside it, it is a standardised measure of discrepancy strength. The estimator has a validity
boundary: when the pruned mass exceeds the opposing sample, the numerator of $\hat B$ goes
negative and the statistic is undefined. This is rare in the balanced configurations used here
($\hat p \approx 0.5$ throughout, §6) but common when references are pooled (§8.2).

> **Placeholder.** A question concerning the implementation of this estimator is being raised
> directly with the EagleEye authors. It is deliberately not documented here, and Appendix C is
> reserved for it. Nothing in this report depends on its resolution: all runs presented are
> balanced ($n_X \approx n_Y$), which is the regime least sensitive to it.

### 1.4 What is inherited and what is new

This report circulates among EagleEye's authors, so the boundary should be explicit.

| Inherited from EagleEye (Springer et al. 2025) | Introduced in this work |
|---|---|
| $\Upsilon$, the binomial kNN statistic | The eight-reference sky construction (§3.1) |
| IDE pruning; multimodal repêchage | $\Gamma = \sum_j \Upsilon^{(j)}$ and its reading as Fisher combination (§3.4) |
| The injection background estimator | Reference equalisation (§4.3) |
| Monte Carlo calibration of $\Upsilon^*$ | The empirical $\Gamma^\star$ bootstrap (§5.2) |
| | $K_M$ resolution under the locality bound, including `"auto"` (§4.4) |
| | Parallelisation with verified bit-level determinism (Appendix B) |

One further item concerning the estimator implementation is being raised with the EagleEye
authors directly and is deliberately not documented here.

---

## 2. Data, coverage and configuration

### 2.1 The *Gaia* query

All data are from `gaiadr3.gaia_source`. We retrieve

`source_id, ra, dec, pmra, pmdec, phot_g_mean_mag, parallax, parallax_error, ruwe`,

adding `bp_rp` when a colour–magnitude diagram is required. The query is a square box in RA/Dec
about the field centre, with RA wrapping handled explicitly:

```sql
SELECT source_id, ra, dec, pmra, pmdec, phot_g_mean_mag,
       parallax, parallax_error, ruwe
FROM gaiadr3.gaia_source
WHERE (ra BETWEEN {ra_min} AND {ra_max})     -- or (ra >= ra_min OR ra <= ra_max) if wrapped
  AND (dec BETWEEN {dec_min} AND {dec_max})
  AND pmra  BETWEEN -5 AND 5
  AND pmdec BETWEEN -5 AND 5
  AND (parallax - 3*parallax_error) < 0.0
```

Two cuts, each doing one job. The proper-motion box $|\mu| < 5\,\mathrm{mas\,yr^{-1}}$ removes the
nearby disc population, which would otherwise dominate the feature space, and bounds the PM
features so that their scatter is set by the halo rather than by a handful of high-velocity
foreground stars. The parallax cut $\varpi - 3\sigma_\varpi < 0$ retains only stars consistent
with being distant, which is the relevant population for satellites at tens of kiloparsecs.

Two things are deliberately *not* cut, and both are open questions. We apply no RUWE cut, so
poorly-behaved astrometric solutions remain in the sample; and we apply no extinction correction,
so the stellar density field retains dust structure. The second is the more consequential, since
a dust-driven density gradient is exactly the kind of large-scale structure the reference
construction is meant to absorb (§3.1) and its residual is what §4 and §7.3 test for.

### 2.2 Coverage: the precondition for every number in this report

The tiling routine does not complain when the $3\times3$ grid runs off the edge of the supplied
table. It simply returns thinner outer tiles. This is the most dangerous failure mode in the
pipeline, because a truncated reference tile contains too few stars, and test stars near the
corresponding edge are then compared against a region with almost no reference — which the
statistic reads as an overwhelming overdensity pinned to the tile boundary.

This is not hypothetical. A run on a $\pm5^\circ$ table with a $(-1,-1)$ grid offset clipped
**26 per cent of the western references' RA extent** and manufactured a 390-star "cluster" at
$S/\sqrt{\hat B} = 239$, in a patch that raw star counts show to be slightly *under*dense. The
result was invalidated entirely.

Two guards now exist. A hard coverage check inside the run driver raises if the supplied table
does not cover $\pm 3h$ about the window centre, where $h$ is the tile half-width. And the query
can be sized in advance: for a scan with box-size factors up to $s_{\max}$ and shift fraction
$\rho$, the half-width that is guaranteed to cover every configuration is

$$\mathrm{half} = h_0\left(1 + s_{\max}(3 + \rho)\right) + \mathrm{margin}.$$

The operational rule, which we state as a rule because the incident above came from breaking it:
**never run the pipeline on a table you did not size in advance.**

### 2.3 Feature construction

Each tile is represented in coordinates local to its own centre,

$$\left[\Delta\alpha \cos\delta_0,\ \Delta\delta,\ \mu_{\alpha*},\ \mu_\delta\right],$$

with $\Delta\alpha$ wrapped to $(-180^\circ, 180^\circ]$. The cosine factor uses the *grid* centre
declination $\delta_0$ for all nine tiles rather than each tile's own declination. This is
deliberate: a per-tile cosine would give each tile a slightly different angular scale, and the
resulting feature spaces would not be commensurable when their $\Upsilon$ values are summed.

### 2.4 Configuration

The three runs in this report use three different configurations. They are collected here as the
single source of truth; later sections refer to this table rather than restating parameters.

| Parameter | Boötes II (§6.3) | Reticulum II (§6.2, §7) | Ret II scan (§7.3) |
|---|---|---|---|
| Tile half-width $h$ [deg] | 1.0000 | 1.3333 | 0.667 – 1.5 |
| $K_M$ | 50 | 100 | `"auto"` (32 – 169) |
| $p_{\rm ext}$ | $10^{-3}$ | $5\times10^{-4}$ | $5\times10^{-3}$ |
| $n_{\rm boot}$ | 10 | 10 | 10 |
| $N_{\rm null}$ | 200 000 | 200 000 | 200 000 |
| DPA $Z$ | 2.65 | 2.65 | 2.65 |
| DBSCAN $\epsilon$ / min\_samples | 1.0 / 5 | 1.0 / 5 | 1.0 / 5 |
| Reference equalisation | **on**, gate $S/\sqrt{\hat B} > 5$ | **off** | on |
| Seed | 42 | 42 | 42 |
| $n_Y$ | 2 099 | 3 037 | 739 – 3 548 |
| $\sum_j n_X^{(j)}$ | 16 517 | 23 897 | — |
| $\Gamma^\star$ | 43.79 | 52.06 | 35.2 – 45.1 |
| Wall time | 32 s | — | — |

**Two warnings follow from this table.**

*The runs are not directly comparable.* $\Gamma^\star$ differs between Boötes II and Reticulum II
(43.79 against 52.06) principally because $p_{\rm ext}$ and $K_M$ differ, not because the fields
differ. Any comparison of significance between §6.2 and §6.3 is therefore a comparison of two
different tests, and we do not make one.

*The Reticulum II run has equalisation switched off.* This is a limitation of that run, not a
choice of method. We checked whether it matters: the nearest catalogued objects capable of
contaminating a reference tile — NGC 1261, Horologium I and Reticulum III — all fall **outside**
the $3\times3$ grid at $h = 1.3333^\circ$ (NGC 1261 lies $5.85^\circ$ west of the grid centre in
raw RA, against a grid half-extent of $4h = 4.0^\circ$ in the tiling coordinate). There is
therefore no known contaminant in that field and no correctness problem. But it does mean the
Reticulum II figures cannot illustrate equalisation, and the Boötes II run carries that load
instead (§4, §6.3).

---

## 3. The multi-reference construction

### 3.1 The $3\times3$ tiling

We partition a square field into a $3\times3$ grid of tiles of half-width $h$. The central tile
is the test sample $\mathcal Y$; the eight surrounding tiles are reference samples
$\mathcal X^{(j)}$, $j = 1,\dots,8$. Every reference is real sky, observed by the same instrument,
subject to the same selection function, at almost the same position on the sky.

The arrangement is chosen so that the references bracket the test tile symmetrically. Four
antipodal pairs (E/W, N/S, and the two diagonals) surround $\mathcal Y$, so that a *linear*
density gradient across the field contributes equally and oppositely to the members of each pair
and cancels in the mean.

That cancellation is only partial, and it is important to say why immediately.
$\Gamma = \sum_j \Upsilon^{(j)}$ is a sum of a non-negative, convex statistic. A linear gradient
cancels in the *mean* of the underlying densities but not in the sum of the rectified scores: the
tile on the sparser side contributes a positive $\Upsilon$ which the denser side cannot offset by
contributing a negative one, because $\Upsilon \ge 0$. A strong gradient therefore leaves a
one-sided residual along whichever edge of $\mathcal Y$ faces the sparser references. This is the
motivation for the geometric tests in §7.3, and it is one of the artefacts the method-level nulls
in §8.1 are designed to expose.

### 3.2 Robust scaling

Features are standardised by the median and median absolute deviation, fitted jointly across all
nine tiles. Fitting on the pooled set rather than on $\mathcal Y$ alone matters because the
scaling defines the metric in which nearest neighbours are found, and it must mean the same thing
in the test tile and in every reference.

The reason the scaling cannot simply be refitted per configuration is that it is not scale-free.
The positional MADs of a uniformly-filled tile scale linearly with the tile half-width, while the
proper-motion MADs are pinned near $1.4\,\mathrm{mas\,yr^{-1}}$ by the $\pm 5$ cut. Both runs
confirm this directly:

| Run | $h$ | $\mathrm{MAD}(\Delta\alpha\cos\delta)$ | $/h$ | $\mathrm{MAD}(\Delta\delta)$ | $/h$ | $\mathrm{MAD}(\mu_{\alpha*})$ | $\mathrm{MAD}(\mu_\delta)$ |
|---|---|---|---|---|---|---|---|
| Boötes II | 1.000 | 0.4894 | 0.489 | 0.4985 | 0.499 | 1.425 | 1.457 |
| Reticulum II | 1.333 | 0.3943 | 0.296 | 0.6675 | 0.501 | 1.452 | 1.394 |

The declination MAD is $0.50h$ in both cases, as expected for a uniform tile of half-width $h$.
The RA MAD is $0.50 h \cos\delta_0$ — $0.489 \approx 0.5\cos(12.9^\circ)$ for Boötes II and
$0.296 \approx 0.5\cos(54.1^\circ)$ for Reticulum II. The positional MADs therefore vary by a
factor of $1.7$ between these two fields at fixed $h$ purely through declination, while the PM
MADs are constant to 4 per cent. The position-to-PM weighting of the Euclidean metric is
consequently *not* a property of the method but of where and at what scale it is pointed.

Two consequences. When comparing configurations at different $h$ — as the robustness scan does —
the metric must be frozen from one configuration and reused, or the scan measures a changing
distance function rather than a changing box. And because the DBSCAN radius $\epsilon$ is
expressed in these scaled units, freezing the metric is also what makes a fixed $\epsilon$
correspond to a fixed angular size.

### 3.3 Per-reference nulls

Each reference has its own cardinality and therefore its own mixture proportion
$\hat p_j = n_Y/(n_Y + n_X^{(j)})$. Each needs its own null distribution and its own threshold
$\Upsilon^*_j$. In the runs presented here the references are well balanced — $\hat p_j$ spans
$0.493$–$0.515$ for Boötes II and $0.490$–$0.519$ for Reticulum II — so the eight thresholds are
close, but they are computed separately regardless.

By contrast $K_M$ is shared across all eight comparisons, and must be. $\Gamma$ sums the
$\Upsilon^{(j)}$, so they must be the same statistic; a per-reference $K_M$ would give each term a
different null *and* a different attainable ceiling (§4.2), so the sum would be weighted by tile
size rather than by evidence.

### 3.4 $\Gamma$ as Fisher combination

The combined statistic is

$$\Gamma_i = \sum_{j=1}^{R} \Upsilon_i^{(j)}, \qquad R = 8.$$

Because $\Upsilon = -\log p$, this is $\Gamma_i = -\log \prod_j p_i^{(j)}$, which is exactly
Fisher's combined-probability statistic (Fisher 1932). If the $R$ tests were independent with
uniform $p$-values, $2\Gamma$ would be distributed as $\chi^2_{2R}$, equivalently
$\Gamma \sim \mathrm{Gamma}(R,1)$.

**They are not independent.** All eight comparisons share the same test sample $\mathcal Y$, so
the Poisson noise of $\mathcal Y$ is common to all of them and the $\Upsilon^{(j)}$ are positively
correlated by construction. The nominal Fisher gain of $\sqrt{R}$ therefore overstates what
stacking actually buys. This is not a small concern: Guth & Namjoo (2026) show that combining four
correlated anomaly tests of the CMB yields a joint $p$-value around $3\times10^{-8}$ that does not
survive a proper treatment of the dependence. The same arithmetic error is available to us here,
and the empirical null of §5.2 is what we use instead of the analytic one precisely to avoid it.

The degree of departure can be read off the fitted shape of the empirical $\Gamma$ null. Matching
a $\mathrm{Gamma}(k,\theta)$ to its first two moments gives

| Run | mean | variance | $k_{\rm fit} = \mathrm{mean}^2/\mathrm{var}$ | $\theta$ | $R$ |
|---|---|---|---|---|---|
| Boötes II | 13.38 | 52.76 | **3.39** | 3.94 | 8 |
| Reticulum II | 15.70 | 61.39 | **4.02** | 3.91 | 8 |

so the null behaves like roughly four independent exponential contributions rather than eight.

**Two cautions on reading that number.** First, $k_{\rm fit}$ is measured on the *synthetic*
uniform bootstrap null, so it characterises the method — the shared-$\mathcal Y$ correlation and
the max-over-$K$ selection — and is close to the same number for any field. It does not tell us
whether *our* references differ from one another. Second, $k_{\rm fit}$ is not the same quantity
as the effective reference count $R_{\rm eff} = R/[1 + (R-1)c]$ obtained from the measured
pairwise correlation $c$ of the $\Upsilon^{(j)}$ on real background stars. The two are related by
$\mathrm{mean}^2/\mathrm{var} = R_{\rm eff}\,(m^2/v)$ for marginal $\Upsilon$ mean $m$ and
variance $v$, and coincide only if $\Upsilon$ were exactly $\mathrm{Exp}(1)$, which it is not.

**$R_{\rm eff}$ is not reported here.** The per-reference $\Upsilon^{(j)}$ arrays are not retained
in the saved output of these runs, so the correlation cannot be recovered from them without
re-running. This is a gap in what we can currently claim — it is the number that answers "how
much independent evidence do eight references actually supply?", and it is the honest counterpart
to the two-algorithm corroboration convention used elsewhere in the satellite-search literature
(e.g. Tan et al. 2026; Overdeck et al. 2026), which asserts agreement without quantifying it.
Retaining `Upsilon_by_ref` in the run output is a one-line change and is listed in §9.

---

## 4. Reference handling

This section covers the failure mode that is specific to using real sky as a reference, and its
repair. It is illustrated throughout by the Boötes II field, where the contaminant is not a
hypothetical but a catalogued dwarf galaxy.

### 4.1 Why a contaminated reference corrupts every test star

$\Upsilon$ tests a *local* neighbourhood composition against a *global* mixture proportion
$\hat p = n_Y/(n_Y + n_X)$. That comparison is valid as long as the two quantities correspond.

A reference tile that is uniformly denser than the test tile is handled correctly: the local
$X{:}Y$ ratio is elevated everywhere by the same factor that elevates $\hat p$, and the two track
each other. But a **concentrated** clump — a globular cluster, or, as here, a dwarf galaxy — is
different. It inflates $n_X$, and therefore deflates $\hat p$, while leaving the local density at
a randomly chosen field position completely unchanged. The global expectation and the local
reality decouple, and *every* point of $\mathcal Y$ acquires an apparent excess of test-sample
neighbours relative to a $\hat p$ that has been dragged down by structure nowhere near it.

The effect is a pedestal added uniformly to $\Upsilon$ across the whole test tile. Measured on
NGC 1261 in a Reticulum II reference tile at a wider grid setting, $\hat p$ fell from $0.524$ to
$0.389$ and a pedestal of $+4.66$ was added to $\Upsilon$ for all 7 911 test stars. A pedestal of
that size is comparable to the detection threshold itself.

### 4.2 The saturation ceiling

There is a second, more insidious consequence. $\Upsilon$ is the negative log of a binomial
right-tail probability on a neighbourhood of at most $K_M$ points, so the largest value it can
attain — realised when every usable neighbour belongs to the test set — is

$$\Upsilon_{\max} = -(K_M - 1)\ln(1 - \hat p).$$

This ceiling depends only on $K_M$ and $\hat p$. It is **independent of how large the contaminant
is**. As a reference grows relative to the test tile, $\hat p \to 0$ and the ceiling collapses.

When it collapses below the detection threshold $\Upsilon^*$, nothing in that reference can ever
be flagged, and the cleaning pass returns "no clusters" — a result **indistinguishable from a
clean reference**. This has been observed: for M3 sitting in a Boötes III reference tile at
$K_M = 50$, $\Upsilon_{\max}$ and $\Upsilon^*$ were both $5.19$, the reference-side flagged set
was empty, and the most massive contaminant in the field was silently ignored.

The remedy is that the cleaning pass must run at its own $K_M$, chosen as large as the locality
bound permits, rather than at the $K_M$ chosen for the measurement. It must also refuse to
proceed rather than return a null result when even that leaves no headroom, and we require
$\Upsilon_{\max} - \Upsilon^* > 0.5$ before a pass is allowed to run.

### 4.3 The equalisation algorithm

Equalisation cuts localised overdensities out of the reference tiles, expressed entirely in the
statistic's own terms. For each reference independently, and iterating:

1. Recompute $K_M$ from the current cardinalities as $\lfloor 0.05\min(n_X^{(j)}, n_Y)\rfloor$,
   subject to a floor of 25 (below which the usable rank range $20 \le K < K_M$ is empty).
2. Check the headroom $\Upsilon_{\max} - \Upsilon^*$ and raise if it is exhausted (§4.2).
3. Run the full EagleEye pass with roles reversed, so that overdensities *in the reference* are
   what is detected.
4. Gate the resulting reference-side clusters on $S/\sqrt{\hat B} > 5$.
5. Remove, for each gated cluster, its **IDE-pruned** set.

Step 5 is the substantive choice. We remove the pruned set and not the repêchaged set, because
repêchage deliberately re-admits the surrounding region (§1.2) and so contains genuine background
by construction. Removing only the excess mass excises the contaminant while leaving the local
background in place, so the repaired reference still samples the true background at that position.
Removing the repêchaged set instead would punch a hole in the reference.

There is deliberately **no cardinality target**. References of differing size are exactly what the
multi-reference construction is built to handle, and a tile that is legitimately denser should
remain denser. The loop stops when EagleEye no longer finds anything in the reference, not when
the tiles match.

**A worked case.** In the Boötes II field, the reference tiles are not blank sky: Boötes I, a
kinematically confirmed dwarf with 204 members in the Battaglia et al. (2022) catalogue, lies
$1.66^\circ$ north and $0.51^\circ$ east of the grid centre. At $h = 1.0^\circ$ this places it in
cell 7, the north-central **reference** tile — outside $\mathcal Y$, but squarely inside the
background against which $\mathcal Y$ is measured. Only 6 of its 204 members reach the test tile.

Equalisation found it. The results for all eight references:

| Reference cell | $n$ before | $n$ after | cut | passes | $S/\sqrt{\hat B}$ | purity |
|---|---|---|---|---|---|---|
| 0 | 2 079 | 2 079 | 0 | 0 | — | — |
| 1 | 2 159 | 2 159 | 0 | 0 | — | — |
| 2 | 2 152 | 2 152 | 0 | 0 | — | — |
| 3 | 2 123 | 2 123 | 0 | 0 | — | — |
| 5 | 2 043 | 2 043 | 0 | 0 | — | — |
| 6 | 1 974 | 1 974 | 0 | 0 | — | — |
| **7** | **2 102** | **2 005** | **97** | **1** | **26.02** | **0.839** |
| 8 | 1 982 | 1 982 | 0 | 0 | — | — |

Seven references were untouched; the one containing Boötes I was flagged at reference-side
$S/\sqrt{\hat B} = 26.0$ and 97 stars were excised in a single pass, at an estimated purity of
0.84. The repaired tile is drawn in Figure 2, where the excised stars are shown separately.

This is the clearest available demonstration that the failure mode is real, that it arises in the
ordinary course of pointing at a field with known neighbours rather than in a contrived case, and
that the method detects its own contamination without being told where to look.

### 4.4 $K_M$ and the locality bound

$\Upsilon$ assumes the $X{:}Y$ ratio is constant inside the $K$-neighbourhood. It is not — two
adjacent sky tiles have a slowly varying ratio — and the neighbourhood radius grows like
$k^{1/d}$, so the larger $K_M$ is, the more of that variation the statistic swallows and reports
as signal. On a synthetic field with a mild ratio gradient and no injected anomaly, the
false-positive rate as a multiple of the nominal $p_{\rm ext}$ is

| $K_M$ | 25 | 50 | 100 | 300 | 600 |
|---|---|---|---|---|---|
| $d = 2$ | 2.0× | 6.7× | 12.5× | 46.0× | 118× |
| $d = 4$ | 1.3× | 1.8× | 2.0× | 2.2× | 6.8× |

The features here are four-dimensional, where the inflation is far milder — a factor of two out to
$K_M = 300$. This is why setting $K_M$ to the locality ceiling is a reasonable default rather than
a reckless one.

Pushing against that, larger $K_M$ is broadly *better* for sensitivity: the threshold $\Upsilon^*$
grows only logarithmically with $K_M$ while the ceiling $\Upsilon_{\max}$ grows linearly, and a
diffuse anomaly accumulates its excess over many neighbours and may be invisible at small $K_M$
regardless of how many stars it contains. The `"auto"` setting resolves $K_M$ to the largest value
the locality bound permits, and it is resolved *after* equalisation — resolving it before would
let a contaminated tile's inflated cardinality set the bound.

The consequences of the bound are visible in the robustness scan of §7.3, where shrinking the box
by a factor $0.67$ reduces $n_Y$ to 739 and drives $K_M^{\rm auto}$ down to 32, close to the floor
of 25.

---

## 5. Thresholding: the complete chain

This section states every threshold between raw *Gaia* rows and a flagged cluster. The chain is

$$\text{rows} \;\to\; \mathcal Y,\ \mathcal X^{(j)} \;\to\; \Upsilon^*_j \;\to\; \Upsilon_i^{(j)}
\;\to\; \Gamma_i \;\to\; \Gamma^\star \;\to\; A_\Gamma \;\to\; \text{DBSCAN} \;\to\;
\hat B_{\rm w} \;\to\; S/\sqrt{\hat B}.$$

### 5.1 $\Upsilon^*_j$ — the per-reference flagging threshold

Because $\Upsilon_i$ is a maximum over $K$, its null distribution is not the tail of any single
binomial and must be obtained by simulation. For each reference we draw $N = 200\,000$ independent
Bernoulli sequences of length $K_M$ at probability $\hat p_j$, compute $\Upsilon$ for each exactly
as for a real point, and take

$$\Upsilon^*_j = \text{quantile}\left(\{\Upsilon\},\ 1 - p_{\rm ext}\right).$$

One threshold per reference, because $\hat p_j$ differs.

**$p_{\rm ext}$ is a per-point exceedance level.** It is not a cluster-level or field-level error
rate, and it does not describe how often the pipeline produces a spurious *cluster*. This is the
most commonly misread parameter in the pipeline and it is the reason §8 exists: converting a
per-point level into a statement about candidate reliability requires the false-alarm measurements
we have not yet made.

### 5.2 $\Gamma^\star$ — the empirical threshold

The threshold on the combined statistic is obtained by bootstrap. For each of $n_{\rm boot} = 10$
realisations we draw $Y_b$ and $X_b^{(j)}$ uniformly on $[0,1]^4$ **at the realised cardinalities
of the actual run**, push them through the identical multi-reference pipeline, and collect the
resulting $\Gamma_i^{(b)}$. Pooling across realisations and points gives the null sample, and

$$\Gamma^\star = \text{quantile}\left(\Gamma_{\rm null},\ 1 - p_{\rm ext}\right).$$

For the two runs here this gives null samples of $20\,990$ and $30\,370$ values respectively, and

| Run | $q_{0.9}$ | $q_{0.99}$ | $q_{0.999}$ | $q_{0.9999}$ | $\Gamma^\star$ used | $\max_i \Gamma_i$ |
|---|---|---|---|---|---|---|
| Boötes II | 23.32 | 35.06 | 43.79 | 49.28 | **43.79** ($p_{\rm ext}=10^{-3}$) | 76.22 |
| Reticulum II | 26.24 | 38.80 | 49.56 | 57.46 | **52.06** ($p_{\rm ext}=5\times10^{-4}$) | 253.08 |

The empirical route is used rather than the analytic $\mathrm{Gamma}(k_{\rm fit},\theta)$ fit
because the true law is a sum of correlated maxima-of-exponentials whose tail the max-over-$K$
selection reshapes. The fitted form tracks the null well to about the $10^{-2}$ level and then
runs high — by roughly 9 per cent at $p_{\rm ext} = 10^{-3}$ and 15 per cent at $10^{-4}$.
Substituting it would buy better run-to-run stability at the cost of a larger bias, in the
conservative direction, discarding real anomalies.

The tail is where $n_{\rm boot}$ matters. At $n_{\rm boot} = 10$ and $n_Y \approx 2\,000$–$3\,000$
the null sample contains $2$–$3\times10^4$ values, so a quantile at $p_{\rm ext} = 10^{-3}$ is
estimated from roughly 20–30 points and at $5\times10^{-4}$ from roughly 10–15. These are thin,
and $\Gamma^\star$ carries a corresponding sampling uncertainty that we have not propagated.
Running at smaller $p_{\rm ext}$ without increasing $n_{\rm boot}$ is not meaningful.

### 5.3 $A_\Gamma$ and clustering

Points exceeding the threshold form the anomalous set $A_\Gamma = \{i : \Gamma_i > \Gamma^\star\}$,
which is then partitioned by DBSCAN in the full four-dimensional scaled feature space with
$\epsilon = 1.0$ and min\_samples $= 5$. Clustering in the same space in which the statistic was
computed — rather than on the sky alone — means a group must be coherent in position *and* proper
motion to survive.

Two properties of this step should be kept in mind. Because $\epsilon$ is in MAD units, its
physical meaning is fixed only if the metric is frozen (§3.2). And DBSCAN's noise label is
retained rather than suppressed: points above threshold that join no cluster are reported as
unclustered and excluded from the cluster statistics. For Boötes II, 20 points exceeded
$\Gamma^\star$ and 18 formed a single cluster, leaving 2 unclustered; for Reticulum II, 111 points
exceeded threshold and all 111 were assigned to two clusters with no unclustered remainder.

### 5.4 $\hat B$ and $S/\sqrt{\hat B}$ per cluster

Each $\Gamma$ cluster needs a background estimate, and the estimate available is per-reference: the
repêchage stage of each of the eight comparisons produces its own clusters, each with its own
$\hat B$ from the injection estimator of §1.3. We combine them by overlap weighting. For a
$\Gamma$ cluster $\alpha$, let $o^{(j)}_\alpha$ be the number of stars it shares with the
overlapping repêchage cluster in reference $j$. Then

$$\hat B_\alpha = \frac{\sum_j o^{(j)}_\alpha \hat B^{(j)}_\alpha}{\sum_j o^{(j)}_\alpha},
\qquad S = |A_\Gamma^{(\alpha)}|, \qquad \text{reported as } S/\sqrt{\hat B_\alpha}.$$

Only references contributing a finite, positive $\hat B$ and a finite, positive significance are
included, so the number of contributing references is itself diagnostic: a cluster supported by
all eight is a different object from one supported by two. In §6 the Reticulum II detection draws
on all 8 references while the unidentified cluster draws on 2, and the Boötes II detection on 5.

**What this number is, and is not.** It is a local estimate of excess over background, and it is a
useful *ranking* statistic. It is not a calibrated significance. It takes no account of how many
independent places in the tile could have produced a comparable clump by chance, so it cannot be
read as a trials-corrected detection significance, and it should not be quoted as a number of
sigma. Converting it into such a statement is what §8 is for, and that work is not done.

---

## 6. Recovery of known systems

We show the pipeline recovering two kinematically confirmed dwarf galaxies from the
Drlica-Wagner et al. (2020) catalogue, using membership from Battaglia et al. (2022) at
$P_{\rm memb} > 0.5$ as ground truth. Members are matched to our *Gaia* DR3 rows by `source_id`,
so the comparison involves no positional tolerance.

For context, an earlier batch application of this pipeline to all 30 class-4 dwarfs with usable
fields recovered 21 at a true-positive rate above 0.5 and 9 not at all, with no intermediate
outcomes. That batch used a different configuration from anything in this report (wider tiles,
$K_M = 300$, $p_{\rm ext} = 10^{-5}$, no equalisation) and its per-system numbers are therefore
not comparable with those below; Boötes II in particular is re-run here under the configuration of
§2.4 and the result differs. We note the aggregate only as an indication of scale and do not draw
on the batch further.

### 6.1 Reticulum II

Reticulum II has 75 Battaglia members, all of which fall inside the test tile. The run returns
**two** $\Gamma$ clusters from 111 points above threshold, with no unclustered remainder.

| | cluster 1 | cluster 0 |
|---|---|---|
| $n$ | 89 | 22 |
| $\hat B$ (overlap-weighted) | 19.96 | 5.34 |
| $S/\sqrt{\hat B}$ | 19.92 | 9.52 |
| Contributing references | 8 of 8 | 2 of 8 |
| Best catalogue match | **Reticulum II** | *unidentified* |
| Members recovered | 62 of 75 | 0 |
| TPR | **0.827** | — |
| Purity against known members | 0.697 | 0.000 |
| Centroid $(\alpha,\delta)$ | $(53.908, -54.036)$ | $(52.884, -55.027)$ |
| Separation from catalogue position | $0.017^\circ$ | $1.142^\circ$ |
| Mean proper motion [mas yr$^{-1}$] | $(+2.42, -1.44)$ | $(-0.40, -0.01)$ |
| PM dispersion [mas yr$^{-1}$] | $(0.38, 0.44)$ | $(0.38, 0.35)$ |
| Median $G$ | 19.68 | 20.29 |

Cluster 1 is Reticulum II. It recovers 62 of 75 known members, its centroid lies $1.0'$ from the
catalogue position, and its mean proper motion matches the published value. It is supported by all
eight references, which is what a genuine, strong overdensity should look like: no choice of
background makes it go away. Of its 89 members, 27 are not in the Battaglia list — a purity of
0.70 — which is the expected behaviour of the repêchage stage, which admits the surrounding
region by construction (§1.2), and is not by itself evidence of contamination.

Cluster 0 matches nothing in the catalogue and is the subject of §7.

**Figure 1** — `figures/fig1_reticulumII_sky.png` (proper-motion counterpart:
`figures/fig1b_reticulumII_pm.png`). The test tile, with the 111 points above
$\Gamma^\star$ marked and the two clusters distinguished. Reticulum II's Battaglia members are
overplotted. Cluster 1 sits on the target at the tile centre; cluster 0 occupies the south-western
corner.

### 6.2 Boötes II

Boötes II has 22 Battaglia members, all inside the test tile. The run returns a single $\Gamma$
cluster from 20 points above threshold, with 2 unclustered.

| | value |
|---|---|
| $n$ | 18 |
| $\hat B$ (overlap-weighted) | 3.93 |
| $S/\sqrt{\hat B}$ | 9.08 |
| Contributing references | 5 of 8 |
| Best catalogue match | **Boötes II** |
| Members recovered | 14 of 22 |
| TPR | **0.636** |
| Purity against known members | 0.778 |
| Centroid $(\alpha,\delta)$ | $(209.537, +12.861)$ |
| Separation from catalogue position | $0.023^\circ$ |
| Mean proper motion [mas yr$^{-1}$] | $(-2.54, -0.36)$ |

Boötes II is a considerably harder target than Reticulum II — 22 catalogued members against 75 —
and the recovery is correspondingly less complete: 14 of 22 members, at $S/\sqrt{\hat B} = 9.08$,
supported by 5 of the 8 references rather than all 8. The centroid is $1.4'$ from the catalogue
position. Of the 18 cluster members, 14 are known members, so the cluster is 78 per cent pure.

The more instructive aspect of this field is what is happening in the *references*. Boötes I lies
$1.66^\circ$ north of the grid centre, placing it inside the north-central reference tile (§4.3),
with 204 catalogued members of which only 6 reach the test tile. Equalisation flagged that tile
at reference-side $S/\sqrt{\hat B} = 26.0$ and removed 97 stars; the other seven references were
untouched. Without that repair, Boötes I's 204 members would have inflated $n_X$ for one of the
eight comparisons, depressing $\hat p$ and adding a pedestal to $\Upsilon^{(j)}$ for all 2 099
test stars (§4.1).

This is the case the multi-reference construction has to handle in order to be usable: any field
chosen because it contains one known satellite is reasonably likely to contain another within a
few degrees.

**Figure 2** — `figures/fig2_bootesII_sky.png` (colour–magnitude counterpart:
`figures/fig2b_bootesII_cmd.png`). The test tile with the recovered cluster, the
Battaglia members of Boötes II (targets) and of Boötes I (interlopers, 6 inside the tile and 198
outside), and the repaired reference tile drawn at the top with its 97 excised stars marked.

---

## 7. An unidentified overdensity in the Reticulum II field

Cluster 0 of §6.1 matches no catalogued system. This section characterises it. We present it as a
**candidate requiring follow-up, not as a detection**, and the evidence below is mixed.

### 7.1 Properties

The cluster contains 22 stars at $(\alpha, \delta) = (52.884, -55.027)$, $1.14^\circ$ from
Reticulum II. It is compact and round — its members span 0.26 and 0.28 of the tile in RA and
declination respectively, an aspect ratio of 1.1:1, which is essentially identical to Reticulum
II's own footprint in the same run (0.27 × 0.29, 1.1:1).

Its proper motion is tightly concentrated at $(-0.40, -0.01)\,\mathrm{mas\,yr^{-1}}$ with
dispersions $(0.38, 0.35)$, again comparable to Reticulum II's $(0.38, 0.44)$. Its members are
about 0.6 mag fainter in median $G$ (20.29 against 19.68).

**It is not associated with Reticulum II.** The two proper motions differ by
$2.9\,\mathrm{mas\,yr^{-1}}$, an order of magnitude larger than either dispersion. At Reticulum
II's distance of roughly 30 kpc that corresponds to a relative transverse velocity of several
hundred km s$^{-1}$, far in excess of anything bound to it.

An overdensity at a consistent position has been seen in this field at other configurations, with
somewhat different membership — 27 stars at $(52.881, -54.936)$ in an earlier run. The right
ascensions agree to $0.003^\circ$ and the declinations differ by $0.09^\circ$. We take these to be
the same feature seen under different tilings, but we have not established that by matching
`source_id` between runs, and it should be checked.

### 7.2 Where the evidence is weak

Two things should be stated plainly before the robustness results.

**It is only supported by 2 of the 8 references**, against 8 of 8 for Reticulum II in the same
run. The overlap-weighted background is therefore built from two repêchage clusters rather than
eight, and $\hat B = 5.34$ is correspondingly less well determined than Reticulum II's 19.96.

**It touches the tile boundary.** Its members come within $0.011h$ of the western edge and
$0.001h$ of the southern edge — that is, they reach the corner of the test tile. This matters in
two ways. Its true extent may be larger than measured, because any members beyond the boundary
are not in $\mathcal Y$ at all; and worse, those members would be sitting in the adjacent
reference tiles, where they would act to suppress the very feature we are trying to measure.
The 22 stars should be read as a lower bound on the membership.

It is *not*, however, the classic tiling artefact. That signature is a thin strip lying flush
along an edge and spanning the full width of the tile — high aspect ratio, large span, centroid
pushed to the boundary. This cluster has an aspect ratio of 1.1 and spans about a quarter of the
tile in each direction. It is a compact clump that happens to lie near a corner, which is a
different thing, and the standard geometric test does not flag it.

### 7.3 The robustness scan

To separate a feature locked to the *sky* from one locked to the *tiling*, the pipeline is re-run
on eight grids anchored on the anomaly itself: the nominal grid, four shifts pushing the anomaly
toward each tile edge, two changes of box size, and one diagonal shift. A real object stays at a
fixed sky position while the grid moves under it; an artefact follows the grid.

The reference set $A_0$ for this scan contains 28 stars.

| Configuration | $h$ | $n_Y$ | $K_M$ | $\Gamma^\star$ | clusters | recovered | overlap | recovery | Jaccard | $S/\sqrt{\hat B}$ |
|---|---|---|---|---|---|---|---|---|---|---|
| nominal | 1.000 | 1 627 | 75 | 39.75 | 3 | yes | 27 | 0.96 | 0.93 | 8.63 |
| shiftE | 1.000 | 1 659 | 75 | 40.43 | 2 | yes | 22 | 0.79 | 0.79 | 6.76 |
| shiftW | 1.000 | 1 585 | 74 | 40.53 | 1 | yes | 26 | 0.93 | 0.84 | 6.53 |
| shiftN | 1.000 | 1 577 | 74 | 40.65 | 2 | yes | 24 | 0.86 | 0.80 | 6.63 |
| shiftS | 1.000 | 1 627 | 75 | 39.94 | 2 | yes | 27 | 0.96 | 0.79 | 7.78 |
| **scale ×0.67** | 0.667 | **739** | **32** | 35.22 | **0** | **no** | 0 | 0.00 | 0.00 | — |
| scale ×1.50 | 1.500 | 3 548 | 169 | 45.09 | 3 | yes | 23 | 0.82 | 0.64 | 6.98 |
| diagonal | 1.000 | 1 596 | 74 | 39.01 | 2 | yes | 27 | 0.96 | 0.84 | 9.00 |

**The formal verdict is ROBUST.** Six of the seven perturbed configurations recover the feature
(0.86, against a requirement of 0.75); the median Jaccard index among recoveries is 0.80, against
a requirement of 0.5; and the significance ranges over 6.53–9.00, a ratio of 1.38, against a
permitted ratio of 3.0. The feature is locked to the sky, not to the tiling.

The single failure is the smallest box, and it is explained by the method rather than being
anomalous. Shrinking the tile to $h = 0.667^\circ$ reduces $n_Y$ to 739, which through the
locality bound of §4.4 drives $K_M^{\rm auto}$ down to 32 — close to the floor of 25 below which
no test exists at all. The attainable $\Upsilon_{\max}$ falls with it, and a feature of this
strength can no longer clear the threshold: 18 per cent of $A_0$ remained above $\Gamma^\star$ but
no cluster formed. This is the documented behaviour of a shrinking box, not evidence against the
feature. It does mean that a single significance threshold applied across all eight configurations
would be biased against the small ones.

**Figure 3** — `figures/fig3_scan_overlay.png`. Left: the sky, with the recovered
anomaly's 50 per cent contour drawn for each configuration and each configuration's test-tile
boundary shown dashed. The contours coincide while the boundaries move, which is the signature of
a sky-locked feature. Right: the same in proper-motion space, where Reticulum II is visible as the
field overdensity near $(+2.4, -1.4)$ and the anomaly contours sit separately near $(-0.4, 0.0)$.
The $\times 0.67$ configuration is absent from both panels because it returned no cluster.

### 7.4 What has been ruled out, and what has not

Two interpretations have been assessed and set aside. A stellar-wake origin — an overdensity of
halo field stars trailing a dark perturber — is not supported: the shear from any plausible
perturber at this distance is orders of magnitude below what is measurable with these data, and
the proper motions are in any case incompatible with a common origin with Reticulum II. A
boundedness argument is also unavailable: the test has no discriminating power at 30 kpc, as
demonstrated by running it on Reticulum II itself, a known bound system, where it fails
identically. Both are reported so they are not attempted again.

What has **not** been established is the alternative that matters: that 22 stars this tightly
grouped in proper motion could not arise by chance within a field whose proper-motion distribution
is itself strongly clustered. Answering that requires the calibration of §8, and until it exists
the candidate stands on the robustness scan alone — which shows the feature is real in the data
and not an artefact of the tiling, but not that it is a physical system.

---

## 8. Calibration — **PARTIAL, FOR LATER COMPLETION**

> This section is deliberately incomplete. It records the machinery that exists and the
> measurements that have not been made. No threshold recommendation should be taken from it.

### 8.1 Nulls implemented

Four nulls exist, in increasing order of realism. Each answers a different question and none
answers all of them.

**Proper-motion scramble.** Permute the $(\mu_{\alpha*}, \mu_\delta)$ pairs among stars in the
window. Positions are untouched, so the spatial density field and all its gradients survive; the
joint PM distribution survives because the pairs are only relabelled; what is destroyed is the
position–PM coherence that a genuine comoving group has. Applied to the Reticulum II-field
candidate this gave an observed $S/\sqrt{\hat B} = 7.05$ against a maximum of 3.54 over 12 blank
trials, $p < 0.08$. Twelve trials is too few to say more. This null can only answer "could the
pipeline have manufactured this from the marginals alone?" — it cannot distinguish a genuine
chance density from a physical system.

**Off-position.** Run the identical pipeline at positions elsewhere on the sky and read the
empirical distribution of $S/\sqrt{\hat B}$ off blank sky. This manipulates nothing and is the
classic off-source measurement. Two hardening steps were necessary after a first pass was found
unsound: the exclusion radius around known objects must be the full $3h$ half-width of the grid
rather than the test tile, because an object just outside $\mathcal Y$ sits in a *reference* and
does exactly the damage described in §4.1; and the DW20 catalogue contains no globular clusters,
so those must be excluded using a separate catalogue (Vasiliev & Baumgardt 2021). Off-positions are also matched in stellar
density to the target field, since surface density varies strongly with Galactic latitude.

**Smooth mock.** Resample the field with replacement and jitter by a kernel wider than any clump
but narrower than the field's real structure. Below the bandwidth the generating density is smooth
by construction, so anything recovered is manufactured by the method, while the large-scale
gradient and tile-to-tile density differences survive. This isolates method artefacts — tile-edge
effects, the rectification described in §3.1, DBSCAN scale coupling — without appealing to what is
or is not in the sky.

**Scan-level persistence.** The one that matters for flagging policy. A marginal cluster in a
single configuration means little; the same cluster reappearing at the same sky position across
all eight configurations of §7.3 is a different claim, and carries power the single-run
significance does not express. Measuring how often blank fields produce such a persistent group
gives the number to flag on. The machinery is written and parallelised. **No number has been
produced.**

### 8.2 Stacking against pooling

There is an alternative use of eight references: concatenate them into a single pooled reference
and run the standard single-comparison pipeline once. This is a genuinely different statistic,
with $\hat p \approx 1/9$ rather than $\approx 1/2$, and it is worth recording as a limitation of
stacking that the two are not uniformly ordered.

On synthetic fields built with the real cardinalities of a *Gaia* tile and deliberately lumpy,
unequal-sized references, recovery rates were:

| Anomaly | Stacked | Pooled |
|---|---|---|
| Compact ($w = 0.035$, $N = 10$) | 0.25 | **0.95** |
| Diffuse ($w = 0.085$, $N = 20$) | **0.43** | 0.23 |

Compact anomalies favour pooling and diffuse anomalies favour stacking. The mechanism is
understandable: stacking is a persistence measure and suppresses variance on a signal present in
every comparison, which is what a diffuse excess looks like; but a $\hat p$ pedestal arising from
reference lumpiness is additive and undiluted under stacking, whereas pooling averages it away.

**These numbers rest on two trials per grid cell and should not be relied on.** The direction of
the effect is suggestive and the mechanism is plausible, but the statistics are thin. We record
it here as a caveat attached to the stacked approach used throughout this report, and note that
the candidate of §7 — compact, 22 stars — sits in the regime where stacking performed worse in
this test. Running the pooled comparison on the real field is a cheap cross-check and has not
been done.

### 8.3 What is missing

- The scan-level persistence false-alarm rate. Staged and parallelised; **no number**.
- The off-position null re-run under the hardened exclusion logic.
- The scramble null beyond 12 trials.
- Whether the stacking/pooling crossover survives more than two trials per cell.
- A pooled cross-check on the Reticulum II field.

**Consequence.** Until these exist there is no defensible detection threshold, and
$S/\sqrt{\hat B}$ should be used to rank candidates, not to decide them. The candidate of §7 is
reported on that basis.

---

## 9. Open items

**Code**

- The per-reference $\Upsilon^{(j)}$ arrays are not retained in the saved run output, so
  $R_{\rm eff}$ cannot be computed after the fact (§3.4). Retaining `Upsilon_by_ref` is a one-line
  change and should be made before the next production runs.
- The injection estimator divides by zero rather than returning NaN when the injected count is
  zero.
- One further item concerning that estimator is pending discussion with the EagleEye authors
  (§1.3, Appendix C).
- The scan-overlay plotting has unguarded RA-wrap exposure.
- The null-calibration routine is unseeded.
- **Nothing is committed.** The EagleEye working copy carries uncommitted changes to
  `EagleEye.py`, `utils_EE.py`, `utils_sistematics.py` and `test_multiBkg_ordered.py`. This should
  be resolved before anything else touches those files, since the runs in this report depend on
  that working state and it is not currently reproducible from version control.

**Analysis**

- Confirm by `source_id` that the 22-star cluster of §7 and the 27-star cluster seen at other
  configurations are the same feature.
- Re-run Reticulum II with equalisation enabled, for consistency with §6.2 even though no
  contaminant is expected (§2.4).
- Run the pooled cross-check on the Reticulum II field (§8.2).
- Establish what distinguishes the fields in which the method recovers a known dwarf from those in
  which it recovers nothing. The batch outcome was strikingly bimodal and the cause is not known.

---

## Appendix A — Hyperparameters

Consolidated from §2.4, with justification.

**$h$, tile half-width.** Sets $n_Y$, and through the locality bound sets the achievable $K_M$ and
hence the achievable significance (§4.4, §7.3). Smaller tiles are not simply "more local": they
are also less sensitive.

**$K_M$.** Bounded above by $0.05\min(n_X, n_Y)$ for locality and below by 25, since the usable
rank range is $20 \le K < K_M$. Larger is broadly better for sensitivity because $\Upsilon^*$
grows logarithmically while $\Upsilon_{\max}$ grows linearly; the cost is gradient leakage, which
in four dimensions is mild (§4.4). Must be shared across all eight references (§3.3), and resolved
after equalisation.

**$p_{\rm ext}$.** A per-point exceedance level, not a cluster or field error rate (§5.1). Smaller
tiles need a looser value: they carry a smaller multiple-comparison burden, so the same physical
excess sits at a less extreme quantile, and simultaneously a lower $\Upsilon_{\max}$ ceiling. A
diffuse overdensity is the case that most needs the looser threshold, since it produces many
mildly elevated points rather than a few extreme ones.

**$n_{\rm boot}$.** Controls how well the tail of the $\Gamma$ null is resolved. At
$n_{\rm boot} = 10$ the quantile at $p_{\rm ext} = 10^{-3}$ rests on 20–30 null values (§5.2).

**DBSCAN $\epsilon$, min\_samples.** In scaled units, so their physical meaning depends on the
metric being frozen (§3.2).

**DPA $Z$.** Cluster-splitting parameter for the repêchage stage, left at the EagleEye default.

## Appendix B — Parallelisation and determinism

Three levels of parallelism exist and their product must stay within the core count:
configurations × references × inner kNN threads.

Only one backend works. Forking breaks OpenBLAS, because numpy initialises its thread pool in the
parent and the child inherits descriptors for threads that do not exist. Spawning re-imports
`__main__`, which is unusable from a notebook. Loky with the inner thread limit pinned to one
works, and is what is used. Note also that joblib disables nested parallelism, so requesting eight
configurations *and* eight references does not give sixty-four workers — the inner level silently
degrades.

Determinism required explicit care. Reseeding each bootstrap worker independently is not
equivalent to the serial computation: it moved $\Gamma^\star$ from 52.38 to 61.66 in one test and
changed the cluster count from two to one. Advancing a single PCG64 stream to the offset each
realisation would have reached serially reproduces the serial result exactly. Measured speedups
are 1.67× at $n_{\rm boot} = 4$ and 3.47× at $n_{\rm boot} = 20$, with
$\max_i |\Delta\Gamma_i| = 1.4\times10^{-14}$ against serial.

## Appendix C — Reserved

Reserved for the estimator implementation item of §1.3, pending discussion with the EagleEye
authors.


---

## Figure manifest

All figures are symlinked into `report/figures/` from the run directories under
`field_diagnostics/`, so the report is self-contained while the underlying products stay with
their runs.

| File | Section | Source run |
|---|---|---|
| `fig1_reticulumII_sky.png` | §6.1 | Reticulum II, h=1.3333, K_M=100, p_ext=5e-4 |
| `fig1b_reticulumII_pm.png` | §6.1 | as above |
| `fig2_bootesII_sky.png` | §6.2, §4.3 | Boötes II, h=1.0, K_M=50, p_ext=1e-3, equalisation on |
| `fig2b_bootesII_cmd.png` | §6.2 | as above |
| `fig3_scan_overlay.png` | §7.3 | Reticulum II scan, gid 0, p_ext=5e-3, K_M auto |
| `fig4_reticulumII_gamma_null.png` | §5.2 | Reticulum II, as fig 1 |

Figure 4 (the empirical Γ null against the analytic Fisher curves) is available and referenced by
§5.2 but is not yet discussed in the text; adding that discussion is a small outstanding item.

---

## References

Battaglia G. et al., 2022, A&A, 657, A54
Darragh-Ford E., Nadler E. O., McLaughlin S., Wechsler R. H., 2021, ApJ, 915, 48
Drlica-Wagner A. et al., 2020, ApJ, 893, 47
d'Errico M., Facco E., Laio A., Rodriguez A., 2021, Information Sciences, 560, 476
Ester M., Kriegel H.-P., Sander J., Xu X., 1996, Proc. KDD-96, 226
Fisher R. A., 1932, Statistical Methods for Research Workers, 4th edn. Oliver & Boyd, Edinburgh
Gaia Collaboration, 2023, A&A, 674, A1
Glielmo A. et al., 2022, Patterns, 3, 100589
Guth A. H., Namjoo M. H., 2026, arXiv:2602.10178
Overdeck K. et al., 2026, arXiv:2606.09975
Rusterucci S., Hammer F., Yang Y., 2026, A&A, accepted (arXiv:2608.21517)
Springer S., Scaffidi A., Autenrieth M., Contardo G., Laio A., Trotta R., Haario H., 2025,
  Scientific Reports (arXiv:2503.23927)
Tan C. Y. et al., 2026, ApJ, 1000, 87
Vasiliev E., Baumgardt H., 2021, MNRAS, 505, 5978

"""Score baseline ATT estimators against the IV gold standard from cexp.py.

The benchmark's `late` column is a Wald ratio `itt_y / itt_d` with the instrument
"CardA was offered". A card cannot be picked when it is absent from the pack, so
noncompliance is one-sided: the compliers are exactly the treated, and LATE
identifies the ATT. That makes it a *gold-standard estimator* rather than ground
truth, and `se_late` is the width of its own uncertainty -- which is why several
of the metrics below weight by it.

Two traps the caller must clear first:

  * **Scale.** `truth_divisor` rescales the gold standard onto whatever scale
    the estimate columns use. It **defaults to 1**, which is right for
    `ivexp` + `baselines`: both report absolute pack-3 counts. It is *not* right
    for `cexp` + `baseline_methods`, where the baselines are rates
    (`count / horizon`) and `late` is a raw count -- there, pass `H_ref`, the
    constant p2p1 horizon, or every squared-error metric is wrong by that factor
    (rank metrics are invariant to it).
  * **Duplicate pairs.** `ivexp.get_single_pack_df` already emits one row per
    `(card_a, group_b)`. `cexp.get_single_pack_df` does not -- it emits `pack`
    and `use_sure_pick` columns, so a frame from it usually holds several rows
    per pair and must be filtered to `pack == 1` (0-indexed pack 2, matching
    p2p1) and one `use_sure_pick` first. `evaluate_methods` warns whenever the
    key is not unique, whichever produced it.

The four metrics
----------------

    rmse, mae, ccc    ignore se
    accuracy95        uses se

`rmse` is root mean squared error and so is dominated by the worst pairs; `mae`
is the robust companion that is not. Read them together: a large gap between
them says a few pairs carry the score.

`ccc` is Lin's concordance correlation coefficient, agreement with the
45-degree line rather than with the best-fitting one:

    ccc = 2*cov(est, truth) / (var(est) + var(truth) + (mean(est) - mean(truth))^2)
        = pearson * bias_factor

The second factor is 1 only when the scales and the locations match, so `ccc`
charges for exactly what a correlation forgives -- a standard-deviation ratio
away from 1, and a non-zero bias. **It is the metric that catches the two
failure modes `rmse` cannot.** An overdispersed method (estimate spread about
twice the truth's) and a shrunk one (regularised learners bias effects toward
zero, and a method predicting ~0 scores a good `rmse` while carrying no
information) both look fine on squared error alone. `est = s*truth` gives
`ccc = 2s/(1 + s^2)` against a correlation of 1 -- 0.8 at `s = 0.5` and at
`s = 2` alike -- so the shortfall from 1 *is* the miscalibration charge. It is
bounded to [-1, 1], which is why it survives methods that have gone badly off
scale.

`accuracy95` is the share of pairs with `|est - late| <= 1.96 * se_late`. It is
the only one that reads the gold standard's own uncertainty, and it is high
whenever the intervals are wide -- `late` is a ratio with a weak-instrument
tail, so check what `null_zero` scores before treating a high value as success.

Every metric carries its own denominator: `accuracy95` needs a finite positive
standard error and `ccc` needs two usable pairs, so `<metric>_n` differs between
them and reporting one shared count would misstate at least one.

Deliberately absent
-------------------
Rank correlations, `r2`, `mean_z2`, `bias`, `expected_rmse`,
`precision_weighted_rmse`, `noise_floor`, `mse_corrected`, `sd_ratio` and
`calibration_slope` were all removed by decision. Two of those removals give up
something real and are worth recording:

  * **`spearman_within`.** A pooled rank correlation can reward tracking group-B
    prevalence rather than causal effects, because across `(card_a, group_b)`
    pairs most outcome-scale variation is how often group B is picked at all.
    Ranking inside each `card_a` was the control for that. Nothing here replaces
    it. `baselines.ControlMean` was that reference point -- it applies no
    treatment contrast at all -- but it was removed with the rest of the
    unscored registry, so reinstating the check means reinstating that column
    first.
  * **`sd_ratio` / `calibration_slope`.** These said *which* miscalibration a
    method had and what rescaling would fix it. `ccc` still detects the
    miscalibration -- that is the whole of its second factor -- but no longer
    reports its direction, so an overdispersed method and a shrunk one at the
    same `s` and `1/s` are indistinguishable by score alone.

Every metric also carries a bootstrap `<metric>_sci` half-width unless
`confidence=False` turns the resampling off. Read the caveats in
`evaluate_methods`: `est +/- sci` is not literally the interval, and
overlapping intervals are not a test of no difference. `evaluate_methods`
resamples whole `card_a` clusters; `compare_baselines` and
`evaluate_methods_ext` still resample rows and so run narrow.
"""

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Normal quantile for a two-sided 95% interval.
Z_95 = 1.959963984540054

# each subset separately: LD -> ld_rmse, GSP -> gsp_rmse, and so on.
MASK_PREFIXES: Tuple[Tuple[str, str, str], ...] = (
    ('LD', 'ld', 'Local difference'),
    ('GSP', 'gsp', 'Group-stratified precision'),
)

# The four scores, everywhere. They cover distinct failure modes: `rmse` is
# squared error and so is dominated by the worst pairs, `mae` is the robust
# companion that is not, `ccc` charges for scale and location disagreement on
# top of correlation, and `accuracy95` reads the gold standard's own
# uncertainty. Rank correlations, `r2`, `mean_z2` and the shrinkage diagnostics
# were dropped by decision -- `ccc` already carries the overdispersion charge
# that `sd_ratio` and `calibration_slope` used to expose separately.
REDUCED_METRICS = ('rmse', 'mae', 'ccc', 'accuracy95')

# Output column order: identity, then the scores, then the two denominators.
METRIC_COLUMNS = (
    'method',
    'n_used',
) + REDUCED_METRICS + (
    'n_used_se',
)

# Metrics that get a bootstrap interval. The count columns are excluded: an
# interval on "how many pairs were usable" is not a quantity anyone reads.
BOOTSTRAP_METRICS = REDUCED_METRICS

PAIR_KEY = ('card_a', 'group_b')

# Columns of one estimator's frame, as `BaselineEnricher.get_estimates` emits it.
ESTIMATE_COLUMNS = PAIR_KEY + ('value',)

_DUPLICATE_PAIRS_WARNING = (
    '%d duplicate (card_a, group_b) rows, so those pairs are counted more than'
    ' once. From cexp, filter to one pack and one use_sure_pick first; ivexp'
    ' emits one row per pair already, so a duplicate there is a merge fault'
)

_COMPARE_COMMON_SUPPORT = (
    '%d of %d pairs dropped so the comparison is paired, leaving %d scored on'
    ' both %r and %r. A large drop means the two are compared on much less than'
    ' the frame suggests'
)

_COMPARE_DEGENERATE = (
    '%d of %d replicates could not score %r on one side or the other and were'
    ' dropped from its interval'
)




def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Column as a float array, with anything non-numeric coerced to nan."""
    return pd.to_numeric(frame[column], errors='coerce').to_numpy(dtype=float)


def _score_reduced(
    estimate: np.ndarray, truth: np.ndarray, truth_se: np.ndarray
) -> Dict[str, Any]:
    """`REDUCED_METRICS`, each with its own valid-observation count.

    The single definition of the four scores, so `evaluate_methods` and
    `evaluate_methods_ext` cannot drift apart on what they mean.

    Together they cover different failure modes: `rmse` is squared error and so
    is dominated by the worst pairs, `mae` is the robust companion that is not,
    `ccc` charges for scale and location disagreement on top of correlation, and
    `accuracy95` reads the gold standard's own uncertainty.

    **Each metric carries its own denominator** as `<metric>_n`, because they
    genuinely differ: `accuracy95` needs a finite positive standard error, `ccc`
    needs two usable pairs, and all of them drop non-finite estimates. Reporting
    one shared count would misstate at least one of them.
    """
    nan = float('nan')
    record: Dict[str, Any] = {name: nan for name in REDUCED_METRICS}
    record.update({f'{name}_n': 0 for name in REDUCED_METRICS})
    record['n_used'] = 0
    record['n_used_se'] = 0

    usable = np.isfinite(estimate) & np.isfinite(truth)
    n_used = int(usable.sum())
    record['n_used'] = n_used
    if n_used == 0:
        return record

    est = estimate[usable]
    tru = truth[usable]
    err = est - tru

    record['rmse'] = float(np.sqrt(np.mean(err ** 2)))
    record['rmse_n'] = n_used
    record['mae'] = float(np.mean(np.abs(err)))
    record['mae_n'] = n_used

    # Lin's concordance correlation: agreement with the 45-degree line rather
    # than with the best-fitting line. It factors as `pearson * bias_factor`,
    # where the second term penalises exactly what `pearson` ignores -- a scale
    # mismatch (`sd_ratio` away from 1) and a location shift (`bias`). That is
    # the pair of failures this benchmark keeps hitting: methods that correlate
    # well with the gold standard while being overdispersed. It is bounded to
    # [-1, 1], so it stays readable when a method is badly miscalibrated -- and
    # it is why the separate `sd_ratio` / `calibration_slope` diagnostics are
    # gone: a proportional shrink shows up here as `pearson * bias_factor` with
    # the second factor below 1.
    #
    # Computed from centred products rather than from raw second moments, which
    # is the numerically stable form: the raw version differences two large
    # nearly-equal numbers when the mean dominates the variance, and these
    # outcomes are small rates around a non-zero prevalence.
    #
    # No spread guard is needed: the denominator carries the truth's variance
    # too, so it survives a constant estimate and correctly returns 0
    # -- which is why the null baselines get a meaningful `ccc` whether or not
    # they are jittered.
    if n_used >= 2:
        est_centred = est - est.mean()
        truth_centred = tru - tru.mean()
        gap = float(est.mean() - tru.mean())
        denominator = (
            float(np.mean(est_centred ** 2))
            + float(np.mean(truth_centred ** 2))
            + gap ** 2
        )
        if denominator > 0:
            covariance = float(np.mean(est_centred * truth_centred))
            record['ccc'] = 2.0 * covariance / denominator
            record['ccc_n'] = n_used

    spread = truth_se[usable]
    has_se = np.isfinite(spread) & (spread > 0)
    n_used_se = int(has_se.sum())
    record['n_used_se'] = n_used_se
    if n_used_se == 0:
        return record
    err_se = err[has_se]
    sigma = spread[has_se]
    record['accuracy95'] = float(np.mean(np.abs(err_se) <= Z_95 * sigma))
    record['accuracy95_n'] = n_used_se
    return record


def _score_method(
    name: str,
    estimate: np.ndarray,
    truth: np.ndarray,
    truth_se: np.ndarray,
) -> Dict[str, Any]:
    """All metrics for one method column.

    A thin wrapper over `_score_reduced` now that the four scores *are* the
    metric table: it adds the method's name and drops the per-metric `<name>_n`
    counts, which `evaluate_methods_ext` reports per subset but the full table
    summarises with `n_used` and `n_used_se`.
    """
    record: Dict[str, Any] = {column: float('nan') for column in METRIC_COLUMNS}
    record.update({
        key: value for key, value in
        _score_reduced(estimate, truth, truth_se).items()
        if key in METRIC_COLUMNS
    })
    record['method'] = name
    return record


def _percentile_interval(
    replicates: np.ndarray, ci: float
) -> Tuple[float, float, int]:
    """Percentile interval over the finite replicates, with how many there were.

    Degenerate replicates are expected rather than exceptional: a resample can
    leave every usable standard error out (accuracy95 -> nan) or shrink a subset
    below the two pairs `ccc` needs. They are dropped instead of poisoning the
    percentile.
    """
    finite = replicates[np.isfinite(replicates)]
    if finite.size < 2:
        return float('nan'), float('nan'), int(finite.size)
    tail = 100.0 * (1.0 - ci) / 2.0
    low, high = np.percentile(finite, [tail, 100.0 - tail])
    return float(low), float(high), int(finite.size)


def _cluster_indices(
    rng: np.random.Generator, clusters: np.ndarray, n_boot: int
) -> List[np.ndarray]:
    """Row indices for `n_boot` cluster resamples: K out of K clusters, all rows.

    Replicates are ragged, since resampled clusters differ in size.
    """
    _, codes = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(codes == k) for k in range(codes.max() + 1)]
    n_clusters = len(members)
    draws = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    return [np.concatenate([members[k] for k in take]) for take in draws]


def _bootstrap_method(
    estimate: np.ndarray,
    truth: np.ndarray,
    truth_se: np.ndarray,
    indices: Iterable[np.ndarray],
    ci: float,
    bounds: bool,
    metric_names: Sequence[str] = BOOTSTRAP_METRICS,
    prefix: str = '',
) -> Dict[str, Any]:
    """Interval columns for one method, over a fixed set of resample indices.

    `indices` is shared across every method, so the widths are mutually
    comparable and a paired method-vs-method comparison stays a small addition.
    """
    rows = []
    for take in indices:
        # Score once per replicate and read all the metrics off that one
        # record -- scoring per metric would repeat the whole computation
        # len(metric_names) times over.
        scored = _score_reduced(estimate[take], truth[take], truth_se[take])
        rows.append([scored[metric] for metric in metric_names])
    replicates = np.array(rows, dtype=float)

    record: Dict[str, Any] = {}
    used = []
    for position, metric in enumerate(metric_names):
        low, high, n_finite = _percentile_interval(replicates[:, position], ci)
        used.append(n_finite)
        if bounds:
            record[f'{prefix}{metric}_lo'] = low
            record[f'{prefix}{metric}_hi'] = high
        else:
            record[f'{prefix}{metric}_sci'] = (high - low) / 2.0
    # The minimum, not the mean: a metric that mostly degenerated should be
    # visible rather than hidden behind the ones that did not.
    record[f'{prefix}n_boot_used'] = min(used) if used else 0
    return record


def add_null_baselines(
    df: pd.DataFrame,
    att_col: str = 'late',    
    prefix: str = 'null_',
    noise: float = 0,
    seed: int = 0,
) -> Tuple[pd.DataFrame, List[str]]:
    """Add near-constant reference columns, and return them alongside the frame.

    Score these next to the real methods to calibrate what a metric is worth:

      * `null_zero` -- 0. If it beats a method on `rmse`, that method is adding
        nothing over predicting no effect at all, and `rmse` is largely
        reporting the scale of the truth rather than anyone's accuracy.
      * `null_mean` -- `mean(truth)`, the MSE-optimal constant predictor. **An
        oracle** (it reads the truth), so it is a floor and not a competitor:
        any method whose `rmse` exceeds it is beaten by a constant.

    `noise` adds `N(0, noise^2)` to each, seeded by `seed` so a rerun reproduces
    the columns exactly, and drawn independently per column so the two nulls are
    not correlated with each other. **It is now vestigial for the four metrics
    that remain**: it existed to break the exact constancy that made `spearman`,
    `pearson` and `calibration_slope` undefined, and none of those is computed
    any more. `rmse`, `mae` and `accuracy95` never needed it, and `ccc` needs no
    help either -- its denominator carries the truth's variance, so a constant
    estimate scores a well-defined 0. Keep `noise` far below the outcome scale
    (the default moves `rmse` by less than its own rounding) or pass `noise=0.0`
    for the exact constants.

    The third null -- prevalence -- is not constant and so cannot live here: it
    needs the row-level data, and `baselines.ControlMean` supplied it until the
    registry was pruned to the nine scored baselines.
    """    
    if noise < 0:
        raise ValueError(f'noise must be non-negative, got {noise!r}')
    truth = _numeric(df, att_col)
    finite = truth[np.isfinite(truth)]
    enriched = df.copy()
    rng = np.random.default_rng(seed)
    zero = np.zeros(len(df))
    mean = np.full(len(df), float(finite.mean()) if finite.size else float('nan'))
    if noise > 0 and len(df):
        # Independent draws per column: shared noise would make the two nulls
        # correlated with each other, which is not what a floor is for.
        zero = zero + rng.normal(scale=noise, size=len(df))
        mean = mean + rng.normal(scale=noise, size=len(df))
    enriched[f'{prefix}zero'] = zero
    enriched[f'{prefix}mean'] = mean
    return enriched, [f'{prefix}zero', f'{prefix}mean']


def _norm_sf(z: np.ndarray) -> np.ndarray:
    """Upper-tail normal probability, `P(Z > z)`.

    Hand-rolled rather than `scipy.stats.norm.sf` because this module imports
    only numpy and pandas, and that is load-bearing: scipy is absent from one of
    the two interpreters this code runs under. `math.erfc` is in the standard
    library and the identity is exact.
    """
    return np.array([0.5 * math.erfc(value / math.sqrt(2.0)) for value in
                     np.asarray(z, dtype=float).ravel()]).reshape(np.shape(z))


def _benjamini_hochberg(pvalues: np.ndarray, alpha: float) -> np.ndarray:
    """Benjamini-Hochberg step-up: which hypotheses to reject at FDR `alpha`.

    Hand-rolled for the same reason as `_norm_sf`, and more pressingly:
    `statsmodels` is not installed in either interpreter here, so importing
    `multipletests` at module scope would make `metrics` unimportable outright.
    That exact failure has already happened once in `cexp.py`.

    Step-up, not step-down: find the largest rank k with `p_(k) <= alpha*k/n`
    and reject everything at or below it -- including any larger p-value that
    failed its own threshold, which is what makes BH more powerful than a naive
    per-test comparison.
    """
    finite = np.isfinite(pvalues)
    reject = np.zeros(len(pvalues), dtype=bool)
    if not finite.any():
        return reject
    values = pvalues[finite]
    n_tests = len(values)
    order = np.argsort(values, kind='stable')
    thresholds = alpha * np.arange(1, n_tests + 1) / n_tests
    passed = np.flatnonzero(values[order] <= thresholds)
    if passed.size:
        keep = np.zeros(n_tests, dtype=bool)
        keep[order[:passed[-1] + 1]] = True
        reject[np.flatnonzero(finite)] = keep
    return reject




def rank_separation_criterion_mask(
    df: pd.DataFrame,
    value_col: str = 'late',
    std_col: str = 'se_late',
    r: float = 0.2,
    threshold: float = 2.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
    apply_bh_correction: bool = False,
    alpha: float = 0.05,
) -> np.ndarray:
    """`local_difference_mask` with the comparison carrying its own uncertainty.

    `local_difference_mask` divides the gap to the `k = int(N*r)`-th neighbour by
    the *candidate's* standard error alone, which treats the neighbour's value as
    if it were known exactly. It is not: which effect sits at a given rank is
    itself uncertain. This bootstraps that:

      1. Resample the `value_col` values with replacement, `n_bootstrap` times.
      2. Sort each replicate, so column `j` holds the `j`-th order statistic.
      3. `rank_mean[j]` and `rank_se[j]` are the mean and SD down that column.

    The candidate keeps its observed `value_col` and `std_col`; the comparison
    at rank `i -+ k` uses `rank_mean` and `rank_se`. A row survives when at least
    one available side separates:

        |candidate_value - rank_mean[j]|
        --------------------------------------  >  threshold
        sqrt(candidate_SE**2 + rank_se[j]**2)

    A separation/resolution criterion, not a formal test -- the null
    distribution of a maximum over two order-statistic contrasts is not normal,
    so `threshold` calibrates a resolution cut rather than a size.

    **`rank_se` is resampling variability only.** The bootstrap draws the point
    estimates, so it measures how much the effect at a given rank moves when the
    *set of cards* is resampled. It does not include the estimation error of the
    card that lands there -- `std_col` never enters the resample. Perturbing the
    draw (`values[idx] + rng.normal(0.0, stds[idx])`) would fold that in; it is
    deliberately not done, because the question this answers is "how well
    resolved is this effect against the local spacing of the reference set",
    and that spacing is a property of the set.

    Two consequences worth knowing before reading the output:

      * `rank_se` shrinks like `1/sqrt(N)`, so on a large frame the comparison
        term nearly vanishes and this converges back to `local_difference_mask`.
        Measured on a 400-row frame with `se_late` around 0.37, `rank_se` ran
        about 0.07 -- a median denominator inflation of ~2%, moving ~2% of the
        decisions. Compare the two masks before assuming this one differs.

        And when they do differ, **it is mostly not the bootstrap**. `LD` keeps
        a row only when *both* neighbours are far (it takes a `minimum`); this
        keeps it when *either* is. On that same frame `LD` kept 169 rows and
        this kept 222, differing on 57 -- of which the `minimum`-to-`maximum`
        change accounts for 57 and the bootstrap for 8. The two masks are not a
        clean ablation of each other.
      * `rank_mean[j]` is a bootstrap-smoothed order statistic while the
        candidate contributes a raw one. Mid-frame the gap between them is
        negligible; in the outer few ranks it reaches a few tenths of a value-SD,
        which is where the effects of interest sit.

    Boundary rows are judged on the side they have: a missing neighbour scores
    zero and loses the `maximum`, rather than vetoing the row. `r` above 0.5
    would leave middle rows with neither neighbour and is rejected.

    `r = 0.5` exactly is the limit and is allowed: the two halves meet, so every
    row has precisely one side and the rule becomes a one-sided check of each row
    against the rank `N/2` away. It is the most permissive `r` -- retention rises
    with the rank step, because a longer step is a wider gap -- but not a vacuous
    one: whether a row survives is still set by its `std_col` against that gap.

    Returned as a boolean array in the frame's own row order, so it can be
    assigned as a column. Rows whose value or standard error is not finite are
    False and take no part in the ranking.
    """
    if n_bootstrap < 2:
        raise ValueError(
            f'n_bootstrap must be at least 2 for a rank SD; got {n_bootstrap}.'
        )

    values_all = _numeric(df, value_col)
    stds_all = _numeric(df, std_col)
    usable = np.isfinite(values_all) & np.isfinite(stds_all)
    mask = np.zeros(len(df), dtype=bool)

    values = values_all[usable]
    stds = stds_all[usable]
    n_rows = len(values)
    if n_rows == 0:
        return mask

    k = int(n_rows * r)
    if k <= 0:
        raise ValueError(
            f'r={r} is too small for {n_rows} usable rows: the neighbour step'
            ' k is 0, so no distance is defined.'
        )
    # Stricter than `local_difference_mask`'s `k >= n_rows`, and it has to be:
    # that rule only asks whether *one* side exists, and this one would leave
    # rows at sorted positions [n_rows - k, k) with neither neighbour, scoring
    # them 0 and dropping them with nothing in the output to show why.
    if 2 * k > n_rows:
        raise ValueError(
            f'r={r} is too large for {n_rows} usable rows: the neighbour step'
            f' k={k} leaves the {2 * k - n_rows} row(s) around the median'
            ' with no comparison on either side.'
        )

    order = np.argsort(values, kind='stable')
    ordered_values = values[order]
    ordered_stds = stds[order]

    # One row per bootstrap replicate, one column per rank after sorting.
    rng = np.random.default_rng(seed)
    replicates = values[rng.integers(0, n_rows, size=(n_bootstrap, n_rows))]
    # In place is safe: fancy indexing above already returned a fresh array.
    replicates.sort(axis=1)
    rank_mean = replicates.mean(axis=0)
    rank_se = replicates.std(axis=0, ddof=1)

    # Built as z directly rather than as separate difference and SE arrays with
    # an `inf` sentinel on the missing side. `local_difference_mask` can use
    # that sentinel because it takes a `minimum`, where `inf` drops out; under
    # the `maximum` this rule needs, `inf / sqrt(finite + inf)` is nan and
    # `np.maximum` propagates it, which would reject every one of the first and
    # last k rows however well separated they are.
    candidate_se = np.where(ordered_stds > 0, ordered_stds, 1e-9)
    backward_z = np.zeros(n_rows)
    forward_z = np.zeros(n_rows)

    backward_z[k:] = np.abs(ordered_values[k:] - rank_mean[:-k]) / np.sqrt(
        candidate_se[k:] ** 2 + rank_se[:-k] ** 2
    )
    forward_z[:-k] = np.abs(ordered_values[:-k] - rank_mean[k:]) / np.sqrt(
        candidate_se[:-k] ** 2 + rank_se[k:] ** 2
    )
    z_scores = np.maximum(backward_z, forward_z)

    if apply_bh_correction:
        # Doubled, unlike `local_difference_mask`'s: its statistic is a gap
        # between sorted values and so is non-negative by construction, while
        # `ordered_values[i] - rank_mean[j]` can fall either way and is put
        # through `abs`. A one-sided tail on it would halve every p-value.
        keep_sorted = _benjamini_hochberg(2.0 * _norm_sf(z_scores), alpha)
    else:
        keep_sorted = z_scores > threshold

    keep = np.zeros(n_rows, dtype=bool)
    keep[order] = keep_sorted
    mask[np.flatnonzero(usable)] = keep
    return mask


def filter_by_rank_separation(
    df: pd.DataFrame,
    value_col: str = 'late',
    std_col: str = 'se_late',
    r: float = 0.2,
    threshold: float = 2.0,
    n_bootstrap: int = 1000,
    seed: int = 0,
    apply_bh_correction: bool = False,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """`rank_separation_criterion_mask` applied, returning the surviving rows.
    """
    mask = rank_separation_criterion_mask(
        df, value_col, std_col, r, threshold, n_bootstrap, seed,
        apply_bh_correction, alpha,
    )
    return df[mask]

def evaluate_methods(
    df: pd.DataFrame,
    method_names: Sequence[str],
    att_col: str = 'late',
    att_se_col: str = 'se_late',
    confidence: bool = True,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
    bounds: bool = False,
    cluster_col: Optional[str] = 'card_a',
) -> pd.DataFrame:
    """Score each method column against the gold-standard ATT.

    df:            one row per (card_a, group_b) pair, carrying the gold standard
                   and one column per method.
    method_names:  the estimate columns to score.
    truth_divisor: the gold standard's scale factor, 1 by default (correct when
        both sides are absolute counts); `H_ref`, the constant p2p1 horizon,
        under `baseline_methods`' rate outcome. `att_col` and `att_se_col`
                   are divided by it so the truth reaches the methods' scale.
    confidence:    add bootstrap intervals. **Set False to skip the bootstrap
                   entirely** -- the frame is then identical to what this
                   returned before intervals existed, and costs the same. That
                   is the mode for iterating; leave it on for anything read as a
                   result.
    n_boot:        replicates. 1000 is the practical floor for a percentile
                   interval; 200 suffices only for a standard error.
    ci:            interval coverage.
    seed:          resample seed, so a rerun reproduces the widths exactly.
    bounds:        emit `<metric>_lo` / `<metric>_hi` instead of `<metric>_sci`.
    cluster_col:   resample whole clusters of this column rather than rows.
                   `None` resamples rows independently.

    Returns one row per method and one column per metric, sorted by `rmse`
    ascending (best first), NaNs last.

    Confidence intervals
    --------------------
    Nonparametric cluster bootstrap: K out of K `cluster_col` clusters with
    replacement, each drawn cluster contributing all its rows, percentile
    method, reported as `<metric>_sci` -- the half-width `(hi - lo) / 2`,
    conventionally the margin of error. Every method is scored on the *same*
    resamples, so the widths are mutually comparable. `n_boot_used` reports the
    finite replicates behind the narrowest metric; resamples too degenerate to
    score a metric are dropped rather than counted.

    Clustering on `card_a` is the honest unit: pairs sharing a `card_a` share
    drafters, the same fitted propensity, and the same natural experiment
    behind `late`, so a row bootstrap (`cluster_col=None`) treats dependent
    pairs as independent and runs narrow.

    Two limits, neither cosmetic:

      * **`est +/- sci` is not the interval.** Percentile intervals are
        asymmetric about the point estimate, so the half-width preserves the
        width but not the endpoints, and every metric here is bounded (`ccc` on
        [-1,1], `rmse` and `mae` non-negative, `accuracy95` on [0,1]), so
        `est +/- sci` can leave the valid range where the raw bounds cannot.
        Pass `bounds=True` for the exact interval.
      * **Overlap is not a test.** Two methods whose marginal intervals overlap
        can still differ reliably -- they are scored on the same pairs, so the
        paired difference is far better determined than either margin. Measured
        on a synthetic where one method's error is uniformly 6% larger, the
        paired interval is **16.7x narrower** and excludes zero while the two
        margins overlap. **Use `compare_baselines` for that question**; these
        columns cannot answer it. And with ~46 method columns, roughly two
        intervals will exclude zero by chance at 95%. No multiplicity
        correction is applied.

    Cost scales as `n_boot x len(method_names)`. Since the metric set was cut
    to four, no rank correlation is computed anywhere, and scoring one replicate
    is a handful of vectorised passes -- so the card count is no longer a term
    at all. `confidence=False` skips the loop entirely.

    `rmse`, `mae` and `ccc` ignore `se`; `accuracy95` needs it, so `n_used_se`
    can be below `n_used` and that metric is then computed on the se-valid
    subset alone.

    Read `rmse` against `mae` and both against `ccc`. The first pair says
    whether a few pairs carry the score; `ccc` says whether a good `rmse` came
    from accuracy or from predicting near zero. `add_null_baselines` gives both
    readings a reference point -- if `null_zero` beats a method on `rmse`, that
    method adds nothing over predicting no effect.

    Weak instruments are **not** filtered: `late` is a ratio, so pairs with a
    small `itt_d` produce enormous `late` and `se_late` and will dominate the
    unweighted `rmse`. `mae` is the robust reading, and the `enrich_scoring_frame`
    masks -- `GSP` in particular, which keeps the most precisely measured pairs
    within each group -- are how to exclude them deliberately.
    """    

    missing = [c for c in (att_col, att_se_col) if c not in df.columns]
    missing += [m for m in method_names if m not in df.columns]
    if confidence and cluster_col is not None and cluster_col not in df.columns:
        missing.append(cluster_col)
    if missing:
        raise KeyError(f"columns not present in df: {missing}")

    if all(k in df.columns for k in PAIR_KEY) and df.duplicated(list(PAIR_KEY)).any():
        n_dup = int(df.duplicated(list(PAIR_KEY)).sum())
        logging.warning(_DUPLICATE_PAIRS_WARNING, n_dup)

    truth = _numeric(df, att_col)
    truth_se = _numeric(df, att_se_col)

    estimates = {name: _numeric(df, name) for name in method_names}
    records: List[Dict[str, Any]] = [
        _score_method(name, estimates[name], truth, truth_se)
        for name in method_names
    ]
    columns = list(METRIC_COLUMNS)

    if confidence and n_boot > 0 and len(df) > 1:
        # One index set up front, so every method sees identical resamples.
        rng = np.random.default_rng(seed)
        if cluster_col is None:
            indices = list(rng.integers(0, len(df), size=(n_boot, len(df))))
        else:
            indices = _cluster_indices(rng, df[cluster_col].to_numpy(), n_boot)
        for record, name in zip(records, method_names):
            record.update(_bootstrap_method(
                estimates[name], truth, truth_se, indices, ci, bounds
            ))
        suffixes = ('_lo', '_hi') if bounds else ('_sci',)
        columns += [f'{m}{s}' for m in BOOTSTRAP_METRICS for s in suffixes]
        columns.append('n_boot_used')

    scores = pd.DataFrame.from_records(records, columns=columns)
    return scores.sort_values(
        'rmse', kind='mergesort', na_position='last'
    ).reset_index(drop=True)


def _estimate_series(name: str, frame: pd.DataFrame) -> pd.Series:
    """One method's `value` column, keyed on (card_a, group_b) and named."""
    missing = [c for c in ESTIMATE_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(
            f"estimates[{name!r}] is missing column(s) {missing}; expected "
            f"{list(ESTIMATE_COLUMNS)} as produced by "
            'BaselineEnricher.get_estimates'
        )
    index = pd.MultiIndex.from_frame(frame[list(PAIR_KEY)])
    if index.has_duplicates:
        repeated = index[index.duplicated()].unique().tolist()[:5]
        raise ValueError(
            f"estimates[{name!r}] has duplicate {list(PAIR_KEY)} rows, e.g. "
            f"{repeated}. Joining it would multiply those pairs; de-duplicate "
            'first.'
        )
    return pd.Series(
        frame['value'].to_numpy(dtype=float), index=index, name=name
    )


def enrich_pairs_estimates(
    pairs_df: pd.DataFrame,
    estimates: Dict[str, pd.DataFrame],
    validate: bool = True,
    require_finite: bool = False,
) -> pd.DataFrame:
    """Left-join `get_estimates`' output onto a pairs frame, one column per method.

    Returns a copy of `pairs_df` with one added column per key of `estimates`,
    named for the method and carrying that method's `value` for the row's
    `(card_a, group_b)`.

    `validate` is a **coverage** check: every pair in `pairs_df` must appear in
    every method's frame, so a missing column value can only mean the estimator
    declined, never that the join failed. It does *not* reject `nan` values --
    a degenerate arm legitimately produces one and `BaselineEnricher._run` logs
    it, so rejecting them by default would raise on nearly every real run.

    `require_finite` is the strict check, off by default: it additionally
    rejects any `nan` in the joined columns, reporting the count per method.
    Use it when a complete table is a precondition rather than an expectation.

    Joined with `DataFrame.join` rather than `pd.merge` so **`pairs_df`'s index
    survives** -- `merge` silently replaces it with a RangeIndex. A `pairs_df`
    holding the same pair twice still receives that pair's estimate on both
    rows, as before.
    """
    if not estimates:
        raise ValueError(
            'estimates is empty, so there is nothing to join. Pass the dict '
            'BaselineEnricher.get_estimates returned.'
        )
    absent = [c for c in PAIR_KEY if c not in pairs_df.columns]
    if absent:
        raise KeyError(f'pairs_df is missing column(s) {absent}')
    # pandas would silently suffix a collision `_x`/`_y`, which reads as a
    # successful join and quietly leaves the old column in place.
    clashing = [name for name in estimates if name in pairs_df.columns]
    if clashing:
        raise ValueError(
            f'pairs_df already has column(s) {clashing} that would collide '
            'with method names. Drop them, or join into a frame that has not '
            'been enriched already.'
        )

    wanted = pd.MultiIndex.from_frame(pairs_df[list(PAIR_KEY)]).unique()
    parts = []
    for name, frame in estimates.items():
        series = _estimate_series(name, frame)
        if validate:
            gap = wanted.difference(series.index)
            if len(gap):
                raise ValueError(
                    f'estimates[{name!r}] is missing {len(gap)} of '
                    f'{len(wanted)} pairs in pairs_df, e.g. '
                    f'{gap.tolist()[:5]}. Pass validate=False to join anyway '
                    'and leave them nan.'
                )
        parts.append(series)

    enriched = pairs_df.join(pd.concat(parts, axis=1), on=list(PAIR_KEY))

    if require_finite:
        counts = {
            name: int(enriched[name].isna().sum()) for name in estimates
        }
        offenders = {n: c for n, c in counts.items() if c}
        if offenders:
            worst = max(offenders, key=lambda n: offenders[n])
            examples = enriched.loc[
                enriched[worst].isna(), list(PAIR_KEY)
            ].head(5).to_records(index=False).tolist()
            raise ValueError(
                f'{len(offenders)} of {len(counts)} methods have nan values: '
                f'{offenders}. Worst is {worst!r}; example pairs {examples}. '
                'These are estimator failures, not join failures -- drop '
                'require_finite to keep them.'
            )
    return enriched

"""Pairwise significance testing between ATT estimators, FDR-corrected.

A leaderboard of point estimates says which method scored best; it does not say
whether the gap is real. This module answers that, for every pair of methods and
every metric, with a paired bootstrap.

    errors = samplewise_errors(df, methods)             # steps 2-3, long format
    matrix = generate_significance_matrix(df, methods)  # steps 4-5, the verdict

Why paired
----------
Two methods scored on the same pairs make *correlated* errors -- same drafts,
same gold standard, same natural experiment -- and

    Var(A - B) = Var(A) + Var(B) - 2*Cov(A, B)

so that shared component cancels in the difference. Comparing two methods by
whether their marginal intervals overlap throws it away, and systematically
under-calls real differences: measured on a synthetic where one method's error
is uniformly 6% larger, the paired interval is **16.7x narrower** and excludes
zero while the two margins overlap.

The pairing is realised by **common random numbers** -- both methods scored on
the identical rows in every replicate. In design terms each sample is a *block*
and the comparison is within-block.

Two levels of pairing, and why `ccc` needs the second
-----------------------------------------------------
Subtraction can happen before or after aggregation:

    per-sample      d_i = Err_A,i - Err_B,i, then bootstrap mean(d)
    per-replicate   score both on one resample, then subtract

**For a linear statistic the two are algebraically identical**, because
subtraction and averaging commute. `rmse` (squared error), `mae` (absolute
error) and `accuracy95` (a 0/1 indicator) are all means of a per-observation
term, so they decompose per sample and can live in the long frame.

`ccc` cannot. It is a ratio of moments,
`2*cov / (var_e + var_t + (mu_e - mu_t)^2)`, so no per-row "ccc error" exists;
by the delta method it is only *asymptotically* linear, through an influence
function. Its test is therefore run per replicate -- **which is no less
paired**: same resamples, same rows, same cancellation, same p-value machinery.
Only the arithmetic differs, and that is stated rather than hidden.

`rmse` sits between: the square root of a linear statistic. Monotone, so testing
it per replicate gives **exactly** the p-value the protocol's squared-error test
would; only the reported magnitude differs.

Self-contained apart from `evaluation`
--------------------------------------
The metric definitions, `Z_95`, the BH step-up and `_numeric` are imported from
`evaluation` rather than copied. That is deliberate: if this module carried its
own `ccc`, the matrix could certify a difference in a quantity the score table
does not report. Sharing the definitions makes that impossible.
"""

import itertools
import logging
import zlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from evaluation import (    
    REDUCED_METRICS,
    Z_95,
    _benjamini_hochberg,    
    _numeric,
)

DEFAULT_TRUTH_DIVISOR=1

# Which direction is good, per metric. `rmse`/`mae` are errors, so lower wins;
# `ccc`/`accuracy95` are agreement, so higher wins. Without this the same sign
# would mean opposite things in different columns of the matrix.
METRIC_ORIENTATION: Dict[str, int] = {
    'rmse': -1, 'mae': -1, 'ccc': +1, 'accuracy95': +1,
}

# What `generate_significance_matrix` writes per pair and metric.
WINS_A, WINS_B, NOT_SIGNIFICANT = '>', '<', '~'

# Replicates scored per pass. The gathers are (block, n_pairs) arrays, so this
# caps the working set rather than the total cost.
_REPLICATE_BLOCK = 2000

_MATRIX_TOO_FEW = (
    '%r and %r share only %d usable pair(s); every metric is reported as not'
    ' significant for that comparison'
)


def _bh_adjusted(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values).

    `evaluation._benjamini_hochberg` answers "reject or not" at one `alpha`;
    this answers "at what FDR would this have been rejected", which is what
    makes a table of symbols auditable. Enforced monotone by a running minimum
    from the largest p downward -- that cumulative minimum is exactly what
    makes BH a step-*up* procedure rather than a per-test comparison.

    Non-finite entries stay non-finite: a pair with no usable comparison has no
    q-value, and imputing one would let it reject.
    """
    adjusted = np.full(len(pvalues), np.nan)
    finite = np.flatnonzero(np.isfinite(pvalues))
    if finite.size == 0:
        return adjusted
    values = pvalues[finite]
    n_tests = len(values)
    order = np.argsort(values, kind='stable')
    scaled = values[order] * n_tests / np.arange(1, n_tests + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted[finite[order]] = np.clip(monotone, 0.0, 1.0)
    return adjusted


def _error_columns(
    estimate: np.ndarray,
    truth: np.ndarray,
    truth_se: np.ndarray,
    weighted: bool = False,
) -> Dict[str, np.ndarray]:
    """Per-sample error contributions, one array per decomposable metric.

    `rmse` here is the per-sample *squared* error, whose mean is `rmse ** 2`;
    `mae` the absolute error; `accuracy95` the 0/1 indicator, nan where
    `se_late` is unusable. `ccc` is absent by necessity -- see the module
    docstring.

    `weighted` divides the error by `se_late` first, so the squared term becomes
    the squared standardized residual and the absolute term its unsquared
    companion: inverse-variance weighting applied at the sample level, so
    imprecise pairs count for less rather than being thresholded out. accuracy
    is left on the raw error, since it is already a statement about `se_late`.
    """
    error = estimate - truth
    usable_se = np.isfinite(truth_se) & (truth_se > 0)
    if weighted:
        with np.errstate(divide='ignore', invalid='ignore'):
            error = error / np.where(usable_se, truth_se, np.nan)
    with np.errstate(invalid='ignore'):
        covered = np.where(
            usable_se,
            (np.abs(estimate - truth) <= Z_95 * truth_se).astype(float),
            np.nan,
        )
    return {'rmse': error ** 2, 'mae': np.abs(error), 'accuracy95': covered}


def samplewise_errors(
    df: pd.DataFrame,
    method_names: Sequence[str],
    att_col: str = 'late',
    att_se_col: str = 'se_late',
    truth_divisor: float = DEFAULT_TRUTH_DIVISOR,
    weighted: bool = False,
) -> pd.DataFrame:
    """Long-format per-sample errors: `sample_id`, `method`, one column per metric.

    The melted frame the paired differences are built from, exposed in its own
    right so they can be inspected, plotted or aggregated outside
    `generate_significance_matrix` -- which computes the same quantities through
    the same `_error_columns`, so the two cannot disagree.

    `sample_id` is the **positional** row index, 0..n-1, not `df.index`: it has
    to be unique for the pairing to align, and a caller's index need not be.
    Join back on position if you need `card_a` / `group_b`.
    """    
    missing = [c for c in (att_col, att_se_col) if c not in df.columns]
    missing += [m for m in method_names if m not in df.columns]
    if missing:
        raise KeyError(f'columns not present in df: {missing}')

    truth = _numeric(df, att_col) / truth_divisor
    truth_se = _numeric(df, att_se_col) / truth_divisor
    sample_id = np.arange(len(df))

    frames = [
        pd.DataFrame({
            'sample_id': sample_id,
            'method': name,
            **_error_columns(_numeric(df, name), truth, truth_se, weighted),
        })
        for name in method_names
    ]
    return pd.concat(frames, ignore_index=True)


def _replicate_differences(
    est_a: np.ndarray,
    est_b: np.ndarray,
    truth: np.ndarray,
    truth_se: np.ndarray,
    indices: np.ndarray,
    weighted: bool = False,
) -> np.ndarray:
    """(n_replicates, 4) array of `metric_A - metric_B` on each shared resample.

    Common random numbers: both methods are scored on the identical rows in
    every replicate, so their shared error cancels in the difference.

    Vectorised over replicates in blocks rather than looped -- at 10,000
    replicates a Python loop calling a scorer twice would dominate everything
    else. The block bounds the working set, not the total work.

    `ccc` is computed from **centred** products, matching
    `evaluation._score_reduced`: the raw-moment form differences two large
    nearly-equal numbers when the mean dominates the variance.
    """
    out = np.full((len(indices), len(REDUCED_METRICS)), np.nan)
    position = {metric: i for i, metric in enumerate(REDUCED_METRICS)}
    usable_se = np.isfinite(truth_se) & (truth_se > 0)
    # Always an array, never None: dividing by exactly 1.0 is a no-op in IEEE,
    # and it keeps the branch out of the inner loop.
    scale = (np.where(usable_se, truth_se, np.nan) if weighted
             else np.ones_like(truth))

    def score(sampled: np.ndarray, tru: np.ndarray,
              take: np.ndarray) -> Dict[str, np.ndarray]:
        raw = sampled - tru
        err = raw / scale[take]
        valid = usable_se[take]
        inside = (np.abs(raw) <= Z_95 * truth_se[take]) & valid
        denominator = valid.sum(axis=1)
        centred_e = sampled - sampled.mean(axis=1, keepdims=True)
        centred_t = tru - tru.mean(axis=1, keepdims=True)
        gap = sampled.mean(axis=1) - tru.mean(axis=1)
        spread = (np.mean(centred_e ** 2, axis=1)
                  + np.mean(centred_t ** 2, axis=1) + gap ** 2)
        with np.errstate(divide='ignore', invalid='ignore'):
            return {
                'rmse': np.sqrt(np.mean(err ** 2, axis=1)),
                'mae': np.mean(np.abs(err), axis=1),
                'ccc': np.where(
                    spread > 0,
                    2.0 * np.mean(centred_e * centred_t, axis=1) / spread,
                    np.nan),
                'accuracy95': np.where(
                    denominator > 0, inside.sum(axis=1) / denominator, np.nan),
            }

    for start in range(0, len(indices), _REPLICATE_BLOCK):
        take = indices[start:start + _REPLICATE_BLOCK]
        tru = truth[take]
        # Both sides scored, then subtracted -- not accumulated in place. A nan
        # on either side must make the difference nan, which `a - b` gives and
        # an accumulator seeded from `a` would silently turn into `-b`.
        scored_a = score(est_a[take], tru, take)
        scored_b = score(est_b[take], tru, take)
        for metric in REDUCED_METRICS:
            out[start:start + len(take), position[metric]] = (
                scored_a[metric] - scored_b[metric]
            )
    return out


def _weighted_differences(
    est_a: np.ndarray,
    est_b: np.ndarray,
    truth: np.ndarray,
    truth_se: np.ndarray,
    weights: np.ndarray,
    weighted: bool = False,
) -> np.ndarray:
    """`_replicate_differences` over integer row *weights* rather than indices.

    A clustered resample draws clusters with replacement, so the number of rows
    it lands on varies from replicate to replicate and cannot be a rectangular
    index matrix without a per-replicate Python loop. Giving each row its
    cluster's draw count as a weight sidesteps that: **with integer weights the
    weighted statistic is exactly the statistic over the duplicated sample**, so
    this is the same bootstrap expressed differently, and `weights` has the same
    `(n_replicates, n_rows)` shape the index matrix had.

    Every metric therefore appears in its weighted form -- including `ccc`,
    whose centred moments are weighted by the same `w`.
    """
    out = np.full((len(weights), len(REDUCED_METRICS)), np.nan)
    position = {metric: i for i, metric in enumerate(REDUCED_METRICS)}
    usable_se = np.isfinite(truth_se) & (truth_se > 0)
    scale = (np.where(usable_se, truth_se, np.nan) if weighted
             else np.ones_like(truth))

    def score(estimate: np.ndarray, block: np.ndarray) -> Dict[str, np.ndarray]:
        total = block.sum(axis=1, keepdims=True)
        raw = estimate - truth
        err = raw / scale
        # Weighted means, all sharing the same denominator.
        mean_e = (block @ estimate)[:, None] / total
        mean_t = (block @ truth)[:, None] / total
        centred_e = estimate - mean_e
        centred_t = truth - mean_t
        gap = (mean_e - mean_t)[:, 0]
        spread = (
            np.einsum('bi,bi->b', block, centred_e ** 2) / total[:, 0]
            + np.einsum('bi,bi->b', block, centred_t ** 2) / total[:, 0]
            + gap ** 2
        )
        covered = usable_se & (np.abs(raw) <= Z_95 * truth_se)
        se_total = block @ usable_se.astype(float)
        with np.errstate(divide='ignore', invalid='ignore'):
            return {
                'rmse': np.sqrt((block @ (err ** 2)) / total[:, 0]),
                'mae': (block @ np.abs(err)) / total[:, 0],
                'ccc': np.where(
                    spread > 0,
                    2.0 * np.einsum('bi,bi->b', block, centred_e * centred_t)
                    / total[:, 0] / spread,
                    np.nan),
                'accuracy95': np.where(
                    se_total > 0,
                    (block @ covered.astype(float)) / se_total, np.nan),
            }

    for start in range(0, len(weights), _REPLICATE_BLOCK):
        block = weights[start:start + _REPLICATE_BLOCK].astype(float)
        scored_a = score(est_a, block)
        scored_b = score(est_b, block)
        for metric in REDUCED_METRICS:
            out[start:start + len(block), position[metric]] = (
                scored_a[metric] - scored_b[metric]
            )
    return out


def _cluster_weights(
    labels: np.ndarray, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """`(n_boot, n_rows)` draw counts from resampling *clusters* with replacement.

    Each replicate draws `n_clusters` clusters with replacement; a row's weight
    is how many times its own cluster came up. Whole clusters move together,
    which is the point -- rows within a `card_a` share drafters, a propensity
    fit and the natural experiment behind `late`, so resampling them
    independently understates the variance.
    """
    codes, _ = pd.factorize(labels)
    n_clusters = int(codes.max()) + 1
    drawn = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    counts = np.zeros((n_boot, n_clusters), dtype=np.int64)
    for replicate in range(n_boot):
        counts[replicate] = np.bincount(drawn[replicate], minlength=n_clusters)
    return counts[:, codes]


def generate_significance_matrix(
    df: pd.DataFrame,
    method_names: Sequence[str],
    att_col: str = 'late',
    att_se_col: str = 'se_late',
    truth_divisor: float = DEFAULT_TRUTH_DIVISOR,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    one_sided: bool = True,
    cluster_col: Optional[str] = None,
    weighted: bool = False,
    with_pvalues: bool = False,
) -> pd.DataFrame:
    """Pairwise paired-bootstrap significance over every metric, FDR-corrected.

    One row per **unordered** pair -- if `(A, B)` is present `(B, A)` is not --
    with `method_a`, `method_b`, `n_pairs`, and one column per metric holding:

        '>'   A beats B, significant after BH at `alpha`
        '<'   B beats A, significant after BH at `alpha`
        '~'   no significant difference

    **The symbol always means "wins", never "is numerically larger".** `rmse`
    and `mae` are errors, so a win there is the *lower* value; `ccc` and
    `accuracy95` are agreement, so a win is the higher one. Reading the raw sign
    instead of the symbol would invert two of the four columns.

    Per pair: restrict to samples where both methods and the truth are finite;
    resample those N-of-N with replacement `n_boot` times; score both on the
    *same* resample; difference; read a one-sided empirical p-value off the
    resulting distribution. Benjamini-Hochberg then runs **per metric** across
    all pairs, so each column controls its own FDR.

    One-sided or two
    ----------------
    `one_sided=True` (the default, and the stated protocol) reports the share of
    replicates that fail to reproduce the *observed* direction. **It is the less
    conservative choice**, and the reason is worth stating: a one-sided test
    whose direction is chosen after seeing the data is not really a one-sided
    test. It behaves like a two-sided test run at `2 * alpha`, so a result
    "significant at 0.05" carries a 0.10 false-positive rate. A *pre-specified*
    direction would be perfectly valid; a data-chosen one is not.

    `one_sided=False` doubles it -- `min(1, 2p)`, the confidence-interval
    inversion of the same bootstrap -- which restores the nominal rate. Use it
    when the number is going into a writeup; the difference is exactly a factor
    of two, so nothing else about the procedure changes.

    Either way the p-values then go through Benjamini-Hochberg at `alpha`,
    per metric, across every pair.

    Rows or clusters
    ----------------
    `cluster_col=None` resamples **rows**, which treats every pair as an
    independent observation. Here they are not: ~35 `group_b` rows share each
    `card_a`, along with the drafters, the fitted propensity and the natural
    experiment behind `late`. Row-level resampling therefore *understates* the
    variance, and if within-card correlation is high the effective sample size
    is nearer the card count than the pair count -- a widening of up to
    `sqrt(rows per card)`.

    `cluster_col='card_a'` resamples whole cards instead, which is the honest
    version. It is off by default only so that numbers already computed do not
    move; **a difference that survives clustering is solid, and one that does
    not was resting on an independence assumption that was never true.** Run
    both and report both. `n_clusters` is emitted beside `n_pairs` so the design
    effect is legible.

    Two further caveats:

      * **p is `(crossings + 1) / (n_finite + 1)`**, never a literal zero. A
        count of 0 would enter BH as infinite certainty when all it means is
        that the bootstrap ran out of resolution at `1 / n_boot`. Doubling for
        the two-sided case happens after that, so the floor is `2 / (n + 1)`.
      * **The resample is over rows, not clusters.** Pairs sharing a `card_a`
        are not independent, so this -- like every interval in `evaluation` --
        runs narrower than the dependence warrants.

    Each pair is seeded from its own **names**, not its position, so a pair's
    verdict does not change when other methods join or leave the list. Note the
    BH correction *does* depend on the list, and legitimately so: it controls
    the false-discovery rate over the comparisons actually made.

    `with_pvalues` adds `<metric>_p` (raw one-sided) and `<metric>_q` (after BH)
    so the symbols can be audited rather than taken on trust.
    """    
    missing = [c for c in (att_col, att_se_col) if c not in df.columns]
    missing += [m for m in method_names if m not in df.columns]
    if missing:
        raise KeyError(f'columns not present in df: {missing}')
    names = list(dict.fromkeys(method_names))
    if len(names) < 2:
        raise ValueError(
            f'need at least two distinct methods to compare, got {names}'
        )
    if n_boot < 2:
        raise ValueError(f'n_boot must be at least 2, got {n_boot!r}')
    if cluster_col is not None and cluster_col not in df.columns:
        raise KeyError(f'cluster_col {cluster_col!r} not present in df')
    labels = (df[cluster_col].to_numpy() if cluster_col is not None else None)

    truth = _numeric(df, att_col) / truth_divisor
    truth_se = _numeric(df, att_se_col) / truth_divisor
    estimates = {name: _numeric(df, name) for name in names}
    with_truth = np.isfinite(truth)

    records: List[Dict[str, Any]] = []
    pvalues: Dict[str, List[float]] = {m: [] for m in REDUCED_METRICS}
    for name_a, name_b in itertools.combinations(names, 2):
        est_a, est_b = estimates[name_a], estimates[name_b]
        paired = np.isfinite(est_a) & np.isfinite(est_b) & with_truth
        n_pairs = int(paired.sum())
        units = (n_pairs if labels is None
                 else int(pd.Series(labels[paired]).nunique()))
        record: Dict[str, Any] = {
            'method_a': name_a, 'method_b': name_b, 'n_pairs': n_pairs,
            'n_clusters': units,
        }
        # Fewer than two resampling units -- rows, or clusters when they are
        # what moves -- leaves nothing to resample.
        if n_pairs < 2 or units < 2:
            logging.warning(_MATRIX_TOO_FEW, name_a, name_b, units)
            for metric in REDUCED_METRICS:
                record[metric] = NOT_SIGNIFICANT
                pvalues[metric].append(float('nan'))
            records.append(record)
            continue

        # `crc32`, not `hash`: Python salts string hashing per process, so a
        # name-derived `hash` would not reproduce across runs.
        rng = np.random.default_rng([
            seed, zlib.crc32(f'{name_a}\x00{name_b}'.encode('utf-8')),
        ])
        arms = (est_a[paired], est_b[paired], truth[paired], truth_se[paired])
        if labels is None:
            differences = _replicate_differences(
                *arms, rng.integers(0, n_pairs, size=(n_boot, n_pairs)),
                weighted,
            )
        else:
            differences = _weighted_differences(
                *arms, _cluster_weights(labels[paired], n_boot, rng), weighted,
            )
        # The observed difference through the same code path as the replicates,
        # so the point estimate and the resamples cannot disagree about how a
        # metric is computed.
        observed = _replicate_differences(
            *arms, np.arange(n_pairs)[None, :], weighted,
        )[0]

        for position, metric in enumerate(REDUCED_METRICS):
            favour = METRIC_ORIENTATION[metric] * observed[position]
            replicates = METRIC_ORIENTATION[metric] * differences[:, position]
            finite = replicates[np.isfinite(replicates)]
            if not np.isfinite(favour) or favour == 0 or finite.size < 2:
                pvalues[metric].append(float('nan'))
                record[metric] = NOT_SIGNIFICANT
                continue
            # How often a resample fails to reproduce the observed direction.
            crossings = int((finite * np.sign(favour) <= 0).sum())
            p_value = (crossings + 1) / (finite.size + 1)
            if not one_sided:
                # Interval inversion: the two-sided p is twice the smaller
                # tail, and the observed direction is the smaller tail by
                # construction unless the difference sits on top of zero --
                # where the clip to 1 is the right answer anyway.
                p_value = min(1.0, 2.0 * p_value)
            pvalues[metric].append(p_value)
            record[metric] = WINS_A if favour > 0 else WINS_B
        records.append(record)

    # BH per metric, across every pair -- each column controls its own FDR.
    for metric in REDUCED_METRICS:
        raw = np.array(pvalues[metric], dtype=float)
        reject = _benjamini_hochberg(raw, alpha)
        adjusted = _bh_adjusted(raw)
        for position, record in enumerate(records):
            if not reject[position]:
                record[metric] = NOT_SIGNIFICANT
            if with_pvalues:
                record[f'{metric}_p'] = raw[position]
                record[f'{metric}_q'] = adjusted[position]

    columns = (['method_a', 'method_b', 'n_pairs', 'n_clusters']
               + list(REDUCED_METRICS))
    if with_pvalues:
        columns += [f'{m}{s}' for m in REDUCED_METRICS for s in ('_p', '_q')]
    return pd.DataFrame(records, columns=columns)

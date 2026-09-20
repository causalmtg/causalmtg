"""Exact-match exclusion diagnostic: the `Z=1,T=0` vs `Z=0` contrast, matched.

A third approach to the question `diagnostic.py` asks -- does `Z` reach `Y`
outside `T` -- alongside `ExclusionDiagnostic` (raw contrast) and
`MatchedExclusionDiagnostic` (K-nearest matching). The estimator here is the
strictest of the three:

  1. Take the `T=0` stratum. `{Z=0, T=1}` is empty by construction -- a card
     absent from the booster cannot be picked -- so the control arm is every
     `Z=0` row and the treated arm is the never-takers, `Z=1, T=0`.
  2. For each treated row, find the `Z=0` rows whose covariate vector is
     **identical** to it (an exact match rather than a caliper).
  3. If that set is empty, **drop the treated row**. Otherwise its control value
     is the plain average of `Y` over the whole matched set.
  4. What survives is two aligned lists -- the treated `Y`, and the matched
     averages -- summarised by the same standardised mean difference
     `ind_diagnostic` uses, plus the share that had to be dropped.

Why exact rather than nearest
-----------------------------
`MatchedExclusionDiagnostic` always finds `K` neighbours, however far away they
are, and pays for a bad match in bias that nothing in its output names. Exact
matching cannot be biased by match quality, because there is no match quality:
either the covariate vector is the same or the row is gone. It moves the whole
problem into one number, `drop_ratio`, which is visible.

**That number is the diagnostic.** A `drop_ratio` near 1 does not mean the
estimate is bad, it means there is no estimate -- the surviving rows are the
handful sitting in the densest cells of `X`, which are not a random subsample of
the never-takers. Read `n_matched` and `drop_ratio` before reading `smd`. This is
why `n_z1t0` is reported alongside: `n_matched / (1 - drop_ratio) == n_z1t0`.

The design
----------
Matching is on `baselines.CORE_COVARIATES` alone, the eight the propensity
reads -- drafters who look alike to the propensity, i.e. the same pack-1
history. Note `diagnostic.CORE_COVARIATES` is a *different* eight-column
constant of the same name; this module uses `baselines`', which is the
propensity's. The design is outcome- and group-independent, so the plan is
built once per CardA and reused across groups.

A caveat on `pooled_sd`, and it matters
---------------------------------------
The control arm is a list of **averages**. An average over a bucket of `k` rows
has roughly `1/k` the variance of a single outcome, so `var_y_z0` is
systematically smaller than the outcome's real spread and `pooled_sd` is
therefore too small. Every `|smd|` here is inflated relative to a textbook one
and **must not be read against the usual 0.1 / 0.25 thresholds**. It is a
within-module comparison statistic -- across pairs -- not an absolute one.
`n_z0_used / n_buckets` is the mean bucket size and says how much inflation to
expect.

No standard error is reported. Buckets are shared between treated rows, so the
matched averages are not independent and the naive `sd/sqrt(n)` understates --
`diagnostic.py`'s module docstring measures that same failure at 0.87x on the
K-nearest estimator, and the correction it applies (Abadie-Imbens) is derived for
a fixed `K` and does not carry over to variable-size exact buckets. An SE that
is wrong in a known direction is worse than none.

Standalone except for the extractor
-----------------------------------
The one dependency is `baselines.DraftDataExtractor`, built by the caller and
passed in: it already drops incomplete drafts, excludes drafters holding CardA
from pack 1 and builds the covariates, and duplicating that is the liability the
`diagnostic.py` docstring flags about its copied helpers.
`RegressionExclusionDiagnostic` takes the same object for the same reason.
Everything else here is stdlib + numpy + pandas, and the `baselines` import is
lazy so the module stays importable where scikit-learn is absent.

Usage::

    extractor = baselines.DraftDataExtractor(
        baselines.ExtractionConfig(set_code), card_db, grouping, card_a_names)
    extractor.process_drafts_file(csv_path)           # or .from_file(path)
    diag = exact_diagnostic.ExactMatchExclusionDiagnostic(extractor)
    out = diag.get_estimates(pairs_df)                # card_a / group_b columns

    out[['card_a', 'group_b', 'n_matched_core', 'drop_ratio_core', 'smd_core']]
"""

import logging
import math
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import (
    TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple,
)

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - annotation only, never imported at runtime
    from baselines import DraftDataExtractor


# The design name, suffixed to every statistic (`smd_core`) so downstream
# readers of the older multi-variant output keep working.
VARIANT = 'core'

# Output statistics, in report order; each is emitted as `<stat>_core`.
STAT_COLUMNS: Tuple[str, ...] = (
    # Sample accounting. `n_matched / (1 - drop_ratio) == n_z1t0`.
    'n_covariates', 'n_z1t0', 'n_matched', 'drop_ratio',
    'n_z0_pool', 'n_z0_used', 'n_buckets', 'mean_bucket',
    # The two arms and their contrast, as `ind_diagnostic` reports them.
    'mean_y_z1', 'mean_y_z0',
    'var_y_z1', 'var_y_z0',
    'sd_y_z1', 'sd_y_z0',
    'mean_diff', 'pooled_sd', 'smd', 'abs_smd',
)

# The outcome statistics, blanked when a plan has fewer than two matched rows.
_OUTCOME_STATS = STAT_COLUMNS[8:]

_MISSING_PAIR_COLUMNS = (
    "pairs_df needs columns %s; got %s. Pass the same frame the baseline "
    "enricher takes."
)

_MISSING_CORE = (
    "the extracted frame is missing %d of the %d propensity covariates: %s. "
    "Matching is on the %d it has, which is a different estimator than the "
    "one named. Re-extract, or read `n_covariates_core`."
)


@lru_cache(maxsize=1)
def _baselines_deps() -> Tuple[
    Callable[[pd.DataFrame], List[str]], Tuple[str, ...],
]:
    """`(covariate_columns, CORE_COVARIATES)`.

    Imported on demand: lazy, never at module scope, so importing this module
    does not drag in `baselines` and through it scikit-learn.
    `diagnostic._regression_deps` is the same pattern for the same reason.
    """
    # pylint: disable=import-outside-toplevel
    from baselines import CORE_COVARIATES, covariate_columns
    return covariate_columns, tuple(CORE_COVARIATES)


def core_columns(all_columns: Sequence[str]) -> List[str]:
    """The propensity's covariates, in frame order, restricted to what is there.

    `all_columns` is expected to be `baselines.covariate_columns(frame)` with
    `not_offered` already removed -- that column *is* `Z`, so matching on it
    would put every treated row in a bucket no control can reach.

    A frame missing some of `CORE_COVARIATES` (one extracted before they
    existed) warns and matches on what it has, rather than raising as
    `baselines._named_view` does: this is a diagnostic, and a narrower design
    that says so in `n_covariates_core` is more useful than no answer at all.
    """
    _, core = _baselines_deps()
    present = [c for c in all_columns if c in set(core)]
    absent = [c for c in core if c not in set(all_columns)]
    if absent:
        logging.warning(_MISSING_CORE, len(absent), len(core), absent,
                        len(present))
    return present


@dataclass
class MatchPlan:
    """One CardA's exact-match assignment.

    **Outcome-independent, and that is what makes this affordable.** The buckets
    depend on the design and the `Z` split alone, so one plan is built per
    CardA and reused across every group -- `contrast_from_plan` is then two
    `bincount`s per group. `diagnostic.MatchPlan` factors the K-nearest
    estimator the same way for the same reason.

    Row indices are into the CardA frame, so `values[treated_rows]` lines up
    with any `Y_<group>` column of it.
    """

    n_covariates: int
    treated_rows: np.ndarray       # kept treated rows, frame indices
    treated_bucket: np.ndarray     # compact bucket id per kept treated row
    control_rows: np.ndarray       # control rows inside a used bucket
    control_bucket: np.ndarray     # compact bucket id per entry of control_rows
    n_buckets: int
    n_treated_total: int           # before dropping, i.e. all of Z=1,T=0
    n_control_pool: int            # all of Z=0, matched or not

    @property
    def n_matched(self) -> int:
        return int(self.treated_rows.size)

    @property
    def drop_ratio(self) -> float:
        """Share of `Z=1,T=0` rows with no exact control. nan if the arm is empty."""
        if not self.n_treated_total:
            return float('nan')
        return 1.0 - self.n_matched / self.n_treated_total

    @property
    def mean_bucket(self) -> float:
        """Controls per used bucket -- how much `var_y_z0` is deflated by."""
        if not self.n_buckets:
            return float('nan')
        return self.control_rows.size / self.n_buckets

    def matched_means(self, values: np.ndarray) -> np.ndarray:
        """Each kept treated row's control value: its bucket's mean of `values`.

        Two `bincount`s give every bucket's mean at once, so a bucket serving
        several treated rows is averaged once rather than once per row. Every
        used bucket holds at least one control by construction, so the division
        cannot hit zero.
        """
        sums = np.bincount(self.control_bucket,
                           weights=values[self.control_rows],
                           minlength=self.n_buckets)
        counts = np.bincount(self.control_bucket, minlength=self.n_buckets)
        return (sums / counts)[self.treated_bucket]


def build_exact_plan(
    design: np.ndarray,
    treated_rows: np.ndarray,
    control_rows: np.ndarray,
) -> MatchPlan:
    """Bucket `Z=0` rows by exact covariate vector and assign the treated ones.

    `np.unique(axis=0)` over the stacked arms does the whole job in one lexsort:
    identical rows land on one id whichever arm they came from, so a treated row
    has an exact control exactly when its id also occurs among the controls.
    That is cheaper and less error-prone than hashing tuples in Python, and it
    fixes the bucket order deterministically.

    Exactness is bit equality. Most covariates are integer counts, and the share
    columns (`oncolor_pip_share` and friends) are deterministic ratios of
    integers -- the same numerator and denominator give the same double -- so
    equal drafters do land in one bucket. Two *different* ratios that happen to
    round to the same double would also merge, which is the harmless direction.

    An empty arm gives an empty plan (`n_matched == 0`) rather than None, so the
    counts it carries -- `n_z1t0`, `n_z0_pool` -- still reach the output.
    """
    n_treated = int(treated_rows.size)
    empty = MatchPlan(
        n_covariates=int(design.shape[1]) if design.ndim == 2 else 0,
        treated_rows=np.empty(0, dtype=np.intp),
        treated_bucket=np.empty(0, dtype=np.intp),
        control_rows=np.empty(0, dtype=np.intp),
        control_bucket=np.empty(0, dtype=np.intp),
        n_buckets=0,
        n_treated_total=n_treated,
        n_control_pool=int(control_rows.size),
    )
    if not n_treated or not control_rows.size or design.shape[1] == 0:
        return empty

    stacked = np.vstack([design[treated_rows], design[control_rows]])
    # `return_inverse`'s shape for axis=0 changed between numpy 2.0 and 2.1;
    # ravel so both give a flat row -> id map.
    _, inverse = np.unique(stacked, axis=0, return_inverse=True)
    inverse = np.asarray(inverse).ravel()
    treated_ids = inverse[:n_treated]
    control_ids = inverse[n_treated:]

    has_control = np.zeros(inverse.max() + 1, dtype=bool)
    has_control[control_ids] = True
    keep = has_control[treated_ids]
    if not keep.any():
        return empty

    # Compact the surviving ids to 0..n_buckets-1 so `bincount` stays dense.
    used = np.unique(treated_ids[keep])
    kept_bucket = np.searchsorted(used, treated_ids[keep])

    in_use = np.isin(control_ids, used)
    control_bucket = np.searchsorted(used, control_ids[in_use])

    return MatchPlan(
        n_covariates=int(design.shape[1]),
        treated_rows=np.asarray(treated_rows)[keep],
        treated_bucket=kept_bucket.astype(np.intp),
        control_rows=np.asarray(control_rows)[in_use],
        control_bucket=control_bucket.astype(np.intp),
        n_buckets=int(used.size),
        n_treated_total=n_treated,
        n_control_pool=int(control_rows.size),
    )


def contrast_from_plan(plan: MatchPlan, values: np.ndarray) -> Dict[str, float]:
    """One group's statistics under a prebuilt plan.

    The only part that touches `Y`, and the only part that runs per group. Two
    `bincount`s give every bucket's control mean at once; the treated rows then
    index into that, so a bucket serving several treated rows is averaged once
    rather than once per row.

    `var_y_z0` is the variance of those **averages**, not of the underlying
    outcomes -- see the module docstring on why `pooled_sd` is consequently too
    small and `smd` is not on the textbook scale.

    Degenerate cells give nan rather than 0: fewer than two matched rows leaves
    the variances undefined, and a zero `pooled_sd` (both arms constant) leaves
    the ratio undefined while `mean_diff` stays exactly 0.
    """
    stats: Dict[str, float] = {
        'n_covariates': float(plan.n_covariates),
        'n_z1t0': float(plan.n_treated_total),
        'n_matched': float(plan.n_matched),
        'drop_ratio': plan.drop_ratio,
        'n_z0_pool': float(plan.n_control_pool),
        'n_z0_used': float(plan.control_rows.size),
        'n_buckets': float(plan.n_buckets),
        'mean_bucket': plan.mean_bucket,
    }
    stats.update({name: float('nan') for name in _OUTCOME_STATS})
    if plan.n_matched < 2:
        return stats

    y1 = values[plan.treated_rows]
    y0 = plan.matched_means(values)
    if not (np.isfinite(y1).all() and np.isfinite(y0).all()):
        return stats

    var_y1 = float(y1.var(ddof=1))
    var_y0 = float(y0.var(ddof=1))
    mean_diff = float(y1.mean() - y0.mean())
    pooled_sd = math.sqrt((var_y1 + var_y0) / 2)
    smd = mean_diff / pooled_sd if pooled_sd > 0 else float('nan')

    stats.update({
        'mean_y_z1': float(y1.mean()),
        'mean_y_z0': float(y0.mean()),
        'var_y_z1': var_y1,
        'var_y_z0': var_y0,
        'sd_y_z1': math.sqrt(var_y1),
        'sd_y_z0': math.sqrt(var_y0),
        'mean_diff': mean_diff,
        'pooled_sd': pooled_sd,
        'smd': smd,
        'abs_smd': abs(smd),
    })
    return stats


@dataclass
class CardPlan:
    """One CardA's exclusion frame and its exact-match plan, shared by every group."""

    frame: pd.DataFrame
    plan: MatchPlan


class ExactMatchExclusionDiagnostic:
    """The exclusion contrast by exact matching on the propensity's covariates.

    Takes a built `baselines.DraftDataExtractor` rather than streaming the CSV,
    exactly as `diagnostic.RegressionExclusionDiagnostic` does: the extractor
    already applies the completeness rule, excludes prior holders of CardA and
    builds the covariates. It also fixes the design -- history is pack 1, `Z` is
    pack 2 pick 0, `Y` is the pack-3 count. No windows.

    The extractor hands back the ATT frame; two lines turn it into the exclusion
    frame, the same two `RegressionExclusionDiagnostic.frame_for` uses::

        diag = df[df['T'] == 0.0]              # the T=0 stratum
        diag['T'] = 1.0 - diag['not_offered']  # T := Z

    `not_offered` is then dropped -- it has become identical to the new `T`, and
    matching on a column that encodes the arm label would leave every treated row
    in a bucket no control can reach.

    Work is dominated by the `np.unique` lexsort, which is outcome- and
    group-independent: one per CardA, reused across every group. There is no
    cross-card state, so a run is **shardable by `card_a`** and bit-identical
    to a whole one.
    """

    def __init__(self, extractor: 'DraftDataExtractor', progress: bool = True):
        self.extractor = extractor
        self.progress = progress

    @property
    def columns(self) -> List[str]:
        """Output columns, in order: the pair, then `<stat>_core` per statistic."""
        return ['card_a', 'group_b'] + [f'{stat}_{VARIANT}' for stat in STAT_COLUMNS]

    def plan_for(self, card_a: str) -> Optional[CardPlan]:
        """The exclusion frame and its exact-match plan for one CardA, or None.

        Built once per CardA -- the frame, the `Z` split and the design depend
        on CardA alone.
        """
        df = self.extractor.get_dataframe(card_a)
        if df.empty or 'not_offered' not in df.columns:
            return None
        frame = df[df['T'] == 0.0].copy()
        if frame.empty:
            return None
        frame['T'] = 1.0 - frame['not_offered'].to_numpy(dtype=float)
        frame = frame.drop(columns=['not_offered'])

        covariate_columns, _ = _baselines_deps()
        all_columns = [c for c in covariate_columns(frame) if c != 'not_offered']
        if not all_columns:
            return None

        columns = core_columns(all_columns)
        design = (frame[columns].fillna(0.0).to_numpy(dtype=float)
                  if columns
                  else np.empty((len(frame), 0), dtype=float))
        treatment = frame['T'].to_numpy(dtype=float)
        plan = build_exact_plan(
            design,
            np.flatnonzero(treatment == 1.0),
            np.flatnonzero(treatment == 0.0),
        )
        return CardPlan(frame=frame, plan=plan)

    def get_estimates(self, pairs_df: pd.DataFrame) -> pd.DataFrame:
        """One row per (card_a, group_b), in `pairs_df` order.

        **Every pair gets a row, nan included.** A CardA with no rows, an empty
        arm, or a group with no `Y_` column produces a nan row rather than a
        dropped one -- so `isna()` means "degenerated" and a missing row
        downstream means exactly "the join did not match". The count columns
        survive where they can: a pair that matched nothing still reports its
        `n_z1t0` and `drop_ratio`, which is the whole point of the estimator.

        Pairs are de-duplicated per CardA with order preserved.
        """
        missing = [c for c in ('card_a', 'group_b') if c not in pairs_df.columns]
        if missing:
            raise ValueError(
                _MISSING_PAIR_COLUMNS % (missing, list(pairs_df.columns))
            )

        rows: List[Dict[str, Any]] = []
        started = time.time()
        cards = list(dict.fromkeys(pairs_df['card_a']))
        for position, (card_a, pairs) in enumerate(
            pairs_df.groupby('card_a', sort=False), start=1
        ):
            built = self.plan_for(card_a)
            groups = list(dict.fromkeys(pairs['group_b']))
            for group_b in groups:
                rows.append(self._one(built, card_a, group_b))
            if self.progress:
                self._report(position, len(cards), card_a, len(groups), started)

        return pd.DataFrame(rows, columns=self.columns)

    def _one(
        self, built: Optional[CardPlan], card_a: str, group_b: str
    ) -> Dict[str, Any]:
        """One pair's row, degenerating to nan rather than raising."""
        row: Dict[str, Any] = {'card_a': card_a, 'group_b': group_b}
        row.update({name: float('nan') for name in self.columns[2:]})
        if built is None:
            return row
        column = f'Y_{group_b}'
        if column not in built.frame.columns:
            return row
        values = built.frame[column].to_numpy(dtype=float)
        row.update({f'{name}_{VARIANT}': value
                    for name, value in contrast_from_plan(built.plan, values).items()})
        return row

    @staticmethod
    def _report(
        position: int, total: int, card_a: str, n_groups: int, started: float
    ) -> None:
        """`[i/N]` per CardA, matching `diagnostic`'s reporting.

        `print` rather than `logging` so it shows in a notebook without
        configuring logging, as `diagnostic.get_single_pack_df` does.
        """
        elapsed = time.time() - started
        remaining = elapsed / position * (total - position)
        print(
            f'[{position}/{total}] {card_a}  ({n_groups} groups)'
            f'  {elapsed:.0f}s elapsed, ~{remaining:.0f}s left',
            flush=True,
        )

"""Pre-treatment history balance for the IV design in `ivexp.py`.

`ivexp` estimates LATE with the instrument Z = "CardA was in the booster at pack
2, pick 0". That rests on Z being handed out by booster collation rather than by
anything about the drafter, so the drafter's pre-treatment history H -- fixed
before the instrument is ever revealed -- must be independent of it:

    H  independent of  Z

This module tests that, one function of H at a time. For each treatment card A
and each history group G, take

    X_G = how many of the drafter's **pack 1 picks** belonged to group G

and compare it across the two arms with the standardised mean difference

    SMD(A, G) = ( E[X_G | Z=1, A] - E[X_G | Z=0, A] )
                / sqrt( ( Var(X_G | Z=1, A) + Var(X_G | Z=0, A) ) / 2 )

Under `H` independent of `Z` every `SMD(A, G)` is zero in expectation, so a
non-trivial `|SMD|` says the instrument is correlated with pre-treatment
behaviour. That is a different failure from the one `diagnostic.py` chases: that
module asks whether Z reaches Y outside T (exclusion), this one asks whether Z
was as-good-as-randomly assigned in the first place. An imbalance here indicts
the LATE regardless of how well exclusion holds.

`X_G` counts what the drafter **took**, not what they were shown. Counting the
`pack_card_*` columns over pack 1 would test collation independence -- a
property of the product -- rather than the behavioural history the independence
assumption is about.

The sample is `ivexp`'s
-----------------------
Deliberately, and this is the point: the same p2p1 requirement, the same
`is_complete_draft` rule, the same prior-holder exclusion. Balance measured on a
wider population would be a true statement about some other experiment. Measured
here, a failure is a failure of the number `ivexp` actually reports, which is why
`n_valid`, `n_invalid`, `n_incomplete` and `n_offered` are carried onto the
output and should agree with `ivexp`'s per card.

One consequence: the prior-holder filter is CardA-specific, so each CardA is
balanced over its *own* retained sample. That is why the SMD is computed per
(CardA, group) pair rather than once per group.

Reading the output
------------------
`get_balance_df()` returns one row per (CardA, group), with both arms' n, mean,
variance and sd alongside the difference, the pooled denominator and the SMD --
so a suspicious ratio can be read back to the pieces it came from rather than
taken on trust. The distribution summary the diagnostic calls for is one line::

    df['abs_smd'].agg(['median', 'max'])
    df['abs_smd'].quantile(0.95)

Standalone by design
--------------------
Nothing from `ivexp`, `baselines` or `diagnostic` is imported; stdlib and pandas
only. `diagnostic.py` keeps the same independence for the same reason, and pays
the same price -- `is_complete_draft` below is a third copy of one rule, and if
`ivexp`'s ever changes this one must change with it or the two modules quietly
stop describing the same population.
"""

import csv
import logging
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

import pandas as pd


# The instrument: pack 2, pick 0 (0-indexed pack number). Everything strictly
# before it is pack 1, which is the history window this module counts over.
ASSIGNMENT_PACK = 1
# Used only by the completeness rule, so that the rule -- and therefore the
# retained sample -- is identical to `ivexp`'s.
OUTCOME_PACK = 2

# Warn above this share of drafts dropped as incomplete. Same value and same
# reason as `ivexp`: the rule assumes all three packs come from one product, and
# a set with genuinely unequal packs would discard the whole corpus with nothing
# in the output to show it.
INCOMPLETE_WARN_RATE = 0.05

_TOO_MANY_INCOMPLETE = (
    "dropped %d of %d drafts (%.1f%%) as incomplete, above the %.0f%% alarm."
    " The rule requires every pack up to pack %d to have as many picks as the"
    " longest pack in the draft; a set whose packs are genuinely unequal, or a"
    " changed export schema, would look like this."
)


def is_complete_draft(rows: List[Dict[str, Any]]) -> bool:
    """Whether a draft ran to the end of the outcome pack.

    A copy of `ivexp.is_complete_draft`, kept here so the module stays
    standalone. The two must agree exactly or this diagnostic describes a
    different population than the estimate it is diagnosing.

    A truncated pack 1 matters directly here: `X_G` is a raw count over that
    window, so a short pack 1 makes every `X_G` mechanically smaller. Nothing
    adjusts for it, and abandonment is plausibly non-random, so such a draft is
    dropped rather than counted.
    """
    lengths = Counter(int(row['pack_number']) for row in rows)
    if not lengths:
        return False
    expected = max(lengths.values())
    return all(lengths.get(pack, 0) == expected
               for pack in range(OUTCOME_PACK + 1))


@dataclass
class HistoryStats:
    """Running sufficient statistics for `X_G` within one arm of one pair."""

    n: int = 0
    sum_x: int = 0
    sum_x_sq: int = 0

    def add_observation(self, x: int) -> None:
        self.n += 1
        self.sum_x += x
        self.sum_x_sq += x ** 2

    @property
    def mean(self) -> float:
        return self.sum_x / self.n if self.n else float('nan')

    @property
    def variance(self) -> float:
        """Sample variance, ddof=1 -- the same form `ivexp` uses for its arms."""
        if self.n < 2:
            return float('nan')
        centred = self.sum_x_sq - self.n * self.mean ** 2
        # Cancellation can push this a hair below zero when the arm is nearly
        # constant, which is exactly the case where a negative variance would
        # otherwise propagate a nan into an sqrt.
        return max(0.0, centred) / (self.n - 1)

    @property
    def sd(self) -> float:
        variance = self.variance
        return float('nan') if math.isnan(variance) else math.sqrt(variance)


@dataclass
class BalanceStats:
    """The two instrument arms for one CardA -> GroupB combination."""

    z0: HistoryStats = field(default_factory=HistoryStats)
    z1: HistoryStats = field(default_factory=HistoryStats)


@dataclass
class DraftCounts:
    """Draft bookkeeping for one CardA. Independent of GroupB.

    `n_valid + n_invalid + n_incomplete` accounts for every draft that reached
    the assignment, so it is always visible which filter did the work. These are
    the same four numbers `ivexp.DraftCounts` carries (minus `n_treated`, since
    the treatment is never read here) and should match it card for card; if they
    do not, the two modules are describing different populations.
    """

    n_valid: int = 0
    n_invalid: int = 0        # CardA already held from pack 1
    n_incomplete: int = 0     # the draft was abandoned part-way
    n_offered: int = 0        # z == 1, among valid

    def add_valid(self, z: int) -> None:
        self.n_valid += 1
        self.n_offered += z


class HistoryBalanceDiagnostic:
    """Accumulates `E[X_G | Z]` by (CardA, GroupB) from a drafts CSV.

    Mirrors `ivexp.PackCausalExperiment`: same streaming pass, same completeness
    rule, same prior-holder exclusion. It differs in what it records -- pack-1
    group counts split by Z, rather than the pack-3 outcome statistics LATE needs
    -- and in what it ignores: the treatment T is never read, because the
    question is whether Z was randomly assigned, not what it caused.
    """

    def __init__(self, set_code: str, bomb_names: List[str],
                 card_metadata: Dict[str, str]):
        # A label only; it performs no filtering.
        self.set_code = set_code
        self.bomb_names = set(bomb_names)
        # Kept as the forward card -> group map and looked up per pick. The
        # reverse `group -> set(cards)` form `ivexp` uses would cost
        # O(picks x groups) per draft, which at ~35 groups dominates.
        self.card_metadata = card_metadata
        # Sorted so the output frame's row order is stable across runs.
        self.groups: List[str] = sorted(set(card_metadata.values()))

        # cardA -> groupB -> BalanceStats. One level shallower than a per-pack
        # design, because the assignment is fixed at p2p1.
        self.state: Dict[str, Dict[str, BalanceStats]] = {
            bomb: {} for bomb in self.bomb_names
        }
        # cardA -> DraftCounts. None of it depends on groupB.
        self.counts: Dict[str, DraftCounts] = {
            bomb: DraftCounts() for bomb in self.bomb_names
        }
        # Draft-level tallies for the incomplete-draft alarm. CardA-independent,
        # so they are counted once rather than per bomb.
        self.n_drafts = 0
        self.n_incomplete = 0

    def process_drafts_file(self, csv_path: str) -> None:
        with open(csv_path, mode='r', encoding='utf-8') as file:
            self.process_drafts(csv.DictReader(file))

    def process_drafts(self, reader: Any) -> None:
        """Accumulate from any iterable of dict rows, grouped by `draft_id`.

        Assumes a draft's rows are contiguous, which a 17lands export is.
        """
        current_draft_id = None
        current_draft_rows: List[Dict[str, Any]] = []

        for ridx, row in enumerate(reader):
            draft_id = row['draft_id']
            if ridx % 200000 == 0:
                logging.info("processed %d", ridx)
            if draft_id != current_draft_id:
                if current_draft_rows:
                    self._process_single_draft(current_draft_rows)
                current_draft_id = draft_id
                current_draft_rows = []
            current_draft_rows.append(row)

        if current_draft_rows:
            self._process_single_draft(current_draft_rows)

        self._warn_if_truncated()

    def _warn_if_truncated(self) -> None:
        """Alarm when the completeness rule discarded an implausible share."""
        if not self.n_drafts:
            return
        rate = self.n_incomplete / self.n_drafts
        if rate > INCOMPLETE_WARN_RATE:
            logging.warning(
                _TOO_MANY_INCOMPLETE, self.n_incomplete, self.n_drafts,
                100 * rate, 100 * INCOMPLETE_WARN_RATE, OUTCOME_PACK,
            )

    def _process_single_draft(self, rows: List[Dict[str, Any]]) -> None:
        p0_row = next(
            (r for r in rows
             if int(r['pack_number']) == ASSIGNMENT_PACK
             and int(r['pick_number']) == 0),
            None,
        )
        if p0_row is None:
            return
        self.n_drafts += 1

        if not is_complete_draft(rows):
            self.n_incomplete += 1
            for bomb in self.bomb_names:
                self.counts[bomb].n_incomplete += 1
            return

        # The history is CardA-independent, so it is built once. "Strictly
        # before the assignment" is exactly pack 1, since the assignment is
        # pick 0 of pack 2.
        history_picks: Set[str] = {
            r['pick'] for r in rows
            if int(r['pack_number']) < ASSIGNMENT_PACK
        }
        group_counts: Counter = Counter()
        for r in rows:
            if int(r['pack_number']) < ASSIGNMENT_PACK:
                group = self.card_metadata.get(r['pick'])
                if group is not None:
                    group_counts[group] += 1

        for bomb in self.bomb_names:
            counts = self.counts[bomb]

            # Drop drafters who already hold CardA, as `ivexp` does: they
            # already have the treatment, so their assignment is not the one the
            # experiment is about.
            #
            # This is also why CardA needs no exclusion from its own group's
            # count below -- it cannot be among `history_picks` in a retained
            # draft, so `X_{group(CardA)}` is already free of it.
            if bomb in history_picks:
                counts.n_invalid += 1
                continue

            # Matches `ivexp` exactly. `fit._offered` reads the same column as
            # `float(v) > 0` instead, which differs on a booster holding two
            # copies; this module follows `ivexp`, since `ivexp`'s design is
            # what is being diagnosed.
            z = 1 if p0_row.get(f"pack_card_{bomb}") in ['1', '1.0'] else 0
            counts.add_valid(z)

            bomb_groups = self.state[bomb]
            for group_name in self.groups:
                if group_name not in bomb_groups:
                    bomb_groups[group_name] = BalanceStats()
                cell = bomb_groups[group_name]
                target = cell.z1 if z == 1 else cell.z0
                target.add_observation(group_counts.get(group_name, 0))

    def get_balance_df(self) -> pd.DataFrame:
        """One row per (CardA, GroupB), with the SMD and every piece of it.

        `mean_diff` is `E[X_G | Z=1] - E[X_G | Z=0]` and `pooled_sd` is
        `sqrt((var_x_z1 + var_x_z0) / 2)`, so `smd = mean_diff / pooled_sd`
        exactly. Both arms' n, mean, variance and sd are reported so a large
        ratio can be traced to a real gap rather than a thin cell.

        `smd` is nan, never 0, in the two degenerate cases:

          * an arm with fewer than two drafts -- the variance is undefined;
          * `pooled_sd == 0`, meaning the group never appears in pack 1 in
            either arm, so both are constant. `mean_diff` is 0 there and stays
            0; it is the ratio that does not exist.

        The count columns describe the CardA sample and do not depend on
        group_b; they are repeated on every row of a card for convenience, so
        take one row per card rather than averaging them.

        Summarise across pairs with `df['abs_smd']` -- `.agg(['median','max'])`
        and `.quantile(0.95)`.
        """
        results = []

        for bomb in sorted(self.bomb_names):
            counts = self.counts[bomb]
            for group_name in self.groups:
                cell = self.state[bomb].get(group_name)
                if cell is None:
                    continue
                z1, z0 = cell.z1, cell.z0

                mean_diff = z1.mean - z0.mean
                pooled_sd = self._pooled_sd(z1.variance, z0.variance)
                smd = (mean_diff / pooled_sd if pooled_sd > 0
                       else float('nan'))

                results.append({
                    "card_a": bomb,
                    "group_b": group_name,
                    "n_z1": z1.n,
                    "n_z0": z0.n,
                    "mean_x_z1": z1.mean,
                    "mean_x_z0": z0.mean,
                    "var_x_z1": z1.variance,
                    "var_x_z0": z0.variance,
                    "sd_x_z1": z1.sd,
                    "sd_x_z0": z0.sd,
                    "mean_diff": mean_diff,
                    "pooled_sd": pooled_sd,
                    "smd": smd,
                    "abs_smd": abs(smd),
                    "n_valid": counts.n_valid,
                    "n_invalid": counts.n_invalid,
                    "n_incomplete": counts.n_incomplete,
                    "n_offered": counts.n_offered,
                })

        return pd.DataFrame(results)

    @staticmethod
    def _pooled_sd(var_z1: float, var_z0: float) -> float:
        """`sqrt((var1 + var0) / 2)`, nan if either arm's variance is."""
        # Either arm being undefined -- fewer than two drafts -- makes the
        # denominator undefined, which is what leaves `smd` nan downstream.
        if math.isnan(var_z1) or math.isnan(var_z0):
            return float('nan')
        return math.sqrt((var_z1 + var_z0) / 2)

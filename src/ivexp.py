"""IV estimation of the bomb-pick effect, with the outcome pivoted to pack 3.

A copy of `cexp.py`, kept separate so the original stays available for
comparison. Three things differ, and they all follow from one decision: **the
outcome is counted over pack 3 alone.**

Why pack 3
----------
The instrument is Z = "CardA was in the pack at pack 2, pick 0" and the
treatment is T = "CardA was picked there". `cexp` counts the outcome over
`(pack 2, pick > 0) | pack 3`, and six of those picks come from packs the
drafter has already seen -- including their own opener, which returns on the
wheel. CardA occupying a slot in that booster displaces whatever card would
otherwise have been there, so `Z` moves those picks mechanically, with no
behavioural response and no covariate able to adjust it away.

Pack 3 is a fresh booster round, collated independently of pack 2, so `Z` is
independent of its supply. The window shrinks from 27 picks to 14 and the
estimates get noisier; that is the price of a window the instrument cannot
touch except through the drafter.

What changes relative to `cexp`
-------------------------------
1. **The outcome window is pack 3**, every pick of it, including its pick 0.
2. **Y no longer excludes CardA.** `cexp` subtracts the bomb from its own
   group's count to stop the treatment leaking into its own outcome. That leak
   ran through the shared pack-2 window; pack 3 is independent of pack 2, so a
   second copy taken there is an ordinary outcome and is counted.
3. **`late` is an absolute count** on the same scale as the `baselines`
   estimators, which now also report counts. `metrics.truth_divisor` is 1.

The instrument is fixed at pack 2 pick 1 and the outcome at pack 3, so the
per-pack loop `cexp` carries is gone: `get_single_pack_df` takes `pack_idx` for
call-site compatibility and rejects anything but `ASSIGNMENT_PACK`.
`get_meta_analysis_df` is dropped with it -- it pooled across packs, and there
is only one. `cexp`'s optional `is_sure_pick` design filter is gone too: it was
only ever run as a pass-through here, and a filter that never fires is a
sample-definition question nobody can see in the output.

Incomplete drafts
-----------------
A draft that was abandoned part-way is dropped outright, counted in
`n_incomplete`. Two truncations matter and they sit at opposite ends: a short
**pack 3** makes `late` a raw count over a shorter window, so it is mechanically
smaller; a short **pack 1** means the covariates on the `baselines` side are
measured over different history lengths. Nothing adjusts for either -- `late` is
an absolute count -- and abandonment is plausibly non-random (a disengaged or
losing player), so this is potential confounding rather than noise.

The rule needs no pack-size parameter, because all three packs come from the
same product: take the longest pack the draft has and require every pack up to
the outcome pack to match it. Picks are sequential and cannot be skipped, so
abandonment truncates a *suffix* -- reaching the last pick of pack 3 implies
every earlier pick exists, which is why one equality test replaces three
separate length checks. The same test catches a mid-draft export hole
(`{0: 14, 1: 15, 2: 15}` fails) for free. Its one blind spot is a uniformly
short draft, which is indistinguishable from a genuinely smaller set -- hence
the drop-rate warning, which fires if the rule ever misfires on a whole corpus.

`n_no_outcome` is subsumed by this and now reads 0: a draft with no pack-3 rows
fails the equality test first. It is kept so the accounting identity
`n_valid + n_invalid + n_no_outcome + n_incomplete` still names every filter.

Split mode
----------
`split_rnd_seed` hands each draft to exactly one of this module and
`baselines.DraftDataExtractor`: both walk the same CSV in the same order and
draw one `rndseq.RndSeq` value per draft, this side keeping `x < 0.5` and the
extractor the rest. The draw comes before every other filter, so the two
streams stay aligned whatever each side later discards; skipped drafts are not
counted in `n_drafts`. None, the default, means no split.
"""

import csv
import json
import logging
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set

import pandas as pd
from scipy import stats  # type: ignore[import-untyped]

from rndseq import RndSeq


# The instrument: pack 2, pick 0 (0-indexed pack number).
ASSIGNMENT_PACK = 1
# The outcome window: pack 3, every pick.
OUTCOME_PACK = 2

# Warn above this share of drafts dropped as incomplete. The completeness rule
# assumes all three packs come from the same product; a set with genuinely
# unequal packs, or a changed export schema, would otherwise discard the whole
# corpus with nothing in the output to show it.
INCOMPLETE_WARN_RATE = 0.05

_TOO_MANY_INCOMPLETE = (
    "dropped %d of %d drafts (%.1f%%) as incomplete, above the %.0f%% alarm."
    " The rule requires every pack up to pack %d to have as many picks as the"
    " longest pack in the draft; a set whose packs are genuinely unequal, or a"
    " changed export schema, would look like this."
)


def is_complete_draft(rows: List[Dict[str, Any]]) -> bool:
    """Whether a draft ran to the end of the outcome pack.

    Take the longest pack the draft has and require every pack up to
    `OUTCOME_PACK` to match it, so no external pack size is needed -- all three
    boosters come from the same product. Picks are sequential and cannot be
    skipped, so abandonment truncates a suffix and this single equality also
    implies packs 1 and 2 are whole. A mid-draft export hole fails it too.

    `baselines._is_complete_draft` is the same rule; the two modules are
    deliberately independent, and both must apply it or they describe different
    populations.
    """
    lengths = Counter(int(row['pack_number']) for row in rows)
    if not lengths:
        return False
    expected = max(lengths.values())
    return all(lengths.get(pack, 0) == expected
               for pack in range(OUTCOME_PACK + 1))


######################
@dataclass
class OutcomeStats:
    n: int = 0
    sum_d: int = 0
    sum_y: int = 0
    sum_y_sq: int = 0
    sum_dy: int = 0

    def add_observation(self, d: int, y: int) -> None:
        self.n += 1
        self.sum_d += d
        self.sum_y += y
        self.sum_y_sq += y ** 2
        self.sum_dy += d * y


@dataclass
class TreatmentStats:
    """Sufficient statistics for one CardA -> GroupB combination."""

    z0: OutcomeStats = field(default_factory=OutcomeStats)
    z1: OutcomeStats = field(default_factory=OutcomeStats)


@dataclass
class DraftCounts:
    """Draft bookkeeping for one CardA. Independent of GroupB.

    `n_valid + n_invalid + n_no_outcome + n_incomplete` accounts for every draft
    that reached the assignment pack, so it is always visible which filter did
    the work. `n_no_outcome` is subsumed by `n_incomplete` and reads 0.

    `horizon_*` measure the pack-3 window. Unlike `cexp` no divisor is derived
    from them -- `late` is reported as an absolute count -- but
    `horizon_min == horizon_max` still says whether drafts are truncated, which
    is the only thing that makes the counts incomparable across drafters. With
    the incomplete-draft filter in place they are equal by construction.
    """

    n_valid: int = 0
    n_invalid: int = 0        # CardA already held from pack 1
    n_incomplete: int = 0     # the draft was abandoned part-way
    n_no_outcome: int = 0     # no pack-3 rows, so no outcome window at all
    n_offered: int = 0        # z == 1, among valid
    n_treated: int = 0        # d == 1, among valid
    horizon_total: int = 0
    horizon_min: int = 0
    horizon_max: int = 0

    def add_valid(self, z: int, d: int, horizon: int) -> None:
        self.n_valid += 1
        self.n_offered += z
        self.n_treated += d
        self.horizon_total += horizon
        if self.n_valid == 1:
            self.horizon_min = self.horizon_max = horizon
        else:
            self.horizon_min = min(self.horizon_min, horizon)
            self.horizon_max = max(self.horizon_max, horizon)


class PackCausalExperiment:
    """LATE for "picked CardA at p2p1" on pack-3 group counts.

    Same Wald estimator as `cexp.PackCausalExperiment`; only the outcome window
    and the CardA exclusion differ. See the module docstring.
    """

    def __init__(self, set_code: str, bomb_names: List[str],
                 card_metadata: Dict[str, str],
                 split_rnd_seed: Optional[int] = None):
        self.set_code = set_code
        self.bomb_names = set(bomb_names)
        self.card_metadata = card_metadata
        # Split mode: one seeded draw per draft, this side keeps `x < 0.5` and
        # `baselines.DraftDataExtractor` with the same seed keeps the rest.
        # Drawn before every other filter so the two streams stay aligned.
        self.split = (RndSeq(split_rnd_seed, take_below=True)
                      if split_rnd_seed is not None else None)

        # Build reverse mapping: group_name -> set of cards
        self.group_to_cards: Dict[str, Set[str]] = {}
        for card, group in card_metadata.items():
            self.group_to_cards.setdefault(group, set()).add(card)

        # cardA -> groupB -> TreatmentStats. One level shallower than `cexp`'s,
        # because the assignment pack is fixed.
        self.state: Dict[str, Dict[str, TreatmentStats]] = {
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
        if self.split is not None and not self.split.get_next_bool():
            return
        p0_row = next(
            (r for r in rows
             if int(r['pack_number']) == ASSIGNMENT_PACK
             and int(r['pick_number']) == 0),
            None,
        )
        if p0_row is None:
            return
        self.n_drafts += 1

        # An abandoned draft has a short outcome window, so its `late` would be
        # mechanically small, and short prior packs, so the `baselines`
        # covariates would be measured over a different history length. Nothing
        # adjusts for either. CardA-independent, so it is checked once.
        if not is_complete_draft(rows):
            self.n_incomplete += 1
            for bomb in self.bomb_names:
                self.counts[bomb].n_incomplete += 1
            return

        # The outcome window is CardA-independent, so it is built once.
        outcome_picks = [
            r['pick'] for r in rows if int(r['pack_number']) == OUTCOME_PACK
        ]
        horizon = len(outcome_picks)

        for bomb in self.bomb_names:
            counts = self.counts[bomb]

            # 0. Drop drafters who already hold CardA. The assignment is pick 0
            # of pack 2, so "strictly earlier" is exactly "pack 1". Such a row
            # is contaminated three ways: the drafter already has the treatment,
            # d = 1 would mean taking a *second* copy, and their later picks are
            # post-treatment behaviour rather than a control observation.
            held_before = any(
                r['pick'] == bomb for r in rows
                if int(r['pack_number']) < ASSIGNMENT_PACK
            )
            if held_before:
                counts.n_invalid += 1
                continue

            # 1. No pack 3 means no outcome window. Counting the draft would
            # contribute a manufactured zero to whichever arm it fell in, so it
            # is dropped and counted rather than scored as an observation.
            if horizon == 0:
                counts.n_no_outcome += 1
                continue

            # 2. Extract Z and D using the boolean columns and 'pick'
            z = 1 if p0_row.get(f"pack_card_{bomb}") in ['1', '1.0'] else 0
            d = 1 if (z == 1 and p0_row['pick'] == bomb) else 0
            counts.add_valid(z, d, horizon)

            # 3. Calculate Y and record. CardA is *not* excluded from its own
            # group: pack 3 is collated independently of the pack the
            # instrument acts on, so a second copy taken there is an ordinary
            # outcome rather than the treatment leaking into itself.
            bomb_groups = self.state[bomb]
            for group_name, group_cards in self.group_to_cards.items():
                y = sum(1 for card in outcome_picks if card in group_cards)
                if group_name not in bomb_groups:
                    bomb_groups[group_name] = TreatmentStats()
                target = (bomb_groups[group_name].z1 if z == 1
                          else bomb_groups[group_name].z0)
                target.add_observation(d, y)

    @staticmethod
    def _check_pack(pack_idx: int) -> None:
        if pack_idx != ASSIGNMENT_PACK:
            raise ValueError(
                f'the instrument is fixed at pack {ASSIGNMENT_PACK} pick 0 and '
                f'the outcome at pack {OUTCOME_PACK}; got pack_idx={pack_idx}. '
                'Use cexp.PackCausalExperiment for the per-pack design.'
            )

    def get_single_pack_df(self, pack_idx: int = ASSIGNMENT_PACK) -> pd.DataFrame:
        """LATE estimates, one row per (CardA, GroupB).

        `late` is an **absolute count** of group-B picks in pack 3 per complier,
        on the same scale the `baselines` estimators report -- so
        `metrics.evaluate_methods` needs no `truth_divisor` (it defaults to 1).

        The count columns describe the CardA sample and do not depend on
        group_b; they are repeated on every row of a card for convenience, so
        take one row per card rather than averaging them.

        `n_valid + n_invalid + n_incomplete + n_no_outcome` accounts for every
        draft that reached pack 2, with `n_no_outcome` subsumed by
        `n_incomplete` and so always 0. Since a card cannot be picked when absent,
        `d0_mean` is zero and so `itt_d == n_treated / n_offered` exactly, which
        ties these counters to the Wald denominator.

        `pack_idx` exists only so call sites written against `cexp` keep
        working; anything but `ASSIGNMENT_PACK` raises.
        """
        self._check_pack(pack_idx)
        results = []

        for bomb in self.bomb_names:
            counts = self.counts[bomb]
            horizon_mean = (
                counts.horizon_total / counts.n_valid
                if counts.n_valid else float('nan')
            )
            for group_name, tg in self.state[bomb].items():
                est = self._calculate_late_stats(tg.z1, tg.z0)
                if est is None:
                    continue

                late, se_late = est['late'], est['se_late']
                z_stat = late / se_late if se_late > 0 else 0
                p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

                results.append({
                    "card_a": bomb,
                    "group_b": group_name,
                    "pack": pack_idx,
                    "itt_y": est['itt_y'],
                    "itt_d": est['itt_d'],
                    "late": late,
                    "se_late": se_late,
                    "z_stat": z_stat,
                    "p_value": p_value,
                    "n_valid": counts.n_valid,
                    "n_invalid": counts.n_invalid,
                    "n_incomplete": counts.n_incomplete,
                    "n_no_outcome": counts.n_no_outcome,
                    "n_offered": counts.n_offered,
                    "n_treated": counts.n_treated,
                    "horizon_total": counts.horizon_total,
                    "horizon_mean": horizon_mean,
                    "horizon_min": counts.horizon_min,
                    "horizon_max": counts.horizon_max,
                })

        return pd.DataFrame(results)

    @staticmethod
    def _calculate_late_stats(
        s1: OutcomeStats, s0: OutcomeStats
    ) -> Optional[Dict[str, float]]:
        if s1.n < 2 or s0.n < 2:
            return None

        y1_mean, y0_mean = s1.sum_y / s1.n, s0.sum_y / s0.n
        d1_mean, d0_mean = s1.sum_d / s1.n, s0.sum_d / s0.n

        itt_y = y1_mean - y0_mean
        itt_d = d1_mean - d0_mean

        if itt_d == 0:
            return None

        late = itt_y / itt_d

        var_y1 = (s1.sum_y_sq - s1.n * y1_mean ** 2) / (s1.n - 1)
        var_y0 = (s0.sum_y_sq - s0.n * y0_mean ** 2) / (s0.n - 1)
        var_d1 = d1_mean * (1 - d1_mean)
        var_d0 = d0_mean * (1 - d0_mean)

        var_itt_y = var_y1 / s1.n + var_y0 / s0.n
        var_itt_d = var_d1 / s1.n + var_d0 / s0.n

        cov_y1_d1 = (s1.sum_dy - s1.n * y1_mean * d1_mean) / (s1.n - 1)
        cov_y0_d0 = (s0.sum_dy - s0.n * y0_mean * d0_mean) / (s0.n - 1)
        cov_itt_y_itt_d = cov_y1_d1 / s1.n + cov_y0_d0 / s0.n

        var_late = (
            (1 / itt_d ** 2) * var_itt_y
            + (itt_y ** 2 / itt_d ** 4) * var_itt_d
            - 2 * (itt_y / itt_d ** 3) * cov_itt_y_itt_d
        )

        return {
            "itt_y": itt_y,
            "itt_d": itt_d,
            "late": late,
            "se_late": math.sqrt(max(0, var_late)),
        }

    def to_json(self, filepath: str) -> None:
        with open(filepath, 'w', encoding='utf-8') as f:
            state_dict = {
                bomb: {group: asdict(tg) for group, tg in groups.items()}
                for bomb, groups in self.state.items()
            }
            payload = {
                "set_code": self.set_code,
                "bomb_names": list(self.bomb_names),
                "card_metadata": self.card_metadata,
                "counts": {b: asdict(c) for b, c in self.counts.items()},
                "state": state_dict,
            }
            json.dump(payload, f)

    @classmethod
    def from_json(cls, filepath: str) -> 'PackCausalExperiment':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        instance = cls(data['set_code'], data['bomb_names'],
                       data['card_metadata'])

        for bomb, group_dict in data['state'].items():
            for group, tg_dict in group_dict.items():
                instance.state[bomb][group] = TreatmentStats(
                    z1=OutcomeStats(**tg_dict['z1']),
                    z0=OutcomeStats(**tg_dict['z0']),
                )
        for bomb, count_dict in data.get('counts', {}).items():
            instance.counts[bomb] = DraftCounts(**count_dict)

        return instance


def enrich_bh(df, pvcol='p_value', alpha=0.05):
    """Benjamini-Hochberg over one p-value column, added in place."""
    # Imported here rather than at module scope: statsmodels is needed by this
    # helper alone, and a top-level import made the whole module unimportable
    # wherever it is absent.
    from statsmodels.stats.multitest import (  # type: ignore[import-not-found]
        multipletests,
    )

    reject, pvals_corrected, _, _ = multipletests(
        df[pvcol].values, alpha=alpha, method='fdr_bh'
    )
    df['p_value_bh'] = pvals_corrected
    df['is_significant_bh'] = reject

    golden_list = df[df['is_significant_bh']]
    print(f"Number of pairs surviving FDR correction: {len(golden_list)}")
    return df

"""Phased ATT baselines for the 17lands causal benchmark, scored on pack 3.

A copy of `baseline_methods.py`, kept separate so the original stays available
for comparison. One decision drives every difference: **the outcome is counted
over pack 3 alone**, as an absolute count. See *Outcome window* below.

Reorganisation of `baselines_reference.py` into separable phases, targeting the
ATT (average treatment effect on the treated) rather than the ATE:

    Phase 1  DraftDataExtractor   streams the drafts CSV once and materialises,
                                  per CardA, a table of treatment,
                                  covariates and raw outcome counts. This is the
                                  slow pass; rerun it only when the covariate set
                                  changes.
    Phase 2  BaselineEnricher     takes a df of (card_a, group_b) pairs and
                                  returns, via `get_estimates`, one frame per
                                  estimator with columns card_a, group_b, value.
                                  `enrich_pairs_estimates` joins those onto the
                                  pairs frame, one column per method -- a
                                  separate step, so an expensive run can be kept
                                  and merged later.

A scoring phase is deliberately absent; it will be added later against the
reference ATT already carried by the pairs df.

Estimand
--------
ATT is `E[Y(1) - Y(0) | T=1]`; the whole problem is imputing `E[Y(0)|T=1]`.
Relative to the ATE code in `baselines_reference.py`:

  * the naive difference and the OLS coefficient on T are unchanged -- the latter
    is the linear S-learner, which asserts a constant effect and so estimates ATE
    and ATT identically;
  * stabilized IPW leaves the treated arm alone (it is already the target
    population) and re-weights only the controls onto it with the odds weight
    `e(X)/(1-e(X))`. The `P(T=1)` stabilization constant cancels in the
    self-normalized (Hajek) ratio, so what survives is the self-normalization;
  * matching keeps only the treated -> control direction;
  * outcome models average the imputed ITE over treated rows only.

Outcome window
--------------
**Y is the count of group-B picks in pack 3**, every pick of it, and nothing
else. `baseline_methods` counts `(pack 2, pick > k) | pack 3`; six of those
picks come from packs the drafter has already seen, including their own opener
returning on the wheel. CardA occupying a slot in that booster displaces
whatever card would otherwise have been there, so the treatment moves those
picks mechanically, with no behavioural response and no covariate able to adjust
it away. Pack 3 is a fresh booster round, collated independently, so it is clean
of that channel.

Two consequences, both simplifications:

  * **No rate normalisation.** The horizon problem that forced `count / horizon`
    was that a decision at pack 2 pick `k` had a `2P-1-k` window, so pooling
    rows at different picks compared incomparable outcomes. With the window
    fixed at pack 3 that problem cannot arise, so the raw count is already
    comparable. Y is an absolute count, on the same
    scale as `ivexp`'s `late`, and `metrics.truth_divisor` is 1.
  * **CardA is no longer excluded from its own group.** That subtraction existed
    because the treatment and its outcome shared the pack-2 window. Pack 3 is
    independent of pack 2, so a second copy taken there is an ordinary outcome.
    `self_count` is gone from storage with it.

`horizon` is still stored, now as the realised pack-3 length. It is metadata,
not a divisor: `horizon_min == horizon_max` says whether drafts are truncated,
which is the only thing that makes raw counts incomparable across drafters. A
draft with no pack 3 at all is dropped rather than contributing a manufactured
zero.

Control groups
--------------
A row is emitted whether or not CardA was in the pack, flagged by
`not_offered`. Such a row has a structurally zero propensity, which is harmless
for an ATT -- an ATT needs `e < 1` on the treated, not `e > 0` everywhere (that
is an ATE requirement). In exchange the outcome models get far more control data
with which to estimate the control surface, which is the whole reason those rows
are admitted.

**The propensity is fitted on the offered rows only**, and `e = 0` outright
elsewhere. Only offered rows carry information about who *takes* CardA rather
than declines it -- 1,971 of 107,502 on a measured card -- so fitting on all of
them spends the model's capacity on a distinction it already knows, and leaves
each structural zero a small non-zero `e` which, multiplied across 105k rows,
took a fifth of the control weight. The estimand is untouched: treated is a
subset of offered by construction, so `treated_pos` is identical either way and
only the control pool moves. `not_offered` is dropped from the propensity design
(it is constant across the fit sample) but stays in the design for the outcome
models, which use every row.

The IPW clip is one-sided (`e <= 1 - clip`, no floor) for the same reason: only
`e -> 1` can make a control weight explode, and a floor would force the correct
zeros back up.

The sample
----------
**One row per draft: pack 2, pick 0.** `baseline_methods` also carried a `p2any`
variant over every pack-2 decision, and it is gone. Its reason for existing was
sample size, and its cost was that surviving to pick `k` without CardA being
taken signals colour-openness -- so its offer is endogenous, its estimand is
take-now-vs-defer rather than the gold standard's, and it was never comparable
to `p2p1` anyway. With the instrument fixed at pack 2 pick 1 there is one
sample, and estimate columns are named for the estimator alone.

Drafters who already took CardA in pack 1 are excluded. That mirrors `ivexp`, so
the gold standard and the baselines describe the same population; without it a
prior holder would be a control while already holding the treatment.

`not_offered` means "CardA was not in this drafter's pack 2 booster". It
predicts the current selection -- a card absent from the pack cannot be taken --
so it forces that row's propensity to ~0. Because pack 2 pick 1 is a fresh
booster, the offer is independent of the pool, which is what licenses fitting
`e` on the offered rows and predicting it everywhere.
"""

import csv
import json
import logging
import os
import zlib
from abc import ABC, abstractmethod
from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import (
    Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional,
    Sequence, Set, Tuple, Union,
)

import numpy as np
import pandas as pd
from sklearn.ensemble import (  # type: ignore[import-untyped]
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (  # type: ignore[import-untyped]
    LogisticRegression,
    Ridge,
)
from sklearn.model_selection import (  # type: ignore[import-untyped]
    KFold,
    StratifiedKFold,
)
from sklearn.neighbors import NearestNeighbors  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from rndseq import RndSeq


######################
# Card metadata and the covariate specification
######################

# The five Magic colors, used for deterministic column ordering.
COLORS = ['W', 'U', 'B', 'R', 'G']


@dataclass
class CovariateSpec:
    """Which covariates to extract from a draft history.

    Derived automatically from a card's metadata by build_covariate_spec, but the
    caller may override it.
    """
    colors: List[str]
    include_num_colors: bool = True
    include_curve: bool = True
    include_creatures: bool = True
    include_instants: bool = False
    include_sorceries: bool = False
    include_artifacts: bool = False
    include_enchantments: bool = False


def build_covariate_spec(card_meta: Dict[str, Any]) -> CovariateSpec:
    """Derive a CovariateSpec from a card's raw (scryfall-like) metadata.

    The colors of interest default to CardA's own colors; the type-inclusion
    flags keep their defaults (num_colors, curve, creatures on).
    """
    colors = [c for c in card_meta.get('colors', []) if c in COLORS]
    return CovariateSpec(colors=colors)


def _card_colors(meta: Optional[Dict[str, Any]]) -> set:
    """Set of Magic colors for a card, from its metadata."""
    if not meta:
        return set()
    return {c for c in meta.get('colors', []) if c in COLORS}


def _mana_symbols(meta: Optional[Dict[str, Any]]) -> List[str]:
    """The 'mana_cost' string split into its individual symbols.

    '{2}{R}{W}' -> ['2', 'R', 'W']. Shared by the cost parse and the pip count so
    the two can never disagree about what a symbol is.
    """
    if not meta:
        return []
    mana_cost = meta.get('mana_cost') or ''
    return [s.strip() for s in mana_cost.replace('}', '{').split('{') if s.strip()]


def _parsed_mana_value(meta: Optional[Dict[str, Any]]) -> float:
    """Mana value recovered from 'mana_cost' alone.

    Generic numeric symbols are summed; every other symbol (colored, hybrid,
    phyrexian) counts as 1. 'X' and unknown/empty costs contribute 0. This is the
    fallback for card data without scryfall's `cmc` -- notably the synthetic test
    fixtures, which carry only colors/mana_cost/type_line.
    """
    total = 0.0
    for symbol in _mana_symbols(meta):
        # Hybrid/phyrexian like '2/W' or 'W/P': take the numeric part if any.
        numeric = next((p for p in symbol.split('/') if p.isdigit()), None)
        if numeric is not None:
            total += int(numeric)
        elif symbol == 'X':
            continue
        else:
            total += 1
    return total


def _mana_value(meta: Optional[Dict[str, Any]]) -> float:
    """A card's mana value, preferring scryfall's authoritative `cmc`.

    `cmc` is exact where the hand parse only approximates -- it already resolves
    hybrid and phyrexian costs, and counts 'X' as 0 the same way. The parse stays
    as the fallback so card data lacking the field still works.
    """
    if not meta:
        return 0.0
    cmc = meta.get('cmc')
    if isinstance(cmc, (int, float)):
        return float(cmc)
    return _parsed_mana_value(meta)


def _mana_pips(meta: Optional[Dict[str, Any]]) -> Dict[str, int]:
    """Colored mana symbols per color, i.e. the card's color *requirements*.

    This is the axis mana value throws away: '{4}{R}' and '{R}{R}{R}' have mana
    values 5 and 3, but the first asks for one red source and the second for a
    deck that is essentially mono-red. A drafter can splash the former and can
    never cast the latter, which is the dominant question when deciding whether
    to take a bomb.

    A hybrid '{W/U}' counts one pip toward *each* of its colors, matching how
    Magic's own devotion works; a consequence is that summing the per-color
    counts slightly over-counts a hybrid card's total requirement. Phyrexian
    '{W/P}' counts as W. Generic, 'X', colorless '{C}' and snow '{S}' contribute
    nothing.
    """
    pips: Dict[str, int] = {}
    for symbol in _mana_symbols(meta):
        if symbol.isdigit() or symbol == 'X':
            continue
        for part in symbol.split('/'):
            if part in COLORS:
                pips[part] = pips.get(part, 0) + 1
    return pips


def _has_type(meta: Optional[Dict[str, Any]], keyword: str) -> bool:
    """Whether a card's type_line contains the given type keyword."""
    if not meta:
        return False
    return keyword in (meta.get('type_line') or '')


def _is_land(meta: Optional[Dict[str, Any]]) -> bool:
    """Whether a card is a land, which several statistics must exclude."""
    return _has_type(meta, 'Land')


def _color_identity(meta: Optional[Dict[str, Any]]) -> FrozenSet[str]:
    """A card's color identity, which is what makes lands legible as fixing.

    A dual land has `colors == []` -- it costs nothing to play -- but a
    `color_identity` naming what it taps for. That is the only signal in the
    metadata that distinguishes a land which fixes CardA's colors from one that
    does not, and fixing is what licenses splashing an off-color bomb. Falls back
    to `colors` for data without the field.
    """
    if not meta:
        return frozenset()
    identity = meta.get('color_identity')
    if identity is None:
        identity = meta.get('colors', [])
    return frozenset(c for c in identity if c in COLORS)


def _is_premium(meta: Optional[Dict[str, Any]]) -> bool:
    """Whether a card is rare or mythic -- a crude proxy for prior investment.

    A drafter already holding two mythics elsewhere is the one who refuses a bomb,
    and pip counting scores a rare exactly like a common.
    """
    if not meta:
        return False
    return (meta.get('rarity') or '') in ('rare', 'mythic')


def _offered(value: Optional[str]) -> bool:
    """Whether a pack_card_<name> cell says the card was in the pack.

    Compares as a count rather than against '1', so packs holding two or more
    copies are not silently dropped.
    """
    if value is None or value == '':
        return False
    try:
        return float(value) > 0
    except ValueError:
        return False


# Card type keywords, paired with the CovariateSpec flag that enables them.
TYPE_FLAGS: Tuple[Tuple[str, str, str], ...] = (
    ('n_creatures', 'Creature', 'include_creatures'),
    ('n_instants', 'Instant', 'include_instants'),
    ('n_sorceries', 'Sorcery', 'include_sorceries'),
    ('n_artifacts', 'Artifact', 'include_artifacts'),
    ('n_enchantments', 'Enchantment', 'include_enchantments'),
)

# Mana-value buckets for the curve. `avg_cmc` reports where a drafter's curve
# sits; these report its *shape*, which is what makes a glut visible -- a drafter
# already holding four five-drops values another expensive card less. Mana values
# of 0 and 1 share a bucket, and everything from 6 up shares the last.
MANA_VALUE_BUCKETS: Tuple[Tuple[str, int, int], ...] = (
    ('n_mv1', 0, 1),
    ('n_mv2', 2, 2),
    ('n_mv3', 3, 3),
    ('n_mv4', 4, 4),
    ('n_mv5', 5, 5),
    ('n_mv6plus', 6, 99),
)

# Pips at or above this in one color count that color as a real commitment, as
# opposed to a splash or a stray early pick. Used only for num_colors_committed.
COMMITTED_PIPS = 3

# Columns of an extracted frame that are not covariates.
#
# `n_prior` and `pick_number` are here rather than dropped from the frame: both
# are constant now that the sample is pack 2 pick 0 of complete drafts alone
# (one full pack, and 0), so as covariates they are zero-variance columns that
# no model can use. They stay *in the frame* because `frame['n_prior'].nunique()`
# is the diagnostic for whether the incomplete-draft filter is doing its job.
NON_COVARIATE_COLUMNS = frozenset({'T', 'horizon', 'n_prior', 'pick_number'})

# The outcome window: pack 3, every pick (0-indexed pack number).
OUTCOME_PACK = 2

# Warn above this share of drafts dropped as incomplete. See
# `_is_complete_draft`; the rule assumes all three packs come from the same
# product, and a set whose packs are genuinely unequal would otherwise discard
# the corpus with nothing in the output to show it.
INCOMPLETE_WARN_RATE = 0.05

_TOO_MANY_INCOMPLETE = (
    "dropped %d of %d drafts (%.1f%%) as incomplete, above the %.0f%% alarm."
    " The rule requires every pack up to pack %d to have as many picks as the"
    " longest pack in the draft; a set whose packs are genuinely unequal, or a"
    " changed export schema, would look like this."
)


def _is_complete_draft(rows: List[Dict[str, Any]]) -> bool:
    """Whether a draft ran to the end of the outcome pack.

    Take the longest pack the draft has and require every pack up to
    `OUTCOME_PACK` to match it -- no external pack size is needed, since all
    three boosters come from the same product. Picks are sequential and cannot
    be skipped, so abandonment truncates a suffix: reaching the last pick of
    pack 3 implies packs 1 and 2 are whole, which is why one equality replaces
    three separate length checks. A mid-draft export hole fails it too.

    An abandoned draft is invalid on both ends. Its pack-3 window is short, so
    `Y` -- a raw count -- is mechanically smaller; and its pack-1 history is
    short, so the covariates are measured over a different prefix. `horizon` is
    in `NON_COVARIATE_COLUMNS`, so nothing adjusts for the first, and
    abandonment is plausibly non-random (a disengaged or losing player), which
    makes this potential confounding rather than noise.

    `ivexp.is_complete_draft` is the same rule. The two modules are deliberately
    independent, and both must apply it or they describe different populations.
    """
    lengths = Counter(int(row['pack_number']) for row in rows)
    if not lengths:
        return False
    expected = max(lengths.values())
    return all(lengths.get(pack, 0) == expected
               for pack in range(OUTCOME_PACK + 1))

# Per-card entry at one decision: (card_a, in_risk_set, T, not_offered).
# Everything else about the decision is shared.
_CardEntry = Tuple[str, bool, float, float]

# Columns of the per-CardA table. Deliberately a table rather than
# three fixed fields: anything genuinely about *this card at this decision*
# (pack-context covariates, say) belongs here too.
PER_CARD_COLUMNS = ('decision_id', 'T', 'not_offered')


######################
# Phase 1 -- extraction
######################


class _HistoryAccumulator:
    """Incremental prefix statistics over the picks made so far in a draft.

    Every covariate is a function of a running total over the picks preceding the
    decision, so one ordered walk suffices instead of re-scanning the history per
    (decision x CardA). The accumulator is CardA-independent -- it tracks all five
    colors and every card type, and a CovariateSpec merely *selects* a subset --
    so nothing is lost by sharing it across cards.
    """

    def __init__(self, card_db: Dict[str, Dict[str, Any]]):
        self._card_db = card_db
        self.n_prior = 0
        self.n_known = 0
        self.n_spells = 0
        self.n_lands = 0
        self.n_rares = 0
        self.sum_cmc = 0.0
        self.color_counts: Dict[str, int] = {color: 0 for color in COLORS}
        self.pip_counts: Dict[str, int] = {color: 0 for color in COLORS}
        self.fix_counts: Dict[str, int] = {color: 0 for color in COLORS}
        self.curve_counts: Dict[str, int] = {name: 0 for name, _, _ in MANA_VALUE_BUCKETS}
        self.type_counts: Dict[str, int] = {name: 0 for name, _, _ in TYPE_FLAGS}
        self.seen_colors: set = set()

    def add(self, card: str):
        """Fold one pick into the running totals."""
        # n_prior counts every pick, including cards missing from card_db, which
        # contribute nothing to the color/type/curve totals.
        self.n_prior += 1
        meta = self._card_db.get(card)
        if meta is None:
            return
        self.n_known += 1
        colors = _card_colors(meta)
        self.seen_colors |= colors
        for color in colors:
            self.color_counts[color] += 1
        for color, count in _mana_pips(meta).items():
            self.pip_counts[color] += count
        if _is_premium(meta):
            self.n_rares += 1
        for name, keyword, _flag in TYPE_FLAGS:
            if _has_type(meta, keyword):
                self.type_counts[name] += 1

        if _is_land(meta):
            # A land has no mana value of its own, so folding it into the curve
            # would drag avg_cmc toward zero and conflate "cheap deck" with
            # "picked lands". Its color identity is what it taps for, which is
            # the only fixing signal the metadata carries.
            self.n_lands += 1
            for color in _color_identity(meta):
                self.fix_counts[color] += 1
            return

        self.n_spells += 1
        mana_value = _mana_value(meta)
        self.sum_cmc += mana_value
        for name, low, high in MANA_VALUE_BUCKETS:
            if low <= mana_value <= high:
                self.curve_counts[name] += 1
                break

    def copy(self) -> '_HistoryAccumulator':
        """Snapshot the current state, so a decision can be scored later."""
        clone = _HistoryAccumulator(self._card_db)
        clone.n_prior = self.n_prior
        clone.n_known = self.n_known
        clone.n_spells = self.n_spells
        clone.n_lands = self.n_lands
        clone.n_rares = self.n_rares
        clone.sum_cmc = self.sum_cmc
        clone.color_counts = dict(self.color_counts)
        clone.pip_counts = dict(self.pip_counts)
        clone.fix_counts = dict(self.fix_counts)
        clone.curve_counts = dict(self.curve_counts)
        clone.type_counts = dict(self.type_counts)
        clone.seen_colors = set(self.seen_colors)
        return clone

    def snapshot(self) -> Dict[str, float]:
        """The full covariate state, with no CardA-specific selection applied.

        Every value here is a property of the draft history alone, so it is
        shared by every CardA at this decision; `CovariateSpec` picks a subset of
        these columns at assembly time (see `DraftDataExtractor.get_dataframe`),
        and the CardA-relative aggregates are built there from the per-color
        columns below.
        """
        pips_total = sum(self.pip_counts.values())
        committed = sum(
            1 for count in self.pip_counts.values() if count >= COMMITTED_PIPS
        )
        feats: Dict[str, float] = {
            'n_prior': float(self.n_prior),
            'num_colors': float(len(self.seen_colors)),
            # num_colors counts every color ever touched, so after a full pack it
            # saturates near 5 for almost everyone and carries little. This counts
            # only colors the drafter is actually invested in.
            'num_colors_committed': float(committed),
            # Divided by non-land picks: see the land branch in `add`.
            'avg_cmc': (self.sum_cmc / self.n_spells) if self.n_spells else 0.0,
            'pips_total': float(pips_total),
            'n_spells': float(self.n_spells),
            'n_lands': float(self.n_lands),
            'n_rares': float(self.n_rares),
        }
        for color in COLORS:
            feats[f'n_{color}'] = float(self.color_counts[color])
            feats[f'pips_{color}'] = float(self.pip_counts[color])
            feats[f'fix_{color}'] = float(self.fix_counts[color])
        for name, _low, _high in MANA_VALUE_BUCKETS:
            feats[name] = float(self.curve_counts[name])
        for name, _keyword, _flag in TYPE_FLAGS:
            feats[name] = float(self.type_counts[name])
        return feats


# The decision: pack 2, pick 0 (0-indexed pack number).
ASSIGNMENT_PACK = 1

# Bumped whenever the on-disk layout changes shape. `from_file` refuses a file
# it was not written for rather than reconstructing a wrong store.
STORE_FORMAT_VERSION = 1

# Candidate integer widths for `_narrow`, smallest first.
_INT_DTYPES = (np.int8, np.int16, np.int32, np.int64)


def _narrow(values: array) -> np.ndarray:
    """The smallest dtype that holds `values` **exactly**.

    Nearly every stored column is a small count sitting in a float64: at the
    real shape 41 of 42 decision columns are integral (only `avg_cmc` is not)
    and span 0..12, and the per-card table is a row index plus two 0/1 flags.
    Writing them as int8/int16/int32 is a ~4x saving on disk -- the same as
    compressing, but with no CPU on either side and exact by construction
    rather than by how well zlib happened to do.

    A column that is not integral, or not finite, or too wide for int64 stays
    float64. The check is `v == round(v)` over finite values, so the narrowing
    can never lose a fraction.

    **The int64 range is tested before the cast, not after.** `astype(np.int64)`
    on a float beyond int64's range wraps silently rather than raising, so
    checking afterwards would compare numbers that had already been corrupted.
    An integral float inside the range converts exactly in both directions --
    it is an integer that float64 provably represents, so int64 holds it and
    the trip back reproduces it.
    """
    v = np.asarray(values)
    if v.dtype.kind == 'f':
        integral = (
            v.size > 0 and bool(np.isfinite(v).all())
            and bool(np.array_equal(v, np.round(v)))
        )
        limits = np.iinfo(np.int64)
        if not integral or (
            v.size and (v.min() < limits.min or v.max() > limits.max)
        ):
            return v.astype(np.float64)
        v = v.astype(np.int64)
    low, high = (int(v.min()), int(v.max())) if v.size else (0, 0)
    for dtype in _INT_DTYPES:
        info = np.iinfo(dtype)
        if info.min <= low and high <= info.max:
            return v.astype(dtype)
    return v.astype(np.int64)


def _rebuild(values: np.ndarray, typecode: str) -> array:
    """A stored column back as the `array` the extractor holds in memory.

    Via `frombytes` rather than element-wise, which matters at 24M values:
    iterating would cost seconds and allocate a Python object per element,
    which is precisely the cost this format exists to avoid.
    """
    target = np.float64 if typecode == 'd' else np.int64
    out = array(typecode)
    out.frombytes(values.astype(target).tobytes())
    return out


@dataclass
class ExtractionConfig:
    """Extraction settings. `set_code` is a label; it performs no filtering.

    `split_rnd_seed` turns on the split mode: one seeded draw per draft, the
    extractor keeps `x >= 0.5` and `ivexp` with the same seed keeps the rest.
    The draw happens before every other filter, so a kept draft can still be
    dropped as incomplete or as a prior holder. None means no split.
    """
    set_code: str
    split_rnd_seed: Optional[int] = None


class DraftDataExtractor:
    """Phase 1: stream the drafts CSV into a global + per-CardA column store.

    Storage is split by what actually varies. Almost everything about a decision
    -- the outcome horizon, the pick index, and every covariate of the draft
    history -- is identical for every CardA, because the history does not depend
    on which card is being treated. `CovariateSpec` only decides *which* of those
    columns a card exposes, so it is a column selector applied at assembly time.

    Only two things differ per CardA: whether it was picked (`T`) and whether it
    was in the pack (`not_offered`). The outcome counts are pack-3 totals and so
    are shared by every CardA *and* by every decision in the draft.

        self.decisions   column -> array, one entry per emitted decision
        self.rows        card_a -> column -> array, keyed by `decision_id`
                         into `self.decisions`

    Storing the group vector once per decision rather than once per CardA is the
    difference between ~8.5 GB and ~170 MB at 100k drafts x 40 CardA x 35 groups.
    Columns are `array('d')`/`array('q')` rather than per-row dicts, which also
    keeps tens of millions of GC-traversable objects out of existence.

    One row per draft: pack 2 pick 0, excluding drafters who already took CardA
    in pack 1.
    """

    def __init__(
        self,
        config: ExtractionConfig,
        card_db: Dict[str, Dict[str, Any]],
        grouping: Dict[str, str],
        card_a_names: Sequence[str],
    ):
        self.config = config
        self.card_db = card_db
        self.grouping = grouping
        self.card_a_names = list(card_a_names)
        self.split = (RndSeq(config.split_rnd_seed, take_below=False)
                      if config.split_rnd_seed is not None else None)

        self.specs: Dict[str, CovariateSpec] = {
            card_a: build_covariate_spec(card_db.get(card_a, {}))
            for card_a in self.card_a_names
        }
        # Sorted for deterministic Y_ column order.
        self.groups: List[str] = sorted(set(grouping.values()))

        self.decisions: Dict[str, array] = {
            column: array('d') for column in self.decision_columns()
        }
        self.rows: Dict[str, Dict[str, array]] = {
            card_a: {
                column: array('q' if column == 'decision_id' else 'd')
                for column in PER_CARD_COLUMNS
            }
            for card_a in self.card_a_names
        }
        # (n_decisions, {column: ndarray}) -- rebuilt when the table grows.
        self._decision_cache: Optional[Tuple[int, Dict[str, np.ndarray]]] = None
        # Draft-level tallies for the incomplete-draft alarm, over drafts that
        # reached the assignment pack. CardA-independent, so counted once.
        self.n_drafts = 0
        self.n_incomplete = 0

    def decision_columns(self) -> List[str]:
        """Global per-decision columns, in storage order.

        All of these are properties of the draft history alone, so they are
        CardA-independent and stored once per decision rather than once per
        (decision x CardA). The CardA-relative columns are derived from them at
        assembly time -- see `_relative_columns`.
        """
        columns = [
            'horizon', 'pick_number', 'n_prior', 'num_colors',
            'num_colors_committed', 'avg_cmc', 'pips_total', 'n_spells',
            'n_lands', 'n_rares',
        ]
        columns += [f'n_{color}' for color in COLORS]
        columns += [f'pips_{color}' for color in COLORS]
        columns += [f'fix_{color}' for color in COLORS]
        columns += [name for name, _low, _high in MANA_VALUE_BUCKETS]
        columns += [name for name, _keyword, _flag in TYPE_FLAGS]
        columns += [f'grp_{group}' for group in self.groups]
        return columns

    # ---- streaming ----
    def process_drafts_file(self, csv_path: str):
        """Extract from a drafts CSV, reading it row by row."""
        with open(csv_path, mode='r', encoding='utf-8') as file:
            self.process_drafts(csv.DictReader(file))

    def process_drafts(self, reader: Iterable[Dict[str, Any]]):
        """Extract from any row iterable, grouping contiguous rows by draft_id."""
        current_draft_id = None
        current_draft_rows: List[Dict[str, Any]] = []

        for ridx, row in enumerate(reader):
            draft_id = row['draft_id']
            if ridx % 200000 == 0:
                logging.info("baseline extraction processed %d rows", ridx)
            if draft_id != current_draft_id:
                if current_draft_rows:
                    self._process_single_draft(current_draft_rows)
                current_draft_id = draft_id
                current_draft_rows = []
            current_draft_rows.append(row)

        if current_draft_rows:
            self._process_single_draft(current_draft_rows)

        self._warn_if_truncated()

    def _warn_if_truncated(self):
        """Alarm when the completeness rule discarded an implausible share."""
        if not self.n_drafts:
            return
        rate = self.n_incomplete / self.n_drafts
        if rate > INCOMPLETE_WARN_RATE:
            logging.warning(
                _TOO_MANY_INCOMPLETE, self.n_incomplete, self.n_drafts,
                100 * rate, 100 * INCOMPLETE_WARN_RATE, OUTCOME_PACK,
            )

    def _process_single_draft(self, rows: List[Dict[str, Any]]):
        """Emit every pack-2 decision row for one draft in two ordered passes."""
        if self.split is not None and not self.split.get_next_bool():
            return
        ordered = sorted(
            rows, key=lambda r: (int(r['pack_number']), int(r['pick_number']))
        )
        position = next(
            (idx for idx, row in enumerate(ordered)
             if int(row['pack_number']) == ASSIGNMENT_PACK
             and int(row['pick_number']) == 0),
            None,
        )
        if position is None:
            return
        self.n_drafts += 1

        # An abandoned draft is invalid on both ends -- a short outcome window
        # and a short covariate prefix. `ivexp` drops the same drafts; if only
        # one side filtered, the two would describe different populations and
        # the `n_valid` == row-count cross-check would break.
        if not _is_complete_draft(ordered):
            self.n_incomplete += 1
            return

        # The outcome is pack 3 in its entirety, so it is the *same* window for
        # every decision in the draft. That is what removes the horizon problem
        # the rate outcome existed to solve, and it replaces the suffix
        # accumulator with one count per draft.
        group_counts: Dict[str, int] = defaultdict(int)
        horizon = 0
        for row in ordered:
            if int(row['pack_number']) != OUTCOME_PACK:
                continue
            group = self.grouping.get(row['pick'])
            if group is not None:
                group_counts[group] += 1
            horizon += 1
        if horizon <= 0:
            # Unreachable once the completeness rule above has run -- a draft
            # with no pack 3 fails it first. Kept as a guard because `horizon`
            # is the denominator-free outcome window and a zero here would
            # contribute a manufactured zero to every group.
            return

        # Which cards the drafter already held. The decision is the first
        # pack-2 row, so "picked at a strictly earlier position" is exactly
        # "picked in pack 1" -- the same exclusion `ivexp` applies, and both
        # sides must apply it or they describe different populations.
        held_before = {row['pick'] for row in ordered[:position]}

        # The prefix state at the decision, accumulated over those same rows.
        prefix = _HistoryAccumulator(self.card_db)
        for row in ordered[:position]:
            prefix.add(row['pick'])

        built = self._build_decision(
            ordered[position], held_before, prefix, group_counts, horizon,
        )
        if built is not None:
            self._store_decision(*built)

    def _store_decision(
        self, decision_values: Dict[str, float], entries: List[_CardEntry]
    ):
        """Append one decision plus its per-card entries to the column store."""
        # Work out the appends first: a decision every CardA excludes must not
        # leave an orphan row in the global table.
        appends = [
            (card_a, treated, not_offered)
            for card_a, in_risk_set, treated, not_offered in entries
            if in_risk_set
        ]
        if not appends:
            return

        decision_id = len(self.decisions['horizon'])
        for column, value in decision_values.items():
            self.decisions[column].append(value)

        for card_a, treated, not_offered in appends:
            table = self.rows[card_a]
            table['decision_id'].append(decision_id)
            table['T'].append(treated)
            table['not_offered'].append(not_offered)

    def _build_decision(
        self,
        decision: Dict[str, Any],
        held_before: Set[str],
        prefix: _HistoryAccumulator,
        group_counts: Dict[str, int],
        horizon: int,
    ) -> Optional[Tuple[Dict[str, float], List[_CardEntry]]]:
        """The shared decision record plus one entry per CardA.

        An entry is produced whether or not CardA was in the pack; `not_offered`
        records which. Such rows have a structurally zero propensity, which is
        harmless for an ATT -- the odds weight `e/(1-e)` sends them to zero --
        and they give the outcome models far more control data. Being
        un-offered does not exclude the entry; **already holding CardA does**,
        which is `held_before` and mirrors `ivexp`'s prior-holder filter.

        `group_counts` and `horizon` describe pack 3, so unlike in
        `baseline_methods` they are not a suffix that shrinks as the pack is
        drafted.
        """
        pick = decision['pick']

        decision_values: Dict[str, float] = {
            'horizon': float(horizon),
            'pick_number': 0.0,
        }
        decision_values.update(prefix.snapshot())
        for group in self.groups:
            decision_values[f'grp_{group}'] = float(group_counts.get(group, 0))

        entries: List[_CardEntry] = []
        for card_a in self.card_a_names:
            offered = _offered(decision.get(f'pack_card_{card_a}'))
            entries.append((
                card_a,
                card_a not in held_before,
                1.0 if pick == card_a else 0.0,
                0.0 if offered else 1.0,
            ))

        return decision_values, entries

    # ---- persistence ----
    def to_file(self, path: Any) -> str:
        """Write the extracted store to an uncompressed `.npz`, returning its path.

        The slow CSV pass is the only expensive step in this module, so losing
        the extractor to a kernel restart is the one thing worth guarding
        against. Everything needed to reconstruct it is written: the column
        store, plus config, grouping, `card_db` and `card_a_names` as one JSON
        blob under the `meta` key.

        **`card_db` is embedded rather than re-supplied on load.** `self.specs`
        derives from it and decides which covariate columns `get_dataframe`
        exposes, so reloading against a different `card_db` would silently
        change the frames. Embedding costs a megabyte or two and makes a
        reloaded extractor provably the one that was written.

        Uncompressed on purpose. The size win comes from `_narrow` picking
        exact integer dtypes (~4x), which costs nothing on either side;
        deflate on top would reach maybe 20% further while adding seconds to
        every read and write.

        Cards are keyed by *position* in `card_a_names`, not by name -- MTG
        names carry commas, apostrophes and `//`, and npz keys become entries
        in a zip.
        """
        meta = {
            'version': STORE_FORMAT_VERSION,
            'set_code': self.config.set_code,
            'split_rnd_seed': self.config.split_rnd_seed,
            'card_a_names': self.card_a_names,
            'grouping': self.grouping,
            'card_db': self.card_db,
            'groups': self.groups,
            'decision_columns': self.decision_columns(),
            'per_card_columns': list(PER_CARD_COLUMNS),
            # Provenance only -- how many drafts the pass saw and how many the
            # completeness rule discarded. Not part of the schema check.
            'n_drafts': self.n_drafts,
            'n_incomplete': self.n_incomplete,
        }
        payload: Dict[str, np.ndarray] = {
            'meta': np.frombuffer(
                json.dumps(meta).encode('utf-8'), dtype=np.uint8
            ),
        }
        for column, values in self.decisions.items():
            payload[f'd_{column}'] = _narrow(values)
        for position, card_a in enumerate(self.card_a_names):
            for column, values in self.rows[card_a].items():
                payload[f'r{position}_{column}'] = _narrow(values)

        target = str(path)
        if not target.endswith('.npz'):
            target += '.npz'
        with open(target, 'wb') as handle:
            np.savez(handle, **payload)
        logging.info(
            'wrote %s: %d decisions, %d cards, %.1f MB',
            target, len(self.decisions['horizon']), len(self.card_a_names),
            os.path.getsize(target) / 1e6,
        )
        return target

    @classmethod
    def from_file(cls, path: Any) -> 'DraftDataExtractor':
        """Reconstruct an extractor written by `to_file`.

        The store is rebuilt into the same `array` typecodes a fresh extraction
        produces, so the loaded object is indistinguishable from one that had
        just read the CSV -- `get_dataframe` included.

        **The schema is validated, not assumed.** `decision_columns()` has
        changed more than once as covariates were added and removed, and a file
        written before such a change would otherwise reload without complaint
        and yield frames with the wrong columns. Version, column list and group
        list all have to match what this code would produce.
        """
        with np.load(str(path)) as data:
            meta = json.loads(bytes(data['meta']).decode('utf-8'))
            if meta.get('version') != STORE_FORMAT_VERSION:
                raise ValueError(
                    f"{path} was written by format version "
                    f"{meta.get('version')!r}, this code reads "
                    f'{STORE_FORMAT_VERSION}. Re-run the extraction.'
                )
            if list(meta['per_card_columns']) != list(PER_CARD_COLUMNS):
                raise ValueError(
                    f"{path} stores per-card columns {meta['per_card_columns']}"
                    f', this code expects {list(PER_CARD_COLUMNS)}.'
                )

            extractor = cls(
                ExtractionConfig(set_code=meta['set_code'],
                                 split_rnd_seed=meta.get('split_rnd_seed')),
                meta['card_db'], meta['grouping'], meta['card_a_names'],
            )
            if extractor.groups != list(meta['groups']):
                raise ValueError(
                    f'{path} was extracted over groups {meta["groups"]}, which '
                    f'differ from the stored grouping. The file is corrupt.'
                )
            expected = extractor.decision_columns()
            if expected != list(meta['decision_columns']):
                only_file = [c for c in meta['decision_columns']
                             if c not in set(expected)]
                only_code = [c for c in expected
                             if c not in set(meta['decision_columns'])]
                raise ValueError(
                    f'{path} has a stale covariate schema: columns only in the '
                    f'file {only_file}, only in this code {only_code}. Re-run '
                    'the extraction.'
                )

            extractor.decisions = {
                column: _rebuild(data[f'd_{column}'], 'd')
                for column in expected
            }
            extractor.rows = {
                card_a: {
                    column: _rebuild(
                        data[f'r{position}_{column}'],
                        'q' if column == 'decision_id' else 'd',
                    )
                    for column in PER_CARD_COLUMNS
                }
                for position, card_a in enumerate(meta['card_a_names'])
            }
            # Absent from a file written before the completeness filter, which
            # is provenance rather than a schema break -- default and move on.
            extractor.n_drafts = int(meta.get('n_drafts', 0))
            extractor.n_incomplete = int(meta.get('n_incomplete', 0))

        sizes = {len(v) for v in extractor.decisions.values()}
        if len(sizes) > 1:
            raise ValueError(
                f'{path} holds decision columns of unequal length {sorted(sizes)}'
            )
        logging.info(
            'loaded %s: %d decisions, %d cards',
            path, len(extractor.decisions['horizon']),
            len(extractor.card_a_names),
        )
        return extractor

    # ---- access ----
    def _decision_arrays(self) -> Dict[str, np.ndarray]:
        """Global decision columns as ndarrays, cached until the table grows."""
        size = len(self.decisions['horizon'])
        if self._decision_cache is None or self._decision_cache[0] != size:
            self._decision_cache = (
                size,
                {column: np.array(values, dtype=float)
                 for column, values in self.decisions.items()},
            )
        return self._decision_cache[1]

    def get_dataframe(self, card_a: str) -> pd.DataFrame:
        """Extracted table for one CardA, or an empty frame.

        Assembles the per-card table against the shared decision table: the
        CardA's `CovariateSpec` selects which covariate columns to expose. `Y`
        is the raw pack-3 count and **includes CardA itself**, since pack 3 is
        independent of the pack the treatment acts in.

        `spec` decides only which *type* counts are exposed. All five colors,
        all five per-color pip counts and the CardA-relative aggregates are
        emitted for every CardA, so the design has the same width whatever
        CardA is -- see the comment in the body for why.
        """
        table = self.rows.get(card_a)
        if table is None or not table['decision_id']:
            return pd.DataFrame()

        index = np.array(table['decision_id'], dtype=np.intp)
        decision = self._decision_arrays()
        spec = self.specs[card_a]

        out: Dict[str, np.ndarray] = {
            'T': np.array(table['T'], dtype=float),
            'horizon': decision['horizon'][index],
        }
        for column in sorted(f'Y_{group}' for group in self.groups):
            out[column] = decision[f'grp_{column[2:]}'][index]

        # **All five colors, not CardA's.** The design is CardA-relative but the
        # outcome is group-B-relative, and emitting `n_<c>` only for CardA's
        # colors left the outcome models with no information at all about the
        # others -- a white CardA against `group_b = B_Creature` could not see a
        # single black pick. That is a variance cost rather than a bias one
        # (`offcolor_pip_share` already carries the part that predicts `T`;
        # which color it is predicts `Y`), and variance is what the measured
        # overdispersion says is hurting. All ten columns were already stored,
        # so this needs no re-extraction.
        #
        # Per-color *fixing* stays in storage: `n_fix_oncolor` is its sum over
        # CardA's colors, and the five would add nothing the aggregate does not.
        for color in COLORS:
            out[f'n_{color}'] = decision[f'n_{color}'][index]
        for color in COLORS:
            out[f'pips_{color}'] = decision[f'pips_{color}'][index]
        out.update(self._relative_columns(spec, decision, index))
        if spec.include_num_colors:
            out['num_colors'] = decision['num_colors'][index]
            out['num_colors_committed'] = decision['num_colors_committed'][index]
        if spec.include_curve:
            out['avg_cmc'] = decision['avg_cmc'][index]
            for name, _low, _high in MANA_VALUE_BUCKETS:
                out[name] = decision[name][index]
        for name, _keyword, flag in TYPE_FLAGS:
            if getattr(spec, flag):
                out[name] = decision[name][index]
        out['n_lands'] = decision['n_lands'][index]
        out['n_rares'] = decision['n_rares'][index]
        out['n_prior'] = decision['n_prior'][index]
        out['pick_number'] = decision['pick_number'][index]
        out['not_offered'] = np.array(table['not_offered'], dtype=float)

        return pd.DataFrame(out)

    @staticmethod
    def _relative_columns(
        spec: CovariateSpec,
        decision: Dict[str, np.ndarray],
        index: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Covariates measured *relative to CardA's colors*.

        These are the columns that make a small propensity model viable. Per-color
        raw counts spend a parameter on each color merely because CardA happens to
        be in it, and they mis-handle a gold card -- an `{R}{W}` pick counts once
        under R and once under W with no way to tell that from two separate picks.
        A share against CardA's own colors states the same domain fact ("how
        committed is this drafter to *this card's* colors") in one column, exactly,
        for any CardA.

        Pips rather than card counts, because pips are the castability question:
        `{4}{R}` asks for one red source and `{R}{R}{R}` asks for a mono-red deck.

        `oncolor_pip_min_share` is the *weakest* of CardA's colors, and it is the
        column that makes a gold CardA legible. `n_oncolor_pips` sums across both
        colors, so a drafter with 10 red pips and 0 white scores identically to
        one with 5 and 5 -- yet only the second can cast an `{R}{W}` card. It is
        the mirror of `offcolor_pip_share`, which takes a `max` for the same
        reason from the other side.

        **It is emitted for every CardA and is 0 when CardA is mono-colored**,
        where `0` means *not applicable* rather than *no commitment*. Emitting it
        only for gold cards would be cheaper but worse on both counts: the design
        width would vary by CardA, and for a mono card the min over one color
        *is* the share, so the column would be a literal duplicate and therefore
        rank-deficient. A constant-0 column contributes 0 to the rank instead.
        """
        pips_total = decision['pips_total'][index]
        # Guard the ratio, not the numerator: a drafter whose picks are all lands
        # or all colorless has no pips at all, and 0/0 must read as 0 commitment
        # rather than nan.
        safe_total = np.where(pips_total > 0, pips_total, 1.0)

        on_colors = [c for c in spec.colors if c in COLORS]
        off_colors = [c for c in COLORS if c not in on_colors]

        oncolor_pips = sum(
            (decision[f'pips_{c}'][index] for c in on_colors),
            start=np.zeros(len(index)),
        )
        # A dual land fixing two of CardA's colors is counted under each, which is
        # deliberate: it is genuinely worth more than a single-color source.
        oncolor_fix = sum(
            (decision[f'fix_{c}'][index] for c in on_colors),
            start=np.zeros(len(index)),
        )
        # The *strongest* other color, not the sum: what deters taking an off-color
        # bomb is being committed somewhere specific, and a drafter scattered across
        # three colors is far likelier to take it than one settled in two.
        offcolor_pips = (
            np.max([decision[f'pips_{c}'][index] for c in off_colors], axis=0)
            if off_colors else np.zeros(len(index))
        )

        # The weakest of CardA's colors. Constant 0 for a mono-colored CardA,
        # where the min over one color would just repeat the share.
        min_pips = (
            np.min([decision[f'pips_{c}'][index] for c in on_colors], axis=0)
            if len(on_colors) > 1 else np.zeros(len(index))
        )

        return {
            'oncolor_pip_share': oncolor_pips / safe_total,
            'oncolor_pip_min_share': min_pips / safe_total,
            'n_oncolor_pips': oncolor_pips,
            'offcolor_pip_share': offcolor_pips / safe_total,
            'n_fix_oncolor': oncolor_fix,
        }


######################
# Phase 2 -- estimation context
######################


@dataclass
class Outcome:
    """One group's pack-3 count, tagged so model fits can be cached by group."""
    group_b: str
    values: np.ndarray


def covariate_columns(df: pd.DataFrame) -> List[str]:
    """Covariate columns of an extracted frame, in frame order."""
    return [
        c for c in df.columns
        if c not in NON_COVARIATE_COLUMNS and not c.startswith('Y_')
    ]


# The columns that only phase-6 extraction produces. Used to tell a frame that
# carries the CardA-relative features from one extracted before they existed,
# which must fall back rather than select an incidental subset.
RELATIVE_COVARIATES: Tuple[str, ...] = (
    'oncolor_pip_share',
    'oncolor_pip_min_share',
    'n_oncolor_pips',
    'offcolor_pip_share',
    'n_fix_oncolor',
)

# The propensity's covariates -- CardA-relative commitment, curve position, and
# two depth terms. Every propensity spec reads exactly this set, so the only
# thing that differs between them is the model class.
#
# Small on purpose: the propensity is fitted on offered rows only, which is
# ~1,971 rows with ~613 events on the measured card, so eight columns sit near
# 68 events per parameter for the logistic fit where the full design sat near 14.
#
# `n_prior` and `pick_number` are gone. Both are constant now that the sample is
# pack 2 pick 0 of complete drafts alone -- one full pack and 0 respectively, by
# construction rather than by observation once the incomplete-draft filter runs
# -- so they were zero-variance columns spending a slot each.
#
# `oncolor_pip_min_share` takes one of the freed slots. It is constant 0 for a
# mono-colored CardA and so still contributes nothing there, but for a gold one
# it is the only column that distinguishes 10-red-0-white from 5-and-5, which is
# exactly the drafter who can and cannot cast an `{R}{W}` bomb.
#
# `n_rares` takes the other. It is already in the outcome design; this puts it in
# the propensity too. Pip counting scores a rare the same as a common, but a
# drafter sitting on two mythics somewhere else is the one who declines a bomb.
# Sparse -- 1-2 after a single pack -- which is why it was left out before.
CORE_COVARIATES: Tuple[str, ...] = (
    'oncolor_pip_share',
    'oncolor_pip_min_share',
    'n_oncolor_pips',
    'offcolor_pip_share',
    'avg_cmc',
    'n_fix_oncolor',
    'num_colors_committed',
    'n_rares',
)

_VIEW_COLUMNS_MISSING = (
    "feature view names columns absent from the frame: %s. Available: %s."
    " A view that silently skipped them would fit on fewer covariates with"
    " nothing in the output to show it."
)


def _named_view(names: Sequence[str]) -> Callable[[Sequence[str]], List[str]]:
    """A view selecting a fixed column list, in the order the frame supplies it.

    Raises on a name the frame does not carry. Filtering by membership alone --
    the obvious implementation -- turns a typo or a renamed column into a
    silently smaller propensity, which shows up nowhere: no error, no warning,
    just a model fitted on fewer covariates than intended.

    The exception is a frame extracted before these covariates existed, which is
    a different situation from a mistake and must keep working. Detect it by
    `RELATIVE_COVARIATES` rather than by "did anything match": `avg_cmc` and
    `n_rares` are generic enough to appear in a hand-built fixture, so a partial
    match is the normal legacy case and not evidence of a typo. Such a frame
    falls back to every column it has, exactly as `scaled_design` does.
    """
    wanted = list(names)

    def select(x_cols: Sequence[str]) -> List[str]:
        if not any(c in set(x_cols) for c in RELATIVE_COVARIATES):
            return list(x_cols)
        missing = [c for c in wanted if c not in set(x_cols)]
        if missing:
            raise KeyError(_VIEW_COLUMNS_MISSING % (missing, list(x_cols)))
        return [c for c in x_cols if c in set(wanted)]

    return select


# How each propensity spec's `features` string resolves to a column subset. A
# registry rather than a branch so a new view cannot silently fall through to the
# wrong one -- the failure phase 2b fixed in `_build_matches`. One entry, since
# every spec reads `core`; that is what makes the model-class comparison clean,
# and the registry stays because the failure it prevents does not depend on how
# many views there are.
FEATURE_VIEWS: Dict[str, Callable[[Sequence[str]], List[str]]] = {
    'core': _named_view(CORE_COVARIATES),
}


# A profile is one model's constructor arguments, and **profile names are builder
# names, one to one** -- so every entry has exactly one model behind it, which is
# what lets `ProfileTuner` take a name and know what to fit and score.
#
# These are the values that were hardcoded in the builders, so the shipped
# `PROFILES` reproduces the previous output exactly.
#
# **Read at call time, never bound at import.** The builders stay nullary, so
# `OUTCOME_LEARNERS` and `PropensitySpec.build` are untouched; applying a tuned
# profile is a `PROFILES` update and takes effect immediately with no registry
# rebuild. Binding with `functools.partial` would freeze the dict at import and
# make that impossible.
#
#  `gbm_classifier` hold *equal values, not a shared entry*. One
# is a classifier on `T` over ~1,971 offered rows, the other a regressor on `Y`
# over the ~1,200 never-takers; the same capacity budget applies for the same
# sample-size reason, but that is a sensible default rather than a derivation, so
# tuning one must not silently move the other.
PROFILES: Dict[str, Dict[str, Any]] = {
    'gbm_classifier': {'max_iter': 100, 'learning_rate': 0.2, 'max_depth': 3, 
                       'min_samples_leaf': 60, 'random_state':42},                        
    'gbm': {'max_iter': 100, 'max_depth': 3, 'min_samples_leaf': 40,
            'learning_rate': 0.05, 'random_state': 42},
    'logit': {'C': 0.35, 'max_iter': 1000},
    'ridge': {'alpha': 1e-8},

    # --- causal forest (`cf_att`), three models, three profiles ---
    #
    # Each is keyed exactly to its own constructor so `**profile` expands
    # cleanly, and none of them shares a key with another by accident: the
    # forest's `min_samples_leaf` governs a within-leaf *contrast*, the
    # nuisances' govern plain prediction, and they are not the same quantity.
    #
    # All three fit on `ctx.support` -- ~1,971 rows at ~39% treated, not the
    # ~107k the `gbm` profile was sized for. That is why these are separate
    # profiles rather than reuses of `gbm` / `gbm_classifier`; see cf.md,
    # "Why the existing profiles cannot be reused".
    'forest': {
        # A *variance* budget, not capacity: bagged trees are grown
        # independently, so more never overfits. 2000 was sized for the 2,523
        # rows of the support-only fit, where each leaf contrast was extremely
        # noisy. With the control cap the sample is ~51k and the leaves are far
        # better populated, so 500 buys the same ensemble stability for a
        # quarter of the cost.
        'n_estimators': 500,
        # **Read this as treated-per-leaf, not rows.** At the cap's 2.50%
        # treated share a leaf of 500 holds ~12 treated and ~488 controls, and
        # `sigma^2 (1/n1 + 1/n0)` makes the contrast almost entirely
        # treated-driven. What that buys is tree shape:
        #
        #     leaves_per_tree = max_samples * honest * n1 / k = 0.225 * n1 / k
        #
        # so with n1 ~ 1,282 and k ~ 12 a tree carries ~24 leaves. Note the
        # control pool *cancels*: resolution is set by the treated count and
        # `k` alone, and `FOREST_CONTROL_CAP` is a variance and speed knob with
        # no effect on it.
        #
        # 3000 (k ~ 30) gave ~10 leaves -- about three binary splits over ~27
        # covariates, too coarse to express the heterogeneity the forest exists
        # to find. Per-leaf noise grows as sqrt(1/k), but honest splitting keeps
        # it out of the split decisions and the ensemble averages it down; GRF's
        # own default `min.node.size` for causal forests is 5, so even k ~ 12 is
        # conservative.
        'min_samples_leaf': 500,
        'max_features': 0.5,
        'max_samples': 0.45,          # econml's default
        'min_balancedness_tol': 0.45,
        'honest': True,               # what makes it a causal forest at all
        'cv': 5,                      # nuisance cross-fitting folds
        'random_state': 42,
    },
    # model_y = E[Y|X], marginal over T -- *not* the control-arm surface `f0`
    # that `gbm` is tuned for. A smaller leaf than the forest's is deliberate:
    # nothing inside it is differenced, so the contrast-stability argument
    # does not apply.
    'forest_y': {
        'n_estimators': 100,          # variance budget; a prediction averages fast
        'min_samples_leaf': 40,       # the module's convention, on the same rows
        'max_features': 0.5,          # ~13 of the ~27 design columns
        'random_state': 42,
    },
    # model_t = E[T|X] on every design column, since econml hands both
    # nuisances `hstack([X, W])`. `gbm_classifier` is tuned on the 8-column
    # `core` view and so is not the same model.
    'forest_t': {
        'n_estimators': 100,
        # `T` is a rare event on the rows econml fits this on. Even after the
        # control cap the treated share is only ~2.50%, so a leaf of 40 holds
        # **one** treated row and `predict_proba` comes back quantized and
        # mostly zero -- which forces `Tres_var ~ 1` whatever is in the data,
        # and `Tres_var ~ 1` is exactly the "no adjustment is possible" reading
        # `_log_tau` tells you to check first. 1200 restores ~30 treated per
        # leaf.
        #
        # This is a workaround, not the fix. The rare event is manufactured by
        # the fit sample: `T = Z*D`, and given `Z=0` the treatment is
        # degenerate, so ~97.5% of these rows carry zero Fisher information
        # about the shape of `e`. Fitting on the offered rows and rescaling by
        # `P(offered)` -- exactly what `_fit_propensity` does -- puts the share
        # back at ~31% and lets this return to 40. econml will not fit a
        # nuisance on a different sample from the forest, so that needs the
        # hand-rolled residualisation recorded in `bs.md` section 4, "Remaining".
        'min_samples_leaf': 1200,
        # ~4 of the 7 core columns. 0.3 would be 2, too few to decorrelate
        # usefully once the view is this narrow.
        'max_features': 0.5,
        'random_state': 42,
    },
}



        

# Which keys each profile accepts, frozen from the defaults above. A tuned
# profile may only override a key that already exists, so a typo raises instead
# of silently leaving the model at its sklearn default -- the same failure class
# `_named_view` guards against for covariates.
PROFILE_PARAMS: Dict[str, FrozenSet[str]] = {
    name: frozenset(params) for name, params in PROFILES.items()
}

_UNKNOWN_PROFILE = "unknown profile %r; available: %s"
_UNKNOWN_PARAM = "profile %r has no parameter %r; it accepts: %s"


def _profile(name: str) -> Dict[str, Any]:
    """A copy of one profile's parameters, looked up at call time."""
    if name not in PROFILES:
        raise KeyError(_UNKNOWN_PROFILE % (name, sorted(PROFILES)))
    return dict(PROFILES[name])


def set_profile(name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Override some of one profile's parameters, returning the new profile.

    Validates every key, so `set_profile('logit', {'c': 3.0})` raises rather
    than leaving `C` at its default and adding a parameter nothing reads.
    """
    if name not in PROFILES:
        raise KeyError(_UNKNOWN_PROFILE % (name, sorted(PROFILES)))
    for key in params:
        if key not in PROFILE_PARAMS[name]:
            raise KeyError(
                _UNKNOWN_PARAM % (name, key, sorted(PROFILE_PARAMS[name]))
            )
    PROFILES[name].update(params)
    return dict(PROFILES[name])


def _fit_ridge() -> Any:
    """Ridge for the outcome models' linear learner.

    `alpha` is doing a **numerical** job, not a statistical one, which is why it
    is frozen and absent from `DEFAULT_SPACES`. The design was badly
    rank-deficient; `LinearRegression` does not damp a null space and returned
    coefficients of order 1e13 with a *worse* residual sum of squares than the
    ridge fit, wrong in the third decimal against a Frisch-Waugh-Lovell check.
    Measured, `alpha` makes no difference at all across 1e-8 to 1e-12.

    The justification has narrowed. It used to also rest on the
    `dm_s_att_lin == linear_att` identity holding only while ridge is
    effectively unregularised -- and both those columns are gone. What is left is
    the conditioning argument, applying to the one surviving Ridge column,
    `dm_impute_t_att_lin`. Dropping the rate columns and the two constants took
    the design from rank 13 of 32 toward full rank, so this is weaker than it
    was: **re-measure the conditioning, and if it is good, `alpha` becomes an
    ordinary regularisation parameter with no reason not to tune it.**
    """
    return Ridge(**_profile('ridge'))


def _fit_gbm() -> Any:
    """The outcome models' tree learner. Fits ~107k rows, so it can afford capacity."""
    return HistGradientBoostingRegressor(**_profile('gbm'))




# The **outcome** learners -- regressors for E[Y|X,T]. The treatment models
# are separate, in PROPENSITY_SPECS below.
OUTCOME_LEARNERS: Dict[str, Callable[[], Any]] = {
    'lin': _fit_ridge, 'gbm': _fit_gbm,
}


def _build_logit() -> Any:
    """Linear logistic regression on the standardised covariates.

    `C` is the only real knob and it is the right one: it controls exactly the
    quantity the propensity is judged on, how dispersed `e` is. Low `C` shrinks
    toward the base rate -- weights near uniform, high ESS, worse balance -- and
    high `C` spreads it out, sharpening the weights at the cost of ESS. `penalty`
    stays `l2` (L1 would select features on *treatment* prediction, which is
    backwards), `class_weight` stays `None` (reweighting destroys the
    calibration IPW consumes), and `max_iter` is a convergence budget rather
    than capacity.
    """
    return make_pipeline(
        StandardScaler(), LogisticRegression(**_profile('logit'))
    )


def _build_gbm_classifier() -> Any:
    """Gradient-boosted trees on the *same* covariates as `_build_logit`.

    Deliberately underfit. `max_iter=15` at `learning_rate=0.05` is total
    shrinkage of 0.75, so predictions stay pulled toward the base rate instead of
    being pushed toward 0 and 1. That matters more than accuracy here, because
    IPW consumes `e` as a probability and the ATT control weight is the odds
    `e/(1-e)`: at e=0.95 that is 19, at e=0.97 it is 32. A model with better AUC
    but overconfident probabilities is strictly worse, so **no calibration layer
    is applied** -- under-confidence is the benign failure direction, and the
    reliability curve is measured rather than assumed.

    `max_depth=7` is mostly not binding: with ~1,971 offered rows and
    `min_samples_leaf=40` the tree caps near 49 leaves whatever the depth, so the
    leaf minimum is the real constraint.

    No `StandardScaler` -- trees are scale-invariant and a pipeline step would
    only obscure that. No `class_weight` or resampling either: both destroy
    calibration, which is the one property required.
    """
    return make_pipeline(
        HistGradientBoostingClassifier(**_profile('gbm_classifier'))
    )


def _build_forest_y() -> Any:
    """`model_y` for the causal forest: E[Y|X] over both arms of the support.

    A different regression from `OUTCOME_LEARNERS['gbm']`, which `_outcome_fit`
    tunes on the control arm alone -- this one is marginal over `T`.
    """
    return RandomForestRegressor(**_profile('forest_y'))


def _build_forest_t() -> Any:
    """`model_t` for the causal forest: E[T|X] on the same design the forest splits on.

    **Not** the 8-column `core` view the standalone propensity uses. DML's
    orthogonality needs `E[T~ | X] = 0` for the `X` the final stage splits on,
    and a treatment model conditioning on fewer columns leaves whatever the
    others predict about `T` inside `T~` -- which survives residualisation as
    bias, not as variance. An earlier version restricted this to `core` on a
    parameter-budget argument; that traded bias for variance in the wrong
    direction.
    """
    return RandomForestClassifier(**_profile('forest_t'))


@dataclass(frozen=True)
class PropensitySpec:
    """How one named propensity score is built.

    `features` selects the covariate view. Every propensity here uses 'core', so
    the specs differ *only* in model class and fold count -- which is the point:
    an earlier round compared a cross-fitted tree against an in-sample logit and
    could not say which of the two differences produced the gap.

    `n_folds` is 0 for an in-sample fit and K for K-fold out-of-fold prediction.
    Both are offered here rather than one being obviously right, because the
    trade-off runs in both directions:

      * **against cross-fitting** -- for IPW, weighting by the *estimated*
        propensity is more efficient than weighting by the true one (Hirano,
        Imbens & Ridder 2003): the estimation error in e_hat partially cancels
        the sampling error in the weighted mean, and splitting the sample severs
        that cancellation. Out-of-fold scores are also noisier for training on
        80% of an already small fit sample.
      * **for it** -- DML needs out-of-fold nuisances outright, and a tree can
        separate the arms sharply in sample, pushing treated `e` toward 1 where
        the odds weight explodes.

    **Every registered spec passes `n_folds=0`.** The field is kept because DML
    is deprecated rather than rejected -- see `DML_FOLDS`.
    """

    features: str
    build: Callable[[], Any]
    n_folds: int


# Folds for the DML nuisances. Cost is linear in it, since each fold is another
# outcome-model fit per group_b.
#
# **No registered estimator reaches this, deliberately.** `dml_att_gbm` and the
# `gbm_cf` propensity spec were removed with the rest of the unscored columns,
# but DML is deprecated rather than rejected and reinstating it must stay a
# two-line change:
#
#     PROPENSITY_SPECS['gbm_cf'] = PropensitySpec(
#         'core', _build_gbm_classifier, DML_FOLDS)
#     DoublyRobustATT('gbm_cf', 0.05, 'gbm', 'dml_att_gbm', n_folds=DML_FOLDS)
#
# So `_cross_fit`, `_out_of_fold_f0`, `PropensitySpec.n_folds`,
# `DoublyRobustATT.n_folds` and the `KFold` / `StratifiedKFold` imports are all
# live code with no caller. Nothing in the registry exercises them, so they are
# covered by direct tests instead -- constructing the estimator above by hand
# and checking that the in-sample mean control residual is ~0 while the
# out-of-fold one is not. Unreachable code with no explanation gets deleted.
DML_FOLDS = 5

_OUTCOME_FOLDS_TOO_SMALL = (
    "outcome models ('%s'): %d controls cannot be split into %d folds --"
    " returning nan rather than an in-sample f0"
)

_ECONML_MISSING = (
    "econml is not installed, so `cf_att` returns nan for every pair. Install"
    " econml to enable it, or drop 'cf_att' from `estimator_names`. This is"
    " logged once, not once per fit."
)

_FOREST_TOO_SMALL = (
    "causal forest: support has %d rows, needs >= %d for %d-fold"
    " cross-fitting -- returning nan"
)

# Keeps the ATT odds weight finite when a pure leaf returns e exactly 0 or 1.
_E_FLOOR = 1e-6

# Controls the causal forest fits, at most. Every treated row is always kept.
#
# The pooled sample is ~1,282 treated against ~137,034 controls -- 107 controls
# per treated row, where case-control designs stop paying at 4 or 5. Variance in
# a contrast goes as `1/n1 + 1/n0`, and at that ratio the control arm supplies
# **0.9%** of the total, so capping at 50,000 costs ~0.8% on the standard error.
#
# What it buys is not just speed. At 0.93% treated a `min_samples_leaf` of a few
# hundred leaves under one treated row per leaf, so `model_t` returns a quantized,
# mostly-zero propensity -- and a near-constant `e` leaves the confounding in
# `T~`, which is the "no adjustment" state. The cap lifts the treated share to
# ~2.5% and makes the propensity estimable again.
#
# The sample is drawn on `T` alone -- never on `X` or `Y` -- so it is a random
# thinning, not a selection, and cannot reintroduce the never-taker bias that
# restricting to the support did.
FOREST_CONTROL_CAP = 50_000

# Fixed so the same controls are drawn for every group of a card, which keeps
# the groups comparable and the run reproducible.
FOREST_SAMPLE_SEED = 42

_CF_TAU = (
    "cf_att %s: rows=%d treated=%d | naive=%+.5f tau=%+.5f adj=%+.5f"
    " tau_att=%+.5f | Yres_var=%.3f Tres_var=%.3f"
    " | e min=%.3f mean=%.3f max=%.3f"
    " | tau sd=%.5f min=%+.5f max=%+.5f"
)

_CF_TOP = "cf_att %s: top modifiers %s"

_FOREST_ARM_TOO_SMALL = (
    "causal forest: %d treated and %d control rows in the support, needs >= %d"
    " in each for %d-fold cross-fitting -- returning nan"
)


@lru_cache(maxsize=1)
def _causal_forest_class() -> Optional[Any]:
    """`CausalForestDML`, or None with a single warning if econml is absent.

    Imported lazily and never at module scope: econml is an optional
    dependency, and a top-level import would make this whole module
    unimportable wherever it is missing -- which is how the `statsmodels`
    defect in `cexp.py` was found.

    `lru_cache` is what makes the warning fire once rather than once per fit;
    at ~1,925 pairs a per-call warning is 1,925 lines of noise.
    """
    try:
        # pylint: disable=import-outside-toplevel
        from econml.dml import (  # type: ignore[import-not-found]
            CausalForestDML,
        )
    except ImportError:
        logging.warning(_ECONML_MISSING)
        return None
    return CausalForestDML


def r_loss(
    y_residual: np.ndarray, t_residual: np.ndarray, tau: np.ndarray
) -> float:
    """Nie & Wager's R-loss: `mean((Y~ - tau(X) * T~)^2)`.

    The model-selection criterion for an estimator whose target is never
    observed. Expanding it shows that, up to a constant free of `tau`, it
    equals `E[Var(T|X) * (tau_hat(X) - tau(X))^2]` -- the mean squared error on
    `tau` after all, weighted by `Var(T|X)`. Nothing unobserved appears in what
    is actually computed.

    **The residuals must be out-of-fold.** In sample they are shrunk toward
    zero -- measured in this module at 8.3e-16 (lin) and 4.4e-11 (gbm) against
    -9.5e-05 and -4.5e-03 out of fold -- and the loss then rewards overfitting.

    Kept as a free function so it is testable without econml, which is needed
    only to produce `tau`.
    """
    residual = y_residual - tau * t_residual
    return float(np.mean(residual ** 2))


_CROSS_FIT_TOO_SMALL = (
    "propensity '%s' needs >= %d rows in each arm to cross-fit, smaller arm"
    " has %d -- returning nan rather than falling back to an in-sample fit"
)

# Two model classes over one covariate set. `logit` and `gbm` share
# `CORE_COVARIATES` and the offered-only fit sample, so the gap between them is
# attributable to the model class and nothing else. The cross-fitted `gbm_cf`
# went with `dml_att_gbm`; see `DML_FOLDS` for how to put both back.
PROPENSITY_SPECS: Dict[str, PropensitySpec] = {
    'logit': PropensitySpec('core', _build_logit, 0),
    'gbm': PropensitySpec('core', _build_gbm_classifier, 0),
}


class TreatmentContext:
    """Everything derivable from one CardA's table, cached.

    Treatment and covariates do not depend on group_b, so the propensity fits and
    every nearest-neighbour search are shared across all outcomes -- the
    reference implementation refits them inside the group_b loop. Only the
    outcome models are outcome-dependent, and those are cached per group.

    **The offer is randomized.** Pack 2 pick 1 is a fresh booster, so being
    offered CardA is independent of the drafter's pool -- the assumption the
    propensity code relies on, and the reason the p2any variant is gone (there,
    surviving to pick k signals colour-openness and the offer is endogenous).

    So `e(X) = P(T=1 | X, offered)` is *fitted* on the offered rows but
    *predicted for every row* -- "what would this drafter have done had
    CardA been in the pack". Since `P(T=1|X) = P(offered) * e(X)` and the offer
    probability is a constant under randomization, `e` and the true propensity
    are the same balancing score up to scale, so it is a valid score for the
    whole sample. That lets IPW and matching draw from all ~106,735 controls
    rather than the ~1,204 offered ones, which is the measured reason the
    propensity family loses to the direct method. It additionally leans on the
    exclusion restriction -- being offered and declining does not itself change
    later picks -- which the IV gold standard already assumes.

    `not_offered` never enters the propensity design: it is an instrument, and
    conditioning on an instrument amplifies residual confounding without
    removing any.
    """

    def __init__(self, df: pd.DataFrame, k_neighbors: int = 3):
        self.df = df
        self.k_neighbors = k_neighbors
        self.treatment = df['T'].to_numpy(dtype=float)
        self.horizon = df['horizon'].to_numpy(dtype=float)
        # No derived rate columns. `rate_<x> = n_<x> / (n_prior + 1)` existed
        # when `n_prior` varied with the decision's pick; the sample is now pack
        # 2 pick 0 of complete drafts alone, so `n_prior` is a constant and every
        # rate was an exact scalar multiple of its count -- measured at
        # correlation 1.000000000000, and the design carried 12 such columns at
        # rank 13 of 32 with an effectively infinite condition number.
        self.covariates = df[covariate_columns(df)].fillna(0.0)
        self.x_cols = list(self.covariates.columns)
        self.design = self.covariates.to_numpy(dtype=float)
        self.treated_pos = np.flatnonzero(self.treatment == 1.0)
        self.control_pos = np.flatnonzero(self.treatment == 0.0)
        self.support = self._propensity_support()

        self._scaled: Optional[np.ndarray] = None
        self._propensity: Dict[str, Optional[np.ndarray]] = {}
        self._matches: Dict[str, Optional[np.ndarray]] = {}
        self._weights: Dict[Tuple[str, float], Optional[np.ndarray]] = {}
        self._outcomes: Dict[str, np.ndarray] = {}
        self._control_fits: Dict[
            Tuple[str, str, str, int], Optional[Tuple[np.ndarray, np.ndarray]]
        ] = {}
        self._forests: Dict[str, Optional[np.ndarray]] = {}
        self._forest_rows: Optional[np.ndarray] = None

    def _propensity_support(self) -> np.ndarray:
        """Rows the propensity is fitted on and predicted for: the offered ones.

        Only offered rows say anything about who *takes* CardA rather than
        declines it -- 1,971 of 107,502 on the measured card. The rest have a
        structurally zero propensity, and including them wastes the model's
        capacity on a split it already knows while leaving each of them a small
        non-zero e that, multiplied by 105k rows, took a real share of the
        control weight.

        **The estimand is unchanged**: treated is a subset of offered by
        construction, since a card absent from the pack cannot be picked, so
        `treated_pos` is identical either way and only the control pool moves.
        The not-offered rows keep the job they were admitted for -- giving the
        outcome models a large sample on which to learn the control surface.

        All rows when `not_offered` is absent, so frames built without it (the
        synthetic ATT fixtures) behave exactly as before.
        """
        if 'not_offered' not in self.covariates.columns:
            return np.arange(len(self.treatment))
        return np.flatnonzero(
            self.covariates['not_offered'].to_numpy(dtype=float) == 0.0
        )

    def control_pool(self, sample: str = 'all') -> np.ndarray:
        """Control rows an estimator may average over.

        'all' is every control; 'offered' is the support, i.e. exactly the rows
        IPW and PS-matching can reach, since everything else carries weight 0.
        """
        if sample == 'all':
            return self.control_pos
        return np.intersect1d(self.control_pos, self.support)

    @property
    def is_valid(self) -> bool:
        """Whether both arms are populated, so an ATT is defined at all."""
        return len(self.treated_pos) > 0 and len(self.control_pos) > 0

    # ---- outcome ----
    def outcome(self, group_b: str) -> Optional[Outcome]:
        """Absolute pack-3 count for a group.

        No division by `horizon`. Every pack-2 decision shares the same pack-3
        window whatever its pick, so raw counts are already comparable across
        rows -- which is what the rate outcome existed to fix. This also puts
        the estimate on `ivexp.late`'s scale directly, so `metrics` needs no
        `truth_divisor`.
        """
        column = f'Y_{group_b}'
        if column not in self.df.columns:
            return None
        if group_b not in self._outcomes:
            self._outcomes[group_b] = self.df[column].to_numpy(dtype=float)
        return Outcome(group_b, self._outcomes[group_b])

    # ---- propensity ----
    def propensity(self, score: str) -> Optional[np.ndarray]:
        """`e(X) = P(T=1|X)` for one named spec in PROPENSITY_SPECS.

        **The unconditional propensity**, on the scale where `e/(1-e)` is the
        ATT odds weight. `_fit_propensity` fits `P(T=1 | X, offered)` and scales
        it by `P(offered)`; that conditioning is an artefact of where the
        likelihood lives and is not a concept this method exposes. Callers see
        `e` and nothing else.

        Only the predictions are used downstream -- do not read the
        coefficients.
        """
        if score not in self._propensity:
            self._propensity[score] = self._fit_propensity(score)
        return self._propensity[score]

    def _fit_propensity(self, score: str) -> Optional[np.ndarray]:
        """`P(T=1|X)`, fitted on the offered rows and rescaled to every row.

        Two steps, and the second is not cosmetic.

        **Fit.** The likelihood factorises. Writing `Z` for the offer and `D`
        for "would take it if offered", `T = Z*D` with `Z` independent of
        everything (pack 2 pick 0 is an unopened booster), so

            L(pi, beta) = pi^n_off (1-pi)^(N-n_off)
                          * prod_{Z=1} e'^T (1-e')^(1-T)

        and the `beta` factor reads only the offered rows. Given `Z=0`, `T` is
        degenerate -- zero variance, zero Fisher information -- so the
        never-offered rows are ancillary for `e'` and fitting on the support is
        the *full-data* MLE, not a subsample of it.

        **Rescale.** `pi = P(offered)` is constant in `X` by randomisation, so

            e(X) = P(T=1|X) = pi * e'(X),   e'(X) = P(T=1 | X, offered)

        Returning `e'` was the defect this fixes. `pi` cancels in the
        self-normalised numerator, but `1-e'` is `P(T=0 | X, offered)` while the
        control pool is ~98% never-offered, and no constant `c` makes
        `1-e' = c(1-pi e')`. So the error does not cancel: `e'/(1-e')` is the
        weight that would be right if every control were a never-taker, which
        is true of ~1.3% of them. `pi` is estimated from N Bernoulli draws and
        is effectively a known constant.

        `pi = 1` when `not_offered` is absent (the synthetic fixtures), so those
        frames are untouched.
        """
        spec = PROPENSITY_SPECS[score]
        view = FEATURE_VIEWS[spec.features](self.x_cols)
        # not_offered is constant 0 across the fit sample, so it carries no
        # information there -- a zero-variance column for the logit and an
        # unsplittable feature for anything else. It is also the instrument:
        # conditioning on it would make the model learn P(T=1 | not offered) = 0
        # exactly, zeroing the weight on ~105k controls and collapsing IPW and
        # the DR correction onto the ~1.2k never-takers.
        cols = [c for c in view if c != 'not_offered']
        if not cols or len(self.support) == 0:
            return None
        full_design = self.covariates[cols].to_numpy(dtype=float)
        design = full_design[self.support]
        treatment = self.treatment[self.support]
        outside = np.setdiff1d(np.arange(len(self.treatment)), self.support)
        try:
            model = spec.build().fit(design, treatment)
            fitted = (
                model.predict_proba(design)[:, 1] if spec.n_folds == 0
                else self._cross_fit(score, spec, design, treatment)
            )
            if fitted is None:
                return None
            scores = np.zeros(len(self.treatment))
            scores[self.support] = fitted
            if len(outside):
                # Extrapolate to rows that never saw CardA: "what would this
                # drafter have done had it been in the pack". Predicted from the
                # whole-support fit, which is out of sample for them either way
                # -- they were in no training set, cross-fitted or not.
                scores[outside] = model.predict_proba(full_design[outside])[:, 1]
        except (ValueError, np.linalg.LinAlgError) as exc:
            logging.warning("propensity '%s' failed to fit: %s", score, exc)
            return None
        # e' -> e. See the docstring: this is the whole difference between a
        # score that ranks correctly and one that weights correctly.
        return (len(self.support) / len(self.treatment)) * scores

    def _cross_fit(
        self,
        score: str,
        spec: PropensitySpec,
        design: np.ndarray,
        treatment: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Out-of-fold e(X) over the support: every row scored by a model that
        never saw it.

        Returns None rather than falling back to an in-sample fit when an arm is
        too small to split -- a silent fallback would quietly reintroduce exactly
        the overfitting bias the split exists to prevent.
        """
        smaller_arm = min(int(treatment.sum()), int((1.0 - treatment).sum()))
        if smaller_arm < spec.n_folds:
            logging.warning(
                _CROSS_FIT_TOO_SMALL, score, spec.n_folds, smaller_arm
            )
            return None
        scores = np.full(len(treatment), np.nan)
        splitter = StratifiedKFold(
            n_splits=spec.n_folds, shuffle=True, random_state=42
        )
        for train, test in splitter.split(design, treatment):
            model = spec.build().fit(design[train], treatment[train])
            scores[test] = model.predict_proba(design[test])[:, 1]
        return scores if np.isfinite(scores).all() else None

    @property
    def scaled_design(self) -> np.ndarray:
        """Standardised covariates, for covariate-space matching.

        Restricted to the `core` view rather than every covariate. Nearest
        neighbours degrade quickly with dimension -- distances concentrate, so
        the "nearest" control stops being meaningfully nearer than the rest --
        and the full design now carries roughly thirty columns, most of them
        raw/rate pairs that duplicate one another. Matching on the CardA-relative
        commitment columns keeps the distance in the space that actually decides
        the pick.

        `not_offered` is excluded by construction (it is not in the view): as a
        coordinate it would keep every treated row (always offered) matching to
        offered controls, which quietly restricts the pool to the ~1.8% of rows
        that saw CardA. It stays in `self.design`, but no model adjusts on it --
        `adjustment_columns` takes it out for the outcome models and the forest
        alike. It is kept in the frame because `_propensity_support` reads it to
        find the support.
        """
        if self._scaled is None:
            # Fall back on the *defining* columns, not on whether the view came
            # back non-empty. `core` also contains generic columns like n_prior,
            # so a frame predating these covariates yields a view of one or two
            # incidental columns -- non-empty, and a far worse matching space
            # than the full design. Keying off a CardA-relative column instead
            # makes the test "was this frame extracted with phase 6 features".
            has_relative = any(c in self.x_cols for c in RELATIVE_COVARIATES)
            view = set(FEATURE_VIEWS['core'](self.x_cols)) if has_relative else set()
            keep = [i for i, c in enumerate(self.x_cols) if c in view]
            if not keep:
                keep = [i for i, c in enumerate(self.x_cols) if c != 'not_offered']
            self._scaled = StandardScaler().fit_transform(self.design[:, keep])
        return self._scaled

    # ---- matching ----
    def match_indices(self, space: str) -> Optional[np.ndarray]:
        """(n_treated, k) **absolute row positions** of each treated row's
        nearest controls, in propensity or covariate space.

        Absolute rather than positions into `control_pos`, because the eligible
        control pool now depends on the space -- see `_build_matches`.
        """
        if space not in self._matches:
            self._matches[space] = self._build_matches(space)
        return self._matches[space]

    def _build_matches(self, space: str) -> Optional[np.ndarray]:
        if space == 'cov':
            source = self.scaled_design
            pool = self.control_pos
        else:
            # Derived from the space name rather than branched on, so an
            # unrecognised space raises instead of silently matching on 'rates'.
            scores = self.propensity(space[len('ps_'):])
            if scores is None:
                return None
            source = scores.reshape(-1, 1)
            # The offer is randomized, so e is defined for every row and a
            # not-offered control at the same score is as good a match as an
            # offered one: the whole pool is eligible.
            pool = self.control_pos

        k = min(self.k_neighbors, len(pool))
        if k <= 0 or len(self.treated_pos) == 0:
            return None
        finder = NearestNeighbors(n_neighbors=k).fit(source[pool])
        found = finder.kneighbors(source[self.treated_pos], return_distance=False)
        return pool[found]

    # ---- IPW ----
    def control_weights(self, score: str, clip: float) -> Optional[np.ndarray]:
        """ATT odds weights `e/(1-e)` over the controls, normalised to sum to 1.

        The treated arm is the target population and so is left unweighted.

        **`clip` defaults to 0, because it cannot bind.** `e = pi * e'` is
        bounded by the offer rate, `pi ~ 0.018`, so against a measured
        `e' <= 0.42` the largest weight is `e ~ 0.008` and `e -> 1` is
        structurally impossible. Any `clip < 0.992` leaves every score
        untouched, so clip variants would be bit-identical columns. There is no
        weight concentration to control either: `w ~ e`, so the spread across
        controls is just the spread of `e'`, roughly 2:1.

        The parameter stays because it is inert, not dead. On a frame without
        `not_offered` (the synthetic fixtures) `pi = 1`, `e = e'`, and a tree
        score genuinely can reach 1.

        A floor is still refused, and now for a cleaner reason than before: a
        never-offered row's `e` is not a structural zero at all but a real
        counterfactual "would this drafter have taken it", so there is nothing
        to lift off zero.
        """
        key = (score, clip)
        if key not in self._weights:
            self._weights[key] = self._build_weights(score, clip)
        return self._weights[key]

    def _build_weights(self, score: str, clip: float) -> Optional[np.ndarray]:
        scores = self.propensity(score)
        if scores is None:
            return None
        # `_E_FLOOR` is a numerical guard, not the clip: at `clip=0` a pure leaf
        # returning exactly 1.0 would make one weight `inf`, hence `total` inf,
        # hence a silently all-nan column for the whole card. Unreachable at
        # `pi ~ 0.018` but live on the `pi = 1` fixtures.
        ceiling = min(1.0 - clip, 1.0 - _E_FLOOR)
        clipped = np.minimum(scores, ceiling)[self.control_pos]
        weights = clipped / (1.0 - clipped)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            return None
        return weights / total

    # ---- outcome models ----
    def adjustment_columns(self) -> List[int]:
        """Positions in `design` of the columns any model may adjust on.

        Everything except `not_offered`, which is the instrument `Z` and must
        never enter a model that predicts at the treated rows.

        For `f0` the failure is concrete rather than theoretical. Within the
        control sample `not_offered == 0` *is* the never-taker indicator: a
        never-offered row has `T=0` forced, with no selection on taste, while an
        offered control chose `T=0` and so reveals `D=0`. So the column splits
        the controls into a clean surface `E[Y(0)|X]` and a contaminated one
        `E[Y(0)|X, D=0]` -- and **every treated row sits at `not_offered = 0`**,
        so `predict(design[treated_pos])` reads off the contaminated surface for
        all of them. Left in, the never-taker gap enters the estimate at weight
        1.0; dropped, it enters only at its honest mixture share of ~1.3%.

        Measured on a synthetic with true ATT 2.0 and a deliberate never-taker
        gap of 4.0: `dm_impute_t_att_lin` returned +5.97 and
        `dm_impute_t_att_gbm` +5.94 with the column in, against +2.05 with it
        out. Both learners, not just the tree -- `PROFILES['ridge']` is
        `alpha=1e-8`, so there is no shrinkage to blunt it.

        `dm_impute_t_att_gbm_nt` is unaffected either way: its fit sample is the
        support, where the column really is constant. That is where the
        "contributes nothing to a linear fit and cannot be split on by a tree"
        argument came from, and it does not survive the move to `sample='all'`.
        """
        return [i for i, c in enumerate(self.x_cols) if c != 'not_offered']

    def control_outcome(
        self,
        outcome: Outcome,
        learner: str,
        sample: str = 'all',
        n_folds: int = 0,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """`f0` predicted at (treated rows, control rows).

        Split out from `outcome_models` because the doubly robust score needs
        only this fit -- requesting it through the full bundle would refit the
        pooled and treated-arm models it never reads, which is two extra GBM
        fits per group_b.

        `n_folds > 0` makes the *control* predictions out of fold. Only they need
        it: `f0` at the treated rows is already out of sample, since `f0` is
        fitted on the controls, and the cross-fitted entry reuses the in-sample
        entry's treated predictions rather than refitting them.
        """
        key = (outcome.group_b, learner, sample, n_folds)
        if key not in self._control_fits:
            self._control_fits[key] = self._fit_control_outcome(
                outcome, learner, sample, n_folds
            )
        return self._control_fits[key]

    def _fit_control_outcome(
        self, outcome: Outcome, learner: str, sample: str, n_folds: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        y = outcome.values
        if not np.isfinite(y).all():
            return None
        control_rows = self.control_pool(sample)
        if len(control_rows) == 0:
            return None
        build = OUTCOME_LEARNERS[learner]
        # `not_offered` is out of the design: it is the never-taker indicator
        # among controls, and every treated row sits at 0. See
        # `adjustment_columns`.
        cols = np.asarray(self.adjustment_columns(), dtype=np.intp)
        try:
            if n_folds == 0:
                model = build().fit(
                    self.design[np.ix_(control_rows, cols)], y[control_rows]
                )
                # Indexed like control_pos, so the DR correction can dot it
                # against the control weights directly. Controls outside the fit
                # sample keep this fit's prediction; their weight is 0 anyway.
                return (
                    model.predict(self.design[np.ix_(self.treated_pos, cols)]),
                    model.predict(self.design[np.ix_(self.control_pos, cols)]),
                )
            if len(control_rows) < n_folds:
                logging.warning(
                    _OUTCOME_FOLDS_TOO_SMALL, learner, len(control_rows), n_folds
                )
                return None
            base = self.control_outcome(outcome, learner, sample, 0)
            if base is None:
                return None
            at_control = base[1].copy()
            inside = np.isin(self.control_pos, control_rows)
            at_control[inside] = self._out_of_fold_f0(
                y, control_rows, build, n_folds
            )
            return base[0], at_control
        except (ValueError, np.linalg.LinAlgError) as exc:
            logging.warning("control outcome ('%s') failed to fit: %s", learner, exc)
            return None

    def _out_of_fold_f0(
        self, y: np.ndarray, control_rows: np.ndarray, build: Any, n_folds: int
    ) -> np.ndarray:
        """f0 at each control row, from a model that did not see that row.

        Matters because the in-sample control residual `y - f0(x)` averages to
        exactly zero for any squared-loss fit with an intercept, which would
        silently zero out the doubly robust correction term.

        Same column set as the in-sample fit in `_fit_control_outcome`, or the
        two halves of `at_control` would come from differently-specified models.
        """
        fitted = np.full(len(control_rows), np.nan)
        cols = np.asarray(self.adjustment_columns(), dtype=np.intp)
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        for train, test in splitter.split(control_rows):
            model = build().fit(
                self.design[np.ix_(control_rows[train], cols)],
                y[control_rows[train]],
            )
            fitted[test] = model.predict(
                self.design[np.ix_(control_rows[test], cols)]
            )
        return fitted


######################
# Phase 2 -- estimators
######################


    # ---- causal forest ----
    def forest_rows(self) -> np.ndarray:
        """Rows the causal forest fits: every treated, plus capped controls.

        See `FOREST_CONTROL_CAP` for why the controls are capped and why doing
        so is a random thinning rather than a selection. Cached, since it does
        not depend on `group_b` and every group of a card must see the same
        rows for their estimates to be comparable.
        """
        if self._forest_rows is None:
            controls = self.control_pos
            if len(controls) > FOREST_CONTROL_CAP:
                rng = np.random.default_rng(FOREST_SAMPLE_SEED)
                controls = rng.choice(
                    controls, FOREST_CONTROL_CAP, replace=False
                )
            self._forest_rows = np.sort(
                np.concatenate([self.treated_pos, controls])
            )
        return self._forest_rows

    def forest_columns(self) -> List[int]:
        """Positions in `design` of the columns the forest splits on.

        Everything except `not_offered`, and excluding it is **load-bearing**
        now that the forest fits every row rather than the support. It is
        randomly assigned by pack collation -- an *instrument*, not a
        confounder -- so conditioning on it amplifies residual confounding
        rather than removing any. Worse, a model that sees it learns
        `P(T=1 | not offered) = 0` exactly, which makes `T~ = 0` on all ~136k
        never-offered rows; they would then carry no weight in
        `sum(Y~ T~) / sum(T~^2)` and the pooled fit would silently collapse
        back to the support-only one it exists to replace.

        Computed once and used for both `fit` and `effect`: they must agree on
        the column set or the design widths do not match.

        The same set `f0` adjusts on -- see `adjustment_columns`, which this
        delegates to so the predicate exists once. The reasons differ (there,
        `not_offered` is the never-taker indicator; here it would zero `T~`) but
        the column set does not.
        """
        return self.adjustment_columns()

    def causal_forest(self, outcome: Outcome) -> Optional[np.ndarray]:
        """`tau_hat` at the treated rows, cached per group.

        **This breaks the class's convention, and deliberately.** Every other
        cached fit here is a *nuisance* that some estimator then combines into
        a contrast; this one produces the contrast itself. It lives on
        `TreatmentContext` because that is where the design, the support and
        the cache are, not because it is a nuisance.

        None when econml is absent or an arm is too thin to fold, so the
        column is nan and no other column is affected.
        """
        if outcome.group_b not in self._forests:
            self._forests[outcome.group_b] = self._fit_causal_forest(outcome)
        return self._forests[outcome.group_b]

    def _fit_causal_forest(self, outcome: Outcome) -> Optional[np.ndarray]:
        """Fit `CausalForestDML` on **every** row and predict at the treated.

        Not the support. Confining the fit there made the control group the
        ~1,241 offered-and-declined rows -- the never-takers -- which at a ~50%
        take rate are roughly the lower half of the latent colour-taste
        distribution among drafters shown the card. Phase 9/11 measured that
        conditioning on `X` does not close that gap, so it is a bias no nuisance
        model can remove.

        Pooling is valid: the pack-2 offer is randomised, so
        `P(T=1 | X, Y(0)) = P(offered) * e(X)` is free of `Y(0)` and
        unconfoundedness survives; positivity holds at `e_pooled ~ 0.009`.

        And it is not even a precision trade. The R-learner's information in
        the treatment contrast goes as `n * E[e(1-e)]`:

            support only : 2,523   x 0.250  =   631
            every row    : 138,316 x 0.0092 = 1,270

        `model_y` also stops needing the pre-fitted-off-support workaround:
        econml now fits it on all ~138k rows itself, and the pooled `E[Y|X]`
        differs from `E[Y(0)|X]` by only `P(T=1|X) tau ~ 0.009 tau`, against
        `0.5 tau` on the support.
        """
        # The profile read and the arm guards come *before* the optional
        # import, so they run whether or not econml is installed. With the
        # import first they were dead code here, which is how a missing `cv`
        # key survived a full verification pass.
        forest_p = _profile('forest')
        cv = int(forest_p['cv'])
        # Guards run on the rows actually fitted, not on the whole table: with
        # the control cap in play those differ, and a guard on rows the forest
        # never sees is not a guard.
        rows = self.forest_rows()
        n_rows = len(rows)
        if n_rows < 2 * cv:
            logging.warning(_FOREST_TOO_SMALL, n_rows, 2 * cv, cv)
            return None
        treated_mask = self.treatment[rows] == 1.0
        n_treated = int(treated_mask.sum())
        n_control = n_rows - n_treated
        if n_treated < cv or n_control < cv:
            logging.warning(
                _FOREST_ARM_TOO_SMALL, n_treated, n_control, cv, cv
            )
            return None

        forest_class = _causal_forest_class()
        if forest_class is None:
            return None

        # One column set for `fit` and `effect` -- they must agree, and
        # `not_offered` has to be out of both. See `forest_columns`.
        cols = np.asarray(self.forest_columns(), dtype=np.intp)
        design = self.design[:, cols]
        # Every profile key is a `CausalForestDML` constructor argument, `cv`
        # included, so the whole dict expands. Filtering one key out by name
        # was what let it go missing unnoticed.
        forest = forest_class(
            model_y=_build_forest_y(),
            model_t=_build_forest_t(),
            discrete_treatment=True,
            **forest_p,
        )
        try:
            # `cache_values` keeps the nuisance residuals so `_log_tau` can
            # report how much of Y and T each nuisance actually explained --
            # the difference between "the forest is not adjusting" and "the
            # forest is adjusting from a worse baseline".
            #
            # Fitted on `rows`, but `effect` is still evaluated at **every**
            # treated row. The cap thins only the control arm, so no treated
            # row is ever dropped and the ATT is averaged over the whole
            # treated population rather than a sample of it.
            forest.fit(outcome.values[rows], self.treatment[rows],
                       X=design[rows], cache_values=True)
            tau = np.asarray(
                forest.effect(design[self.treated_pos]), dtype=float
            )
            self._log_tau(outcome, forest, tau, rows)
            return tau
        except (ValueError, np.linalg.LinAlgError) as exc:
            logging.exception("...")
            logging.warning('causal forest failed for %s: %s',
                            outcome.group_b, exc)
            raise
            return None

    def _log_tau(self, outcome: Outcome, forest: Any, tau: np.ndarray,
                 rows: np.ndarray) -> None:
        """Report where the estimate came from, not just what it was.

        Six numbers, chosen to partition the ways this can go wrong:

          * `naive`  -- the unadjusted difference **on the rows the forest
            fitted**, so `adj = naive - tau` states in one number how much
            adjustment actually happened. That is the symptom measured
            directly rather than inferred from a scatter plot afterwards.
          * `Tres_var`, `Yres_var` -- the share of variance each residual
            keeps. **`Tres_var` near 1 is the diagnostic one**: it means `e(X)`
            explains nothing, and since bias enters through `e` and not `m`
            (any function-of-X error in `m` is annihilated by `E[T~|X] = 0`),
            no adjustment is possible in that state.
          * `e` min/mean/max -- a narrow band is the under-dispersion phase 7
            measured on the standalone propensity, recurring here.
          * `tau_att` -- `mean(tau)` over the treated reweighted by the ATT
            odds `e/(1-e)`. The R-learner targets the `Var(T|X)`-weighted
            average; if `tau_att` moves and `tau` does not, the gap is the
            estimand rather than the adjustment.

        No exception guard: a missing `residuals_` should stop the run loudly
        rather than degrade every line to nan and be discovered later.

        **Everything here indexes `rows`, the fitted sample.** `residuals_` has
        that length, so `y_fit`, `treated` and the `e` stats must too, or every
        number silently describes a different sample from the one the forest
        saw. `tau` is the exception: it is evaluated at *all* treated rows, so
        `tau_att` maps them into row-space with `searchsorted`. Misaligned
        diagnostics would be worse than none.
        """
        y_res, t_res = (np.asarray(v, dtype=float).ravel()
                        for v in forest.residuals_[:2])
        y_fit = outcome.values[rows]
        treated = self.treatment[rows] == 1.0
        naive = float(y_fit[treated].mean() - y_fit[~treated].mean())
        tau_mean = float(np.mean(tau))

        # The cap thins controls only, so every treated row survives into
        # `rows` and this lookup is total. Assert it rather than trust it: a
        # miss returns a plausible neighbouring index instead of failing.
        treated_at = np.searchsorted(rows, self.treated_pos)
        if not np.array_equal(rows[treated_at], self.treated_pos):
            raise AssertionError(
                'forest_rows() dropped a treated row; tau_att would be '
                'computed against the wrong residuals'
            )
        propensity = np.clip(1.0 - t_res[treated_at], _E_FLOOR, 1.0 - _E_FLOOR)
        odds = propensity / (1.0 - propensity)
        e_all = np.clip(self.treatment[rows] - t_res, 0.0, 1.0)

        logging.info(
            _CF_TAU, outcome.group_b, len(rows), int(treated.sum()),
            naive, tau_mean, naive - tau_mean,
            float(np.sum(odds * tau) / np.sum(odds)),
            float(np.var(y_res) / np.var(y_fit)),
            float(np.var(t_res) / np.var(self.treatment[rows])),
            float(e_all.min()), float(e_all.mean()), float(e_all.max()),
            float(np.std(tau)), float(np.min(tau)), float(np.max(tau)),
        )
        importances = getattr(forest, 'feature_importances_', None)
        if importances is None:
            return
        names = [self.x_cols[i] for i in self.forest_columns()]
        order = np.argsort(np.asarray(importances, dtype=float))[::-1][:3]
        logging.info(
            _CF_TOP, outcome.group_b,
            [(names[i], round(float(importances[i]), 3)) for i in order],
        )


class ATTEstimator(ABC):
    """One baseline estimate of the ATT from a prepared context and outcome."""

    name: str = ''

    @abstractmethod
    def estimate(self, ctx: TreatmentContext, outcome: Outcome) -> float:
        """Return the estimate, or nan when it is not defined."""


class NaiveDiff(ATTEstimator):
    """Difference in means. Plugs E[Y|T=0] in for E[Y(0)|T=1]; still biased.

    `sample='offered'` averages only the controls inside the support, which is
    the same ~1,204 rows IPW and PS-matching are limited to. That makes
    `satt_offered`, not `satt`, the fair naive comparator for them: `satt`
    averages all 106,735 controls and so enjoys an 89x variance advantage that
    has nothing to do with the quality of any adjustment.

    Reading the pair: `sipw` beating `satt_offered` means the covariate
    adjustment works and the loss to `satt` was pool size. The two coming out
    level means conditioning on *being offered* already removes most of the
    confounding and the pool covariates add little on top -- a finding about the
    draft rather than a failure of the benchmark.
    """

    def __init__(self, sample: str = 'all'):
        self.sample = sample
        self.name = 'satt' if sample == 'all' else f'satt_{sample}'

    def estimate(self, ctx: TreatmentContext, outcome: Outcome) -> float:
        values = outcome.values
        pool = ctx.control_pool(self.sample)
        if len(pool) == 0:
            return float('nan')
        return float(values[ctx.treated_pos].mean() - values[pool].mean())

class StabilizedIpwATT(ATTEstimator):
    """Self-normalized IPW for the ATT, re-weighting only the controls."""

    def __init__(self, score: str, clip: float, name: str):
        self.score = score
        self.clip = clip
        self.name = name

    def estimate(self, ctx: TreatmentContext, outcome: Outcome) -> float:
        weights = ctx.control_weights(self.score, self.clip)
        if weights is None:
            return float('nan')
        values = outcome.values
        imputed = float(np.dot(weights, values[ctx.control_pos]))
        return float(values[ctx.treated_pos].mean() - imputed)


class MatchingATT(ATTEstimator):
    """K-nearest-neighbour matching, treated -> control only.

    On a tree-based score (`ps_gbm`) the scores are piecewise constant and tie
    heavily, so neighbours within a tie block are arbitrary and this is blunter
    than on a continuous score. Tolerable -- tied e genuinely is tied -- but it
    is why `match_cov` remains the sharper matching baseline.
    """

    def __init__(self, space: str, k: int, name: str):
        self.space = space
        self.k = k
        self.name = name

    def estimate(self, ctx: TreatmentContext, outcome: Outcome) -> float:
        matches = ctx.match_indices(self.space)
        if matches is None:
            return float('nan')
        values = outcome.values
        imputed = values[matches].mean(axis=1)
        return float((values[ctx.treated_pos] - imputed).mean())


class OutcomeModelATT(ATTEstimator):
    """Direct-method ATT: `mean over treated of Y - f0(X)`, the Oaxaca-Blinder form.

    `f0` is the control-arm fit, so the observed treated outcome is kept and only
    the counterfactual is modelled. That is what makes this the one direct-method
    variant immune to the S-learner degeneracy: `mean[g(X,1) - g(X,0)]` returns
    *exactly* 0.0 whenever no tree in the ensemble splits on `T`, which at a 0.7%
    treated rate is the expected outcome rather than an edge case, and `pehe`
    actively rewards a column that predicts no effect when the truth is small.
    Anchoring on the observed treated mean cannot collapse that way.

    (The registry once carried three further variants -- pooled `g(X,1) - g(X,0)`,
    per-arm `f1 - f0`, and `Y - g(X,0)`. `f1 - f0` was numerically identical to
    this one, since `Y - f0` differs from it by the treated-arm model's in-sample
    mean residual, which is exactly zero for any squared-loss fit with an
    intercept. The two `g`-based variants collapse onto the OLS coefficient on
    `T` under a linear learner. None of the three was scored.)

    `sample='offered'` restricts the fit to the support. The direct method's
    advantage over IPW and matching is confounded with sample size: all three
    share the same treated mean and differ only in what imputes E[Y(0)|T=1],
    and on the measured card that is 106,735 controls for the outcome models
    against ~1,204 for the propensity-based ones, because IPW gives every
    not-offered row weight exactly 0. An offered-only fit equalises that pool,
    so the gap between a column and its `_offered` twin measures how much of the
    advantage is data volume -- and, equally, how much the 105k rows outside the
    treated population's support are moving the answer by extrapolation. Every
    treated row has `not_offered = 0`, so a fit on all rows is 98% out of
    overlap on the one covariate that determines treatment.

    `not_offered` is **dropped from the design** by `adjustment_columns`. Among
    controls it is the never-taker indicator, and every treated row sits at 0,
    so leaving it in made `f0` impute from the never-taker surface for all of
    them -- measured at +5.97 (lin) and +5.94 (gbm) against a true ATT of 2.0 on
    a synthetic with a deliberate 4.0 never-taker gap, versus +2.05 with it out.
    An earlier note here argued it was harmless because "across the support it
    is all zeros"; that is true of `sample='offered'` alone, and these columns
    fit on every control.

    `sample='offered'` -- registered as `dm_impute_t_att_gbm_nt` -- deserves its
    own note, because the two properties pull in opposite
    directions. **What it buys:** the control pool is exactly the never-takers
    (Z=1, T=0), so treated and control rows are alike in having been *offered*
    CardA. Every other `dm_*` column imputes from a pool that is ~98%
    never-offered, which is a covariate no treated row ever takes. **What it
    assumes:** that never-takers stand in for compliers at the same X. The
    exclusion diagnostic argues they do not. Compliers are by construction the
    drafters who would take a bomb in CardA's colours, so the never-taker cell
    is the bottom `1 - P(T=1|Z=1)` of that preference -- roughly 0.63 sd below
    the mean at the measured ~39% compliance -- and the gap survives matching on
    the covariates because complier status is latent. So expect this column to
    **understate** effects aligned with CardA's colours. It is here as a
    contrast against `dm_impute_t_att_gbm`, whose pool is the opposite extreme,
    not as the column to read on its own.

    The lighter `gbm15` learner goes with it: the never-taker pool is ~1,200
    rows against 106,735, and 100 trees at that size fit noise.
    """

    def __init__(self, learner: str, name: str, sample: str = 'all'):
        self.learner = learner
        self.name = name
        self.sample = sample

    def estimate(self, ctx: TreatmentContext, outcome: Outcome) -> float:
        # `control_outcome` rather than a three-model bundle: `f0` is the only
        # fit this reads, and requesting it through the bundle also fitted the
        # pooled and treated-arm models and discarded both -- two wasted GBM
        # fits per (card_a, group_b), on the dominant cost of the enrichment.
        control = ctx.control_outcome(outcome, self.learner, self.sample)
        if control is None:
            return float('nan')
        at_treated, _at_control = control
        return float((outcome.values[ctx.treated_pos] - at_treated).mean())


class DoublyRobustATT(ATTEstimator):
    """AIPW for the ATT: the direct method with an IPW correction.

        tau = mean_{T=1}[Y - m0(X)]
              - sum_{T=0} w_i (Y_i - m0(X_i)) / sum_{T=0} w_i,    w = e/(1-e)

    The first term is exactly `dm_impute_t_att`; the second corrects whatever
    imbalance `m0` left behind. Consistent if **either** nuisance is right, which
    is why it belongs here: the outcome model is the strong side (it gets all
    106,735 controls) and the propensity is the weak one (~1,971 informative
    rows), so neither alone is trustworthy but their combination is.

    `m0` is fitted on every row, keeping the direct method's data-volume
    advantage; the weights are zero off the support, so the correction only ever
    operates among the offered.

    This score is also the Neyman-orthogonal one, so **DML here is exactly this
    estimator with cross-fitted nuisances** -- `n_folds > 0` plus a cross-fitted
    propensity spec, and nothing else. Note the two sets of folds are drawn
    independently (the propensity's inside the support, the outcome's inside the
    control arm): every prediction is out of fold for its own nuisance, which is
    what kills own-observation bias, but a canonical DML would share one
    partition across both.
    """

    def __init__(
        self, score: str, clip: float, learner: str, name: str, n_folds: int = 0
    ):
        self.score = score
        self.clip = clip
        self.learner = learner
        self.name = name
        self.n_folds = n_folds

    def estimate(self, ctx: TreatmentContext, outcome: Outcome) -> float:
        weights = ctx.control_weights(self.score, self.clip)
        control = ctx.control_outcome(outcome, self.learner, 'all', self.n_folds)
        if weights is None or control is None:
            return float('nan')
        at_treated, at_control = control
        values = outcome.values
        treated_residual = values[ctx.treated_pos] - at_treated
        control_residual = values[ctx.control_pos] - at_control
        return float(
            treated_residual.mean() - np.dot(weights, control_residual)
        )


class CausalForestATT(ATTEstimator):
    """ATT as the average of a learned `tau(X)` over the treated rows.

    The only estimator here that *learns* how the effect varies instead of
    imposing a form on it. `mean(tau_hat(X_i))` over the treated is the ATT by
    definition -- averaging the CATE over the treated distribution.

    There is no "fit on the treated only" option and there cannot be: a causal
    forest forms its contrast *within* each leaf, so every leaf needs both
    arms. ATT-targeted APIs change the averaging weights, not the fit sample.

    **Comparison class.** It fits on the support, so its control pool is the
    ~1,204 offered-and-declined rows, not the ~107k the `dm_*` family uses.
    Read it against `sipw_*` and `match_ps_*`, not against `dm_impute_t_att_gbm`
    -- and if it is quoted against the latter, state the pool difference.
    Fitting off the support is not an option: `e == 0` there, so those rows
    carry no treatment variation at all.

    **When it adds nothing.** If `tau` is near-constant in `X` the forest
    reduces to a constant-effect estimator, which is not the ATE but the
    `Var(T|X)`-weighted average -- the same estimand `linear_att` reports.
    `sd(tau_hat)` over the treated rows, and the forest's
    `feature_importances_`, are how you find out.
    """

    name = 'cf_att'

    def estimate(self, ctx: TreatmentContext, outcome: Outcome) -> float:
        tau = ctx.causal_forest(outcome)
        return float(np.mean(tau)) if tau is not None else float('nan')


def default_estimators(
    k: int = 3, include_causal_forest: bool = True
) -> List[ATTEstimator]:
    """The nine baselines: two `sipw_*`, three matching, two direct-method,
    `dr_att_gbm`, and `dm_impute_t_att_gbm_nt` for diagnostic.

    Several columns exist only to be read as pairs, each isolating one factor:

        sipw_gbm               vs sipw_logit            the propensity model class
        match_ps_gbm_k3        vs match_ps_logit_k3     the same, in match space
        match_cov_k3           vs match_ps_*            the matching space
        dm_impute_t_att_gbm    vs dm_impute_t_att_lin   the outcome learner
        dr_att_gbm             vs dm_impute_t_att_gbm   the IPW correction term        

    **Every propensity reads the same covariates** (`CORE_COVARIATES`) and the
    same offered-only fit sample, so the first pair isolates linear-versus-tree
    and nothing else. An earlier round compared a cross-fitted tree against an
    in-sample logit, concluded the tree was worse, and could not separate the
    model class from the fold split; that confound is what this arrangement
    removes.

    The registry was pruned to the set actually scored. Gone with the columns:
    the unadjusted and diagnostic baselines (`corr`, `satt`, `std_satt`,
    `ctrl_mean` and their `_offered` twins, `linear_att`), the extra IPW clips,
    the three other direct-method variants under both learners, and
    `dml_att_gbm`. Null baselines have not been lost -- `metrics.add_null_baselines`
    emits `null_zero` and `null_mean`, and ranks them in the same table.
    """
    estimators: List[ATTEstimator] = [
        NaiveDiff(),
        # `clip=0`: `e = pi * e'` is bounded by the offer rate, so `e -> 1` is
        # structurally impossible and any clip below 0.992 is a no-op. The
        # column names drop the `_c05` suffix with it. See `control_weights`.
        StabilizedIpwATT('logit', 0.0, 'sipw_logit'),
        StabilizedIpwATT('gbm', 0.0, 'sipw_gbm'),
        MatchingATT('ps_logit', k, f'match_ps_logit_k{k}'),
        MatchingATT('ps_gbm', k, f'match_ps_gbm_k{k}'),
        MatchingATT('cov', k, f'match_cov_k{k}'),
        OutcomeModelATT('lin', 'dm_impute_t_att_lin'),
        OutcomeModelATT('gbm', 'dm_impute_t_att_gbm'),
        # Doubly robust: `dm_impute_t_att_gbm` plus an IPW correction for
        # whatever imbalance `m0` left behind. It reuses the control-arm fit
        # that column already paid for, so it costs nothing extra.
        DoublyRobustATT('gbm', 0.0, 'gbm', 'dr_att_gbm'),
        
    ]
    # Opt-in, because it dominates everything else: per pair it is `cv` pairs
    # of nuisance forests plus one causal forest, against the single
    # HistGradientBoosting fits the rest of the registry shares. `DML_FOLDS=5`
    # alone was 53% of enrichment runtime (91.0s with, 43.2s without, on 10
    # pairs) and a causal forest is heavier still. Returns nan without econml.
    if include_causal_forest:
        estimators.append(CausalForestATT())
    return estimators


######################
# Phase 2 -- orchestration
######################





class BaselineEnricher:
    """Phase 2: run every estimator over a df of (card_a, group_b) pairs.

    Produces estimates only; `enrich_pairs_estimates` joins them onto a frame.
    """

    def __init__(
        self,
        extractor: DraftDataExtractor,
        estimators: Optional[Sequence[ATTEstimator]] = None,
        k_neighbors: int = 3,
        estimator_names: Optional[Sequence[str]] = None,
    ):
        """`estimator_names` keeps only the estimators it names.

        Applied to whatever `estimators` resolved to, so it narrows a custom
        list as readily as the default registry, and the result stays in
        *registry* order rather than the caller's -- so `baseline_columns()`
        keeps a stable ordering however the subset is written.

        Names are validated. A typo would otherwise drop a method from the
        sweep silently, and the whole point of naming a subset is that the
        expensive run covers exactly what was asked for.
        """
        self.extractor = extractor
        self.k_neighbors = k_neighbors
        self.estimators = (
            list(estimators) if estimators is not None
            else default_estimators(k_neighbors)
        )
        if estimator_names is not None:
            available = {estimator.name for estimator in self.estimators}
            unknown = [n for n in estimator_names if n not in available]
            if unknown:
                raise ValueError(
                    f'unknown estimator name(s) {unknown}; available: '
                    f'{sorted(available)}'
                )
            wanted = set(estimator_names)
            self.estimators = [e for e in self.estimators if e.name in wanted]            
            if not self.estimators:
                raise ValueError(
                    'estimator_names selected nothing; pass None to keep all '
                    f'{len(available)} estimators.'
                )
        logging.info(f"estimators: {[e.name for e in self.estimators]}")

    def baseline_columns(self) -> List[str]:
        """Names of the frames `get_estimates` returns, in output order.

        The estimator name alone -- there is one sample (pack 2 pick 1), so the
        `p2p1_` prefix `baseline_methods` carried would be constant.
        """
        return [estimator.name for estimator in self.estimators]

    def get_estimates(self, pairs_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """One frame per estimator: `card_a`, `group_b`, `value`.

        Estimation only -- joining the results onto a pairs frame is
        `enrich_pairs_estimates`. Separating them means a run can be kept,
        cached or persisted per method, and a later run of a *subset* of
        estimators can be merged in without recomputing the rest.

        **Every frame carries a row for every pair, `nan` included.** So all of
        them come out the same length and key order and can be concatenated or
        compared directly, `value.isna()` says which cells degenerated, and
        "missing" downstream can mean exactly "the join did not match". `nan` is
        an ordinary result here: a CardA with no rows or an empty arm, a group
        with no `Y_` column, or an estimator that could not fit all produce one,
        and `_run` logs each.

        Pairs are de-duplicated per CardA, so a `pairs_df` holding the same
        `(card_a, group_b)` twice still yields one row per pair.
        """
        names = self.baseline_columns()
        cards: List[Any] = []
        groups: List[Any] = []
        values: Dict[str, List[float]] = {name: [] for name in names}

        for card_a, pairs in pairs_df.groupby('card_a', sort=False):
            logging.info("running: %s, %d", card_a, len(pairs))
            ctx = self._build_context(card_a)
            # dict.fromkeys de-duplicates while preserving order.
            for group_b in dict.fromkeys(pairs['group_b']):
                cards.append(card_a)
                groups.append(group_b)
                for estimator in self.estimators:
                    values[estimator.name].append(
                        self._run(estimator, ctx, group_b, card_a)
                    )

        return {
            name: pd.DataFrame({
                'card_a': cards, 'group_b': groups, 'value': values[name],
            })
            for name in names
        }

    def _build_context(self, card_a: str) -> Optional[TreatmentContext]:
        df = self.extractor.get_dataframe(card_a)
        if df.empty:
            return None
        ctx = TreatmentContext(df, self.k_neighbors)
        return ctx if ctx.is_valid else None

    @staticmethod
    def _run(
        estimator: ATTEstimator,
        ctx: Optional[TreatmentContext],
        group_b: str,
        card_a: str,
    ) -> float:
        """Run one estimator, mapping any failure to nan rather than raising."""
        if ctx is None:
            return float('nan')
        outcome = ctx.outcome(group_b)
        if outcome is None:
            return float('nan')
        try:
            value = estimator.estimate(ctx, outcome)
        except Exception as exc:  # pylint: disable=broad-except
            # A single ill-conditioned (CardA, group) must not abort the sweep.
            logging.exception("...")
            logging.warning(
                "estimator %s failed for %s / %s: %s",
                estimator.name, card_a, group_b, exc,
            )
            raise
            return float('nan')
        return float(value) if value is not None and np.isfinite(value) else float('nan')



######################
# Phase 3 -- hyperparameter tuning
######################

# Search spaces per profile: an explicit tuple of candidate values per
# parameter, so a space is a **finite grid** rather than a range to sample from.
# Ranges were replaced because pinning a value meant faking a degenerate window
# (`('log_int', 14, 16)` to mean 15), and because a five-value axis says what
# will actually be tried instead of leaving it to the draw.
#
# Every set **contains that profile's shipped value** from `PROFILES`, enforced
# at import by `_check_spaces_cover_profiles`. So the incumbent is always one of
# the candidates and a completed search can never return something worse than
# the default -- a guarantee sampling from a range could not give.
#
# Five values on four axes is 5^4 = 625 combinations, so the default `n=40`
# still samples rather than enumerates; `full_grid` runs all of them. `logit`
# has 5 in total, and `random_grid` returns the whole set rather than drawing 40
# with 35 duplicates.
#
# **`ridge` is deliberately absent.** `ridge`'s `alpha` is a
# numerical setting, not a statistical one -- see `_fit_ridge`. 
# `gbm_classifier`. Both stay tunable with an explicit grid.
DEFAULT_SPACES: Dict[str, Dict[str, Tuple[Any, ...]]] = {
    # Fits ~1,971 offered rows with ~613 events. The measured failure is
    # *under*-dispersion (predicted e spanning 0.209-0.421 against realized take
    # rates of 0.121-0.553), so the interesting direction is more capacity --
    # hence the shipped `max_iter` of 15 sitting at the bottom of its axis and
    # `max_depth` of 7 at the top of its own.
    'gbm_classifier': {
        'max_iter': (15, 25, 40, 60, 100),
        'learning_rate': (0.03, 0.05, 0.08, 0.12, 0.2),
        'max_depth': (3, 4, 5, 6, 7),
        'min_samples_leaf': (20, 30, 40, 60, 100),
    },

    # Fits ~107k rows, so it can afford finer leaves than `gbm_classifier`:
    # `min_samples_leaf` is an absolute count, and 40 of 107k is 0.04% against
    # 2% of 1,971.
    'gbm': {
        'max_iter': (50, 100, 150, 250, 300),
        'learning_rate': (0.02, 0.035, 0.05, 0.08, 0.12),
        'max_depth': (3, 4, 5, 6, 8),
        'min_samples_leaf': (20, 40, 80, 100),
    },

    # `C` is inverse regularisation strength, so it moves `e`'s dispersion --
    # exactly what a proper scoring rule reads. Bracketing the shipped 0.35 by
    # roughly a factor of 3 each way; the old 1e-3..1e3 range spent most of its
    # draws where the model is effectively unregularised. `penalty`,
    # `class_weight`, `solver` and `max_iter` are excluded by decision, not by
    # omission.
    'logit': {'C': (0.1, 0.2, 0.35, 0.6, 1.0)},

    # --- causal forest ---
    # `n_estimators`, `cv`, `honest` and `random_state` are deliberately absent:
    # the first two are budgets (cost is linear in them, exactly as `DML_FOLDS`
    # is), and `honest=False` would silently turn `cf_att` into a DR-learner --
    # the same class of change as `class_weight` on a propensity.
    'forest': {
        # In treated-per-leaf `k` at the cap's 2.50% share, and the leaves per
        # tree each implies (0.225 * n1 / k, at n1 ~ 1,282):
        #
        #     leaf   300   500  1000  2000  5000
        #     k      7.5  12.5    25    50   125
        #     leaves  38    23    12     6     2
        #
        # The previous space started at 1000, so every candidate sat at <=12
        # leaves and the top half was effectively a constant-effect estimator.
        # A search confined there cannot tell "tau is flat" from "the forest
        # was too coarse to see it", which is the question `cf_att` exists to
        # answer. This reaches k ~ 7.
        'min_samples_leaf': (300, 500, 1000, 2000, 5000),
        'max_features': (0.2, 0.3, 0.5, 0.7, 1.0),
        'max_samples': (0.2, 0.3, 0.4, 0.45, 0.5),
        'min_balancedness_tol': (0.05, 0.15, 0.25, 0.35, 0.45),
    },
    'forest_y': {
        'min_samples_leaf': (5, 10, 20, 40, 80),
        'max_features': (0.2, 0.3, 0.5, 0.7, 1.0),
    },
    'forest_t': {
        # Centred on 1200 rather than 40: at the ~2.50% treated share of the
        # fitted rows, 40 holds one treated row. See the profile.
        'min_samples_leaf': (400, 800, 1200, 2000, 3000),
        'max_features': (0.2, 0.3, 0.5, 0.7, 1.0),
    },
}

# Which kind of model sits behind each profile, and so how a candidate is
# scored: Brier on held-out offered rows for a treatment model, plain held-out
# MSE on the control arm for an outcome model.
PROFILE_KIND: Dict[str, str] = {
    'gbm_classifier': 'propensity',
    'logit': 'propensity',
    'gbm': 'outcome',
    'ridge': 'outcome',
    # The causal forest's two nuisances score like any other observed target,
    # but on the forest's own fit sample -- the support, both arms, full design
    # -- so they need their own scorers rather than reusing the two above.
    'forest_y': 'forest_y',
    'forest_t': 'forest_t',
    # `tau` is never observed, so there is no held-out column to difference
    # against. Scored on the R-loss instead; see `ProfileTuner._score_forest`.
    'forest': 'forest',
}

# The builder behind each profile, so a candidate is evaluated through exactly
# the code the estimators use -- pipeline, scaler and all.
PROFILE_BUILDERS: Dict[str, Callable[[], Any]] = {
    'gbm_classifier': _build_gbm_classifier,
    'logit': _build_logit,
    'gbm': _fit_gbm,
    'ridge': _fit_ridge,
    'forest_y': _build_forest_y,
    'forest_t': _build_forest_t,
}

_NO_DEFAULT_SPACE = (
    "profile %r has no default search space, so `tune` needs an explicit grid."
    " %s"
)

_RIDGE_FROZEN = (
    "`alpha` does a numerical job here, not a statistical one; see _fit_ridge."
    " Pass grids={'ridge': [...]} to override that judgement deliberately."
)

_TUNER_NO_ROWS = "profile %r: no card produced a usable train/test split"

_UNKNOWN_OBJECTIVE = (
    "objective %r is not scored for profile %r (kind %r); it reports: %s."
    " Balance objectives are propensity-only."
)

_NO_CANDIDATE_SCORED = (
    "tuning %r on %r: every candidate scored nan, so none could win. For a"
    " balance objective this usually means no card's holdout had both arms."
)

_SPACE_UNKNOWN_PARAM = (
    "DEFAULT_SPACES[%r] names %r, which is not a parameter of PROFILES[%r];"
    " it accepts: %s"
)

_SPACE_MISSES_DEFAULT = (
    "DEFAULT_SPACES[%r][%r] does not contain the shipped value %r, so a search"
    " could return something worse than the default. Its values are %s."
)


def _check_spaces_cover_profiles() -> None:
    """Every search space must contain its profile's shipped value.

    Run at import: the two literals sit next to each other, so editing one and
    not the other should fail immediately rather than silently produce a search
    that cannot reach the incumbent.
    """
    for profile, space in DEFAULT_SPACES.items():
        shipped = PROFILES.get(profile)
        if shipped is None:
            continue
        for key, values in space.items():
            if key not in shipped:
                raise ValueError(
                    _SPACE_UNKNOWN_PARAM % (profile, key, profile,
                                            sorted(shipped))
                )
            if shipped[key] not in values:
                raise ValueError(
                    _SPACE_MISSES_DEFAULT % (profile, key, shipped[key],
                                             list(values))
                )


_check_spaces_cover_profiles()


@dataclass(frozen=True)
class TunedProfile:
    """One profile's tuned parameters, its score, and how it was obtained.

    Provenance sits inside the profile rather than beside it, which is what lets
    files from separate runs merge as a plain dict without losing which cards
    and grid produced each entry. Without it a tuned profile is an unexplainable
    magic dict in six months.
    """

    params: Dict[str, Any]
    score: float                    # held-out MSE / Brier, lower is better
    provenance: Dict[str, Any]


def _space_axes(
    space: Mapping[str, Sequence[Any]]
) -> Tuple[List[str], List[Tuple[Any, ...]], int]:
    """A space as `(parameter names, value tuples, combination count)`."""
    if not space:
        raise ValueError('a search space must name at least one parameter')
    keys = list(space)
    values: List[Tuple[Any, ...]] = []
    total = 1
    for key in keys:
        choices = tuple(space[key])
        if not choices:
            raise ValueError(f'the search space for {key!r} lists no values')
        values.append(choices)
        total *= len(choices)
    return keys, values, total


def _combination(
    keys: List[str], values: List[Tuple[Any, ...]], index: int
) -> Dict[str, Any]:
    """Decode a flat combination index into one parameter dict.

    Mixed radix over the axes, so index `i` names the same combination for
    both `full_grid` and `random_grid` and the two orderings agree.
    """
    params: Dict[str, Any] = {}
    for key, choices in zip(keys, values):
        index, offset = divmod(index, len(choices))
        params[key] = choices[offset]
    return params


def full_grid(space: Mapping[str, Sequence[Any]]) -> List[Dict[str, Any]]:
    """Every combination of a space's explicit values."""
    keys, values, total = _space_axes(space)
    return [_combination(keys, values, index) for index in range(total)]


def random_grid(
    space: Mapping[str, Sequence[Any]], n: int, seed: int = 0
) -> List[Dict[str, Any]]:
    """`n` distinct parameter dicts drawn from a space, deterministically.

    Sampling is **without replacement**, so no candidate is ever evaluated
    twice -- which matters now that a space is finite, since 40 draws from
    `logit`'s 5 combinations would otherwise be 35 wasted fits. When `n`
    reaches the number of combinations the whole grid comes back, so a small
    space is enumerated rather than sampled with no call-site change.

    The result is in combination-index order, which keeps `_search`'s
    lowest-index tie-break meaningful and reproducible.
    """
    if n < 1:
        raise ValueError(f'n must be at least 1; got {n}')
    keys, values, total = _space_axes(space)
    if n >= total:
        return [_combination(keys, values, index) for index in range(total)]
    rng = np.random.default_rng(seed)
    picked = rng.choice(total, size=n, replace=False)
    return [_combination(keys, values, int(index)) for index in sorted(picked)]


def to_params(result: Dict[str, TunedProfile]) -> Dict[str, Dict[str, Any]]:
    """Just the parameters, in the shape `PROFILES` takes."""
    return {name: dict(tuned.params) for name, tuned in result.items()}


def save_profiles(result: Dict[str, TunedProfile], path: str) -> str:
    """Write tuned profiles as JSON, keyed by profile name."""
    payload = {
        name: {
            'params': tuned.params,
            'score': tuned.score,
            'provenance': tuned.provenance,
        }
        for name, tuned in result.items()
    }
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def load_profiles(path: str, apply: bool = True) -> Dict[str, TunedProfile]:
    """Read tuned profiles, by default pushing their `params` into `PROFILES`.

    Only `params` is applied; `score` and `provenance` are carried for the
    caller. Every key is validated by `set_profile`, so a stale file naming a
    parameter this code no longer has raises rather than being ignored.
    """
    with open(path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    result = {
        name: TunedProfile(
            params=entry['params'],
            score=float(entry.get('score', float('nan'))),
            provenance=entry.get('provenance', {}),
        )
        for name, entry in payload.items()
    }
    if apply:
        for name, tuned in result.items():
            set_profile(name, tuned.params)
    return result


class ProfileTuner:
    """Random search over one or more `PROFILES` entries, by held-out loss.

    One rule for both kinds of model: split each card's rows into train and
    test, fit on train, score squared error on test, and pool across every fit
    the profile drives. Lowest pooled test loss wins.

      * **Propensity** (`logit`, `gbm_classifier`) -- `mean((e_hat - T)^2)` over
        held-out offered rows. That is the Brier score, a **proper scoring
        rule**, decomposing into calibration plus resolution. Calibration is the
        property IPW actually needs, since it consumes `e` as a probability and
        the ATT control weight is the odds `e/(1-e)`: a model that ranks well
        and is wrong about levels is strictly worse. It should also catch the
        one failure on record -- an under-dispersed tree is miscalibrated, and a
        proper scoring rule penalises that. **AUC would not**: it is rank-only
        and blind to exactly this.
      * **Outcome** (`gbm`, `ridge`) -- `mean((y_hat - y)^2)` over
        held-out control rows, which is what `f0` is for and what
        `dm_impute_t_att` and the DR correction consume.

    **The benchmark can never be the objective.** Selecting profiles by scoring
    against `late` would be fitting the gold standard, and every downstream
    comparison would become meaningless. That is enforced structurally rather
    than by convention: this class takes the extractor and a list of `card_a`
    names, **never a pairs frame**, so there is no code path from here to
    `late`.

    Balance and Kish ESS are computed alongside every propensity candidate and
    returned by `score`, but they are **diagnostics, not the target**. If the
    loss-optimal profile and the best-balancing one disagree, that disagreement
    is worth seeing rather than hiding: balance is what a propensity exists for,
    so a large divergence says the score and the purpose have come apart.

    **Deviation from the plan, recorded.** `tune.md` asked for one draft split
    shared across every card *and* stratification by `T` within each card. Those
    two cannot both hold: a treated row for one CardA is a control row for
    another, so no single draft partition is `T`-stratified for all of them.
    Stratification wins, because it has an operational reason -- at a ~0.7%
    treated rate an unstratified split can leave a card with an empty arm. The
    split is drawn per card and is deterministic in `(seed, card_a)`, so it is
    identical across every candidate, which preserves the pairing that made a
    shared split attractive.

    Profiles are tuned **independently, not jointly**: `dr_att_gbm` reads
    `gbm_classifier` for `e` and `gbm` for `m0`, so the jointly optimal pair need
    not be the two individually optimal ones. Joint search is combinatorial and
    not worth it at this scale, but it makes a tuned `dr_att_gbm` a
    coordinate-wise optimum rather than a global one.
    """

    def __init__(
        self,
        extractor: DraftDataExtractor,
        card_a_names: Sequence[str],
        test_fraction: float = 0.3,
        seed: int = 0,
    ):
        if not 0.0 < test_fraction < 1.0:
            raise ValueError(
                f'test_fraction must lie in (0, 1); got {test_fraction}'
            )
        self.extractor = extractor
        self.card_a_names = list(card_a_names)
        self.test_fraction = test_fraction
        self.seed = seed
        self._contexts: Dict[str, Optional[TreatmentContext]] = {}
        self._splits: Dict[str, Optional[Tuple[np.ndarray, np.ndarray]]] = {}
        # Set by `tune`; None means every extracted group.
        self._group_names: Optional[List[str]] = None

    # ---- shared state ----
    def context(self, card_a: str) -> Optional[TreatmentContext]:
        """The card's `TreatmentContext`, built once and reused by every candidate."""
        if card_a not in self._contexts:
            df = self.extractor.get_dataframe(card_a)
            ctx = TreatmentContext(df) if not df.empty else None
            self._contexts[card_a] = ctx if ctx is not None and ctx.is_valid else None
        return self._contexts[card_a]

    def split(self, card_a: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """(train, test) row positions for one card, stratified by `T`.

        Deterministic in `(seed, card_a)` and cached, so every candidate in
        every profile sees the same split and comparisons between them are
        paired. Returns None when either arm would be empty on either side --
        such a card cannot score a propensity at all and is skipped with a
        warning rather than silently contributing a degenerate fit.
        """
        if card_a in self._splits:
            return self._splits[card_a]
        ctx = self.context(card_a)
        if ctx is None:
            self._splits[card_a] = None
            return None
        # crc32, not `hash`: Python randomises string hashing per process, so
        # `hash` would give a different split on every run and quietly break the
        # determinism the whole search rests on.
        rng = np.random.default_rng(
            zlib.crc32(f'{self.seed}:{card_a}'.encode('utf-8'))
        )
        train_parts, test_parts = [], []
        for arm in (ctx.treated_pos, ctx.control_pos):
            shuffled = rng.permutation(arm)
            n_test = int(round(self.test_fraction * len(shuffled)))
            if n_test == 0 or n_test == len(shuffled):
                self._splits[card_a] = None
                return None
            test_parts.append(shuffled[:n_test])
            train_parts.append(shuffled[n_test:])
        self._splits[card_a] = (
            np.sort(np.concatenate(train_parts)),
            np.sort(np.concatenate(test_parts)),
        )
        return self._splits[card_a]

    # ---- scoring ----
    def score(self, profile: str, params: Dict[str, Any]) -> Dict[str, float]:
        """Held-out loss for one candidate, plus diagnostics.

        Returns `loss` (Brier or MSE, lower better), `n` (held-out rows scored),
        `n_fits`, and for a propensity profile the balance and effective-sample
        diagnostics `smd_max`, `smd_mean` and `ess_share`, each averaged over
        the cards that contributed.
        """
        kind = self._require_kind(profile)
        for key in params:
            if key not in PROFILE_PARAMS[profile]:
                raise KeyError(
                    _UNKNOWN_PARAM % (profile, key,
                                      sorted(PROFILE_PARAMS[profile]))
                )
        saved = dict(PROFILES[profile])
        PROFILES[profile] = {**saved, **params}
        try:
            if kind == 'propensity':
                return self._score_propensity(profile)
            if kind == 'forest_t':
                return self._score_forest_t(profile)
            if kind == 'forest_y':
                return self._score_forest_y(profile)
            if kind == 'forest':
                return self._score_forest(profile)
            return self._score_outcome(profile)
        finally:
            PROFILES[profile] = saved

    def _score_propensity(self, profile: str) -> Dict[str, float]:
        build = PROFILE_BUILDERS[profile]
        total, count, fits = 0.0, 0, 0
        keys = ('smd_max', 'smd_mean', 'smd_rms', 'ess_share')
        collected: Dict[str, List[float]] = {key: [] for key in keys}
        for card_a in self.card_a_names:
            piece = self._propensity_card(card_a, build)
            if piece is None:
                continue
            squared, rows, diagnostics = piece
            total += squared
            count += rows
            fits += 1
            for key in keys:
                collected[key].append(diagnostics[key])
        if not count:
            raise ValueError(_TUNER_NO_ROWS % profile)
        measured = {
            'loss': total / count,
            'n': float(count),
            'n_fits': float(fits),
        }
        # nanmean: a card whose holdout has an empty arm contributes no balance
        # number, and must not turn the pooled diagnostic into nan.
        for key in keys:
            measured[key] = float(np.nanmean(collected[key]))
        return measured

    def _propensity_card(
        self, card_a: str, build: Callable[[], Any]
    ) -> Optional[Tuple[float, int, Dict[str, float]]]:
        """Squared error summed over one card's held-out offered rows."""
        ctx = self.context(card_a)
        parts = self.split(card_a)
        if ctx is None or parts is None:
            return None
        # The same view and the same fit sample the shipped propensity uses,
        # so a tuned profile is tuned on the model that will actually run.
        cols = [c for c in FEATURE_VIEWS['core'](ctx.x_cols) if c != 'not_offered']
        design = ctx.covariates[cols].to_numpy(dtype=float)
        support = set(ctx.support.tolist())
        train = np.array([i for i in parts[0] if i in support], dtype=np.intp)
        test = np.array([i for i in parts[1] if i in support], dtype=np.intp)
        if len(train) < 2 or len(test) == 0:
            return None
        if len(np.unique(ctx.treatment[train])) < 2:
            return None
        try:
            model = build().fit(design[train], ctx.treatment[train])
            predicted = model.predict_proba(design[test])[:, 1]
            everywhere = model.predict_proba(design)[:, 1]
        except (ValueError, np.linalg.LinAlgError) as exc:
            logging.warning('tuning fit failed for %s: %s', card_a, exc)
            return None
        squared = float(np.sum((predicted - ctx.treatment[test]) ** 2))
        return squared, len(test), self._diagnostics(
            ctx, everywhere, self._balance_design(ctx, cols),
            np.asarray(parts[1], dtype=np.intp),
        )

    @staticmethod
    def _balance_design(
        ctx: TreatmentContext, fit_cols: Sequence[str]
    ) -> np.ndarray:
        """Covariates to check balance on: the ones the propensity did *not* fit.

        Balance measured on the fit covariates is close to circular -- the
        balancing property guarantees that weighting by `e(X)` balances that
        `X`, and a logistic score's first-order conditions push it further
        still. So a fit-view SMD structurally favours `logit` over `gbm`, which
        makes it useless for the one comparison the registry exists to isolate.

        The frame carries ~28 covariates against `core`'s 8, so ~20 columns are
        genuinely held out. Falls back to the fit columns when there are none
        (the synthetic fixtures, whose frames are `CORE_COVARIATES` exactly);
        the number then means what it used to and is comparable within a run,
        just not across schemas.
        """
        wanted = [c for c in ctx.x_cols
                  if c != 'not_offered' and c not in set(fit_cols)]
        if not wanted:
            wanted = [c for c in fit_cols if c != 'not_offered']
        return ctx.covariates[wanted].to_numpy(dtype=float)

    @staticmethod
    def _diagnostics(
        ctx: TreatmentContext,
        scores: np.ndarray,
        design: np.ndarray,
        holdout: np.ndarray,
    ) -> Dict[str, float]:
        """Weighted covariate balance and Kish ESS under the ATT odds weight.

        **Held out on both axes**, which is what makes these usable for
        selection rather than only for reading:

          * *rows* -- computed on `holdout` alone, so the scores come from a
            model that never saw them. Previously this ran over every row,
            training rows included, so it rewarded overfitting exactly the way
            an in-sample Brier would.
          * *columns* -- `design` is `_balance_design`, the covariates outside
            the propensity's fit view. Balance on the fit view is close to
            circular; see there.

        `scores` arrives as `e'` -- raw `predict_proba` from `_propensity_card`,
        which does not go through `TreatmentContext.propensity` -- so it is
        rescaled by `P(offered)` exactly as `_fit_propensity` does. Without that
        the reported balance would describe weights no estimator uses.

        Three summaries of the same SMD vector, because they answer different
        questions. **`smd_rms` is the one to select on**: smooth enough for a
        search to rank candidates stably, but quadratic, so one badly balanced
        covariate cannot be traded away against small gains elsewhere.
        `smd_max` is an extreme order statistic over ~20 columns and jumps to
        whichever happens to be worst, which makes it a pass/fail guardrail (the
        conventional 0.1) rather than an objective. `smd_mean` is stable but
        lets a single failure hide.
        """
        nan = {'smd_max': float('nan'), 'smd_mean': float('nan'),
               'smd_rms': float('nan'), 'ess_share': float('nan')}
        treated_rows = np.intersect1d(holdout, ctx.treated_pos)
        control_rows = np.intersect1d(holdout, ctx.control_pos)
        if not len(treated_rows) or not len(control_rows) or not design.shape[1]:
            return nan
        pi = len(ctx.support) / len(ctx.treatment)
        ceiling = 1.0 - _E_FLOOR
        clipped = np.minimum(pi * scores, ceiling)[control_rows]
        weights = clipped / (1.0 - clipped)
        total = weights.sum()
        if not np.isfinite(total) or total <= 0:
            return nan
        weights = weights / total
        treated = design[treated_rows]
        controls = design[control_rows]
        spread = treated.std(axis=0, ddof=0)
        spread = np.where(spread > 0, spread, 1.0)
        smd = np.abs(treated.mean(axis=0) - weights @ controls) / spread
        return {
            'smd_max': float(np.max(smd)),
            'smd_mean': float(np.mean(smd)),
            'smd_rms': float(np.sqrt(np.mean(smd ** 2))),
            'ess_share': float(1.0 / np.sum(weights ** 2) / len(control_rows)),
        }

    def _score_outcome(self, profile: str) -> Dict[str, float]:
        build = PROFILE_BUILDERS[profile]
        total, count, fits = 0.0, 0, 0
        for card_a in self.card_a_names:
            for group_b in self._groups_for():
                piece = self._outcome_fit(card_a, group_b, build)
                if piece is None:
                    continue
                total += piece[0]
                count += piece[1]
                fits += 1
        if not count:
            raise ValueError(_TUNER_NO_ROWS % profile)
        return {'loss': total / count, 'n': float(count), 'n_fits': float(fits)}

    def _outcome_fit(
        self, card_a: str, group_b: str, build: Callable[[], Any]
    ) -> Optional[Tuple[float, int]]:
        """Squared error summed over one (card, group)'s held-out control rows.

        On `adjustment_columns`, the same design `_fit_control_outcome` uses --
        otherwise a tuned profile is tuned on a model that never runs.
        """
        ctx = self.context(card_a)
        parts = self.split(card_a)
        if ctx is None or parts is None:
            return None
        outcome = ctx.outcome(group_b)
        if outcome is None or not np.isfinite(outcome.values).all():
            return None
        controls = set(ctx.control_pos.tolist())
        train = np.array([i for i in parts[0] if i in controls], dtype=np.intp)
        test = np.array([i for i in parts[1] if i in controls], dtype=np.intp)
        if len(train) < 2 or len(test) == 0:
            return None
        cols = np.asarray(ctx.adjustment_columns(), dtype=np.intp)
        try:
            model = build().fit(
                ctx.design[np.ix_(train, cols)], outcome.values[train]
            )
            predicted = model.predict(ctx.design[np.ix_(test, cols)])
        except (ValueError, np.linalg.LinAlgError) as exc:
            logging.warning(
                'tuning fit failed for %s / %s: %s', card_a, group_b, exc
            )
            return None
        squared = float(np.sum((predicted - outcome.values[test]) ** 2))
        return squared, len(test)

    # ---- causal forest ----
    #
    # Three models, and they are not in the same situation. `forest_y` and
    # `forest_t` predict something observed, so they score exactly like the
    # other nuisances -- but on the forest's own sample (`forest_rows()`, both
    # arms, `forest_columns()`), which is why they cannot reuse
    # `_score_outcome` and `_score_propensity`. `tau` is never observed for
    # anybody, so the forest scores on the R-loss instead.
    #
    # **Order matters, and nothing enforces it.** `_fit_causal_forest` reads
    # `PROFILES['forest_y']` and `PROFILES['forest_t']` at call time, so tune
    # and `set_profile` both nuisances *before* tuning `forest`. Otherwise its
    # R-loss is computed from residuals produced by the default nuisances --
    # ranking candidates against residuals that will not be used in
    # production. Nothing raises; the search silently answers another question.
    def _card_split(
        self, card_a: str
    ) -> Optional[Tuple[TreatmentContext, np.ndarray, np.ndarray, np.ndarray]]:
        """Train/test rows and the column set, matching what the forest fits.

        Both restrictions matter, and all three forest scorers share them:

          * **rows** -- `forest_rows()`, not every row. `_fit_causal_forest`
            keeps every treated row plus a capped control sample, so a tuner
            scoring the whole table measures a model on ~2.7x the data.
          * **columns** -- `forest_columns()`, not the full design. econml is
            handed `design[:, cols]`, and leaving `not_offered` in was the
            worse of the two: it predicts `T` almost perfectly (`T=0` wherever
            `not_offered=1`), so held-out Brier collapsed toward zero for every
            candidate and the `forest_t` ranking was noise.
        """
        ctx = self.context(card_a)
        parts = self.split(card_a)
        if ctx is None or parts is None:
            return None
        fitted = set(ctx.forest_rows().tolist())
        train = np.array([i for i in parts[0] if i in fitted], dtype=np.intp)
        test = np.array([i for i in parts[1] if i in fitted], dtype=np.intp)
        if len(train) < 2 or len(test) == 0:
            return None
        return ctx, train, test, np.asarray(ctx.forest_columns(), dtype=np.intp)

    def _score_forest_t(self, profile: str) -> Dict[str, float]:
        """Held-out Brier for `model_t`, on the forest's own rows and columns."""
        total, count, fits = 0.0, 0, 0
        for card_a in self.card_a_names:
            piece = self._card_split(card_a)
            if piece is None:
                continue
            ctx, train, test, cols = piece
            if len(np.unique(ctx.treatment[train])) < 2:
                continue
            try:
                model = PROFILE_BUILDERS[profile]().fit(
                    ctx.design[np.ix_(train, cols)], ctx.treatment[train]
                )
                predicted = model.predict_proba(
                    ctx.design[np.ix_(test, cols)]
                )[:, 1]
            except (ValueError, np.linalg.LinAlgError) as exc:
                logging.warning('tuning fit failed for %s: %s', card_a, exc)
                continue
            total += float(np.sum((predicted - ctx.treatment[test]) ** 2))
            count += len(test)
            fits += 1
        if not count:
            raise ValueError(_TUNER_NO_ROWS % profile)
        return {'loss': total / count, 'n': float(count), 'n_fits': float(fits)}

    def _score_forest_y(self, profile: str) -> Dict[str, float]:
        """Held-out MSE for `model_y` = E[Y|X], marginal over `T`.

        Both arms, which is what makes it a different fit from `_outcome_fit`'s
        control-arm-only `f0` -- but on the forest's rows and columns, not the
        whole table.
        """
        total, count, fits = 0.0, 0, 0
        for card_a in self.card_a_names:
            piece = self._card_split(card_a)
            if piece is None:
                continue
            ctx, train, test, cols = piece
            for group_b in self._groups_for():
                outcome = ctx.outcome(group_b)
                if outcome is None or not np.isfinite(outcome.values).all():
                    continue
                try:
                    model = PROFILE_BUILDERS[profile]().fit(
                        ctx.design[np.ix_(train, cols)], outcome.values[train]
                    )
                    predicted = model.predict(ctx.design[np.ix_(test, cols)])
                except (ValueError, np.linalg.LinAlgError) as exc:
                    logging.warning('tuning fit failed for %s / %s: %s',
                                    card_a, group_b, exc)
                    continue
                total += float(np.sum((predicted - outcome.values[test]) ** 2))
                count += len(test)
                fits += 1
        if not count:
            raise ValueError(_TUNER_NO_ROWS % profile)
        return {'loss': total / count, 'n': float(count), 'n_fits': float(fits)}

    def _score_forest(self, profile: str) -> Dict[str, float]:
        """Held-out R-loss for the causal forest.

        econml's `.score(Y, T, X)` residualises the held-out rows with the
        nuisances fitted on train and returns exactly `r_loss`, so the
        residuals are out-of-sample by construction -- the condition that
        makes the criterion valid rather than a reward for overfitting.

        Unlike the estimator path, a missing econml **raises** here: returning
        a nan loss would silently make every candidate tie.
        """
        forest_p = _profile(profile)
        cv = int(forest_p['cv'])
        forest_class = _causal_forest_class()
        if forest_class is None:
            raise ImportError(_ECONML_MISSING)
        total, count, fits = 0.0, 0, 0
        for card_a in self.card_a_names:
            piece = self._card_split(card_a)
            if piece is None:
                continue
            ctx, train, test, cols = piece
            treated = int(np.sum(ctx.treatment[train] == 1.0))
            if treated < cv or len(train) - treated < cv:
                continue
            for group_b in self._groups_for():
                outcome = ctx.outcome(group_b)
                if outcome is None or not np.isfinite(outcome.values).all():
                    continue
                try:
                    forest = forest_class(
                        model_y=_build_forest_y(),
                        model_t=_build_forest_t(),
                        discrete_treatment=True, **forest_p,
                    )
                    forest.fit(outcome.values[train], ctx.treatment[train],
                               X=ctx.design[np.ix_(train, cols)])
                    loss = float(forest.score(
                        outcome.values[test], ctx.treatment[test],
                        X=ctx.design[np.ix_(test, cols)],
                    ))
                except (ValueError, np.linalg.LinAlgError) as exc:
                    logging.warning('tuning fit failed for %s / %s: %s',
                                    card_a, group_b, exc)
                    continue
                total += loss * len(test)      # row-weighted, as elsewhere
                count += len(test)
                fits += 1
        if not count:
            raise ValueError(_TUNER_NO_ROWS % profile)
        return {'loss': total / count, 'n': float(count), 'n_fits': float(fits)}

    def _groups_for(self) -> List[str]:
        groups = self._group_names
        return groups if groups is not None else list(self.extractor.groups)

    # ---- search ----
    def tune(
        self,
        profiles: Union[str, Sequence[str]],
        grids: Optional[Dict[str, Sequence[Dict[str, Any]]]] = None,
        n: int = 40,
        group_b_names: Optional[Sequence[str]] = None,
        objective: str = 'loss',
    ) -> Dict[str, TunedProfile]:
        """Tune one profile or several. **Always** returns a dict keyed by name.

        So a one-profile run and a five-profile run merge identically, which is
        the point: separate runs are combined with `{**a, **b}` and written with
        `save_profiles`.

        `group_b_names` is optional because a propensity profile does not need
        it -- `e(X)` is group-independent, so `logit` and `gbm_classifier` sample
        cards alone while outcome profiles sample `(card, group)` pairs.

        `objective` names the key of `score` to minimise. `'loss'` (held-out
        Brier or MSE) is the default and the only one every profile kind
        produces. A propensity profile also offers `'smd_rms'`, `'smd_mean'`
        and `'smd_max'` -- balance is what the propensity exists for, and it is
        a defensible target now that `_diagnostics` is held out on both rows
        and columns. Of the three, prefer **`smd_rms`**: `smd_max` is an
        extreme order statistic and ranks candidates by which covariate happens
        to be worst, `smd_mean` lets one bad column hide.

        Whatever is chosen, it is still not the benchmark -- see the class
        docstring. Every option here is computable without `late`.
        """
        names = [profiles] if isinstance(profiles, str) else list(profiles)
        self._group_names = list(group_b_names) if group_b_names else None
        result: Dict[str, TunedProfile] = {}
        for name in names:
            grid = self._grid_for(name, grids, n)
            result[name] = self._search(name, grid, len(grid), objective)
        return result

    def _grid_for(
        self,
        name: str,
        grids: Optional[Dict[str, Sequence[Dict[str, Any]]]],
        n: int,
    ) -> List[Dict[str, Any]]:
        self._require_kind(name)
        if grids and name in grids:
            return [dict(candidate) for candidate in grids[name]]
        if name not in DEFAULT_SPACES:
            hint = _RIDGE_FROZEN if name == 'ridge' else (
                'Its capacity follows gbm_classifier by the sample-size '
            )
            raise ValueError(_NO_DEFAULT_SPACE % (name, hint))
        return random_grid(DEFAULT_SPACES[name], n, self.seed)

    def _search(
        self, name: str, grid: Sequence[Dict[str, Any]], n: int,
        objective: str = 'loss',
    ) -> TunedProfile:
        if not grid:
            raise ValueError(f'tuning {name!r} got an empty grid')
        best_params: Dict[str, Any] = {}
        best: Dict[str, float] = {objective: float('inf')}
        for index, candidate in enumerate(grid):
            measured = self.score(name, candidate)
            if objective not in measured:
                raise KeyError(
                    _UNKNOWN_OBJECTIVE % (objective, name, PROFILE_KIND[name],
                                          sorted(measured))
                )
            logging.info(
                'tune %s [%d/%d] %s=%.6f (loss=%.6f) %s',
                name, index + 1, len(grid), objective, measured[objective],
                measured['loss'], candidate,
            )
            # A nan candidate must never win, and `nan < x` is False, so the
            # comparison already refuses it -- but only in this direction.
            if measured[objective] < best[objective]:
                best, best_params = measured, dict(candidate)
        if not best_params:
            raise ValueError(_NO_CANDIDATE_SCORED % (name, objective))
        provenance: Dict[str, Any] = {
            'kind': PROFILE_KIND[name],
            'objective': objective,
            'objective_meaning': (
                'held-out Brier' if objective == 'loss'
                and PROFILE_KIND[name] == 'propensity'
                else 'held-out MSE' if objective == 'loss'
                else 'held-out weighted covariate balance, out-of-view columns'
            ),
            'cards': list(self.card_a_names),
            'groups': self._group_names,
            'grid_size': n,
            'seed': self.seed,
            'test_fraction': self.test_fraction,
            'baseline': dict(PROFILES[name]),
        }
        provenance.update({
            key: value for key, value in best.items() if key != objective
        })
        # `score` is the value of whatever was optimised, so two runs are only
        # comparable when their `provenance['objective']` agrees.
        return TunedProfile(best_params, float(best[objective]), provenance)

    @staticmethod
    def _require_kind(profile: str) -> str:
        if profile not in PROFILE_KIND:
            raise KeyError(_UNKNOWN_PROFILE % (profile, sorted(PROFILE_KIND)))
        return PROFILE_KIND[profile]
    


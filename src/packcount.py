"""Joint distribution of pack lengths over a 17lands drafts CSV.

Answers one question: for each draft, how many picks did it record in pack 1,
pack 2 and pack 3, and how many drafts share each combination?

    counter = PackCounter()
    counter.process_drafts_file(DRAFTS_CSV)
    counter.get_counts()      # {(15, 15, 15): 94210, (15, 15, 7): 312, ...}

or in one call, `count_pack_lengths(DRAFTS_CSV)`. From the shell::

    python packcount.py <drafts_csv> <dest.json>

which writes the same dict as JSON. Object keys must be strings there, so
`(15, 15, 15)` is written as `"15,15,15"` -- see `to_json_dict`.

Why this exists
---------------
`ivexp.is_complete_draft` drops a draft unless every pack up to the outcome
pack has as many picks as the longest pack in that draft, and
`baselines._is_complete_draft` mirrors it. Both report only a count of what
they dropped, and `ivexp`'s docstring names the blind spot that leaves: a
*uniformly* short draft passes the rule and is kept, because it is
indistinguishable from a genuinely smaller set. Only the joint distribution
shows it -- a `(14, 14, 14)` key sitting next to a modal `(15, 15, 15)` is a
truncated draft that both modules silently keep.

Two identities tie this module to those:

  * `is_complete_draft` keeps a draft exactly when all three lengths are equal,
    so `n_uneven()` is `ivexp`'s `n_incomplete` -- exactly, not approximately.
  * if the modal key covers essentially every draft after filtering, then
    `n_prior` and `horizon` are constant by construction, which is what the
    `covs.md` suggestions to drop the `rate_*` columns and the two constant
    covariates rest on.

Reading the CSV
---------------
`pack_number` is **0-indexed in the file**; the tuple is 1-indexed in its name,
so slot 0 of a key is pack 1.

A pack's length is the number of rows carrying that `pack_number` -- the same
definition `is_complete_draft` uses, which is what makes the identities above
exact rather than close. A duplicated export row would inflate both alike; this
counts rows, not distinct `pick_number`s, and the two would diverge there.

A pack with no rows contributes `0` rather than a missing slot, so a draft
abandoned during pack 3 reads `(15, 15, 0)`.

Reading follows `ivexp` exactly -- `csv.DictReader`, one row at a time, flushed
when `draft_id` changes and again at end of file -- so the two modules stay the
same shape and `process_drafts` accepts the same reader.

One difference, and it is a reduction: the draft's rows are not buffered. `ivexp`
accumulates `current_draft_rows` because it needs them; only a per-draft tally
of `pack_number` is needed here, so memory is constant in draft size.

Recorded, not acted on: a drafts CSV carries a `pack_card_<name>` and a
`pool_<name>` column per card, so `DictReader` builds a several-hundred-key dict
per row to reach the two fields this uses. If that ever dominates, `csv.reader`
with the two column positions resolved from the header is the cheaper read.

Repeated `process_drafts_file` calls accumulate, so a set split across files
sums correctly.
"""

import argparse
import csv
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple


import pandas as pd


# A draft is three boosters.
N_PACKS = 3

DRAFT_COLUMN = 'draft_id'
PACK_COLUMN = 'pack_number'

# JSON object keys must be strings, so a `(15, 15, 15)` key is written as
# `"15,15,15"`. Read one back with
# `tuple(int(part) for part in key.split(KEY_SEPARATOR))`.
KEY_SEPARATOR = ','

_NO_DRAFTS = 'no drafts processed yet; call process_drafts_file first'

_MISSING_COLUMN = (
    "the drafts file has no %r column; it has %d columns beginning %s. This is"
    " not a 17lands drafts export, or its schema changed."
)

_NON_CONTIGUOUS = (
    "draft %r reappeared after its rows were already tallied, so it is counted"
    " more than once. Either a file is not grouped by %s -- sort it before"
    " counting -- or the same draft appears in two of the processed files."
)

_SUMMARY = 'pack lengths: %d combination(s) over %d drafts, %d rows, %d file(s)'

_OUT_OF_RANGE = (
    "ignored %d row(s) whose %s was outside 0..%d. The key is a fixed %d-tuple,"
    " so those picks are in no pack; raise n_packs if the set really has more."
)


class PackCounter:
    """Counts drafts by their `(pack 1, pack 2, pack 3)` pick counts.

    Streams the file row by row, grouping by `draft_id`, exactly as
    `ivexp.PackCausalExperiment` does -- a draft is tallied when the id changes
    and again at end of file. That assumes the file is grouped by draft, which
    a 17lands export is; a violation is detected and warned about rather than
    silently double-counting.
    """

    def __init__(self, n_packs: int = N_PACKS) -> None:
        if n_packs < 1:
            raise ValueError(f'n_packs must be at least 1; got {n_packs}')
        self.n_packs = n_packs
        self.counts: Dict[Tuple[int, ...], int] = {}
        self.n_rows = 0
        self.n_out_of_range = 0
        self.sources: List[str] = []
        self._seen: set = set()
        self._warned_repeat = False

    def process_drafts_file(self, csv_path: str) -> None:
        """Tally every draft in one drafts CSV, accumulating across calls."""
        with open(csv_path, mode='r', encoding='utf-8') as file:
            self.process_drafts(csv.DictReader(file))
        self.sources.append(csv_path)

    def process_drafts(self, reader: Any) -> None:
        """Tally from an open `csv.DictReader`, one row at a time."""
        current_id: Optional[str] = None
        lengths: Dict[int, int] = {}
        checked = False

        for index, row in enumerate(reader):
            if not checked:
                self._check_columns(row)
                checked = True
            if index % 200000 == 0:
                logging.info("processed %d", index)
            draft_id = row[DRAFT_COLUMN]
            if draft_id != current_id:
                self._flush(current_id, lengths)
                current_id = draft_id
                lengths = {}
            pack = int(row[PACK_COLUMN])
            lengths[pack] = lengths.get(pack, 0) + 1
            self.n_rows += 1

        self._flush(current_id, lengths)

        if self.n_out_of_range:
            logging.warning(
                _OUT_OF_RANGE, self.n_out_of_range, PACK_COLUMN,
                self.n_packs - 1, self.n_packs,
            )

    def get_counts(self) -> Dict[Tuple[int, ...], int]:
        """`{(p1, p2, p3): n_drafts}`, a copy so callers cannot mutate state."""
        if not self.counts:
            raise ValueError(_NO_DRAFTS)
        return dict(self.counts)

    def n_drafts(self) -> int:
        """How many drafts were tallied."""
        return sum(self.get_counts().values())

    def n_uneven(self) -> int:
        """Drafts whose packs are not all the same length.

        `ivexp.is_complete_draft` keeps a draft exactly when every pack matches
        the longest one, so this is that rule's drop count. It does **not**
        include a uniformly short draft, which that rule keeps -- compare the
        keys themselves against the modal one to see those.
        """
        return sum(count for key, count in self.get_counts().items()
                   if len(set(key)) != 1)

    def get_dataframe(self) -> pd.DataFrame:
        """One row per distinct combination, commonest first.

        Columns `p1 .. p<n_packs>`, `n_drafts`, `share`. Ties in `n_drafts`
        break on the key, so the frame is stable across runs.
        """
        counts = self.get_counts()
        total = sum(counts.values())
        rows: List[Dict[str, Any]] = []
        for key in sorted(counts, key=lambda entry: (-counts[entry], entry)):
            row: Dict[str, Any] = {
                f'p{pack + 1}': key[pack] for pack in range(self.n_packs)
            }
            row['n_drafts'] = counts[key]
            row['share'] = counts[key] / total
            rows.append(row)
        logging.info(
            _SUMMARY, len(rows), total, self.n_rows, len(self.sources)
        )
        return pd.DataFrame(rows)

    def _flush(self, draft_id: Optional[str], lengths: Dict[int, int]) -> None:
        """Fold one finished draft into `counts`."""
        if draft_id is None or not lengths:
            return
        if draft_id in self._seen and not self._warned_repeat:
            logging.warning(_NON_CONTIGUOUS, draft_id, DRAFT_COLUMN)
            self._warned_repeat = True
        self._seen.add(draft_id)
        self.n_out_of_range += sum(
            count for pack, count in lengths.items()
            if not 0 <= pack < self.n_packs
        )
        key = tuple(lengths.get(pack, 0) for pack in range(self.n_packs))
        self.counts[key] = self.counts.get(key, 0) + 1

    @staticmethod
    def _check_columns(row: Dict[str, Any]) -> None:
        """Fail on the first row rather than KeyError-ing deep in the walk."""
        for name in (DRAFT_COLUMN, PACK_COLUMN):
            if name not in row:
                raise ValueError(
                    _MISSING_COLUMN % (name, len(row), list(row)[:5])
                )


def count_pack_lengths(
    csv_path: str, n_packs: int = N_PACKS
) -> Dict[Tuple[int, ...], int]:
    """`{(p1, p2, p3): n_drafts}` for one drafts CSV."""
    counter = PackCounter(n_packs)
    counter.process_drafts_file(csv_path)
    return counter.get_counts()


def to_json_dict(counts: Dict[Tuple[int, ...], int]) -> Dict[str, int]:
    """`counts` with JSON-legal keys, ordered by pack lengths ascending.

    The ordering is done here rather than by `json.dump(sort_keys=True)`,
    which would sort the keys as *strings* and so put `"9,9,9"` after
    `"15,15,15"`.
    """
    return {
        KEY_SEPARATOR.join(str(length) for length in key): count
        for key, count in sorted(counts.items())
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Count drafts by their per-pack pick counts.'
    )
    parser.add_argument('drafts_csv', help='path to a 17lands drafts CSV')
    parser.add_argument('dest', help='path to write the JSON result to')
    parser.add_argument(
        '--n-packs', type=int, default=N_PACKS,
        help=f'boosters per draft (default {N_PACKS})',
    )
    return parser.parse_args(argv)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s'
    )
    counter = PackCounter(args.n_packs)
    counter.process_drafts_file(args.drafts_csv)
    counts = counter.get_counts()
    with open(args.dest, mode='w', encoding='utf-8') as handle:
        json.dump(to_json_dict(counts), handle, indent=2)
        handle.write('\n')
    logging.info(
        'wrote %d combination(s) over %d drafts (%d uneven) to %s',
        len(counts), counter.n_drafts(), counter.n_uneven(), args.dest,
    )
    return 0


if __name__ == '__main__':
    sys.exit(_main())

"""A seeded stream of doubles, independent of every other random source.

`RndSeq` exists to split a drafts CSV between `ivexp` and `baselines` without
either side knowing about the other: both walk the same file in the same order
and draw exactly one value per draft, so the same seed hands each draft to
exactly one side. `ivexp` keeps `x < SPLIT_THRESHOLD`, the extractor keeps the
rest; `take_below` picks the side.

It wraps a private `random.Random(seed)`. Python guarantees `Random.random()`
gives the same sequence for the same seed across versions, and an instance is
untouched by `random.seed`, numpy or any other randomness in the process --
which is the "consistent across runs, independent of other random operations"
requirement in one line. Stdlib only, so both modules can import it at the top.
"""

import random
from typing import List

# The default split point; `ivexp` takes below, the extractor above.
SPLIT_THRESHOLD = 0.5


class RndSeq:
    """Deterministic stream of doubles in [0, 1) from a seed."""

    def __init__(self, seed: int, threshold: float = SPLIT_THRESHOLD,
                 take_below: bool = True):
        self.seed = seed
        self.threshold = threshold
        self.take_below = take_below
        self._rng = random.Random(seed)

    def get_next(self, n: int) -> List[float]:
        """The next `n` doubles."""
        return [self._rng.random() for _ in range(n)]

    def get_next_bool(self) -> bool:
        """One draw: whether it falls on this instance's side of `threshold`."""
        return (self._rng.random() < self.threshold) == self.take_below

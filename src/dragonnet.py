"""DragonNet ATT baseline: one multi-output net per CardA, two ATT forms.

A neural baseline next to the `baselines.py` estimators, on the same
extractor and with the same enricher surface, so it drops into
`evaluation.enrich_pairs_estimates` unchanged::

    enricher  = dragonnet.DragonNetEnricher(extractor)
    estimates = enricher.get_estimates(pairs_df)
    # {'dragonnet_imp_att':  DataFrame(card_a, group_b, value),
    #  'dragonnet_cate_att': DataFrame(card_a, group_b, value)}
    enricher.fit_reports_df()      # one audit row per CardA, see FitReport

**Written for another environment.** torch is not installed where this was
authored and the module has never been executed there; only pylint and mypy
were run. The synthetic check under "Verification" is what to run first.

What the net is
---------------
Shi, Blei & Veitch (2019): a shared trunk `Phi(x)` and three heads --
`g(Phi)` the propensity logit, `Q0(Phi) = E[Y | T=0, x]`,
`Q1(Phi) = E[Y | T=1, x]`. The propensity head is a regulariser: it forces
the representation to keep whatever predicts treatment, i.e. the confounders.
CATE per row is `Q1 - Q0`. No targeted regularisation / epsilon head here;
simplicity first.

**The outcome heads are multi-output.** One CardA has ~35 `Y_<group>`
columns, so `Q0, Q1 : R^64 -> R^G` and one fit per CardA covers every group.
A run is ~40 fits rather than ~1,400, and the propensity head is shared
across groups by construction, which is what DragonNet intends anyway.

Two ATT forms from the one fit
------------------------------
  * `dragonnet_imp_att`  = mean over treated of `Y - Q0(x)`, the imputation
                           form `dm_impute_t_att_*` uses.
  * `dragonnet_cate_att` = mean over treated of `Q1(x) - Q0(x)`.

`baselines.py` documents that these were identical for its squared-loss fits
with an intercept (zero mean residual on the treated). A net has no such
guarantee, and `Q1` is trained on ~1.3k treated rows against ~137k controls,
so they can differ; both are read off the same `q0`, `q1` arrays for two
vector operations per pair. A large gap on the same fit says the treated-arm
head is under-determined.

Every row, `not_offered` dropped
--------------------------------
Training uses every row -- treated plus all controls, offered or not -- as
`baselines._fit_causal_forest` does, with `not_offered` **removed from the
design** (`TreatmentContext.adjustment_columns`). That exclusion is
load-bearing: with the column in, a model learns `P(T=1 | not offered) = 0`
and the never-taker gap, which `adjustment_columns`' docstring measured at
ATT ~6 against a true 2. The propensity head then learns the pooled
`P(T=1|x) = pi * e(x)`, the same balancing score up to scale that the
forest's `model_t` learns. Masking the BCE to offered rows is a one-line
change if a support-only variant is ever wanted; not done.

Autonomous training
-------------------
Fits run unsupervised, many in parallel, so nothing here is a guess that
needs watching:

  * Adam at `lr` (1e-3, safe because X and Y are standardised);
    `ReduceLROnPlateau` on val loss halves it on a plateau, floor `lr_min`.
  * Early stopping after `stop_patience` epochs without val improvement;
    `max_epochs` is a cap, not a target, and hitting it is recorded.
  * Gradient-norm clipping; a non-finite loss aborts the fit (`diverged`).
  * The val split is stratified by `T`, so `Q1` always has val rows.
  * A trivial baseline -- predict each group's mean and the treatment base
    rate -- is computed on the val set; a fit that does not beat it is
    `degenerate`. Degenerate and diverged cards return nan for every pair.
  * Every fit returns a `FitReport`; the enricher keeps one per CardA in
    `fit_reports`, so a parallel run is audited afterwards by scanning that
    frame: `best_epoch` of 1-2 says the LR was too high at the start,
    `hit_cap` with `best_epoch` near the end says under-trained,
    `n_lr_drops` says how hard the schedule worked.

Per CardA, shardable
--------------------
Each fit builds its own `TreatmentContext`, net and seeded generator; there
is no cross-card state. A `pairs_df` holding one `card_a` gives bit-identical
values to the same rows of a full run, so one process per CardA and
concatenation is the same estimator. `num_threads` pins torch's CPU threads
per fit so parallel workers do not each take every core.

Verification (for the environment that has torch)
-------------------------------------------------
  1. Synthetic: `Y = 2*T + X @ b + noise`, ~50k rows, ~1% treated; both
     estimators should return ~2.0 within a few tenths, `degenerate` False.
  2. Schema: both frames have exactly `card_a, group_b, value`, one row per
     pair, same order, and pass `evaluation._estimate_series`.
  3. Real store: compare `dragonnet_imp_att` to `dm_impute_t_att_gbm` and
     `dragonnet_cate_att` to `cf_att` on the same pairs.
"""

import copy
import logging
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from baselines import DraftDataExtractor, TreatmentContext

# Output keys, in report order.
IMP_NAME = 'dragonnet_imp_att'
CATE_NAME = 'dragonnet_cate_att'
NAMES: Tuple[str, ...] = (IMP_NAME, CATE_NAME)

_TORCH_MISSING = (
    "torch is not importable; DragonNetEnricher will return nan for every "
    "pair. Install torch in this environment to use it."
)

_MISSING_PAIR_COLUMNS = (
    "pairs_df needs columns %s; got %s. Pass the same frame the baseline "
    "enricher takes."
)

_FLAGGED = (
    "%s: DragonNet fit flagged (degenerate=%s, diverged=%s; val_loss_best=%.4f"
    " vs trivial %.4f, epochs_run=%d, best_epoch=%d); every pair -> nan"
)


@lru_cache(maxsize=1)
def _torch() -> Optional[Any]:
    """The `torch` module, or None with a single warning if it is absent.

    Same pattern as `baselines._causal_forest_class`: lazy, never at module
    scope, so this module imports wherever `baselines` does. `lru_cache`
    makes the warning fire once rather than once per CardA.
    """
    try:
        # pylint: disable=import-outside-toplevel,import-error
        import torch  # type: ignore[import-not-found]
    except ImportError:
        logging.warning(_TORCH_MISSING)
        return None
    return torch


@dataclass(frozen=True)
class DragonNetConfig:
    """The one fixed configuration. Not tuned, not meant to be.

    ~27 standardised inputs, ~1.1e5-1.4e5 rows and ~35 outputs per CardA;
    the trunk plus heads is ~12k parameters. Everything under `lr` is a
    mechanism rather than a number to get right -- see the module docstring.
    """

    trunk: Tuple[int, ...] = (64, 64)
    head: int = 32
    lr: float = 1e-3
    lr_min: float = 1e-5
    lr_factor: float = 0.5
    lr_patience: int = 3
    stop_patience: int = 10
    max_epochs: int = 200
    clip_norm: float = 1.0
    weight_decay: float = 0.0
    batch_size: int = 512
    val_frac: float = 0.1
    alpha: float = 1.0           # weight of the propensity BCE
    seed: int = 42
    device: str = 'cpu'
    num_threads: Optional[int] = None
    predict_chunk: int = 65536


@dataclass
class FitReport:
    """One fit's audit row. `DragonNetEnricher.fit_reports` holds one per CardA."""

    n_rows: int
    n_treated: int
    n_groups: int
    epochs_run: int
    best_epoch: int
    hit_cap: bool
    lr_final: float
    n_lr_drops: int
    train_loss_first: float
    train_loss_best: float
    val_loss_best: float
    val_loss_trivial: float
    degenerate: bool
    diverged: bool
    seconds: float

    @property
    def usable(self) -> bool:
        return not (self.degenerate or self.diverged)


@dataclass
class FittedDragonNet:
    """Predictions for every row of one CardA's frame, un-standardised.

    Plain numpy, so nothing downstream needs torch. Column `j` of `q0` / `q1`
    is group `groups[j]`.
    """

    q0: np.ndarray               # (n_rows, n_groups)
    q1: np.ndarray               # (n_rows, n_groups)
    groups: List[str]
    report: FitReport


def _build_net(torch: Any, n_in: int, n_groups: int, cfg: DragonNetConfig) -> Any:
    """The trunk and three heads as one `nn.Module`.

    Defined inside a function because `torch.nn.Module` only exists after the
    lazy import. `forward(x)` returns `(g_logit[N], q0[N, G], q1[N, G])`.
    """
    nn = torch.nn

    def mlp(sizes: Sequence[int]) -> Any:
        layers: List[Any] = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ELU()]
        return nn.Sequential(*layers)

    class DragonNet(nn.Module):  # type: ignore[misc,name-defined]
        def __init__(self) -> None:
            super().__init__()
            width = cfg.trunk[-1]
            self.trunk = mlp((n_in,) + tuple(cfg.trunk))
            self.g = nn.Linear(width, 1)
            self.q0 = nn.Sequential(mlp((width, cfg.head)),
                                    nn.Linear(cfg.head, n_groups))
            self.q1 = nn.Sequential(mlp((width, cfg.head)),
                                    nn.Linear(cfg.head, n_groups))

        def forward(self, x: Any) -> Tuple[Any, Any, Any]:
            phi = self.trunk(x)
            return self.g(phi).squeeze(-1), self.q0(phi), self.q1(phi)

    return DragonNet()


def _standardise(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column z-scores with the (mean, std) used; std floored at 1e-8."""
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    return (values - mean) / std, mean, std


def _stratified_split(
    torch: Any, treatment: np.ndarray, val_frac: float, gen: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """(train_idx, val_idx) with `val_frac` of the treated *and* of the controls.

    At ~1% treated a plain split leaves the `Q1` head's validation to chance;
    permuting each arm separately fixes its share. Every arm keeps at least
    one training row.
    """
    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    for arm in (np.flatnonzero(treatment == 1.0), np.flatnonzero(treatment == 0.0)):
        if arm.size == 0:
            continue
        perm = arm[torch.randperm(arm.size, generator=gen).numpy()]
        n_val = min(int(round(val_frac * arm.size)), arm.size - 1)
        val_parts.append(perm[:n_val])
        train_parts.append(perm[n_val:])
    return np.concatenate(train_parts), np.concatenate(val_parts)


def _loss(torch: Any, net: Any, x: Any, t: Any, y: Any, alpha: float) -> Any:
    """`mse(q_T, y)` over rows and groups plus `alpha * bce(g, t)`."""
    g_logit, q0, q1 = net(x)
    q_t = torch.where(t.unsqueeze(-1) > 0.5, q1, q0)
    mse = torch.mean((q_t - y) ** 2)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(g_logit, t)
    return mse + alpha * bce


def _trivial_val_loss(
    t_train: np.ndarray, y_train: np.ndarray,
    t_val: np.ndarray, y_val: np.ndarray, alpha: float,
) -> float:
    """Val loss of predicting each group's train mean and the train base rate.

    Computed rather than assumed to be 1.0: the standardisation used every
    row, the train subset's mean is not exactly 0 and the val subset's
    variance is not exactly 1.
    """
    mse = float(np.mean((y_val - y_train.mean(axis=0)) ** 2))
    p = float(np.clip(t_train.mean(), 1e-6, 1 - 1e-6))
    bce = float(-np.mean(t_val * np.log(p) + (1 - t_val) * np.log(1 - p)))
    return mse + alpha * bce


def fit_dragonnet(
    design: np.ndarray,
    treatment: np.ndarray,
    outcomes: np.ndarray,
    groups: Sequence[str],
    cfg: DragonNetConfig = DragonNetConfig(),
) -> FittedDragonNet:
    """Fit one DragonNet and predict `q0`, `q1` for every row.

    `design` is `(n, p)` with `not_offered` already removed, `treatment` is
    `(n,)` in {0, 1}, `outcomes` is `(n, G)` with column `j` being
    `Y_<groups[j]>`. Standardisation of X and of each Y column happens here
    and is undone on the way out. All randomness comes from one private
    generator seeded with `cfg.seed`; the global torch seed is untouched.

    Raises `RuntimeError` if torch is absent -- the enricher checks first.
    """
    torch = _torch()
    if torch is None:
        raise RuntimeError(_TORCH_MISSING)
    started = time.time()
    if cfg.num_threads is not None:
        torch.set_num_threads(cfg.num_threads)
    device = torch.device(cfg.device)
    gen = torch.Generator().manual_seed(cfg.seed)

    x_std, _, _ = _standardise(np.asarray(design, dtype=np.float64))
    y_std, y_mean, y_sd = _standardise(np.asarray(outcomes, dtype=np.float64))
    t_all = np.asarray(treatment, dtype=np.float64)
    n_rows, n_groups = y_std.shape

    train_idx, val_idx = _stratified_split(torch, t_all, cfg.val_frac, gen)
    trivial = _trivial_val_loss(t_all[train_idx], y_std[train_idx],
                                t_all[val_idx], y_std[val_idx], cfg.alpha)

    def as_t(a: np.ndarray) -> Any:
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    x_tr, t_tr, y_tr = as_t(x_std[train_idx]), as_t(t_all[train_idx]), as_t(y_std[train_idx])
    x_va, t_va, y_va = as_t(x_std[val_idx]), as_t(t_all[val_idx]), as_t(y_std[val_idx])

    # Parameter init draws from torch's global RNG; fork it so the fit is
    # reproducible without touching the caller's global state.
    with torch.random.fork_rng(devices=[] if cfg.device == 'cpu' else None):
        torch.manual_seed(cfg.seed)
        net = _build_net(torch, x_std.shape[1], n_groups, cfg).to(device)

    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=cfg.lr_factor, patience=cfg.lr_patience,
        min_lr=cfg.lr_min,
    )

    n_train = int(train_idx.size)
    best_state = copy.deepcopy(net.state_dict())
    best_val = float('inf')
    best_epoch = 0
    best_train = float('nan')
    first_train = float('nan')
    epochs_run = 0
    since_best = 0
    n_lr_drops = 0
    diverged = False

    for epoch in range(1, cfg.max_epochs + 1):
        net.train()
        order = torch.randperm(n_train, generator=gen).to(device)
        total = 0.0
        for start in range(0, n_train, cfg.batch_size):
            rows = order[start:start + cfg.batch_size]
            opt.zero_grad()
            loss = _loss(torch, net, x_tr[rows], t_tr[rows], y_tr[rows], cfg.alpha)
            if not torch.isfinite(loss):
                diverged = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.clip_norm)
            opt.step()
            total += float(loss) * rows.numel()
        if diverged:
            break
        epochs_run = epoch
        train_loss = total / n_train
        if epoch == 1:
            first_train = train_loss

        net.eval()
        with torch.no_grad():
            val_loss = float(_loss(torch, net, x_va, t_va, y_va, cfg.alpha))
        if not np.isfinite(val_loss):
            diverged = True
            break

        lr_before = opt.param_groups[0]['lr']
        sched.step(val_loss)
        if opt.param_groups[0]['lr'] < lr_before:
            n_lr_drops += 1

        if val_loss < best_val:
            best_val, best_epoch, best_train = val_loss, epoch, train_loss
            best_state = copy.deepcopy(net.state_dict())
            since_best = 0
        else:
            since_best += 1
            if since_best >= cfg.stop_patience:
                break

    net.load_state_dict(best_state)
    net.eval()
    q0 = np.empty((n_rows, n_groups), dtype=np.float64)
    q1 = np.empty((n_rows, n_groups), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, n_rows, cfg.predict_chunk):
            stop = min(start + cfg.predict_chunk, n_rows)
            _, a, b = net(as_t(x_std[start:stop]))
            q0[start:stop] = a.cpu().numpy()
            q1[start:stop] = b.cpu().numpy()
    q0 = q0 * y_sd + y_mean
    q1 = q1 * y_sd + y_mean

    report = FitReport(
        n_rows=n_rows,
        n_treated=int((t_all == 1.0).sum()),
        n_groups=n_groups,
        epochs_run=epochs_run,
        best_epoch=best_epoch,
        hit_cap=epochs_run >= cfg.max_epochs and since_best < cfg.stop_patience,
        lr_final=float(opt.param_groups[0]['lr']),
        n_lr_drops=n_lr_drops,
        train_loss_first=first_train,
        train_loss_best=best_train,
        val_loss_best=best_val,
        val_loss_trivial=trivial,
        degenerate=not best_val < trivial,
        diverged=diverged,
        seconds=time.time() - started,
    )
    return FittedDragonNet(q0=q0, q1=q1, groups=list(groups), report=report)


class DragonNetEnricher:
    """`BaselineEnricher`'s surface over one DragonNet fit per CardA.

    `get_estimates` returns `{name: DataFrame(card_a, group_b, value)}` for
    both `NAMES`, every pair present (nan included), rows in `pairs_df` key
    order and aligned across the two frames. Degenerate, diverged or failed
    fits give nan for the whole card and one WARNING; the reason lives in
    `fit_reports`.
    """

    def __init__(self, extractor: DraftDataExtractor,
                 config: DragonNetConfig = DragonNetConfig()):
        self.extractor = extractor
        self.config = config
        self.fit_reports: List[Dict[str, Any]] = []

    @staticmethod
    def baseline_columns() -> List[str]:
        return list(NAMES)

    def fit_reports_df(self) -> pd.DataFrame:
        """One audit row per CardA fitted so far; see `FitReport`."""
        return pd.DataFrame(self.fit_reports)

    def get_estimates(self, pairs_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        missing = [c for c in ('card_a', 'group_b') if c not in pairs_df.columns]
        if missing:
            raise ValueError(_MISSING_PAIR_COLUMNS % (missing, list(pairs_df.columns)))

        cards: List[str] = []
        groups: List[str] = []
        values: Dict[str, List[float]] = {name: [] for name in NAMES}
        for card_a, pairs in pairs_df.groupby('card_a', sort=False):
            logging.info("running: %s, %d", card_a, len(pairs))
            fitted, y_at_treated, treated_pos = self._fit_card(str(card_a))
            for group_b in dict.fromkeys(pairs['group_b']):
                cards.append(str(card_a))
                groups.append(str(group_b))
                imp, cate = self._pair(fitted, y_at_treated, treated_pos, str(group_b))
                values[IMP_NAME].append(imp)
                values[CATE_NAME].append(cate)

        return {
            name: pd.DataFrame({'card_a': cards, 'group_b': groups,
                                'value': values[name]})
            for name in NAMES
        }

    def _fit_card(
        self, card_a: str
    ) -> Tuple[Optional[FittedDragonNet], Optional[pd.DataFrame], Optional[np.ndarray]]:
        """`(fit, Y frame at treated rows, treated_pos)`, or Nones for nan.

        Any failure inside the fit is logged and mapped to nan for the card,
        the contract `BaselineEnricher._run` documents.
        """
        if _torch() is None:
            return None, None, None
        df = self.extractor.get_dataframe(card_a)
        if df.empty:
            return None, None, None
        ctx = TreatmentContext(df)
        if not ctx.is_valid:
            return None, None, None
        y_cols = [c for c in df.columns if c.startswith('Y_')]
        if not y_cols:
            return None, None, None

        design = ctx.design[:, np.asarray(ctx.adjustment_columns(), dtype=np.intp)]
        outcomes = df[y_cols].to_numpy(dtype=float)
        try:
            fitted = fit_dragonnet(design, ctx.treatment, outcomes,
                                   [c[2:] for c in y_cols], self.config)
        except Exception:  # pylint: disable=broad-exception-caught
            logging.exception("DragonNet fit failed for %s; every pair -> nan", card_a)
            return None, None, None

        row = {'card_a': card_a}
        row.update(asdict(fitted.report))
        self.fit_reports.append(row)
        r = fitted.report
        if not r.usable:
            logging.warning(_FLAGGED, card_a, r.degenerate, r.diverged,
                            r.val_loss_best, r.val_loss_trivial,
                            r.epochs_run, r.best_epoch)
            return None, None, None
        return fitted, df[y_cols].iloc[ctx.treated_pos], ctx.treated_pos

    @staticmethod
    def _pair(
        fitted: Optional[FittedDragonNet],
        y_at_treated: Optional[pd.DataFrame],
        treated_pos: Optional[np.ndarray],
        group_b: str,
    ) -> Tuple[float, float]:
        """`(imp, cate)` for one group, nan when undefined."""
        nan = (float('nan'), float('nan'))
        if fitted is None or y_at_treated is None or treated_pos is None:
            return nan
        if group_b not in fitted.groups:
            return nan
        j = fitted.groups.index(group_b)
        q0 = fitted.q0[treated_pos, j]
        q1 = fitted.q1[treated_pos, j]
        y = y_at_treated[f'Y_{group_b}'].to_numpy(dtype=float)
        imp = float(np.mean(y - q0))
        cate = float(np.mean(q1 - q0))
        return (imp if np.isfinite(imp) else float('nan'),
                cate if np.isfinite(cate) else float('nan'))

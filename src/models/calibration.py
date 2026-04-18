"""
Post-hoc probability calibration for the Monte Carlo simulator's win
probability outputs (Wave 4 Phase 3).

Why: the Wave 3.5 backtest measured systematic over-confidence -
predicted-95% buckets won 33% of the time, predicted-5% won 40%. That's
a textbook case for post-hoc calibration on top of the simulator's raw
output, applied AFTER the Monte Carlo aggregation.

Method: per-(format, gender) Platt scaling.

  logit_calibrated = a * logit_raw + b

Two scalars (a, b) per combo, fit by minimising NLL on a held-out slice
of historical matches. `a < 1` shrinks toward 0.5 (softens overconfidence);
`a > 1` sharpens (rarely needed). `b` shifts the diagonal off centre when
there's a systematic favourite/underdog bias.

Decoupled by design: refitting calibration is a 5-second job that doesn't
need a model retrain. We can A/B "with vs without calibration" cleanly
via the backtest harness.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CalibrationParams:
    """Per-(format, gender) Platt scalars."""
    a: float = 1.0
    b: float = 0.0
    n_train: int = 0
    n_iter: int = 0
    nll_before: Optional[float] = None
    nll_after: Optional[float] = None


@dataclass
class CalibrationBundle:
    """All four (format, gender) calibrations + metadata."""
    params: Dict[str, CalibrationParams] = field(default_factory=dict)
    fit_at: Optional[str] = None
    notes: Optional[str] = None

    def route_key(self, format_type: str, gender: str) -> str:
        return f"{format_type.upper()}_{gender.lower()}"

    def get(self, format_type: str, gender: str) -> CalibrationParams:
        """Identity calibration for unknown routes (preserves raw probs)."""
        return self.params.get(self.route_key(format_type, gender), CalibrationParams())

    def to_json(self) -> Dict:
        return {
            "params": {k: asdict(v) for k, v in self.params.items()},
            "fit_at": self.fit_at,
            "notes": self.notes,
        }

    @classmethod
    def from_json(cls, data: Dict) -> "CalibrationBundle":
        bundle = cls(fit_at=data.get("fit_at"), notes=data.get("notes"))
        for k, v in (data.get("params") or {}).items():
            bundle.params[k] = CalibrationParams(**v)
        return bundle

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fp:
            json.dump(self.to_json(), fp, indent=2)

    @classmethod
    def load(cls, path: str) -> "CalibrationBundle":
        with open(path, "r") as fp:
            return cls.from_json(json.load(fp))


# ============================================================================
# Apply calibration
# ============================================================================


def _safe_logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def _safe_sigmoid(x: float) -> float:
    if x > 0:
        e = math.exp(-x)
        return 1.0 / (1.0 + e)
    e = math.exp(x)
    return e / (1.0 + e)


def calibrate_probability(p: float, params: CalibrationParams) -> float:
    """Apply Platt scaling to a single win probability."""
    if params.a == 1.0 and params.b == 0.0:
        return p  # identity - nothing to do
    return _safe_sigmoid(params.a * _safe_logit(p) + params.b)


def calibrate_probabilities(probs: np.ndarray, params: CalibrationParams) -> np.ndarray:
    """Vectorised version. Identity early-out when a=1, b=0."""
    if params.a == 1.0 and params.b == 0.0:
        return probs.copy()
    eps = 1e-6
    p = np.clip(probs.astype(np.float64), eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p))
    z = params.a * logits + params.b
    # Numerically-stable sigmoid
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    ).astype(probs.dtype)


# ============================================================================
# Fit calibration
# ============================================================================


def _bce_loss(a: float, b: float, logits: np.ndarray, y: np.ndarray) -> float:
    """Mean binary cross-entropy after Platt scaling."""
    z = a * logits + b
    # Stable log(sigmoid(z))
    log_sig = np.where(z >= 0, -np.log1p(np.exp(-z)), z - np.log1p(np.exp(z)))
    log_one_minus_sig = np.where(z >= 0, -z - np.log1p(np.exp(-z)), -np.log1p(np.exp(z)))
    return float(-np.mean(y * log_sig + (1 - y) * log_one_minus_sig))


def _bce_grad(a: float, b: float, logits: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Gradient of mean BCE wrt (a, b)."""
    z = a * logits + b
    p = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
    err = p - y
    da = float(np.mean(err * logits))
    db = float(np.mean(err))
    return da, db


def fit_platt(
    raw_probs: np.ndarray,
    y_true: np.ndarray,
    *,
    max_iter: int = 200,
    lr: float = 0.5,
    tol: float = 1e-6,
) -> CalibrationParams:
    """Fit Platt scalars (a, b) by gradient descent on BCE.

    Args:
        raw_probs: shape (N,), the simulator's predicted win prob for team1.
        y_true:    shape (N,), 0/1 (1 if team1 actually won).

    Returns:
        CalibrationParams with NLL before/after for the report.
    """
    raw_probs = np.asarray(raw_probs, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    if raw_probs.shape != y_true.shape:
        raise ValueError(f"shape mismatch: probs {raw_probs.shape} vs y {y_true.shape}")
    n = len(raw_probs)
    if n < 5:
        logger.warning(f"Only {n} samples; returning identity calibration")
        return CalibrationParams(a=1.0, b=0.0, n_train=n)

    eps = 1e-6
    p = np.clip(raw_probs, eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p))

    # Identity baseline NLL
    nll_before = _bce_loss(1.0, 0.0, logits, y_true)

    a, b = 1.0, 0.0
    prev_loss = float("inf")
    for it in range(max_iter):
        da, db = _bce_grad(a, b, logits, y_true)
        # Plain gradient descent with a small lr; this loss surface is convex
        # in (a, b) so it converges quickly.
        a -= lr * da
        b -= lr * db
        # Constrain a >= 0 (negative a would invert probabilities; that's not
        # a real-world calibration outcome, only an optimiser pathology).
        a = max(a, 0.05)
        if it % 10 == 0:
            cur = _bce_loss(a, b, logits, y_true)
            if abs(prev_loss - cur) < tol:
                break
            prev_loss = cur

    nll_after = _bce_loss(a, b, logits, y_true)
    if nll_after > nll_before:
        # Calibration made it worse (degenerate sample or fit instability).
        # Fall back to identity rather than ship a regression.
        logger.warning(
            f"Platt fit increased NLL ({nll_before:.4f} -> {nll_after:.4f}); "
            "falling back to identity"
        )
        return CalibrationParams(
            a=1.0, b=0.0, n_train=n, n_iter=it + 1,
            nll_before=nll_before, nll_after=nll_before,
        )

    return CalibrationParams(
        a=float(a), b=float(b), n_train=n, n_iter=it + 1,
        nll_before=nll_before, nll_after=nll_after,
    )


def fit_bundle_from_backtest(
    backtest_csv: str,
    bundle: Optional[CalibrationBundle] = None,
) -> CalibrationBundle:
    """Read a backtest_simulator.py CSV and fit per-(format, gender) Platt scalars.

    Expected columns (subset):
      match_type, gender, sim_team1_win_prob, team1_won
    """
    import csv
    from datetime import datetime

    if bundle is None:
        bundle = CalibrationBundle()

    by_route: Dict[str, Tuple[List[float], List[int]]] = {}
    with open(backtest_csv, "r") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if not row.get("team1_won"):  # no-result, skip
                continue
            key = bundle.route_key(row["match_type"], row["gender"])
            probs, ys = by_route.setdefault(key, ([], []))
            try:
                p = float(row["sim_team1_win_prob"])
                y = int(row["team1_won"])
            except (TypeError, ValueError):
                continue
            probs.append(p)
            ys.append(y)

    for key, (probs, ys) in by_route.items():
        params = fit_platt(np.asarray(probs), np.asarray(ys))
        bundle.params[key] = params
        logger.info(
            f"[{key}] n={params.n_train}  a={params.a:.3f}  b={params.b:.3f}  "
            f"nll {params.nll_before:.4f} -> {params.nll_after:.4f}"
        )

    bundle.fit_at = datetime.utcnow().isoformat() + "Z"
    return bundle

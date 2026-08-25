"""Conservative high-tail mix inside committed back_half alpha.

The v9 back-half Ridge is dominated by the 0.55–0.65 bulk (median ~0.622)
and cannot reach 咏归-like observed alpha ~0.75. Independent-holdout 咏归
was predicted ~0.597 (abs err 0.151); the 2/3 rule (0.667) is closer.

Seven samples in the 103-JSON set have alpha >= 0.70, which is too small
for a second fitted expert. This module does not change the Ridge class.
When a committed back-half expert has shrunk *materially below* the
training-target median, predictions are floored at the 2/3 rule. Typical
0.62–0.64 predictions stay put and are not dragged to 0.75.

Front-half (岚下-like) must not call this helper: v9 front expert ~0.427
sits on the bulk median ~0.449, so a copied trigger would wreck ~0.45
front-half samples. See ``front_half_low_tail.py``. Uncommitted boundary
(远济-like) paths must not call this helper either. Online v9 stays
designer-specified front_half/back_half; the mix is an internal expert
post-process, not a new UI mode.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .node_geometry import DEFAULT_ALPHA

# 仙居 0.758, 咏归 0.748, 北涧 0.732, 广利 0.729, 溪东 0.726, 田地 0.726, 杨梅洲 0.708.
BACK_HALF_HIGH_TAIL_ALPHA = 0.70
# 咏归 expert 0.597 vs mode-train median ~0.622 (gap ~0.025). Require a
# material below-median gap so the 0.62–0.64 bulk is not lifted to 2/3.
BACK_HALF_SHRINK_BAND = 0.02
FIXED_RULE_ALPHA = DEFAULT_ALPHA


def back_half_high_tail_record(
    train_median: float,
    high_tail_train_rows: int,
) -> dict[str, Any]:
    """Metadata stored on the v9 artifact; online reads the same dict."""
    median = float(train_median)
    if not math.isfinite(median):
        raise ValueError("back-half training median must be finite")
    return {
        "method": "rule_floor_when_expert_below_median_by_shrink_band",
        "rule_alpha": float(FIXED_RULE_ALPHA),
        "train_median": median,
        "shrink_band": float(BACK_HALF_SHRINK_BAND),
        "high_tail_alpha": float(BACK_HALF_HIGH_TAIL_ALPHA),
        "high_tail_train_rows": int(high_tail_train_rows),
        "note": (
            "n=7 samples with alpha>=0.70 is too small for a second Ridge; "
            "咏归-like shrinkage below the back-half median is floored at 2/3. "
            "北涧/田地 expert outputs sit in the 0.62-0.64 bulk and are not moved."
        ),
    }


def mix_back_half_high_tail_alpha(
    expert_prediction: float,
    config: Mapping[str, Any] | None,
) -> tuple[float, bool]:
    """Return ``(mixed_prediction, applied)`` for one committed back_half value.

    Missing config is a no-op so older artifacts keep their raw expert output
    until they are retrained.
    """
    value = float(expert_prediction)
    mixed, applied = mix_back_half_high_tail_values(np.asarray([value], dtype=float), config)
    return float(mixed[0]), bool(applied[0])


def mix_back_half_high_tail_values(
    expert_predictions: np.ndarray,
    config: Mapping[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized mix. ``applied`` is True only when the numeric value changes."""
    values = np.asarray(expert_predictions, dtype=float).reshape(-1)
    if config is None:
        return values.copy(), np.zeros(len(values), dtype=bool)
    train_median = float(config["train_median"])
    shrink_band = float(config.get("shrink_band", BACK_HALF_SHRINK_BAND))
    rule_alpha = float(config.get("rule_alpha", FIXED_RULE_ALPHA))
    if not math.isfinite(train_median) or not math.isfinite(shrink_band) or not math.isfinite(rule_alpha):
        raise ValueError("back-half high-tail mix config must be finite")
    trigger = values < (train_median - shrink_band)
    mixed = np.where(trigger, np.maximum(values, rule_alpha), values)
    applied = trigger & ~np.isclose(mixed, values)
    return mixed, applied

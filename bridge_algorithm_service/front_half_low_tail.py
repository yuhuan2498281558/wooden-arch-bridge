"""Documented refusal of a front-half low-tail mix (岚下).

Re-annotation 2026-08-25 on the live workbench (canonical JSON span 15.8 m):
Q4 moved ~3 px toward C; this is not a mis-labeled 0.45 node.
Observed alpha L/R = 0.277 / 0.187, mean = 0.232, symmetry ~0.089.
Unique lowest of 103 samples; next is 洋坑桥 0.293 (development).
Front-half n=36, median ~0.449. Frozen-holdout v9 front expert ~0.427.

Why there is no mix
-------------------
The 咏归 high-tail mix fires only when the back-half expert is already
0.02 below the training median (the model is trying to go high, then
shrinks). The symmetric test for 岚下 is: expert 0.427 vs median 0.449,
gap ~0.022. That sits on the front-half bulk, not below it. Copying the
0.02 trigger would fire on typical front predictions (~0.43) and drag
the 0.45 mass toward 0.23, which is worse than leaving 岚下 alone.

n=1 (or 2 with 洋坑) cannot support a second fitted expert. There is no
low-side analog of the 2/3 rule. Span 15.8 m is ordinary. Keep the
sample; do not delete it; leave the front Ridge unchanged.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .back_half_high_tail import BACK_HALF_SHRINK_BAND

# Isolation band covering 岚下 0.232 and 洋坑 0.293; not a mix threshold.
FRONT_HALF_LOW_TAIL_ALPHA = 0.30
# Verified on the 103-sample set / frozen v9 holdout (not a mix trigger).
FRONT_HALF_TRAIN_MEDIAN = 0.449
LANXIA_V9_FRONT_EXPERT = 0.427
LANXIA_OBSERVED_ALPHA = 0.232
LANXIA_OBSERVED_LEFT = 0.277
LANXIA_OBSERVED_RIGHT = 0.187
LANXIA_SPAN_M = 15.8
FRONT_HALF_LOW_TAIL_MIX_ENABLED = False


def front_half_low_tail_record(
    train_median: float,
    low_tail_train_rows: int,
) -> dict[str, Any]:
    """Metadata stored on the v9 artifact. ``enabled`` is always False."""
    median = float(train_median)
    if not math.isfinite(median):
        raise ValueError("front-half training median must be finite")
    return {
        "method": "none_isolated_holdout_outlier",
        "enabled": FRONT_HALF_LOW_TAIL_MIX_ENABLED,
        "train_median": median,
        "low_tail_alpha": float(FRONT_HALF_LOW_TAIL_ALPHA),
        "low_tail_train_rows": int(low_tail_train_rows),
        "note": (
            "岚下 re-annotation 2026-08-25: L/R 0.277/0.187, mean 0.232, "
            "span 15.8 m. Unique lowest of 103; next 洋坑 0.293. v9 front "
            "expert ~0.427 vs median ~0.449 sits on the bulk. A copied "
            "咏归 0.02-below-median trigger would drag typical ~0.45 "
            "front-half predictions toward 0.23. n=1 is too small for a "
            "second Ridge. Leave the front expert unchanged; keep the sample."
        ),
    }


def copied_high_tail_trigger_would_fire(
    expert_prediction: float,
    *,
    train_median: float = FRONT_HALF_TRAIN_MEDIAN,
    shrink_band: float = BACK_HALF_SHRINK_BAND,
) -> bool:
    """True if a naive copy of the 咏归 trigger would fire on this expert output.

    Used only to document why that trigger is not installed for front_half.
    """
    value = float(expert_prediction)
    median = float(train_median)
    band = float(shrink_band)
    if not math.isfinite(value) or not math.isfinite(median) or not math.isfinite(band):
        raise ValueError("front-half trigger probe values must be finite")
    return value < (median - band)


def mix_front_half_low_tail_alpha(
    expert_prediction: float,
    config: Mapping[str, Any] | None = None,
) -> tuple[float, bool]:
    """Intentional no-op: 岚下 is a verified isolated outlier, not a mix."""
    del config
    return float(expert_prediction), False


def mix_front_half_low_tail_values(
    expert_predictions: np.ndarray,
    config: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized no-op matching the back-half mix signature."""
    del config
    values = np.asarray(expert_predictions, dtype=float).reshape(-1)
    return values.copy(), np.zeros(len(values), dtype=bool)

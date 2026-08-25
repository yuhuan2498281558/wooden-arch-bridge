"""Uncommitted alpha: band center, not the v6 bulk model.

Local v9 rerun 2026-08-25: 远济 is correctly ``uncommitted`` (gate OK) but
``predicted_alpha`` was 0.622 — exactly ``v6_single_model``. True mean is
0.503; abs err 0.119. The 22 boundary-band rows have |mean α−0.5|≤0.03
by construction (~0.47–0.53). v6 is span-only and lives on the ~0.62
bulk, so it is the wrong expert for the band.

This module:
- predicts a documented constant near 0.5 for band / knife-edge rows
  (training band median, clamped inside ±0.03; default 0.5);
- interpolates the two committed-half experts without a 0.5 clip for
  uncommitted rows *outside* the band (25−22=3 on that run);
- never sends uncommitted rows through v6.

Committed front_half / back_half (咏归 mix, 岚下 no-mix) do not use this.
Online: designer-specified front/back unchanged; ``uncommitted`` or a
missing half uses the same band predictor. No new UI.

Keep these numbers identical to ``structure_gated_alpha``
``STRUCTURE_THRESHOLD`` / ``STRUCTURE_BOUNDARY_BAND``. This module must
not import the training pipeline (online ``node_model`` loads it).
"""
from __future__ import annotations

import math
from typing import Any, Mapping

UNCOMMITTED_BAND_CENTER = 0.5
UNCOMMITTED_BOUNDARY_BAND = 0.03


def clamp_uncommitted_band_alpha(value: float) -> float:
    """Keep the band predictor inside ±0.03 of 0.5 so it cannot drift to ~0.62."""
    center = float(UNCOMMITTED_BAND_CENTER)
    band = float(UNCOMMITTED_BOUNDARY_BAND)
    number = float(value)
    if not math.isfinite(number) or abs(number - center) > band:
        return center
    return number


def uncommitted_alpha_record(
    band_median: float,
    band_rows: int,
    off_band_rows: int,
) -> dict[str, Any]:
    """Metadata stored on the v9 artifact; online reads ``band_alpha``."""
    return {
        "method": "boundary_band_center_else_ungated_expert_mix",
        "band_center": float(UNCOMMITTED_BAND_CENTER),
        "band_alpha": clamp_uncommitted_band_alpha(band_median),
        "boundary_band": float(UNCOMMITTED_BOUNDARY_BAND),
        "band_rows": int(band_rows),
        "off_band_uncommitted_rows": int(off_band_rows),
        "note": (
            "Do not use v6 (~0.62 bulk) for |α-0.5|<=0.03. 远济 holdout "
            "mean 0.503 was predicted 0.622 by v6. Band rows use the "
            "training band median clamped to the band (default 0.5). "
            "Off-band uncommitted rows interpolate the two half-experts "
            "without a 0.5 clip."
        ),
    }


def row_uses_uncommitted_band_predictor(row: Mapping[str, Any]) -> bool:
    """True for boundary-band / missing-label rows; False for off-band uncommitted.

    Online feature dicts have no observed alpha: treat as the band (designer
    did not commit a half). Historical rows use ``design_target_alpha``.
    """
    mean = row.get("design_target_alpha")
    if mean is None:
        return True
    try:
        value = float(mean)
    except (TypeError, ValueError):
        return True
    if not math.isfinite(value):
        return True
    return abs(value - UNCOMMITTED_BAND_CENTER) <= UNCOMMITTED_BOUNDARY_BAND


def uncommitted_band_prediction(config: Mapping[str, Any] | None) -> float:
    if not config:
        return float(UNCOMMITTED_BAND_CENTER)
    return clamp_uncommitted_band_alpha(float(config.get("band_alpha", UNCOMMITTED_BAND_CENTER)))

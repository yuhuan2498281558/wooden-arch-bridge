from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


DEFAULT_ALPHA = 2.0 / 3.0
DEFAULT_BETA = 1.0 / 4.0
COORDINATE_SYSTEM = "three-miao-parallel-chord-ratio-v2"
_EPSILON = 1e-12


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def as_dict(self, digits: int = 6) -> dict[str, float]:
        return {"x": round(float(self.x), digits), "y": round(float(self.y), digits)}


@dataclass(frozen=True)
class Projection:
    fraction: float
    perpendicular_distance: float
    normalized_distance: float


@dataclass(frozen=True)
class ParallelChordFit:
    fraction: float
    signed_start_offset: float
    normalized_start_offset: float
    legacy_projection_fraction: float


@dataclass(frozen=True)
class RatioMeasurement:
    alpha_left: float
    alpha_right: float
    beta_left: float
    beta_right: float
    alpha: float
    beta: float
    alpha_symmetry_error: float
    beta_symmetry_error: float
    max_normalized_projection_error: float
    alpha_legacy_projection_left: float
    alpha_legacy_projection_right: float
    outer_start_offset_left_normalized: float
    outer_start_offset_right_normalized: float
    warnings: tuple[str, ...]

    def as_dict(self, digits: int = 6) -> dict[str, Any]:
        values = {
            "alpha_left": self.alpha_left,
            "alpha_right": self.alpha_right,
            "beta_left": self.beta_left,
            "beta_right": self.beta_right,
            "alpha": self.alpha,
            "beta": self.beta,
            "alpha_symmetry_error": self.alpha_symmetry_error,
            "beta_symmetry_error": self.beta_symmetry_error,
            "max_normalized_projection_error": self.max_normalized_projection_error,
            "alpha_legacy_projection_left": self.alpha_legacy_projection_left,
            "alpha_legacy_projection_right": self.alpha_legacy_projection_right,
            "outer_start_offset_left_normalized": self.outer_start_offset_left_normalized,
            "outer_start_offset_right_normalized": self.outer_start_offset_right_normalized,
        }
        return {
            **{key: round(float(value), digits) for key, value in values.items()},
            "warnings": list(self.warnings),
        }


def _point(value: Point | Mapping[str, Any]) -> Point:
    if isinstance(value, Point):
        return value
    try:
        return Point(float(value["x"]), float(value["y"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Point must contain finite numeric x and y values") from exc


def _assert_finite(point: Point, name: str) -> None:
    if not math.isfinite(point.x) or not math.isfinite(point.y):
        raise ValueError(f"{name} must contain finite coordinates")


def project_to_segment(point: Point, start: Point, end: Point) -> Projection:
    _assert_finite(point, "point")
    _assert_finite(start, "start")
    _assert_finite(end, "end")

    dx = end.x - start.x
    dy = end.y - start.y
    length_squared = dx * dx + dy * dy
    if length_squared <= _EPSILON:
        raise ValueError("Cannot project onto a zero-length segment")

    px = point.x - start.x
    py = point.y - start.y
    fraction = (px * dx + py * dy) / length_squared
    projected_x = start.x + fraction * dx
    projected_y = start.y + fraction * dy
    distance = math.hypot(point.x - projected_x, point.y - projected_y)
    segment_length = math.sqrt(length_squared)
    return Projection(
        fraction=fraction,
        perpendicular_distance=distance,
        normalized_distance=distance / segment_length,
    )


def fit_parallel_chord(
    point: Point,
    start: Point,
    end: Point,
    axis_start: Point,
    axis_end: Point,
    span_start: Point,
    span_end: Point,
) -> ParallelChordFit:
    """Fit a point using a chord direction fixed parallel to ``start -> end``.

    The five-miao chord may start above the three-miao springing point because
    of member thickness.  Its fraction is therefore measured along the bridge
    axis, while the remaining bridge-normal component is retained separately
    as the inferred start-height offset.
    """
    for name, value in {
        "point": point,
        "start": start,
        "end": end,
        "axis_start": axis_start,
        "axis_end": axis_end,
        "span_start": span_start,
        "span_end": span_end,
    }.items():
        _assert_finite(value, name)

    axis_dx = axis_end.x - axis_start.x
    axis_dy = axis_end.y - axis_start.y
    axis_length = math.hypot(axis_dx, axis_dy)
    if axis_length <= _EPSILON:
        raise ValueError("Cannot define a bridge axis from coincident B and C")
    ux, uy = axis_dx / axis_length, axis_dy / axis_length

    chord_dx = end.x - start.x
    chord_dy = end.y - start.y
    if math.hypot(chord_dx, chord_dy) <= _EPSILON:
        raise ValueError("Cannot fit a zero-length chord segment")
    longitudinal_chord = chord_dx * ux + chord_dy * uy
    if abs(longitudinal_chord) <= _EPSILON:
        raise ValueError("Chord direction has no longitudinal component")

    point_dx = point.x - start.x
    point_dy = point.y - start.y
    fraction = (point_dx * ux + point_dy * uy) / longitudinal_chord
    residual_x = point_dx - fraction * chord_dx
    residual_y = point_dy - fraction * chord_dy

    # In image coordinates y grows downwards. (uy, -ux) is the bridge-normal
    # direction pointing upward for the usual left-to-right drawing axis.
    normal_up_x, normal_up_y = uy, -ux
    signed_offset = residual_x * normal_up_x + residual_y * normal_up_y

    span_dx = span_end.x - span_start.x
    span_dy = span_end.y - span_start.y
    longitudinal_span = abs(span_dx * ux + span_dy * uy)
    if longitudinal_span <= _EPSILON:
        raise ValueError("Bridge span has no longitudinal component")

    legacy = project_to_segment(point, start, end)
    return ParallelChordFit(
        fraction=fraction,
        signed_start_offset=signed_offset,
        normalized_start_offset=signed_offset / longitudinal_span,
        legacy_projection_fraction=legacy.fraction,
    )


def ratio_warnings(alpha: float, beta: float) -> tuple[str, ...]:
    warnings: list[str] = []
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        warnings.append("alpha must be between 0 and 1")
    if not math.isfinite(beta) or not 0.0 < beta < 0.5:
        warnings.append("beta must be between 0 and 0.5")
    return tuple(warnings)


def reconstruct_nodes(
    a: Point | Mapping[str, Any],
    b: Point | Mapping[str, Any],
    c: Point | Mapping[str, Any],
    d: Point | Mapping[str, Any],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> dict[str, Point]:
    anchors = {name: _point(value) for name, value in {"A": a, "B": b, "C": c, "D": d}.items()}
    for name, point in anchors.items():
        _assert_finite(point, name)

    warnings = ratio_warnings(float(alpha), float(beta))
    if warnings:
        raise ValueError("; ".join(warnings))

    a_p, b_p, c_p, d_p = anchors["A"], anchors["B"], anchors["C"], anchors["D"]
    return {
        "Q1": Point(a_p.x + alpha * (b_p.x - a_p.x), a_p.y + alpha * (b_p.y - a_p.y)),
        "Q2": Point(b_p.x + beta * (c_p.x - b_p.x), b_p.y + beta * (c_p.y - b_p.y)),
        "Q3": Point(c_p.x + beta * (b_p.x - c_p.x), c_p.y + beta * (b_p.y - c_p.y)),
        "Q4": Point(d_p.x + alpha * (c_p.x - d_p.x), d_p.y + alpha * (c_p.y - d_p.y)),
    }


def measure_ratios(
    a: Point | Mapping[str, Any],
    b: Point | Mapping[str, Any],
    c: Point | Mapping[str, Any],
    d: Point | Mapping[str, Any],
    q1: Point | Mapping[str, Any],
    q2: Point | Mapping[str, Any],
    q3: Point | Mapping[str, Any],
    q4: Point | Mapping[str, Any],
) -> RatioMeasurement:
    points = {
        name: _point(value)
        for name, value in {
            "A": a,
            "B": b,
            "C": c,
            "D": d,
            "Q1": q1,
            "Q2": q2,
            "Q3": q3,
            "Q4": q4,
        }.items()
    }

    alpha_left = fit_parallel_chord(
        points["Q1"], points["A"], points["B"], points["B"], points["C"], points["A"], points["D"]
    )
    alpha_right = fit_parallel_chord(
        points["Q4"], points["D"], points["C"], points["B"], points["C"], points["A"], points["D"]
    )
    beta_left = project_to_segment(points["Q2"], points["B"], points["C"])
    beta_right = project_to_segment(points["Q3"], points["C"], points["B"])

    alpha = (alpha_left.fraction + alpha_right.fraction) / 2.0
    beta = (beta_left.fraction + beta_right.fraction) / 2.0
    warnings = list(ratio_warnings(alpha, beta))
    projection_error = max(
        project_to_segment(points["Q1"], points["A"], points["B"]).normalized_distance,
        project_to_segment(points["Q4"], points["D"], points["C"]).normalized_distance,
        beta_left.normalized_distance,
        beta_right.normalized_distance,
    )
    if projection_error > 0.015:
        warnings.append("one or more target points have a large member-axis offset")

    return RatioMeasurement(
        alpha_left=alpha_left.fraction,
        alpha_right=alpha_right.fraction,
        beta_left=beta_left.fraction,
        beta_right=beta_right.fraction,
        alpha=alpha,
        beta=beta,
        alpha_symmetry_error=abs(alpha_left.fraction - alpha_right.fraction),
        beta_symmetry_error=abs(beta_left.fraction - beta_right.fraction),
        max_normalized_projection_error=projection_error,
        alpha_legacy_projection_left=alpha_left.legacy_projection_fraction,
        alpha_legacy_projection_right=alpha_right.legacy_projection_fraction,
        outer_start_offset_left_normalized=alpha_left.normalized_start_offset,
        outer_start_offset_right_normalized=alpha_right.normalized_start_offset,
        warnings=tuple(warnings),
    )


def geometry_payload(
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    *,
    source: str = "rule_fallback",
) -> dict[str, Any]:
    warnings = ratio_warnings(float(alpha), float(beta))
    if warnings:
        raise ValueError("; ".join(warnings))
    return {
        "coordinate_system": COORDINATE_SYSTEM,
        "five_miao_bullhead": {
            "ratios": {
                "alpha_outer": round(float(alpha), 6),
                "beta_inner": round(float(beta), 6),
            },
            "source": source,
            "thickness_mode": "ignored",
        },
    }


def ratios_from_result(result: Mapping[str, Any] | None) -> tuple[float, float, str]:
    if not isinstance(result, Mapping):
        return DEFAULT_ALPHA, DEFAULT_BETA, "rule_fallback"

    geometry = result.get("geometry")
    if not isinstance(geometry, Mapping):
        return DEFAULT_ALPHA, DEFAULT_BETA, "rule_fallback"
    bullhead = geometry.get("five_miao_bullhead")
    if not isinstance(bullhead, Mapping):
        return DEFAULT_ALPHA, DEFAULT_BETA, "rule_fallback"
    ratios = bullhead.get("ratios")
    if not isinstance(ratios, Mapping):
        return DEFAULT_ALPHA, DEFAULT_BETA, "rule_fallback"

    try:
        alpha = float(ratios.get("alpha_outer", DEFAULT_ALPHA))
        beta = float(ratios.get("beta_inner", DEFAULT_BETA))
    except (TypeError, ValueError):
        return DEFAULT_ALPHA, DEFAULT_BETA, "rule_fallback"
    if ratio_warnings(alpha, beta):
        return DEFAULT_ALPHA, DEFAULT_BETA, "rule_fallback"
    return alpha, beta, str(bullhead.get("source") or "unknown")

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


# Screening assumptions.  These are deliberately explicit and uncalibrated:
# they give a transparent first-order load-bearing envelope, not a formal
# structural design or a statutory safety assessment.
DEAD_LOAD_KPA = 2.0
LIVE_LOAD_KPA = 3.5
LOAD_COMBINATION_FACTOR = 1.3
SELF_WEIGHT_ALLOWANCE = 1.1
COMPRESSIVE_STRENGTH_MPA = 10.0
ELASTIC_MODULUS_MPA = 9000.0
SECTION_REDUCTION_FACTOR = 0.80
BUCKLING_SAFETY_FACTOR = 2.0
MAX_DECAY_HORIZON_YEARS = 300

DECAY_SCENARIOS = {
    "protected": {
        "label": "遮蔽/防护良好",
        "radial_loss_mm_per_year": 0.05,
    },
    "sheltered": {
        "label": "常规遮蔽",
        "radial_loss_mm_per_year": 0.15,
    },
    "exposed": {
        "label": "露天/临水暴露",
        "radial_loss_mm_per_year": 0.40,
    },
}


@dataclass(frozen=True)
class MemberCheck:
    name: str
    system: str
    length_m: float
    angle_deg: float
    diameter_mm: float
    axial_demand_kn: float
    stress_mpa: float
    stress_capacity_mpa: float
    buckling_capacity_kn: float
    stress_dcr: float
    buckling_dcr: float
    dcr: float
    governing: str
    passes: bool


def _number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _mid_diameter(lo: Any, hi: Any) -> float:
    lo_value = _number(lo, 0.0)
    hi_value = _number(hi, 0.0)
    if lo_value > 0 and hi_value > 0:
        return (lo_value + hi_value) / 2.0
    if lo_value > 0:
        return lo_value
    if hi_value > 0:
        return hi_value
    raise ValueError("member diameter range is unavailable")


def _section_area_mm2(diameter_mm: float) -> float:
    return math.pi * diameter_mm**2 / 4.0


def _inertia_mm4(diameter_mm: float) -> float:
    return math.pi * diameter_mm**4 / 64.0


def _buckling_capacity_kn(
    diameter_mm: float,
    length_m: float,
    *,
    elastic_modulus_mpa: float,
    safety_factor: float,
) -> float:
    if length_m <= 0:
        return math.inf
    # Pinned-pinned first mode: Pcr = pi^2 E I / L^2.
    inertia_m4 = _inertia_mm4(diameter_mm) / 1e12
    critical_n = math.pi**2 * elastic_modulus_mpa * 1e6 * inertia_m4 / length_m**2
    return critical_n / (1000.0 * safety_factor)


def _check_member(
    name: str,
    system: str,
    length_m: float,
    angle_deg: float,
    axial_demand_kn: float,
    diameter_mm: float,
) -> MemberCheck:
    area_mm2 = _section_area_mm2(diameter_mm)
    stress_mpa = axial_demand_kn * 1000.0 / area_mm2
    stress_capacity_mpa = COMPRESSIVE_STRENGTH_MPA * SECTION_REDUCTION_FACTOR
    buckling_capacity_kn = _buckling_capacity_kn(
        diameter_mm,
        length_m,
        elastic_modulus_mpa=ELASTIC_MODULUS_MPA,
        safety_factor=BUCKLING_SAFETY_FACTOR,
    )
    stress_dcr = stress_mpa / stress_capacity_mpa if stress_capacity_mpa > 0 else math.inf
    buckling_dcr = axial_demand_kn / buckling_capacity_kn if buckling_capacity_kn > 0 else math.inf
    dcr = max(stress_dcr, buckling_dcr)
    governing = "stress" if stress_dcr >= buckling_dcr else "buckling"
    return MemberCheck(
        name=name,
        system=system,
        length_m=length_m,
        angle_deg=angle_deg,
        diameter_mm=diameter_mm,
        axial_demand_kn=axial_demand_kn,
        stress_mpa=stress_mpa,
        stress_capacity_mpa=stress_capacity_mpa,
        buckling_capacity_kn=buckling_capacity_kn,
        stress_dcr=stress_dcr,
        buckling_dcr=buckling_dcr,
        dcr=dcr,
        governing=governing,
        passes=dcr <= 1.0,
    )


def _member_payload(check: MemberCheck) -> dict[str, Any]:
    return {
        "name": check.name,
        "system": check.system,
        "length_m": round(check.length_m, 3),
        "angle_deg": round(check.angle_deg, 2),
        "diameter_mm": round(check.diameter_mm, 1),
        "axial_demand_kn": round(check.axial_demand_kn, 3),
        "stress_mpa": round(check.stress_mpa, 4),
        "stress_capacity_mpa": round(check.stress_capacity_mpa, 2),
        "buckling_capacity_kn": round(check.buckling_capacity_kn, 2),
        "stress_dcr": round(check.stress_dcr, 4),
        "buckling_dcr": round(check.buckling_dcr, 4),
        "dcr": round(check.dcr, 4),
        "governing": check.governing,
        "passes": check.passes,
    }


def _decay_life_years(check: MemberCheck, radial_loss_mm_per_year: float) -> float:
    if radial_loss_mm_per_year <= 0:
        return float(MAX_DECAY_HORIZON_YEARS)

    def dcr_at_year(year: float) -> float:
        diameter = max(1.0, check.diameter_mm - 2.0 * radial_loss_mm_per_year * year)
        decayed = _check_member(
            check.name,
            check.system,
            check.length_m,
            check.angle_deg,
            check.axial_demand_kn,
            diameter,
        )
        return decayed.dcr

    if dcr_at_year(0) > 1.0:
        return 0.0
    if dcr_at_year(float(MAX_DECAY_HORIZON_YEARS)) <= 1.0:
        return float(MAX_DECAY_HORIZON_YEARS)
    low, high = 0.0, float(MAX_DECAY_HORIZON_YEARS)
    for _ in range(60):
        mid = (low + high) / 2.0
        if dcr_at_year(mid) <= 1.0:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def compute_structural_check(
    params: Mapping[str, Any],
    result: Mapping[str, Any],
    *,
    radial_loss_mm_per_year: float | None = None,
) -> dict[str, Any]:
    span = _number(params.get("span"), 0.0)
    width = _number(params.get("width"), 0.0)
    n1 = int(_number(params.get("n1"), 0.0))
    n2 = int(_number(params.get("n2"), 0.0))
    rise = _number(result.get("rise_height"), 0.0)
    if min(span, width, rise) <= 0 or n1 <= 0 or n2 <= 0:
        raise ValueError("structural check requires positive span/width/rise and system counts")

    alpha = _number(
        result.get("geometry", {}).get("five_miao_bullhead", {}).get("ratios", {}).get("alpha_outer"),
        2.0 / 3.0,
    )
    beta = _number(
        result.get("geometry", {}).get("five_miao_bullhead", {}).get("ratios", {}).get("beta_inner"),
        0.25,
    )
    alpha = min(0.999, max(0.001, alpha))
    beta = min(0.499, max(0.001, beta))

    third_span = span / 3.0
    theta = math.atan(rise / third_span)
    three_diag_len = math.hypot(third_span, rise)
    q1_x = alpha * third_span
    q1_y = alpha * rise
    q2_x = (1.0 + beta) * third_span
    q2_y = rise
    inner_phi = math.atan2(max(q2_y - q1_y, 1e-6), max(q2_x - q1_x, 1e-6))

    design_pressure = (DEAD_LOAD_KPA + LIVE_LOAD_KPA) * LOAD_COMBINATION_FACTOR
    total_line_load = design_pressure * width * SELF_WEIGHT_ALLOWANCE
    total_vertical_kn = total_line_load * span
    # Each "系" is one side-view plane frame; the total deck load is shared by
    # the n1+n2 plane systems as a conservative screening assumption.
    frame_vertical_kn = total_vertical_kn / (n1 + n2)
    half_frame = frame_vertical_kn / 2.0

    def mid(result_lo: str, result_hi: str) -> float:
        return _mid_diameter(result.get(result_lo), result.get(result_hi))

    members: list[MemberCheck] = [
        _check_member(
            "三节苗平弦",
            "three_miao",
            third_span,
            0.0,
            half_frame / math.tan(theta),
            mid("s1_flat_lo", "s1_flat_hi"),
        ),
        _check_member(
            "三节苗斜弦",
            "three_miao",
            three_diag_len,
            math.degrees(theta),
            half_frame / math.sin(theta),
            mid("s1_diag_lo", "s1_diag_hi"),
        ),
        _check_member(
            "五节苗外斜弦",
            "five_miao",
            _number(result.get("s2_diag1_len"), alpha * three_diag_len),
            math.degrees(theta),
            half_frame / math.sin(theta),
            mid("s2_diag1_lo", "s2_diag1_hi"),
        ),
        _check_member(
            "五节苗内斜弦",
            "five_miao",
            _number(result.get("s2_diag2_len"), math.hypot(q2_x - q1_x, q2_y - q1_y)),
            math.degrees(inner_phi),
            half_frame / math.sin(inner_phi),
            mid("s2_diag2_lo", "s2_diag2_hi"),
        ),
        _check_member(
            "五节苗平弦",
            "five_miao",
            _number(result.get("s2_flat_len"), (1.0 - 2.0 * beta) * third_span),
            0.0,
            half_frame * (1.0 / math.tan(theta) + 1.0 / math.tan(inner_phi)),
            mid("s2_flat_lo", "s2_flat_hi"),
        ),
    ]

    max_dcr = max(member.dcr for member in members)
    decay_life: dict[str, Any] = {}
    for key, scenario in DECAY_SCENARIOS.items():
        years = [
            (member.name, _decay_life_years(member, float(scenario["radial_loss_mm_per_year"])))
            for member in members
        ]
        critical_name, critical_years = min(years, key=lambda item: item[1])
        decay_life[key] = {
            "label": scenario["label"],
            "radial_loss_mm_per_year": scenario["radial_loss_mm_per_year"],
            "critical_member": critical_name,
            "years_until_dcr_1": round(critical_years, 1),
            "status": "screening_assumption_not_calibrated",
        }

    selected_decay_life: dict[str, Any] | None = None
    if radial_loss_mm_per_year is not None:
        selected_years = [
            (member.name, _decay_life_years(member, max(0.001, float(radial_loss_mm_per_year))))
            for member in members
        ]
        selected_name, selected_value = min(selected_years, key=lambda item: item[1])
        selected_decay_life = {
            "label": "当前安全情景",
            "radial_loss_mm_per_year": round(float(radial_loss_mm_per_year), 4),
            "critical_member": selected_name,
            "years_until_dcr_1": round(selected_value, 1),
            "status": "screening_assumption_not_calibrated",
        }

    return {
        "status": "screening_pass" if max_dcr <= 1.0 else "screening_fail",
        "passes": max_dcr <= 1.0,
        "method": "planar-system-envelope-v1",
        "model_version": "structural-screening-v1",
        "max_dcr": round(max_dcr, 4),
        "capacity_load_factor": round(1.0 / max_dcr, 2) if max_dcr > 0 else None,
        "load_model": {
            "total_design_load_kn": round(total_vertical_kn, 2),
            "load_per_plane_system_kn": round(frame_vertical_kn, 2),
            "plane_systems": n1 + n2,
            "sharing_rule": "deck load shared equally by n1+n2 plane systems",
        },
        "assumptions": {
            "dead_load_kpa": DEAD_LOAD_KPA,
            "live_load_kpa": LIVE_LOAD_KPA,
            "load_combination_factor": LOAD_COMBINATION_FACTOR,
            "self_weight_allowance": SELF_WEIGHT_ALLOWANCE,
            "compressive_strength_mpa": COMPRESSIVE_STRENGTH_MPA,
            "elastic_modulus_mpa": ELASTIC_MODULUS_MPA,
            "section_reduction_factor": SECTION_REDUCTION_FACTOR,
            "buckling_safety_factor": BUCKLING_SAFETY_FACTOR,
            "force_model": "worst-case axial envelope: each inclined member checked for half-frame vertical reaction",
            "life_model": "deterministic section-loss proxy; decay rates are screening assumptions, not calibrated field data",
        },
        "members": [_member_payload(member) for member in members],
        "decay_life_proxy": decay_life,
        "selected_decay_life": selected_decay_life,
        "limitations": [
            "不替代正式结构设计、荷载试验或有限元分析",
            "未包含节点榫卯、偏心、横桥向分布、风/地震/人群动力效应",
            "荷载与材料强度均为待甲方确认的方案阶段假设",
            "寿命为给定年腐朽深度下的截面能力筛选年限，不是实际案例寿命",
        ],
    }

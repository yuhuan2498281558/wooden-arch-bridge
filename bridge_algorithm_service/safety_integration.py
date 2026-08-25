from __future__ import annotations

import math
from typing import Any, Mapping


EXPOSURE_DECAY_RATES = {
    "protected": 0.05,
    "sheltered": 0.15,
    "exposed": 0.40,
}

MAINTENANCE_MULTIPLIERS = {
    "good": 0.70,
    "regular": 1.00,
    "poor": 1.30,
}

TREATMENT_MULTIPLIERS = {
    True: 0.70,
    False: 1.30,
}


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "on"}
    return bool(value)


def select_decay_profile(evidence: Mapping[str, Any] | None) -> dict[str, Any]:
    evidence = evidence or {}
    exposure_class = str(evidence.get("exposure_class") or "sheltered")
    maintenance_level = str(evidence.get("maintenance_level") or "regular")
    preservative_treatment = _bool(evidence.get("preservative_treatment"), True)

    base = EXPOSURE_DECAY_RATES.get(exposure_class, EXPOSURE_DECAY_RATES["sheltered"])
    maintenance = MAINTENANCE_MULTIPLIERS.get(maintenance_level, MAINTENANCE_MULTIPLIERS["regular"])
    treatment = TREATMENT_MULTIPLIERS[preservative_treatment]
    rate = max(0.005, base * maintenance * treatment)
    return {
        "exposure_class": exposure_class,
        "maintenance_level": maintenance_level,
        "preservative_treatment": preservative_treatment,
        "base_radial_loss_mm_per_year": base,
        "maintenance_multiplier": maintenance,
        "treatment_multiplier": treatment,
        "radial_loss_mm_per_year": round(rate, 4),
        "status": "screening_assumption_not_calibrated",
    }


def _structural_score_from_dcr(max_dcr: float) -> float:
    if max_dcr <= 0:
        return 100.0
    load_factor = 1.0 / max_dcr
    # Score is tied to the capacity/demand ratio rather than a linear DCR
    # penalty.  DCR=0.27 (load factor 3.7) is a healthy envelope result and
    # should not be reported as a 60-point structural score.
    bands = [
        (1.00, 60.0),
        (1.25, 75.0),
        (1.50, 82.0),
        (2.00, 88.0),
        (3.00, 93.0),
        (4.00, 95.0),
        (5.00, 96.0),
    ]
    if load_factor >= bands[-1][0]:
        return bands[-1][1]
    if load_factor < 1.0:
        return max(0.0, 60.0 * load_factor)
    for (lower, lower_score), (upper, upper_score) in zip(bands, bands[1:]):
        if lower <= load_factor <= upper:
            ratio = (load_factor - lower) / (upper - lower)
            return lower_score + ratio * (upper_score - lower_score)
    return 60.0


def _years_score(years: float) -> float:
    if years >= 100:
        return 95.0
    if years >= 60:
        return 85.0
    if years >= 30:
        return 70.0
    if years >= 15:
        return 55.0
    return 40.0


def _risk_from_years(years: float) -> tuple[str, str]:
    if years >= 80:
        return "低", "12 个月"
    if years >= 40:
        return "中低", "12 个月"
    if years >= 20:
        return "中等", "6 个月"
    return "较高", "3 个月"


def integrate_safety_analysis(
    *,
    service_life: Mapping[str, Any],
    structural_check: Mapping[str, Any],
    decay_profile: Mapping[str, Any],
) -> dict[str, Any]:
    selected = structural_check.get("selected_decay_life")
    if not isinstance(selected, Mapping):
        raise ValueError("structural_check is missing selected_decay_life")

    max_dcr = float(structural_check.get("max_dcr", 1.0))
    structural_years = float(selected.get("years_until_dcr_1", 0.0))
    bayesian_years = float(service_life.get("estimated_life", 0.0))
    remaining_years = float(service_life.get("remaining_life", 0.0))

    governing_years = min(bayesian_years, structural_years)
    if structural_years <= bayesian_years:
        governing_source = "section_loss_screening"
        governing_limit = "给定腐朽速率下构件截面 DCR 先达到 1.0"
    else:
        governing_source = "bayesian_prior"
        governing_limit = "贝叶斯先验预测中位数更早到达终点"

    structural_score = max(0.0, min(100.0, _structural_score_from_dcr(max_dcr)))
    durability_score = _years_score(structural_years)
    life_score = _years_score(bayesian_years)
    overall_score = round(0.55 * structural_score + 0.25 * durability_score + 0.20 * life_score, 1)
    grade = "A" if overall_score >= 85 else "B" if overall_score >= 70 else "C"
    label = "安全" if overall_score >= 85 else "可用" if overall_score >= 70 else "需复核"

    risk_level, inspect_cycle = _risk_from_years(structural_years)
    return {
        "method": "structure-durability-life-integrated-v1",
        "governing_life_years": round(governing_years, 1),
        "governing_life_source": governing_source,
        "governing_limit_state": governing_limit,
        "remaining_life_years": round(max(0.0, min(remaining_years, structural_years)), 1),
        "overall_score": {
            "score": overall_score,
            "grade": grade,
            "label": label,
            "color": "#27ae60" if overall_score >= 85 else "#e6a23c" if overall_score >= 70 else "#dc2626",
            "structural_score": round(structural_score, 1),
            "durability_score": round(durability_score, 1),
            "life_score": round(life_score, 1),
            "weights": {"structural_dcr": 0.55, "section_life": 0.25, "bayesian_life": 0.20},
        },
        "decay_risk": {
            "overall_risk": risk_level,
            "inspect_cycle": inspect_cycle,
            "selected_decay_profile": decay_profile,
            "critical_member": selected.get("critical_member"),
            "structural_years_until_dcr_1": structural_years,
            "bayesian_life_years": bayesian_years,
        },
        "chain": [
            {
                "step": 1,
                "name": "结构包络验算",
                "input": "预测几何、根径中值、n1/n2 平面系",
                "output": f"max DCR = {max_dcr:.4f}",
                "feeds": "构件验算表与综合评分",
            },
            {
                "step": 2,
                "name": "截面退化寿命筛选",
                "input": "暴露、维护、防腐情景换算的年径向腐朽深度",
                "output": f"DCR=1 年限 {structural_years:g} 年，临界构件 {selected.get('critical_member')}",
                "feeds": "耐久评分与腐朽风险",
            },
            {
                "step": 3,
                "name": "贝叶斯寿命区间",
                "input": "Weibull–Gamma 先验及当前服役年限右删失证据",
                "output": f"中位数 {bayesian_years:g} 年，90% 区间 {service_life.get('ci_90')}",
                "feeds": "寿命评分与结论取小值",
            },
            {
                "step": 4,
                "name": "综合结论",
                "input": "截面寿命与贝叶斯寿命取较不利者",
                "output": f"控制寿命 {governing_years:g} 年，来源 {governing_source}",
                "feeds": "总分、等级、风险提示",
            },
        ],
        "limitations": [
            "综合评分是三块证据的显式加权汇总，不是结构可靠度或安全鉴定结论",
            "荷载、材料强度和腐朽速率均为待确认方案阶段假设",
            "贝叶斯寿命与截面退化寿命口径不同，取较小值只是保守筛选规则",
        ],
    }

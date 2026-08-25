from __future__ import annotations

import math
from typing import Any, Mapping


# V1 deliberately keeps the legacy screening median and exposes every numeric
# adjustment as an uncalibrated engineering assumption.  Chinese common wood
# names alone do not identify species, heartwood/sapwood, grade or treatment,
# so they must not silently select invented species-specific lifetime tables.
PRIOR_MEDIAN_YEARS = 35.0
PRIOR_SHAPE = 4.0
WEIBULL_SHAPE = 3.0

EXPOSURE_FACTORS = {
    "protected": 1.20,
    "sheltered": 1.00,
    "exposed": 0.72,
}

MAINTENANCE_FACTORS = {
    "good": 1.15,
    "regular": 1.00,
    "poor": 0.78,
}

WOOD_LABELS = {
    "default": "混合/未细分木材",
    "杉木": "杉木（俗名，材料细节待确认）",
    "马尾松": "马尾松（俗名，材料细节待确认）",
    "柏木": "柏木（俗名，材料细节待确认）",
}


def _as_non_negative_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def _prior_rate(median: float, gamma_shape: float, weibull_shape: float) -> float:
    return median**weibull_shape / (2 ** (1 / gamma_shape) - 1)


def _conditional_quantile(
    probability: float,
    *,
    current_age: float,
    gamma_shape: float,
    rate: float,
    weibull_shape: float,
) -> float:
    # T|lambda ~ Weibull(lambda, k), lambda ~ Gamma(a, b rate).
    # Survival to current_age is a right-censored observation, hence
    # P(T>t|T>c)=((b+c^k)/(b+t^k))^a for t >= c.
    numerator = rate + current_age**weibull_shape
    lifetime_power = numerator / ((1 - probability) ** (1 / gamma_shape)) - rate
    return max(current_age, lifetime_power ** (1 / weibull_shape))


def _conditional_survival(
    lifetime: float,
    *,
    current_age: float,
    gamma_shape: float,
    rate: float,
    weibull_shape: float,
) -> float:
    if lifetime <= current_age:
        return 1.0
    numerator = rate + current_age**weibull_shape
    denominator = rate + lifetime**weibull_shape
    return (numerator / denominator) ** gamma_shape


def _conditional_mean(
    *,
    current_age: float,
    gamma_shape: float,
    rate: float,
    weibull_shape: float,
) -> float:
    # Integrate E[T|T>c] = c + integral_c^infinity S(t|T>c) dt.
    upper = _conditional_quantile(
        0.9999,
        current_age=current_age,
        gamma_shape=gamma_shape,
        rate=rate,
        weibull_shape=weibull_shape,
    )
    steps = 1200
    width = (upper - current_age) / steps
    area = 0.0
    previous = 1.0
    for index in range(1, steps + 1):
        lifetime = current_age + index * width
        survival = _conditional_survival(
            lifetime,
            current_age=current_age,
            gamma_shape=gamma_shape,
            rate=rate,
            weibull_shape=weibull_shape,
        )
        area += (previous + survival) * width / 2
        previous = survival
    return current_age + area


def estimate_service_life(
    wood_type: str,
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    evidence = evidence or {}
    wood_key = wood_type if wood_type in WOOD_LABELS else "default"
    exposure_class = str(evidence.get("exposure_class") or "sheltered")
    maintenance_level = str(evidence.get("maintenance_level") or "regular")
    preservative_treatment = _as_bool(evidence.get("preservative_treatment"), True)
    current_age = _as_non_negative_float(evidence.get("current_age"))

    exposure_factor = EXPOSURE_FACTORS.get(exposure_class, EXPOSURE_FACTORS["sheltered"])
    maintenance_factor = MAINTENANCE_FACTORS.get(maintenance_level, MAINTENANCE_FACTORS["regular"])
    treatment_factor = 1.15 if preservative_treatment else 0.85
    scenario_median = PRIOR_MEDIAN_YEARS * exposure_factor * maintenance_factor * treatment_factor
    rate = _prior_rate(scenario_median, PRIOR_SHAPE, WEIBULL_SHAPE)

    q05 = _conditional_quantile(
        0.05,
        current_age=current_age,
        gamma_shape=PRIOR_SHAPE,
        rate=rate,
        weibull_shape=WEIBULL_SHAPE,
    )
    q50 = _conditional_quantile(
        0.50,
        current_age=current_age,
        gamma_shape=PRIOR_SHAPE,
        rate=rate,
        weibull_shape=WEIBULL_SHAPE,
    )
    q95 = _conditional_quantile(
        0.95,
        current_age=current_age,
        gamma_shape=PRIOR_SHAPE,
        rate=rate,
        weibull_shape=WEIBULL_SHAPE,
    )
    mean = _conditional_mean(
        current_age=current_age,
        gamma_shape=PRIOR_SHAPE,
        rate=rate,
        weibull_shape=WEIBULL_SHAPE,
    )
    survival_10y = _conditional_survival(
        current_age + 10,
        current_age=current_age,
        gamma_shape=PRIOR_SHAPE,
        rate=rate,
        weibull_shape=WEIBULL_SHAPE,
    )

    posterior_updated = current_age > 0
    update_mode = "right_censored_update" if posterior_updated else "prior_predictive"
    interval_label = "90% 后验预测区间" if posterior_updated else "90% 先验预测区间"
    if posterior_updated:
        confidence = (
            f"已把服役至 {current_age:g} 年仍在役作为右删失证据更新。"
            "数值先验和情景因子尚未用本项目长期巡检数据校准，仅用于方案筛选。"
        )
    else:
        confidence = (
            "当前没有在役年限证据，结果为方案阶段先验预测。"
            "数值先验和情景因子尚未用本项目长期巡检数据校准。"
        )

    conservative_year = max(current_age, q05)
    return {
        "wood_label": WOOD_LABELS[wood_key],
        "endpoint": "真菌腐朽达到维修/更换阈值（阈值待项目确认）",
        "method": "解析 Weibull–Gamma 贝叶斯生存模型",
        "model_version": "service-life-weibull-gamma-v1",
        "model_status": "screening_prior_requires_calibration",
        "update_mode": update_mode,
        "posterior_updated": posterior_updated,
        "interval_label": interval_label,
        "prior_life": round(PRIOR_MEDIAN_YEARS, 1),
        "adjusted_prior_life": round(scenario_median, 1),
        "estimated_life": round(q50, 1),
        "posterior_mean_life": round(mean, 1),
        "ci_90": [round(q05, 1), round(q95, 1)],
        "current_age": round(current_age, 1),
        "remaining_life": round(max(0.0, q50 - current_age), 1),
        "remaining_ci_90": [round(max(0.0, value - current_age), 1) for value in (q05, q95)],
        "survival_probability_next_10y": round(survival_10y, 4),
        "evidence_used": ["current_age_right_censored"] if posterior_updated else [],
        "factors_used": ["exposure_class", "maintenance_level", "preservative_treatment"],
        "factors_not_used": ["span", "width", "rise", "condition_state"],
        "missing_factors": [
            "endpoint_threshold",
            "botanical_species_and_heartwood",
            "climate_and_moisture_history",
            "decay_depth_or_section_loss",
        ],
        "inputs": {
            "wood_type": wood_type,
            "exposure_class": exposure_class,
            "maintenance_level": maintenance_level,
            "preservative_treatment": preservative_treatment,
        },
        "prior_parameters": {
            "median_years": PRIOR_MEDIAN_YEARS,
            "gamma_shape": PRIOR_SHAPE,
            "weibull_shape": WEIBULL_SHAPE,
            "source": "legacy_screening_assumption_pending_approval",
        },
        "scenario_factors": {
            "exposure": exposure_factor,
            "maintenance": maintenance_factor,
            "preservative_treatment": treatment_factor,
            "status": "draft_engineering_assumptions_pending_calibration",
        },
        "confidence": confidence,
        "assumptions": [
            "木材俗名不用于虚构物种级寿命排序",
            "环境、维护和防腐处理只形成显式的方案先验情景",
            "当前年龄按右删失证据处理；无标定依据的定性状态不进入似然",
            "跨度、桥宽和矢高不直接映射为腐朽寿命",
        ],
        "phases": [
            {
                "phase": "常规巡检阶段",
                "range": f"当前—{round(conservative_year, 1):g} 年",
                "status": "按计划巡检",
                "action": "记录含水率、腐朽深度、节点状态和维修事件，作为后续贝叶斯更新证据。",
                "color": "#27ae60",
            },
            {
                "phase": "保守复核节点",
                "range": f"约 {round(conservative_year, 1):g} 年",
                "status": "专项检测",
                "action": "在寿命分布第 5 百分位附近安排材料与连接专项检测。",
                "color": "#e6a23c",
            },
            {
                "phase": "滚动更新窗口",
                "range": f"{round(conservative_year, 1):g}—{round(q50, 1):g} 年",
                "status": "以后验结果滚动更新",
                "action": "结合实测退化量重新计算剩余寿命，不以当前方案先验直接决定报废。",
                "color": "#c0392b",
            },
        ],
    }

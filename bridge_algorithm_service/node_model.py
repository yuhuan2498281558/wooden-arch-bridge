from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

from .node_geometry import DEFAULT_ALPHA, DEFAULT_BETA, ratio_warnings
from .back_half_high_tail import mix_back_half_high_tail_alpha
from .front_half_low_tail import mix_front_half_low_tail_alpha


MODEL_ENABLE_ENV = "BRIDGE_NODE_MODEL_ENABLED"
MODEL_DIR_ENV = "BRIDGE_NODE_MODEL_DIR"
DESIGN_MODE_MODEL_ENABLE_ENV = "BRIDGE_NODE_DESIGN_MODE_MODEL_ENABLED"
DESIGN_MODE_MODEL_DIR_ENV = "BRIDGE_NODE_DESIGN_MODE_MODEL_DIR"
DESIGN_MODE_MODEL_FILE = "alpha_design_mode_model.joblib"
SUPPORTED_DESIGN_MODES = {"front_half", "back_half"}
SUPPORTED_FEATURES = {
    "span_m",
    "three_miao_rise_span_ratio",
    "three_miao_rise_m",
    "span_count",
    "is_edge_span",
    "span_center_distance",
}


class NodeModelError(RuntimeError):
    """Base exception for a configured but unusable node-ratio model."""


class NodeModelOutOfDistribution(NodeModelError):
    """Raised when one or more design features are outside the training range."""


@dataclass(frozen=True)
class NodeRatioDecision:
    alpha: float
    beta: float
    source: str
    metadata: dict[str, Any]


def _enabled(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _artifact_prediction(artifact: Mapping[str, Any], feature_values: Mapping[str, float]) -> float:
    feature_names = tuple(str(name) for name in artifact.get("feature_names", []))
    if not feature_names or any(name not in SUPPORTED_FEATURES for name in feature_names):
        raise NodeModelError(f"unsupported feature contract: {feature_names!r}")
    missing = [name for name in feature_names if name not in feature_values]
    if missing:
        raise NodeModelError(f"missing model features: {', '.join(missing)}")

    values = np.asarray([[float(feature_values[name]) for name in feature_names]], dtype=float)
    if not np.all(np.isfinite(values)):
        raise NodeModelError("model features must be finite")

    ranges = artifact.get("training_feature_range")
    if not isinstance(ranges, Mapping):
        raise NodeModelError("artifact has no training_feature_range")
    outside: list[str] = []
    for index, name in enumerate(feature_names):
        bounds = ranges.get(name)
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise NodeModelError(f"invalid training range for {name}")
        lower, upper = float(bounds[0]), float(bounds[1])
        value = float(values[0, index])
        if not lower <= value <= upper:
            outside.append(f"{name}={value:g} outside [{lower:g}, {upper:g}]")
    if outside:
        raise NodeModelOutOfDistribution("; ".join(outside))

    estimator = artifact.get("estimator")
    if estimator is None:
        constant = artifact.get("constant")
        if constant is None:
            raise NodeModelError("constant artifact has no constant value")
        prediction = float(constant)
    else:
        prediction = float(artifact["rule_baseline"]) + float(estimator.predict(values)[0])

    bounds = artifact.get("bounds")
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        raise NodeModelError("artifact has invalid prediction bounds")
    prediction = float(np.clip(prediction, float(bounds[0]), float(bounds[1])))
    if not math.isfinite(prediction):
        raise NodeModelError("model produced a non-finite prediction")
    return prediction


def _local_domain_status(
    artifact: Mapping[str, Any],
    feature_values: Mapping[str, float],
) -> dict[str, Any]:
    domain = artifact.get("applicability_domain")
    if not isinstance(domain, Mapping):
        return {"available": False, "within_local_domain": None}
    feature_names = tuple(str(name) for name in artifact.get("feature_names", []))
    try:
        center = np.asarray(domain["center"], dtype=float)
        scale = np.asarray(domain["scale"], dtype=float)
        training_points = np.asarray(domain["normalized_training_points"], dtype=float)
        values = np.asarray([float(feature_values[name]) for name in feature_names], dtype=float)
        threshold = float(domain["sparse_region_threshold"])
        configured_k = int(domain.get("nearest_neighbor_k", 5))
    except (KeyError, TypeError, ValueError) as exc:
        raise NodeModelError(f"invalid applicability domain: {exc}") from exc
    if center.shape != values.shape or scale.shape != values.shape:
        raise NodeModelError("applicability domain shape mismatch")
    if training_points.ndim != 2 or training_points.shape[1] != len(values) or len(training_points) == 0:
        raise NodeModelError("applicability domain training points are invalid")
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise NodeModelError("applicability domain normalization is invalid")
    normalized = (values - center) / scale
    distances = np.linalg.norm(training_points - normalized, axis=1)
    k = min(max(configured_k, 1), len(distances))
    mean_distance = float(np.mean(np.partition(distances, kth=k - 1)[:k]))
    return {
        "available": True,
        "method": domain.get("method"),
        "mean_neighbor_distance": mean_distance,
        "sparse_region_threshold": threshold,
        "within_local_domain": mean_distance <= threshold,
    }


class NodeRatioModelBundle:
    def __init__(self, alpha_artifact: Mapping[str, Any], beta_artifact: Mapping[str, Any]) -> None:
        self.artifacts = {"alpha": dict(alpha_artifact), "beta": dict(beta_artifact)}
        for target, artifact in self.artifacts.items():
            if artifact.get("target") != target:
                raise NodeModelError(f"{target} artifact target mismatch")
            if not str(artifact.get("artifact_version") or "").startswith("five-miao-node-pilot-"):
                raise NodeModelError(f"unsupported {target} artifact version")
            if artifact.get("status") != "pilot_not_for_production":
                raise NodeModelError(f"unexpected {target} artifact status")

    @classmethod
    def from_directory(cls, directory: Path) -> "NodeRatioModelBundle":
        directory = directory.resolve()
        if not directory.is_dir():
            raise NodeModelError(f"model directory does not exist: {directory}")
        paths = {
            target: directory / f"{target}_model.joblib"
            for target in ("alpha", "beta")
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise NodeModelError("missing model artifact(s): " + ", ".join(missing))
        # joblib artifacts are executable pickle payloads.  Only load files
        # produced by this repository's trusted offline training pipeline.
        return cls(joblib.load(paths["alpha"]), joblib.load(paths["beta"]))

    def predict(self, feature_values: Mapping[str, float]) -> NodeRatioDecision:
        alpha = _artifact_prediction(self.artifacts["alpha"], feature_values)
        beta = _artifact_prediction(self.artifacts["beta"], feature_values)
        warnings = ratio_warnings(alpha, beta)
        if warnings:
            raise NodeModelError("; ".join(warnings))
        versions = sorted({str(artifact["artifact_version"]) for artifact in self.artifacts.values()})
        predictions = {"alpha": alpha, "beta": beta}
        local_domains = {
            target: _local_domain_status(artifact, feature_values)
            for target, artifact in self.artifacts.items()
        }
        sparse_targets = [
            target
            for target, status in local_domains.items()
            if status.get("within_local_domain") is False
        ]
        return NodeRatioDecision(
            alpha=alpha,
            beta=beta,
            source="pilot_model",
            metadata={
                "model_status": "pilot_not_for_production",
                "model_versions": versions,
                "target_models": {
                    target: {
                        "model_name": artifact.get("model_name"),
                        "feature_set": artifact.get("feature_set"),
                        "feature_names": list(artifact.get("feature_names", [])),
                        "oof_absolute_error_quantiles": artifact.get("oof_absolute_error_quantiles", {}),
                        "independent_holdout_metrics": artifact.get("independent_holdout_metrics", {}),
                        "prediction": predictions[target],
                        "prediction_interval_90": [
                            max(float(artifact["bounds"][0]), predictions[target] - float(artifact.get("oof_absolute_error_quantiles", {}).get("q90", 0.0))),
                            min(float(artifact["bounds"][1]), predictions[target] + float(artifact.get("oof_absolute_error_quantiles", {}).get("q90", 0.0))),
                        ],
                        "local_domain": local_domains[target],
                    }
                    for target, artifact in self.artifacts.items()
                },
                "feature_values": {name: float(value) for name, value in feature_values.items()},
                "within_training_range": True,
                "within_local_domain": not sparse_targets,
                "model_warnings": [f"sparse_training_region:{target}" for target in sparse_targets],
            },
        )


class DesignModeAlphaModel:
    def __init__(self, artifact: Mapping[str, Any]) -> None:
        self.artifact = dict(artifact)
        version = str(self.artifact.get("artifact_version") or "")
        if version != "five-miao-node-pilot-v9-designer-selected-mode":
            raise NodeModelError("unsupported design-mode alpha artifact version")
        if self.artifact.get("status") != "research_designer_mode_not_for_deployment":
            raise NodeModelError("unexpected design-mode alpha artifact status")
        if self.artifact.get("target") != "alpha":
            raise NodeModelError("design-mode artifact target mismatch")
        if self.artifact.get("deployment_mode_source_required") != "designer_selected":
            raise NodeModelError("design-mode artifact does not require designer selection")
        if self.artifact.get("integration_gate_passed") is not True:
            raise NodeModelError("design-mode artifact did not pass its integration gate")
        experts = self.artifact.get("experts")
        if not isinstance(experts, Mapping) or set(experts) != SUPPORTED_DESIGN_MODES:
            raise NodeModelError("design-mode artifact must contain front/back experts")
        self.experts = {name: dict(experts[name]) for name in SUPPORTED_DESIGN_MODES}

    @classmethod
    def from_directory(cls, directory: Path) -> "DesignModeAlphaModel":
        directory = directory.resolve()
        if not directory.is_dir():
            raise NodeModelError(f"design-mode model directory does not exist: {directory}")
        path = directory / DESIGN_MODE_MODEL_FILE
        if not path.is_file():
            raise NodeModelError(f"missing design-mode model artifact: {path}")
        # This is a trusted joblib produced by this repository's offline pipeline.
        return cls(joblib.load(path))

    def predict(self, feature_values: Mapping[str, float], design_mode: str) -> tuple[float, dict[str, Any]]:
        if design_mode not in SUPPORTED_DESIGN_MODES:
            raise NodeModelError(f"unsupported outer-node design mode: {design_mode}")
        expert = self.experts[design_mode]
        feature_names = tuple(str(name) for name in expert.get("feature_names", []))
        if any(name not in SUPPORTED_FEATURES for name in feature_names):
            raise NodeModelError(f"unsupported design-mode feature contract: {feature_names!r}")
        missing = [name for name in feature_names if name not in feature_values]
        if missing:
            raise NodeModelError(f"missing design-mode features: {', '.join(missing)}")
        values = np.asarray([[float(feature_values[name]) for name in feature_names]], dtype=float)
        if not np.all(np.isfinite(values)):
            raise NodeModelError("design-mode features must be finite")

        ranges = expert.get("training_feature_range")
        if not isinstance(ranges, Mapping):
            raise NodeModelError("design-mode expert has no training_feature_range")
        outside: list[str] = []
        for index, name in enumerate(feature_names):
            bounds = ranges.get(name)
            if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                raise NodeModelError(f"invalid design-mode training range for {name}")
            lower, upper = float(bounds[0]), float(bounds[1])
            value = float(values[0, index])
            if not lower <= value <= upper:
                outside.append(f"{name}={value:g} outside [{lower:g}, {upper:g}]")
        if outside:
            raise NodeModelOutOfDistribution("; ".join(outside))

        estimator = expert.get("estimator")
        if estimator is None:
            constant = expert.get("constant")
            if constant is None:
                raise NodeModelError("design-mode constant expert has no value")
            prediction = float(constant)
        else:
            prediction = float(estimator.predict(values)[0])
        high_tail_applied = False
        low_tail_applied = False
        if design_mode == "back_half":
            prediction, high_tail_applied = mix_back_half_high_tail_alpha(
                prediction,
                self.artifact.get("back_half_high_tail"),
            )
        elif design_mode == "front_half":
            prediction, low_tail_applied = mix_front_half_low_tail_alpha(
                prediction,
                self.artifact.get("front_half_low_tail"),
            )
        output_bounds = expert.get("output_bounds")
        if not isinstance(output_bounds, (list, tuple)) or len(output_bounds) != 2:
            raise NodeModelError("design-mode expert has invalid output bounds")
        lower, upper = float(output_bounds[0]), float(output_bounds[1])
        if design_mode == "front_half":
            lower = math.nextafter(lower, upper)
            upper = math.nextafter(upper, lower)
        else:
            upper = math.nextafter(upper, lower)
        prediction = float(np.clip(prediction, lower, upper))
        if not math.isfinite(prediction):
            raise NodeModelError("design-mode expert produced a non-finite prediction")

        local_domain = _local_domain_status(expert, feature_values)
        metrics = (
            self.artifact.get("development_metrics", {})
            .get("by_mode", {})
            .get(design_mode, {})
        )
        q90 = float(metrics.get("absolute_error_quantiles", {}).get("q90", 0.0))
        return prediction, {
            "model_name": expert.get("model_name"),
            "feature_set": expert.get("feature_set"),
            "feature_names": list(feature_names),
            "prediction": prediction,
            "high_tail_mix_applied": high_tail_applied,
            "low_tail_mix_applied": low_tail_applied,
            "prediction_interval_90": [
                max(lower, prediction - q90),
                min(upper, prediction + q90),
            ],
            "local_domain": local_domain,
            "within_local_domain": local_domain.get("within_local_domain"),
            "model_warnings": (
                ["sparse_training_region:alpha_design_mode"]
                if local_domain.get("within_local_domain") is False
                else []
            ),
        }


@lru_cache(maxsize=1)
def _configured_bundle() -> NodeRatioModelBundle | None:
    if not _enabled(os.environ.get(MODEL_ENABLE_ENV)):
        return None
    configured = str(os.environ.get(MODEL_DIR_ENV) or "").strip()
    if not configured:
        raise NodeModelError(f"{MODEL_DIR_ENV} is not configured")
    return NodeRatioModelBundle.from_directory(Path(configured))


@lru_cache(maxsize=1)
def _configured_design_mode_model() -> DesignModeAlphaModel | None:
    if not _enabled(os.environ.get(DESIGN_MODE_MODEL_ENABLE_ENV)):
        return None
    configured = str(os.environ.get(DESIGN_MODE_MODEL_DIR_ENV) or "").strip()
    if not configured:
        raise NodeModelError(f"{DESIGN_MODE_MODEL_DIR_ENV} is not configured")
    return DesignModeAlphaModel.from_directory(Path(configured))


def reset_node_model_cache() -> None:
    _configured_bundle.cache_clear()
    _configured_design_mode_model.cache_clear()


def resolve_node_ratios(
    feature_values: Mapping[str, float],
    design_mode: str | None = None,
) -> NodeRatioDecision:
    if not _enabled(os.environ.get(MODEL_ENABLE_ENV)):
        metadata: dict[str, Any] = {
            "model_enabled": False,
            "fallback_reason": "model_disabled",
        }
        if design_mode is not None:
            metadata["design_mode"] = {
                "requested": design_mode,
                "selection_source": "designer_selected",
                "applied": False,
                "fallback_reason": "node_model_disabled",
            }
        return NodeRatioDecision(
            DEFAULT_ALPHA,
            DEFAULT_BETA,
            "rule_fallback",
            metadata,
        )
    try:
        bundle = _configured_bundle()
        if bundle is None:
            raise NodeModelError("model is disabled")
        base_decision = bundle.predict(feature_values)
    except NodeModelOutOfDistribution as exc:
        reason = "out_of_training_range"
        detail = str(exc)
    except Exception as exc:
        reason = "model_unavailable"
        detail = str(exc)
    else:
        if design_mode is None:
            return base_decision
        metadata = dict(base_decision.metadata)
        if design_mode not in SUPPORTED_DESIGN_MODES:
            metadata["design_mode"] = {
                "requested": design_mode,
                "applied": False,
                "fallback_reason": "invalid_design_mode",
            }
            return NodeRatioDecision(
                base_decision.alpha,
                base_decision.beta,
                base_decision.source,
                metadata,
            )
        if not _enabled(os.environ.get(DESIGN_MODE_MODEL_ENABLE_ENV)):
            metadata["design_mode"] = {
                "requested": design_mode,
                "selection_source": "designer_selected",
                "applied": False,
                "fallback_reason": "design_mode_model_disabled",
            }
            return NodeRatioDecision(
                base_decision.alpha,
                base_decision.beta,
                base_decision.source,
                metadata,
            )
        try:
            design_model = _configured_design_mode_model()
            if design_model is None:
                raise NodeModelError("design-mode model is disabled")
            alpha, alpha_metadata = design_model.predict(feature_values, design_mode)
        except NodeModelOutOfDistribution as exc:
            mode_reason = "out_of_training_range"
            mode_detail = str(exc)
        except Exception as exc:
            mode_reason = "model_unavailable"
            mode_detail = str(exc)
        else:
            target_models = dict(metadata.get("target_models") or {})
            if "alpha" in target_models:
                target_models["alpha_v6_baseline"] = target_models["alpha"]
            target_models["alpha"] = {
                **alpha_metadata,
                "artifact_version": design_model.artifact["artifact_version"],
                "model_status": design_model.artifact["status"],
                "design_mode": design_mode,
                "selection_source": "designer_selected",
            }
            versions = sorted({
                *[str(value) for value in metadata.get("model_versions", [])],
                str(design_model.artifact["artifact_version"]),
            })
            model_warnings = list(metadata.get("model_warnings") or [])
            model_warnings.extend(alpha_metadata.get("model_warnings") or [])
            metadata.update({
                "model_status": design_model.artifact["status"],
                "model_versions": versions,
                "target_models": target_models,
                "within_local_domain": (
                    metadata.get("within_local_domain") is not False
                    and alpha_metadata.get("within_local_domain") is not False
                ),
                "model_warnings": model_warnings,
                "design_mode": {
                    "requested": design_mode,
                    "selection_source": "designer_selected",
                    "applied": True,
                    "historical_mode_source": design_model.artifact.get("historical_mode_source"),
                },
            })
            return NodeRatioDecision(
                alpha,
                base_decision.beta,
                "designer_mode_pilot",
                metadata,
            )
        metadata["design_mode"] = {
            "requested": design_mode,
            "selection_source": "designer_selected",
            "applied": False,
            "fallback_reason": mode_reason,
            "fallback_detail": mode_detail,
        }
        return NodeRatioDecision(
            base_decision.alpha,
            base_decision.beta,
            base_decision.source,
            metadata,
        )
    fallback_metadata: dict[str, Any] = {
        "model_enabled": True,
        "within_training_range": False,
        "fallback_reason": reason,
        "fallback_detail": detail,
    }
    if design_mode is not None:
        fallback_metadata["design_mode"] = {
            "requested": design_mode,
            "selection_source": "designer_selected",
            "applied": False,
            "fallback_reason": reason,
            "fallback_detail": detail,
        }
    return NodeRatioDecision(
        DEFAULT_ALPHA,
        DEFAULT_BETA,
        "rule_fallback",
        fallback_metadata,
    )


def _design_mode_model_status() -> dict[str, Any]:
    if not _enabled(os.environ.get(DESIGN_MODE_MODEL_ENABLE_ENV)):
        return {"enabled": False, "available": False, "reason": "model_disabled"}
    try:
        model = _configured_design_mode_model()
        return {
            "enabled": True,
            "available": model is not None,
            "version": model.artifact.get("artifact_version") if model else None,
            "status": model.artifact.get("status") if model else None,
            "requires_designer_selection": True,
            "supported_modes": sorted(SUPPORTED_DESIGN_MODES),
        }
    except Exception as exc:
        return {"enabled": True, "available": False, "reason": str(exc)}


def node_model_status() -> dict[str, Any]:
    design_mode_status = _design_mode_model_status()
    if not _enabled(os.environ.get(MODEL_ENABLE_ENV)):
        return {
            "enabled": False,
            "available": False,
            "reason": "model_disabled",
            "design_mode_model": design_mode_status,
        }
    try:
        bundle = _configured_bundle()
        return {
            "enabled": True,
            "available": bundle is not None,
            "versions": sorted({
                str(artifact["artifact_version"])
                for artifact in bundle.artifacts.values()
            }) if bundle else [],
            "design_mode_model": design_mode_status,
        }
    except Exception as exc:
        return {
            "enabled": True,
            "available": False,
            "reason": str(exc),
            "design_mode_model": design_mode_status,
        }

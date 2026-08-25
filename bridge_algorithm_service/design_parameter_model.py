from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Any, Mapping

import numpy as np
import xgboost as xgb


SCHEMA_VERSION = "mentor-design-parameter-model-v1"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "mentor_models"

_TRAIN_BOUNDS = {
    "span": {"core": (10.4, 35.0), "valid": (5.9, 37.6)},
    "width": {"core": (3.8, 5.8), "valid": (3.2, 6.83)},
    "length": {"core": (16.0, 70.0), "valid": (11.6, 115.0)},
    "n1": {"core": (7, 9), "valid": (5, 11)},
    "n2": {"core": (6, 8), "valid": (4, 10)},
}
_N1_COVERAGE = {5: 0.007, 7: 0.267, 8: 0.022, 9: 0.696, 11: 0.007}
_N2_COVERAGE = {4: 0.007, 6: 0.267, 7: 0.022, 8: 0.696, 10: 0.007}
_SATURATION_MIN_RANGE = {
    "s1_flat": 10.0,
    "s1_diag": 5.0,
    "s2_flat": 15.0,
    "s2_diag1": 5.0,
    "s2_diag2": 3.0,
}


class DesignParameterModelUnavailable(RuntimeError):
    pass


def _enabled() -> bool:
    return os.environ.get("BRIDGE_DESIGN_MODEL_ENABLED", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _model_dir() -> Path:
    configured = os.environ.get("BRIDGE_DESIGN_MODEL_DIR", "").strip()
    return Path(configured).expanduser().resolve() if configured else DEFAULT_MODEL_DIR


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_result(
    length: float,
    width: float,
    span: float,
    n1: int,
    n2: int,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    warnings_in: list[str] = []
    warnings_out: list[str] = []
    score = 100

    for feature, value in (("span", span), ("width", width), ("length", length)):
        core_lo, core_hi = _TRAIN_BOUNDS[feature]["core"]
        valid_lo, valid_hi = _TRAIN_BOUNDS[feature]["valid"]
        if not valid_lo <= value <= valid_hi:
            warnings_in.append(
                f"{feature}={value} 超出训练数据范围 [{valid_lo}, {valid_hi}]，预测为外推，可信度低"
            )
            score -= 35
        elif not core_lo <= value <= core_hi:
            warnings_in.append(
                f"{feature}={value} 在训练数据边缘区间（核心区 {core_lo}~{core_hi}），可信度中等"
            )
            score -= 12

    n1_coverage = _N1_COVERAGE.get(n1, 0.0)
    n2_coverage = _N2_COVERAGE.get(n2, 0.0)
    if n1_coverage < 0.02:
        warnings_in.append(f"n1={n1} 在训练集中样本极少，预测参考价值有限")
        score -= 20
    elif n1_coverage < 0.10:
        warnings_in.append(f"n1={n1} 样本较少（主流为 7 或 9），可信度中等")
        score -= 8
    if n2_coverage < 0.02:
        warnings_in.append(f"n2={n2} 在训练集中样本极少，预测参考价值有限")
        score -= 20
    if n1 != n2 + 1:
        warnings_in.append(f"n2={n2} 不等于 n1-1={n1 - 1}，偏离原训练数据的常规组合")
        score -= 10

    rise_span = float(result.get("rise_span", 0.0))
    rise_span_ok = 0.13 <= rise_span <= 0.22
    if not 0.08 <= rise_span <= 0.30:
        warnings_out.append(f"矢跨比={rise_span:.4f} 超出传统木拱桥常用区间 0.08~0.30")
        score -= 25
    elif not rise_span_ok:
        warnings_out.append(f"矢跨比={rise_span:.4f} 偏离原数据主流区间 0.13~0.22")
        score -= 10

    saturated: list[str] = []
    for key, minimum_width in _SATURATION_MIN_RANGE.items():
        lo = float(result.get(f"{key}_lo", 0.0))
        hi = float(result.get(f"{key}_hi", 0.0))
        if hi - lo < minimum_width:
            saturated.append(key)
    if saturated:
        field_names = {
            "s1_flat": "三节苗平弦",
            "s1_diag": "三节苗斜弦",
            "s2_flat": "五节苗平弦",
            "s2_diag1": "五节苗斜弦1",
            "s2_diag2": "五节苗斜弦2",
        }
        warnings_out.append(
            "、".join(field_names.get(key, key) for key in saturated)
            + "根径预测区间偏窄，建议人工复核"
        )
        score -= 15 * min(len(saturated), 3)

    if float(result.get("s1_flat_lo", 0.0)) >= 285:
        warnings_out.append("三节苗平弦根径下限接近模型输出上界，建议人工复核")
        score -= 10

    score = max(0, min(100, score))
    if score >= 80:
        trust_level = "high"
        note = "输入位于原训练数据核心范围，结果可用于传统骨架方案参考。"
    elif score >= 55:
        trust_level = "medium"
        note = "输入部分偏离原训练数据核心范围，建议结合相似古桥资料复核。"
    else:
        trust_level = "low"
        note = "输入明显偏离原训练数据，当前结果属于外推，应重点复核。"

    return {
        "trust_level": trust_level,
        "trust_score": round(score),
        "input_warnings": warnings_in,
        "output_warnings": warnings_out,
        "saturated_fields": saturated,
        "rise_span_ok": rise_span_ok,
        "note": note,
    }


class MentorDesignParameterModel:
    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR):
        self.model_dir = Path(model_dir).resolve()
        manifest_path = self.model_dir / "manifest.json"
        try:
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise DesignParameterModelUnavailable(f"cannot read model manifest: {exc}") from exc

        if self.manifest.get("schema_version") != SCHEMA_VERSION:
            raise DesignParameterModelUnavailable(
                f"unsupported model schema: {self.manifest.get('schema_version')}"
            )

        stage1 = self.manifest["stage1"]
        stage2 = self.manifest["stage2"]
        self.stage1_path = self.model_dir / stage1["artifact"]
        self.stage2_path = self.model_dir / stage2["artifact"]
        for path, expected_hash in (
            (self.stage1_path, stage1["artifact_sha256"]),
            (self.stage2_path, stage2["artifact_sha256"]),
        ):
            if not path.is_file():
                raise DesignParameterModelUnavailable(f"model artifact is missing: {path.name}")
            actual_hash = _sha256(path)
            if actual_hash != expected_hash:
                raise DesignParameterModelUnavailable(f"model artifact hash mismatch: {path.name}")

        self.booster = xgb.Booster()
        try:
            self.booster.load_model(self.stage1_path)
        except Exception as exc:
            raise DesignParameterModelUnavailable(f"cannot load stage-1 XGBoost model: {exc}") from exc

        self.feature_names = tuple(stage1["feature_names"])
        self.feature_min = np.asarray(stage1["feature_min"], dtype=np.float64)
        self.feature_max = np.asarray(stage1["feature_max"], dtype=np.float64)
        self.output_bounds = tuple(float(value) for value in stage1["output_bounds"])
        self.missing_rise_ratio = float(stage1["missing_rise_policy"]["ratio"])

        try:
            with np.load(self.stage2_path, allow_pickle=False) as arrays:
                self.w1 = arrays["w1"].astype(np.float64)
                self.w2 = arrays["w2"].astype(np.float64)
                self.wl = arrays["wl"].astype(np.float64)
                self.b1 = arrays["b1"].astype(np.float64)
                self.b2 = arrays["b2"].astype(np.float64)
                self.xmeans = arrays["xmeans"].astype(np.float64)
        except Exception as exc:
            raise DesignParameterModelUnavailable(f"cannot load CF-BPNN matrices: {exc}") from exc

        expected_shapes = {
            "w1": (15, 14),
            "w2": (10, 14),
            "wl": (10, 15),
            "b1": (15,),
            "b2": (10,),
            "xmeans": (8,),
        }
        actual_shapes = {
            "w1": self.w1.shape,
            "w2": self.w2.shape,
            "wl": self.wl.shape,
            "b1": self.b1.shape,
            "b2": self.b2.shape,
            "xmeans": self.xmeans.shape,
        }
        if actual_shapes != expected_shapes:
            raise DesignParameterModelUnavailable(f"unexpected CF-BPNN matrix shapes: {actual_shapes}")

        self.stage2_mapping = stage2["input_mapping"]
        self.output_names = tuple(stage2["output_names"])
        self.output_ranges = np.asarray(stage2["output_ranges_mm"], dtype=np.float64)

    @property
    def model_version(self) -> str:
        return str(self.manifest["model_version"])

    def _stage1_predict(
        self,
        *,
        length: float,
        width: float,
        span: float,
        rise: float | None,
        n1: int,
        n2: int,
    ) -> tuple[float, float, str]:
        if rise is None:
            rise_input = span * self.missing_rise_ratio
            rise_input_mode = "legacy_span_ratio_seed"
        else:
            rise_input = float(rise)
            rise_input_mode = "provided"

        raw = np.asarray([[length, width, span, rise_input, n1, n2]], dtype=np.float64)
        scaled = (raw - self.feature_min) / (self.feature_max - self.feature_min)
        predicted = float(self.booster.inplace_predict(scaled)[0])
        rise_span = float(np.clip(predicted, self.output_bounds[0], self.output_bounds[1]))
        return rise_span, span * rise_span, rise_input_mode

    def _stage2_predict(self, *, width: float, span: float, n1: int, n2: int) -> dict[str, float]:
        normalized: dict[str, float] = {}
        for name, value in (("width", width), ("span", span), ("n1", n1), ("n2", n2)):
            lo, hi = (float(item) for item in self.stage2_mapping[name]["range"])
            normalized[name] = float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0))

        y15 = np.zeros(15, dtype=np.float64)
        y15[0] = 1.0
        y15[1], y15[2] = normalized["width"], 0.0
        y15[3], y15[4] = normalized["span"], 0.0
        y15[5], y15[6] = self.xmeans[3], 0.0
        y15[7], y15[8] = self.xmeans[4], 1.0
        y15[9], y15[10] = self.xmeans[5], 1.0
        y15[11], y15[12] = normalized["n1"], 0.0
        y15[13], y15[14] = normalized["n2"], 0.0

        network_input = 2.0 * y15[1:] - 1.0
        hidden = np.tanh(self.w1 @ network_input + self.b1)
        output = self.w2 @ network_input + self.wl @ hidden + self.b2
        output_01 = (output + 1.0) / 2.0
        physical = self.output_ranges[:, 0] + np.clip(output_01, 0.0, 1.0) * (
            self.output_ranges[:, 1] - self.output_ranges[:, 0]
        )
        result = {name: round(float(physical[index]), 1) for index, name in enumerate(self.output_names)}
        for lo_key, hi_key in (
            ("s1_flat_lo", "s1_flat_hi"),
            ("s1_diag_lo", "s1_diag_hi"),
            ("s2_flat_lo", "s2_flat_hi"),
            ("s2_diag1_lo", "s2_diag1_hi"),
            ("s2_diag2_lo", "s2_diag2_hi"),
        ):
            if result[lo_key] > result[hi_key]:
                result[lo_key], result[hi_key] = result[hi_key], result[lo_key]
        return result

    def predict(
        self,
        *,
        length: float,
        width: float,
        span: float,
        n1: int,
        n2: int,
        rise: float | None = None,
    ) -> dict[str, Any]:
        rise_span, rise_height, rise_input_mode = self._stage1_predict(
            length=length,
            width=width,
            span=span,
            rise=rise,
            n1=n1,
            n2=n2,
        )
        result: dict[str, Any] = {
            "rise_span": round(rise_span, 4),
            "rise_height": round(rise_height, 3),
        }
        result.update(self._stage2_predict(width=width, span=span, n1=n1, n2=n2))
        validation = _validate_result(length, width, span, n1, n2, result)
        validation.update(
            {
                "method": "SSA-XGBoost + CF-BPNN",
                "model_status": "active",
                "model_version": self.model_version,
                "rise_input_mode": rise_input_mode,
                "within_training_range": not validation["input_warnings"],
                "stage1": {
                    "name": self.manifest["stage1"]["name"],
                    "runtime_model": self.manifest["stage1"]["runtime_model"],
                    "artifact_sha256": self.manifest["stage1"]["artifact_sha256"],
                    "feature_names": list(self.feature_names),
                    "metrics": self.manifest["stage1"]["metrics_from_legacy_bundle"],
                },
                "stage2": {
                    "name": self.manifest["stage2"]["name"],
                    "runtime_model": self.manifest["stage2"]["runtime_model"],
                    "artifact_sha256": self.manifest["stage2"]["artifact_sha256"],
                },
            }
        )
        result["validation"] = validation
        return result


_cache_lock = threading.Lock()
_cached_key: tuple[bool, str] | None = None
_cached_model: MentorDesignParameterModel | None = None
_cached_error: str | None = None


def _load_cached_model() -> MentorDesignParameterModel:
    global _cached_key, _cached_model, _cached_error
    enabled = _enabled()
    model_dir = _model_dir()
    key = (enabled, str(model_dir))
    with _cache_lock:
        if _cached_key != key:
            _cached_key = key
            _cached_model = None
            _cached_error = None
            if enabled:
                try:
                    _cached_model = MentorDesignParameterModel(model_dir)
                except Exception as exc:
                    _cached_error = str(exc)
        if not enabled:
            raise DesignParameterModelUnavailable("mentor design model is disabled")
        if _cached_model is None:
            raise DesignParameterModelUnavailable(_cached_error or "mentor design model is unavailable")
        return _cached_model


def predict_design_parameters(**kwargs: Any) -> dict[str, Any]:
    try:
        return _load_cached_model().predict(**kwargs)
    except DesignParameterModelUnavailable:
        raise
    except Exception as exc:
        raise DesignParameterModelUnavailable(f"mentor design inference failed: {exc}") from exc


def design_parameter_model_status() -> dict[str, Any]:
    enabled = _enabled()
    status: dict[str, Any] = {
        "enabled": enabled,
        "available": False,
        "model_dir": str(_model_dir()),
        "schema_version": SCHEMA_VERSION,
    }
    if not enabled:
        status["reason"] = "disabled"
        return status
    try:
        model = _load_cached_model()
    except DesignParameterModelUnavailable as exc:
        status["reason"] = str(exc)
        return status
    status.update(
        {
            "available": True,
            "model_version": model.model_version,
            "stage1": model.manifest["stage1"]["name"],
            "stage2": model.manifest["stage2"]["name"],
        }
    )
    return status


def _reset_model_cache_for_tests() -> None:
    global _cached_key, _cached_model, _cached_error
    with _cache_lock:
        _cached_key = None
        _cached_model = None
        _cached_error = None

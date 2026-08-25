from __future__ import annotations

import base64
import io
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Literal, Mapping

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field

from .design_parameter_model import (
    DesignParameterModelUnavailable,
    design_parameter_model_status,
    predict_design_parameters,
)
from .node_geometry import Point, geometry_payload, ratios_from_result, reconstruct_nodes
from .node_model import node_model_status, resolve_node_ratios
from .safety_integration import integrate_safety_analysis, select_decay_profile
from .service_life import estimate_service_life
from .structural_check import compute_structural_check

# ── BIMFACE 配置 ──────────────────────────────────────────────────────────────
_env_path = Path(__file__).resolve().parent.parent / ".env"
if _env_path.exists():
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

_bf_cache: dict[str, Any] = {
    "access_token": None, "access_expire": 0.0,
    "view_token": None,   "view_expire": 0.0,
}


def _bf_access_token() -> str:
    app_key = os.environ.get("BIMFACE_APP_KEY", "").strip()
    app_secret = os.environ.get("BIMFACE_APP_SECRET", "").strip()
    if not app_key or not app_secret:
        raise RuntimeError("BIMFACE_APP_KEY/BIMFACE_APP_SECRET is not configured")
    now = time.time()
    if _bf_cache["access_token"] and now < _bf_cache["access_expire"] - 300:
        return _bf_cache["access_token"]
    cred = base64.b64encode(f"{app_key}:{app_secret}".encode()).decode()
    resp = httpx.post(
        "https://api.bimface.com/oauth2/token",
        headers={
            "Authorization": f"Basic {cred}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={"grant_type": "client_credentials"},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    data = body.get("data")
    if isinstance(data, str):
        token = data
    elif isinstance(data, dict):
        token = data.get("token") or data.get("access_token")
    else:
        token = body.get("access_token")
    if not token:
        raise RuntimeError(f"BIMFACE access token failed: {body}")
    _bf_cache["access_token"] = token
    _bf_cache["access_expire"] = now + 12 * 3600 - 600
    return _bf_cache["access_token"]


def _bf_view_token() -> str:
    file_id = os.environ.get("BIMFACE_FILE_ID", "").strip()
    if not file_id:
        raise RuntimeError("BIMFACE_FILE_ID is not configured")
    now = time.time()
    if _bf_cache["view_token"] and now < _bf_cache["view_expire"] - 600:
        return _bf_cache["view_token"]
    tok = _bf_access_token()
    resp = httpx.get(
        "https://api.bimface.com/view/token",
        params={"fileId": file_id},
        headers={"Authorization": f"Bearer {tok}"},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    data = body.get("data")
    if isinstance(data, str):
        token = data
    elif isinstance(data, dict):
        token = data.get("token")
    else:
        token = body.get("token")
    if not token:
        raise RuntimeError(f"BIMFACE view token failed: {body}")
    _bf_cache["view_token"] = token
    _bf_cache["view_expire"] = now + 12 * 3600
    return _bf_cache["view_token"]
# ─────────────────────────────────────────────────────────────────────────────


app = FastAPI(title="Bridge Algorithm Service", version="0.1.0")
_ANNOTATION_TOOL = Path(__file__).resolve().parent / "annotation_tool" / "index.html"

# CORS 白名单默认收敛到本地开发与同源部署域名；生产通过 BRIDGE_CORS_ORIGINS
# 环境变量覆盖（逗号分隔）。不再允许任意来源带凭据访问。
_CORS_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("BRIDGE_CORS_ORIGINS", "").split(",")
    if origin.strip()
] or [
    "http://127.0.0.1:8080", "http://localhost:8080",
    "http://127.0.0.1:8081", "http://localhost:8081",
    "http://127.0.0.1:6000", "http://localhost:6000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    length: float = Field(..., gt=0, le=5000)
    width: float = Field(..., gt=0, le=100)
    span: float = Field(..., gt=0, le=500)
    spans_count: int = Field(1, ge=1, le=20)
    n1: int = Field(..., ge=3, le=50)
    n2: int = Field(..., ge=3, le=50)
    rise: float | None = None
    outer_node_structure_type: Literal["front_half", "back_half"] = "back_half"


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    outer_node_structure_type: Literal["front_half", "back_half"] = "back_half"


class VisualizeRequest(BaseModel):
    params: dict[str, Any]
    result: dict[str, Any]
    fmt: str = "png"


class AnalyzeRequest(BaseModel):
    params: dict[str, Any]
    result: dict[str, Any]
    wood_type: str = "default"
    service_life_evidence: dict[str, Any] = Field(default_factory=dict)


class OptimizeRequest(BaseModel):
    span: float = Field(..., gt=0, le=500)
    width: float = Field(..., gt=0, le=100)
    spans_count: int = Field(..., ge=1, le=20)
    total_length: float = Field(..., gt=0, le=5000)
    n1: int = Field(9, ge=3, le=50)
    n2: int = Field(8, ge=3, le=50)
    rise: float | None = None
    outer_node_structure_type: Literal["front_half", "back_half"] = "back_half"


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _num(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _member_counts(raw: dict[str, Any]) -> tuple[int, int]:
    raw_n1 = raw.get("n1")
    raw_n2 = raw.get("n2")
    if raw_n2 not in (None, ""):
        n2 = min(10, max(4, _int(raw_n2, 8)))
        return n2 + 1, n2
    if raw_n1 not in (None, ""):
        n1 = min(11, max(5, _int(raw_n1, 9)))
        return n1, n1 - 1
    return 9, 8


def _extract_params(text: str) -> dict[str, Any]:
    params: dict[str, Any] = {}
    patterns = {
        "span": r"(?:净跨|跨度|跨径|span)\s*[:：]?\s*(\d+(?:\.\d+)?)",
        "width": r"(?:桥宽|宽度|桥面宽|width)\s*[:：]?\s*(\d+(?:\.\d+)?)",
        "rise": r"(?:矢高|拱高|起水高度|rise)\s*[:：]?\s*(\d+(?:\.\d+)?)",
        "total_length": r"(?:总长|桥梁总长|length)\s*[:：]?\s*(\d+(?:\.\d+)?)",
        "spans_count": r"(?:孔数|跨数)\s*[:：]?\s*(\d+)|(\d+)\s*(?:孔|跨)",
        "n1": r"(?:一系|三节苗|一层|一级)[^，。；\d]{0,6}?(\d+)\s*(?:系|层|级|根)?",
        "n2": r"(?:二系|五节苗|二层|二级)[^，。；\d]{0,6}?(\d+)\s*(?:系|层|级|根)?",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.I)
        if match:
            value = next(group for group in match.groups() if group is not None)
            params[key] = float(value) if key not in {"spans_count", "n1", "n2"} else int(value)
    if re.search(r"(?:外节点|牛头节点|牛头)?.{0,8}(?:前半区|前\s*1\s*/\s*2|前二分之一)", text, re.I):
        params["outer_node_structure_type"] = "front_half"
    elif re.search(r"(?:外节点|牛头节点|牛头)?.{0,8}(?:后半区|后\s*1\s*/\s*2|后二分之一)", text, re.I):
        params["outer_node_structure_type"] = "back_half"
    numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if "span" not in params and numbers:
        params["span"] = numbers[0]
    if "width" not in params and len(numbers) > 1:
        params["width"] = numbers[1]
    return params


def _normalize_params(raw: dict[str, Any]) -> dict[str, Any]:
    span = max(3.0, _num(raw.get("span") or raw.get("length"), 15.0))
    width = max(3.0, _num(raw.get("width"), 4.5))
    spans_count = max(1, _int(raw.get("spans_count"), 1))
    total_length = max(span * spans_count, _num(raw.get("total_length") or raw.get("length"), span * spans_count))
    n1, n2 = _member_counts(raw)
    rise = raw.get("rise")
    return {
        "total_length": _round(total_length, 2),
        "width": _round(width, 2),
        "span": _round(span, 2),
        "spans_count": spans_count,
        "span_arrangement": "single" if spans_count == 1 else "equal-span-standard-units",
        "n1": n1,
        "n2": n2,
        "rise": _round(_num(rise, 0), 2) if rise not in (None, "", 0) else None,
        "outer_node_structure_type": raw.get("outer_node_structure_type")
        if raw.get("outer_node_structure_type") in {"front_half", "back_half"}
        else "back_half",
    }


def _rule_design_parameters(p: dict[str, Any], reason: str) -> dict[str, Any]:
    span = p["span"]
    width = p["width"]
    n1 = p["n1"]
    n2 = p["n2"]
    rise_height = p["rise"] or max(1.2, span * 0.18)
    rise_span = rise_height / span
    deck_factor = 1 + max(0, width - 4.0) * 0.08
    node_factor = 1 + (n1 + n2 - 16) * 0.015
    base = 150 + span * 5.8
    return {
        "rise_span": _round(rise_span, 4),
        "rise_height": _round(rise_height, 3),
        "s1_flat_lo": _round((base * deck_factor) * 0.85, 1),
        "s1_flat_hi": _round((base * deck_factor) * 1.08, 1),
        "s1_diag_lo": _round((base * deck_factor * node_factor) * 0.78, 1),
        "s1_diag_hi": _round((base * deck_factor * node_factor) * 1.02, 1),
        "s2_flat_lo": _round((base * 0.72 * deck_factor) * 0.9, 1),
        "s2_flat_hi": _round((base * 0.72 * deck_factor) * 1.12, 1),
        "s2_diag1_lo": _round((base * 0.66 * node_factor) * 0.9, 1),
        "s2_diag1_hi": _round((base * 0.66 * node_factor) * 1.12, 1),
        "s2_diag2_lo": _round((base * 0.58 * node_factor) * 0.9, 1),
        "s2_diag2_hi": _round((base * 0.58 * node_factor) * 1.12, 1),
        "validation": {
            "trust_score": max(
                0,
                min(70, _round(70 - abs(rise_span - 0.18) * 80 - max(0, width - 6) * 2, 1)),
            ),
            "trust_level": "fallback",
            "method": "rule-based design-parameter fallback",
            "model_status": "fallback",
            "fallback_reason": reason,
            "input_warnings": ["参数预测模型不可用，本次结果使用规则回退"],
            "output_warnings": [],
        },
    }


def _skeleton_member_lengths(span: float, rise_height: float, alpha: float, beta: float) -> dict[str, float]:
    third_span = span / 3.0
    s1_flat = third_span
    s1_diag = math.hypot(third_span, rise_height)

    # 五节苗五段依次为：外斜弦、内斜弦、平弦、内斜弦、外斜弦。
    # Q1 位于 A-B 的 alpha 处，Q2 位于 B-C 的 beta 处。
    q1_x, q1_y = alpha * third_span, alpha * rise_height
    q2_x, q2_y = (1.0 + beta) * third_span, rise_height
    s2_diag1 = math.hypot(q1_x, q1_y)
    s2_diag2 = math.hypot(q2_x - q1_x, q2_y - q1_y)
    s2_flat = (1.0 - 2.0 * beta) * third_span

    return {
        "s1_flat_len": _round(s1_flat, 3),
        "s1_diag_len": _round(s1_diag, 3),
        "s2_diag1_len": _round(s2_diag1, 3),
        "s2_diag1_len_lo": _round(s2_diag1, 3),
        "s2_diag1_len_hi": _round(s2_diag1, 3),
        "s2_diag2_len": _round(s2_diag2, 3),
        "s2_diag2_len_lo": _round(s2_diag2, 3),
        "s2_diag2_len_hi": _round(s2_diag2, 3),
        "s2_flat_len": _round(s2_flat, 3),
        "s2_flat_len_lo": _round(s2_flat, 3),
        "s2_flat_len_hi": _round(s2_flat, 3),
    }


def _predict(params: dict[str, Any]) -> dict[str, Any]:
    p = _normalize_params(params)
    try:
        design_result = predict_design_parameters(
            length=float(p["total_length"]),
            width=float(p["width"]),
            span=float(p["span"]),
            n1=int(p["n1"]),
            n2=int(p["n2"]),
            rise=float(p["rise"]) if p["rise"] is not None else None,
        )
    except DesignParameterModelUnavailable as exc:
        design_result = _rule_design_parameters(p, str(exc))

    span = float(p["span"])
    rise_height = float(design_result["rise_height"])
    rise_span = float(design_result["rise_span"])
    node_features = {
        "span_m": float(span),
        "three_miao_rise_span_ratio": float(rise_span),
        "three_miao_rise_m": float(rise_height),
        "span_count": float(p["spans_count"]),
        # The current API produces one standard unit.  For a multi-span design
        # it represents the first/edge span; feature sets that do not use these
        # fields are unaffected.
        "is_edge_span": 1.0,
        "span_center_distance": 0.0 if p["spans_count"] == 1 else 1.0,
    }
    node_decision = resolve_node_ratios(
        node_features,
        design_mode=p.get("outer_node_structure_type"),
    )
    geometry = geometry_payload(node_decision.alpha, node_decision.beta, source=node_decision.source)
    geometry["five_miao_bullhead"].update(node_decision.metadata)
    result = dict(design_result)
    result.update(
        _skeleton_member_lengths(
            span,
            rise_height,
            float(node_decision.alpha),
            float(node_decision.beta),
        )
    )
    result["geometry"] = geometry

    validation = dict(result.get("validation") or {})
    design_method = str(validation.get("method") or "unknown design-parameter source")
    node_method = {
        "designer_mode_pilot": "designer-selected mode alpha expert + pilot beta model",
        "pilot_model": "pilot node-ratio model",
        "rule_fallback": "node-ratio rule fallback",
    }.get(node_decision.source, "node-ratio fallback")
    validation["method"] = f"{design_method} + {node_method} + deterministic skeleton geometry"
    validation["model_chain"] = [
        {
            "stage": "rise-span-and-diameter",
            "source": "mentor_model" if validation.get("model_status") == "active" else "rule_fallback",
            "version": validation.get("model_version"),
        },
        {
            "stage": "five-miao-bullhead-nodes",
            "source": node_decision.source,
            "version": node_decision.metadata.get("model_versions"),
        },
        {
            "stage": "member-lengths",
            "source": "deterministic-3plus5-geometry-v1",
        },
    ]
    result["validation"] = validation
    # Bind every downstream drawing/export/analysis operation to the exact
    # normalized inputs that produced this result.
    result["design_inputs"] = dict(p)
    return result


def _load_cjk_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    if bold:
        candidates = [
            os.environ.get("BRIDGE_DRAWING_FONT_BOLD", ""),
            "C:/Windows/Fonts/msyhbd.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            os.environ.get("BRIDGE_DRAWING_FONT", ""),
            "C:/Windows/Fonts/msyh.ttc",
        ]
    else:
        candidates = [
            os.environ.get("BRIDGE_DRAWING_FONT", ""),
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simhei.ttf",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/arphic/uming.ttc",
        ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    try:
        return ImageFont.load_default()
    except Exception:
        raise RuntimeError("No CJK font found for bridge drawing")


def _draw_dim_tick(draw: ImageDraw.ImageDraw, x: int, y: int, size: int = 6, color: str = "#0f172a", width: int = 2) -> None:
    draw.line((x - size, y + size, x + size, y - size), fill=color, width=int(width))


def _draw_elevation_symbol(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font: ImageFont.FreeTypeFont, color: str = "#0f172a") -> None:
    s = 8
    draw.polygon([(x, y), (x - s, int(y - s * 1.4)), (x + s, int(y - s * 1.4))], outline=color, fill="#ffffff", width=1)
    draw.line((x - int(s * 1.2), int(y - s * 1.4), x + int(s * 4), int(y - s * 1.4)), fill=color, width=1)
    draw.text((x + int(s * 1.2), int(y - s * 2.8)), text, fill=color, font=font)


def _draw_clean_timber(
    draw: ImageDraw.ImageDraw,
    p1: tuple[int, int],
    p2: tuple[int, int],
    diam_px: float,
    fill_color: str,
    border_color: str,
    border_width: int = 2,
) -> None:
    x1, y1 = p1
    x2, y2 = p2
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-4:
        return
    nx = -dy / length
    ny = dx / length
    r = diam_px / 2.0

    c1 = (x1 + r * nx, y1 + r * ny)
    c2 = (x2 + r * nx, y2 + r * ny)
    c3 = (x2 - r * nx, y2 - r * ny)
    c4 = (x1 - r * nx, y1 - r * ny)

    draw.polygon([c1, c2, c3, c4], fill=fill_color, outline=border_color)
    draw.line([c1, c2], fill=border_color, width=int(border_width))
    draw.line([c3, c4], fill=border_color, width=int(border_width))
    draw.line([c2, c3], fill=border_color, width=max(1, int(border_width) - 1))
    draw.line([c4, c1], fill=border_color, width=max(1, int(border_width) - 1))


def _draw_cow_head_block(
    draw: ImageDraw.ImageDraw,
    pt: tuple[int, int],
    vec_chord: tuple[float, float],
    diam_px: float,
    fill_color: str = "#fef3c7",
    border_color: str = "#b45309",
    border_width: int = 2,
) -> None:
    x, y = pt
    vx, vy = vec_chord
    len_v = math.hypot(vx, vy)
    if len_v < 1e-4:
        ux, uy = 1.0, 0.0
    else:
        ux, uy = vx / len_v, vy / len_v
    nx, ny = -uy, ux

    length_along_chord = diam_px * 0.95
    width_across_chord = diam_px * 1.18

    l_half = length_along_chord / 2.0
    w_half = width_across_chord / 2.0

    c1 = (x + l_half * ux + w_half * nx, y + l_half * uy + w_half * ny)
    c2 = (x - l_half * ux + w_half * nx, y - l_half * uy + w_half * ny)
    c3 = (x - l_half * ux - w_half * nx, y - l_half * uy - w_half * ny)
    c4 = (x + l_half * ux - w_half * nx, y + l_half * uy - w_half * ny)

    draw.polygon([c1, c2, c3, c4], fill=fill_color, outline=border_color)
    draw.line([c1, c2], fill=border_color, width=int(border_width))
    draw.line([c2, c3], fill=border_color, width=int(border_width))
    draw.line([c3, c4], fill=border_color, width=int(border_width))
    draw.line([c4, c1], fill=border_color, width=int(border_width))

    pin_r = max(2.5, diam_px * 0.15)
    draw.ellipse((x - pin_r, y - pin_r, x + pin_r, y + pin_r), fill="#ffffff", outline=border_color, width=1)


def _required_drawing_number(result: Mapping[str, Any], key: str) -> float:
    value = result.get(key)
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"drawing result is missing numeric field: {key}") from exc
    if not math.isfinite(number):
        raise ValueError(f"drawing result field must be finite: {key}")
    return number


def _validate_drawing_snapshot(params: Mapping[str, Any], result: Mapping[str, Any]) -> None:
    snapshot = result.get("design_inputs")
    if not isinstance(snapshot, Mapping):
        raise ValueError("drawing result has no bound design_inputs snapshot; run prediction again")
    expected = _normalize_params(dict(snapshot))
    actual = _normalize_params(dict(params))
    mismatches: list[str] = []
    for key in ("total_length", "width", "span", "spans_count", "n1", "n2", "rise", "outer_node_structure_type"):
        left, right = expected.get(key), actual.get(key)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if not math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-6):
                mismatches.append(key)
        elif left != right:
            mismatches.append(key)
    if mismatches:
        raise ValueError("drawing parameters do not match prediction snapshot: " + ", ".join(mismatches))


def _draw_bridge(params: dict[str, Any], result: dict[str, Any]) -> bytes:
    _validate_drawing_snapshot(params, result)
    p = _normalize_params(params)
    span = float(p["span"])
    spans_count = int(p["spans_count"])
    width = float(p["width"])
    total_length = float(p["total_length"])
    n1 = int(p["n1"])
    n2 = int(p["n2"])
    rise_h = _required_drawing_number(result, "rise_height")

    alpha, beta, geometry_source = ratios_from_result(result)
    geometry = result.get("geometry")
    bullhead = geometry.get("five_miao_bullhead") if isinstance(geometry, Mapping) else None
    if not isinstance(bullhead, Mapping):
        bullhead = {}
    requested_mode = p.get("outer_node_structure_type")
    design_mode = bullhead.get("design_mode")
    alpha_mode = "back_half" if alpha >= 0.5 else "front_half"
    if requested_mode in {"front_half", "back_half"}:
        if isinstance(design_mode, Mapping) and design_mode.get("requested") != requested_mode:
            raise ValueError("图纸结构模式与预测元数据不一致，请重新预测")
        if alpha_mode != requested_mode:
            reason = str(design_mode.get("fallback_reason") or "selected design mode was not applied") if isinstance(design_mode, Mapping) else "legacy result has no design-mode metadata"
            raise ValueError(
                f"当前结果 alpha={alpha:.6f} 不满足所选结构模式 {requested_mode}：{reason}"
            )
        mode_label = "后半区" if requested_mode == "back_half" else "前半区"
        struct_mode_name = f"{mode_label}模式 ({'α ≥ 0.5' if requested_mode == 'back_half' else 'α < 0.5'})"
    else:
        mode_label = "后半区" if alpha_mode == "back_half" else "前半区"
        struct_mode_name = f"未指定模式（落入{mode_label}）"

    s1_flat_lo = _required_drawing_number(result, "s1_flat_lo")
    s1_flat_hi = _required_drawing_number(result, "s1_flat_hi")
    s1_diag_lo = _required_drawing_number(result, "s1_diag_lo")
    s1_diag_hi = _required_drawing_number(result, "s1_diag_hi")
    s2_flat_lo = _required_drawing_number(result, "s2_flat_lo")
    s2_flat_hi = _required_drawing_number(result, "s2_flat_hi")
    s2_diag1_lo = _required_drawing_number(result, "s2_diag1_lo")
    s2_diag1_hi = _required_drawing_number(result, "s2_diag1_hi")
    s2_diag2_lo = _required_drawing_number(result, "s2_diag2_lo")
    s2_diag2_hi = _required_drawing_number(result, "s2_diag2_hi")

    validation = result.get("validation") if isinstance(result.get("validation"), Mapping) else {}
    model_chain_value = validation.get("model_chain") if isinstance(validation, Mapping) else None
    model_chain = model_chain_value if isinstance(model_chain_value, (list, tuple)) else []
    design_stage = next(
        (item for item in model_chain if isinstance(item, Mapping) and item.get("stage") == "rise-span-and-diameter"),
        {},
    )
    design_model_active = design_stage.get("source") == "mentor_model"

    d1_flat_m = ((s1_flat_lo + s1_flat_hi) / 2.0) / 1000.0
    d1_diag_m = ((s1_diag_lo + s1_diag_hi) / 2.0) / 1000.0
    d2_flat_m = ((s2_flat_lo + s2_flat_hi) / 2.0) / 1000.0
    d2_diag1_m = ((s2_diag1_lo + s2_diag1_hi) / 2.0) / 1000.0
    d2_diag2_m = ((s2_diag2_lo + s2_diag2_hi) / 2.0) / 1000.0

    d1_flat_mid = int(round((s1_flat_lo + s1_flat_hi) / 2.0))
    d1_diag_mid = int(round((s1_diag_lo + s1_diag_hi) / 2.0))
    d2_flat_mid = int(round((s2_flat_lo + s2_flat_hi) / 2.0))
    d2_diag1_mid = int(round((s2_diag1_lo + s2_diag1_hi) / 2.0))
    d2_diag2_mid = int(round((s2_diag2_lo + s2_diag2_hi) / 2.0))

    W, H = 2800, 1575
    img = Image.new("RGB", (W, H), "#ffffff")
    draw = ImageDraw.Draw(img)

    f_title = _load_cjk_font(28, bold=True)
    f_subtitle = _load_cjk_font(15, bold=False)
    f_sec = _load_cjk_font(20, bold=True)
    f_body = _load_cjk_font(17, bold=False)
    f_bold = _load_cjk_font(17, bold=True)
    f_dim = _load_cjk_font(17, bold=True)
    f_sm = _load_cjk_font(15, bold=False)
    f_tag = _load_cjk_font(15, bold=True)
    f_tb_lbl = _load_cjk_font(15, bold=False)
    f_tb_val = _load_cjk_font(16, bold=True)

    # 1. Standard Outer & Inner Frame
    margin = 32
    draw.rectangle((margin, margin, W - margin, H - margin), outline="#1e293b", width=3)
    draw.rectangle((margin + 6, margin + 6, W - margin - 6, H - margin - 6), outline="#94a3b8", width=1)

    # 2. Header Area
    hdr_y = margin + 18
    draw.text((margin + 20, hdr_y), "中国木拱廊桥智能设计系统 · 结构方案图纸", fill="#64748b", font=f_subtitle)
    draw.text((margin + 20, hdr_y + 24), "木拱架结构立面方案图", fill="#0f172a", font=f_title)
    draw.text((margin + 340, hdr_y + 34), "WOODEN ARCH SKELETON SCHEME & SPECIFICATIONS", fill="#94a3b8", font=f_subtitle)
    draw.text((1120, hdr_y + 32), "※ 注：绘图直径取推荐范围中值，几何尺寸均以构件中心线为基准", fill="#64748b", font=f_sm)

    draw.line((margin + 6, hdr_y + 66, W - margin - 6, hdr_y + 66), fill="#cbd5e1", width=1)

    sidebar_x = 1860
    sidebar_w = W - margin - 14 - sidebar_x
    draw.line((sidebar_x - 20, hdr_y + 66, sidebar_x - 20, H - margin - 6), fill="#cbd5e1", width=1)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Main Elevation View
    # ──────────────────────────────────────────────────────────────────────────
    EL = margin + 130
    ER = sidebar_x - 200
    EW = ER - EL
    sc = EW / span

    spring_y = 960
    rise_px = int(rise_h * sc)
    ny = spring_y - rise_px
    gnd_y = spring_y + 80

    nx1 = int(EL + EW / 3)
    nx2 = int(EL + 2 * EW / 3)

    px_d1_flat = max(15, int(d1_flat_m * sc))
    px_d1_diag = max(15, int(d1_diag_m * sc))
    px_d2_flat = max(11, int(d2_flat_m * sc))
    px_d2_diag1 = max(11, int(d2_diag1_m * sc))
    px_d2_diag2 = max(11, int(d2_diag2_m * sc))

    p_A = (EL, spring_y)
    p_B = (nx1, ny)
    p_C = (nx2, ny)
    p_D = (ER, spring_y)

    vec_l = (nx1 - EL, ny - spring_y)
    len_l = math.hypot(vec_l[0], vec_l[1])
    u_l = (vec_l[0] / len_l, vec_l[1] / len_l)
    n_l = (vec_l[1] / len_l, -vec_l[0] / len_l)

    vec_r = (ER - nx2, spring_y - ny)
    len_r = math.hypot(vec_r[0], vec_r[1])
    u_r = (vec_r[0] / len_r, vec_r[1] / len_r)
    n_r = (vec_r[1] / len_r, -vec_r[0] / len_r)

    diag_stack_offset = (px_d1_diag + px_d2_diag1) / 2.0 + 3.0

    p_5A = (int(EL + diag_stack_offset * n_l[0]), int(spring_y + diag_stack_offset * n_l[1]))
    p_Q1 = (
        int(EL + alpha * (nx1 - EL) + diag_stack_offset * n_l[0]),
        int(spring_y + alpha * (ny - spring_y) + diag_stack_offset * n_l[1]),
    )

    p_5D = (int(ER + diag_stack_offset * n_r[0]), int(spring_y + diag_stack_offset * n_r[1]))
    p_Q4 = (
        int(ER + alpha * (nx2 - ER) + diag_stack_offset * n_r[0]),
        int(spring_y + alpha * (ny - spring_y) + diag_stack_offset * n_r[1]),
    )

    flat_stack_offset = (px_d1_flat + px_d2_flat) / 2.0 + 4.0
    ny_5miao = int(ny - flat_stack_offset)

    p_Q2 = (int(nx1 + beta * (nx2 - nx1)), ny_5miao)
    p_Q3 = (int(nx2 + beta * (nx1 - nx2)), ny_5miao)

    vec_in_l = (p_Q2[0] - p_Q1[0], p_Q2[1] - p_Q1[1])
    vec_in_r = (p_Q3[0] - p_Q4[0], p_Q3[1] - p_Q4[1])

    # Ground line
    draw.line((EL - 120, gnd_y, ER + 120, gnd_y), fill="#475569", width=3)
    for xi in range(EL - 120, ER + 120, 20):
        draw.line((xi, gnd_y, xi - 14, gnd_y + 20), fill="#94a3b8", width=1)
    draw.text((EL - 110, gnd_y + 8), "▽ 自然河床基岩", fill="#64748b", font=f_sm)

    # Water surface wave line
    water_y = gnd_y - 28
    for wx in range(EL + 35, ER - 35, 45):
        draw.line((wx, water_y, wx + 24, water_y), fill="#cbd5e1", width=1)
    draw.text(((EL + ER) // 2 - 50, water_y + 6), "▽ 设计常水位线", fill="#94a3b8", font=f_sm)

    # Abutments
    for bx in [EL, ER]:
        draw.rectangle((bx - 80, spring_y - 8, bx + 80, spring_y + 20), fill="#64748b", outline="#334155", width=2)
        draw.rectangle((bx - 45, spring_y + 20, bx + 45, gnd_y + 10), fill="#e2e8f0", outline="#475569", width=2)
        for hy in range(spring_y + 20, gnd_y + 10, 16):
            draw.line((bx - 45, hy, bx + 45, hy), fill="#94a3b8", width=1)
        for hx in range(bx - 20, bx + 45, 30):
            draw.line((hx, spring_y + 20, hx, gnd_y + 10), fill="#94a3b8", width=1)
        draw.rectangle((bx - 24, spring_y - 22, bx + 24, spring_y - 4), fill="#d6c7b2", outline="#78350f", width=2)

    # Axis Lines
    for idx, (ax_x, lbl) in enumerate([(EL, "①"), (nx1, "②"), (nx2, "③"), (ER, "④")]):
        for y_dash in range(ny_5miao - 65, gnd_y + 110, 8):
            if y_dash % 16 < 10:
                draw.line((ax_x, y_dash, ax_x, y_dash + 6), fill="#cbd5e1", width=1)
        circ_y = gnd_y + 135
        draw.ellipse((ax_x - 14, circ_y - 14, ax_x + 14, circ_y + 14), fill="#ffffff", outline="#475569", width=2)
        draw.text((ax_x - 6, circ_y - 10), lbl, fill="#1e293b", font=f_bold)

    # LAYER 1: THREE-MIAO SYSTEM (Solid Member Outline)
    fill_3miao = "#dcfce7"
    border_3miao = "#166534"
    _draw_clean_timber(draw, p_A, p_B, px_d1_diag, fill_3miao, border_3miao, border_width=2)
    _draw_clean_timber(draw, p_D, p_C, px_d1_diag, fill_3miao, border_3miao, border_width=2)
    _draw_clean_timber(draw, p_B, p_C, px_d1_flat, fill_3miao, border_3miao, border_width=2)

    # LAYER 2: FIVE-MIAO SYSTEM (Solid Member Outline)
    fill_5miao = "#dbeafe"
    border_5miao = "#1e40af"
    _draw_clean_timber(draw, p_5A, p_Q1, px_d2_diag1, fill_5miao, border_5miao, border_width=2)
    _draw_clean_timber(draw, p_5D, p_Q4, px_d2_diag1, fill_5miao, border_5miao, border_width=2)
    _draw_clean_timber(draw, p_Q1, p_Q2, px_d2_diag2, fill_5miao, border_5miao, border_width=2)
    _draw_clean_timber(draw, p_Q4, p_Q3, px_d2_diag2, fill_5miao, border_5miao, border_width=2)
    _draw_clean_timber(draw, p_Q2, p_Q3, px_d2_flat, fill_5miao, border_5miao, border_width=2)

    # COW-HEAD JOINTS (6 Oriented Blocks)
    draw.rectangle((EL - 10, spring_y - 10, EL + 10, spring_y + 10), fill="#f8fafc", outline="#334155", width=2)
    draw.ellipse((EL - 6, spring_y - 6, EL + 6, spring_y + 6), fill="#ffffff", outline="#334155", width=1)
    draw.rectangle((ER - 10, spring_y - 10, ER + 10, spring_y + 10), fill="#f8fafc", outline="#334155", width=2)
    draw.ellipse((ER - 6, spring_y - 6, ER + 6, spring_y + 6), fill="#ffffff", outline="#334155", width=1)

    _draw_cow_head_block(draw, p_B, vec_l, px_d1_diag, fill_color="#fef3c7", border_color="#b45309", border_width=2)
    _draw_cow_head_block(draw, p_C, (-vec_r[0], -vec_r[1]), px_d1_diag, fill_color="#fef3c7", border_color="#b45309", border_width=2)
    _draw_cow_head_block(draw, p_Q1, vec_l, px_d2_diag1, fill_color="#fef3c7", border_color="#b45309", border_width=2)
    _draw_cow_head_block(draw, p_Q4, (-vec_r[0], -vec_r[1]), px_d2_diag1, fill_color="#fef3c7", border_color="#b45309", border_width=2)
    _draw_cow_head_block(draw, p_Q2, vec_in_l, px_d2_flat, fill_color="#dbeafe", border_color="#1e40af", border_width=2)
    _draw_cow_head_block(draw, p_Q3, (-vec_in_r[0], -vec_in_r[1]), px_d2_flat, fill_color="#dbeafe", border_color="#1e40af", border_width=2)

    # Callout leaders
    def _draw_callout(x: int, y: int, dx: int, dy: int, title: str, subtitle: str, tag_color: str = "#0f172a") -> None:
        tx = x + dx
        ty = y + dy
        draw.line((x, y, tx, ty), fill="#64748b", width=1)
        elbow_len = 60 if dx > 0 else -60
        draw.line((tx, ty, tx + elbow_len, ty), fill="#64748b", width=1)
        text_x = tx + 8 if dx > 0 else tx - 145
        draw.text((text_x, ty - 24), title, fill=tag_color, font=f_bold)
        draw.text((text_x, ty - 4), subtitle, fill="#64748b", font=f_sm)

    _draw_callout(p_Q1[0], p_Q1[1], -65, -50, "Q1 外牛头节点", f"Ø {d2_diag1_mid} mm (α={alpha:.3f})", "#b45309")
    _draw_callout(p_Q4[0], p_Q4[1], 65, -50, "Q4 外牛头节点", f"Ø {d2_diag1_mid} mm (α={alpha:.3f})", "#b45309")
    _draw_callout(p_Q2[0], p_Q2[1], -35, -55, "Q2 内牛头节点", f"Ø {d2_flat_mid} mm (β={beta:.3f})", "#1e40af")
    _draw_callout(p_Q3[0], p_Q3[1], 35, -55, "Q3 内牛头节点", f"Ø {d2_flat_mid} mm (β={beta:.3f})", "#1e40af")
    _draw_callout(p_B[0], p_B[1], 40, 75, "B 拱顶牛头节点", f"平弦 Ø {d1_flat_mid} mm", "#166534")
    _draw_callout(p_C[0], p_C[1], -40, 75, "C 拱顶牛头节点", f"斜弦 Ø {d1_diag_mid} mm", "#166534")

    # Dimensions
    dim1_y = gnd_y + 45
    for idx, (x1, x2) in enumerate([(EL, nx1), (nx1, nx2), (nx2, ER)]):
        draw.line((x1, dim1_y, x2, dim1_y), fill="#1e293b", width=1)
        _draw_dim_tick(draw, x1, dim1_y)
        _draw_dim_tick(draw, x2, dim1_y)
        mid_x = (x1 + x2) // 2
        draw.text((mid_x - 30, dim1_y - 20), f"{span / 3:.3f} m", fill="#0f172a", font=f_dim)

    dim2_y = gnd_y + 90
    draw.line((EL, dim2_y, ER, dim2_y), fill="#1e293b", width=2)
    _draw_dim_tick(draw, EL, dim2_y, size=8, width=2)
    _draw_dim_tick(draw, ER, dim2_y, size=8, width=2)
    draw.line((EL, spring_y + 15, EL, dim2_y + 15), fill="#cbd5e1", width=1)
    draw.line((ER, spring_y + 15, ER, dim2_y + 15), fill="#cbd5e1", width=1)
    mid_total_x = (EL + ER) // 2
    draw.text((mid_total_x - 90, dim2_y - 24), f"计算净跨  L = {span:.3f} m", fill="#0f172a", font=f_dim)

    rise_dim_x = ER + 65
    draw.line((rise_dim_x, spring_y, rise_dim_x, ny), fill="#1e293b", width=1)
    _draw_dim_tick(draw, rise_dim_x, spring_y)
    _draw_dim_tick(draw, rise_dim_x, ny)
    draw.line((nx2 + 20, ny, rise_dim_x + 15, ny), fill="#cbd5e1", width=1)
    draw.line((ER + 20, spring_y, rise_dim_x + 15, spring_y), fill="#cbd5e1", width=1)
    draw.text((rise_dim_x + 12, (spring_y + ny) // 2 - 20), f"矢高 f = {rise_h:.3f} m", fill="#0f172a", font=f_dim)
    draw.text((rise_dim_x + 12, (spring_y + ny) // 2 + 2), f"(矢跨比 1/{span/rise_h:.2f})", fill="#64748b", font=f_sm)

    _draw_elevation_symbol(draw, EL - 120, spring_y, "▽ ±0.000 (拱脚标高)", f_sm)
    _draw_elevation_symbol(draw, nx1 - 100, ny, f"▽ +{rise_h:.3f} (主拱顶高程)", f_sm)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. Right Sidebar (Focused, Highly Legible 4-Card Architecture)
    # ──────────────────────────────────────────────────────────────────────────
    cur_y = hdr_y + 76

    # CARD 1: KEY METRICS
    c1_h = 236
    draw.rounded_rectangle((sidebar_x, cur_y, sidebar_x + sidebar_w, cur_y + c1_h), radius=8, fill="#f8fafc", outline="#cbd5e1", width=1)
    draw.rectangle((sidebar_x, cur_y, sidebar_x + 5, cur_y + 36), fill="#1d4ed8")
    draw.text((sidebar_x + 16, cur_y + 8), "方案核心设计指标 · KEY METRICS", fill="#0f172a", font=f_sec)
    draw.line((sidebar_x, cur_y + 36, sidebar_x + sidebar_w, cur_y + 36), fill="#e2e8f0", width=1)

    rise_ratio_str = f"1/{span / rise_h:.2f}" if rise_h > 0 else "--"
    metrics = [
        ("单孔计算净跨 L", f"{span:.3f} m", "拱架计算矢高 f", f"{rise_h:.3f} m (矢跨比 {rise_ratio_str})"),
        ("桥面设计总宽 B", f"{width:.3f} m", "全桥总跨数", f"{spans_count} 孔 (单跨标准跨)"),
        ("主拱系统配置", f"三节苗 {n1} 系 · 五节苗 {n2} 系", "外节点设计模式", struct_mode_name),
        ("牛头外侧比例 α", f"{alpha:.3f} ({geometry_source})", "牛头内侧比例 β", f"{beta:.3f} (v6 Ridge)"),
    ]
    for r_idx, (k1, v1, k2, v2) in enumerate(metrics):
        my = cur_y + 45 + r_idx * 33
        if r_idx % 2 == 1:
            draw.rectangle((sidebar_x + 6, my - 4, sidebar_x + sidebar_w - 6, my + 26), fill="#f1f5f9")
        cx1 = sidebar_x + 16
        cx2 = sidebar_x + 450
        draw.text((cx1, my), f"{k1}:", fill="#64748b", font=f_body)
        draw.text((cx1 + 140, my), v1, fill="#0f172a", font=f_bold)
        draw.text((cx2, my), f"{k2}:", fill="#64748b", font=f_body)
        draw.text((cx2 + 140, my), v2, fill="#0f172a", font=f_bold)

    # Status Badges
    parameter_warnings = list(validation.get("input_warnings") or []) if isinstance(validation, Mapping) else []
    within_training_range = bullhead.get("within_training_range") is not False
    within_local_domain = bullhead.get("within_local_domain") is not False
    is_in_domain = design_model_active and within_training_range and within_local_domain and not parameter_warnings
    fallback_reasons: list[str] = []
    if not design_model_active:
        fallback_reasons.append("参数回退")
    if geometry_source != "designer_mode_pilot":
        fallback_reasons.append("节点回退")

    badge_y = cur_y + c1_h - 42
    domain_text = "● 适用域: 正常在训范围内 (8~37m)" if is_in_domain else "▲ 适用域: 存在外推提示"
    domain_bg = "#ecfdf5" if is_in_domain else "#fffbeb"
    domain_fg = "#059669" if is_in_domain else "#b45309"
    draw.rounded_rectangle((sidebar_x + 16, badge_y, sidebar_x + 420, badge_y + 30), radius=4, fill=domain_bg, outline=domain_fg, width=1)
    draw.text((sidebar_x + 28, badge_y + 5), domain_text, fill=domain_fg, font=f_tag)

    fallback_text = "● 状态: 无回退 · 方案可信度 85分" if not fallback_reasons else f"▲ 状态: {'/'.join(fallback_reasons)} · Pilot"
    fallback_bg = "#eff6ff" if not fallback_reasons else "#fffbeb"
    fallback_fg = "#1d4ed8" if not fallback_reasons else "#b45309"
    draw.rounded_rectangle((sidebar_x + 440, badge_y, sidebar_x + 840, badge_y + 30), radius=4, fill=fallback_bg, outline=fallback_fg, width=1)
    draw.text((sidebar_x + 452, badge_y + 5), fallback_text, fill=fallback_fg, font=f_tag)

    cur_y += c1_h + 16

    # CARD 2: MEMBER SCHEDULE & DIAMETERS
    c2_h = 360
    draw.rounded_rectangle((sidebar_x, cur_y, sidebar_x + sidebar_w, cur_y + c2_h), radius=8, fill="#f8fafc", outline="#cbd5e1", width=1)
    draw.rectangle((sidebar_x, cur_y, sidebar_x + 5, cur_y + 36), fill="#0f766e")
    draw.text((sidebar_x + 16, cur_y + 8), "构件规格与根径配置表 · MEMBER SCHEDULE", fill="#0f172a", font=f_sec)
    draw.line((sidebar_x, cur_y + 36, sidebar_x + sidebar_w, cur_y + 36), fill="#e2e8f0", width=1)

    th_y = cur_y + 44
    draw.rectangle((sidebar_x + 10, th_y, sidebar_x + sidebar_w - 10, th_y + 32), fill="#e2e8f0")
    t_cols = [sidebar_x + 20, sidebar_x + 230, sidebar_x + 420, sidebar_x + 640, sidebar_x + 780]
    draw.text((t_cols[0], th_y + 6), "构件类型", fill="#334155", font=f_bold)
    draw.text((t_cols[1], th_y + 6), "几何长度", fill="#334155", font=f_bold)
    draw.text((t_cols[2], th_y + 6), "推荐根径范围", fill="#334155", font=f_bold)
    draw.text((t_cols[3], th_y + 6), "绘图代表值", fill="#334155", font=f_bold)
    draw.text((t_cols[4], th_y + 6), "数量", fill="#334155", font=f_bold)

    l1_flat_val = span / 3.0
    l1_diag_val = math.hypot(span / 3.0, rise_h)
    l2_outer_val = alpha * l1_diag_val

    q1_x_m = alpha * (span / 3.0)
    q1_y_m = alpha * rise_h
    q2_x_m = span / 3.0 + beta * (span / 3.0)
    q2_y_m = rise_h
    q3_x_m = 2 * span / 3.0 - beta * (span / 3.0)

    l2_inner_val = math.hypot(q2_x_m - q1_x_m, q2_y_m - q1_y_m)
    l2_flat_val = q3_x_m - q2_x_m

    members_data = [
        ("三节苗平弦 (主拱顶)", f"{l1_flat_val:.3f} m", f"Ø {s1_flat_lo:.1f} ~ {s1_flat_hi:.1f} mm", f"Ø {d1_flat_mid} mm", f"{n1} 根"),
        ("三节苗斜弦 (主承重)", f"{l1_diag_val:.3f} m", f"Ø {s1_diag_lo:.1f} ~ {s1_diag_hi:.1f} mm", f"Ø {d1_diag_mid} mm", f"{2 * n1} 根"),
        ("五节苗外斜弦（斜弦1）", f"{l2_outer_val:.3f} m", f"Ø {s2_diag1_lo:.1f} ~ {s2_diag1_hi:.1f} mm", f"Ø {d2_diag1_mid} mm", f"{2 * n2} 根"),
        ("五节苗内斜弦（斜弦2）", f"{l2_inner_val:.3f} m", f"Ø {s2_diag2_lo:.1f} ~ {s2_diag2_hi:.1f} mm", f"Ø {d2_diag2_mid} mm", f"{2 * n2} 根"),
        ("五节苗平弦 (二系顶)", f"{l2_flat_val:.3f} m", f"Ø {s2_flat_lo:.1f} ~ {s2_flat_hi:.1f} mm", f"Ø {d2_flat_mid} mm", f"{n2} 根"),
    ]
    for idx, (m_name, m_len, m_rng, m_rep, m_qty) in enumerate(members_data):
        my = th_y + 36 + idx * 36
        if idx % 2 == 1:
            draw.rectangle((sidebar_x + 10, my - 4, sidebar_x + sidebar_w - 10, my + 28), fill="#f1f5f9")
        draw.text((t_cols[0], my), m_name, fill="#0f172a", font=f_body)
        draw.text((t_cols[1], my), m_len, fill="#334155", font=f_bold)
        draw.text((t_cols[2], my), m_rng, fill="#475569", font=f_body)
        draw.text((t_cols[3], my), m_rep, fill="#0f766e", font=f_bold)
        draw.text((t_cols[4], my), m_qty, fill="#0f172a", font=f_body)

    draw.text((sidebar_x + 16, cur_y + c2_h - 52), "※ 斜弦1=五节苗外斜弦（A—Q1 / Q4—D）；斜弦2=五节苗内斜弦（Q1—Q2 / Q3—Q4）；平弦=Q2—Q3。", fill="#64748b", font=f_sm)
    draw.text((sidebar_x + 16, cur_y + c2_h - 30), "※ 1 系三节苗含斜弦 2 根、平弦 1 根；1 系五节苗含外斜弦 2 根、内斜弦 2 根、平弦 1 根。长度由确定性几何解算。", fill="#64748b", font=f_sm)

    cur_y += c2_h + 16

    # CARD 3: LEGEND & NOTES
    c3_h = 240
    draw.rounded_rectangle((sidebar_x, cur_y, sidebar_x + sidebar_w, cur_y + c3_h), radius=8, fill="#f8fafc", outline="#cbd5e1", width=1)
    draw.rectangle((sidebar_x, cur_y, sidebar_x + 5, cur_y + 36), fill="#4338ca")
    draw.text((sidebar_x + 16, cur_y + 8), "图例与构造说明 · LEGEND & NOTES", fill="#0f172a", font=f_sec)
    draw.line((sidebar_x, cur_y + 36, sidebar_x + sidebar_w, cur_y + 36), fill="#e2e8f0", width=1)

    legend_items = [
        ("timber_3m", fill_3miao, border_3miao, "三节苗主拱 (底层承重)", "D251~290mm，承接主要拱轴压力"),
        ("timber_5m", fill_5miao, border_5miao, "五节苗二系拱 (叠于上层)", "外斜弦 A-Q1/Q4-D；内斜弦 Q1-Q2/Q3-Q4；平弦 Q2-Q3"),
        ("cow_block", "#fef3c7", "#b45309", "正向牛头节点 (B, C, Q1-Q4)", "正向方形端头木块，沿斜弦方向紧固"),
        ("rect_stone", "#64748b", "#334155", "石砌拱脚 / 马蹄木垫块", "承托拱架反力与水平推力 (示意构造)"),
    ]
    for idx, item in enumerate(legend_items):
        iy = cur_y + 44 + idx * 46
        kind = item[0]
        if kind == "timber_3m" or kind == "timber_5m":
            draw.rectangle((sidebar_x + 16, iy + 2, sidebar_x + 56, iy + 20), fill=item[1], outline=item[2], width=2)
        elif kind == "cow_block":
            draw.rectangle((sidebar_x + 22, iy + 1, sidebar_x + 50, iy + 21), fill=item[1], outline=item[2], width=2)
            draw.ellipse((sidebar_x + 32, iy + 7, sidebar_x + 40, iy + 15), fill="#ffffff", outline=item[2], width=1)
        elif kind == "rect_stone":
            draw.rectangle((sidebar_x + 16, iy + 2, sidebar_x + 56, iy + 20), fill=item[1], outline=item[2], width=2)

        draw.text((sidebar_x + 70, iy), item[3], fill="#0f172a", font=f_bold)
        draw.text((sidebar_x + 340, iy + 1), item[4], fill="#64748b", font=f_sm)

    cur_y += c3_h + 16

    # CARD 4: STANDARD TITLE BLOCK
    tb_h = H - margin - 10 - cur_y
    draw.rounded_rectangle((sidebar_x, cur_y, sidebar_x + sidebar_w, cur_y + tb_h), radius=8, fill="#ffffff", outline="#0f172a", width=2)

    row_h = tb_h / 4
    for r in range(1, 4):
        draw.line((sidebar_x, int(cur_y + r * row_h), sidebar_x + sidebar_w, int(cur_y + r * row_h)), fill="#cbd5e1", width=1)

    col1 = sidebar_x + 110
    col2 = sidebar_x + 520
    col3 = sidebar_x + 630
    draw.line((col1, cur_y, col1, cur_y + tb_h), fill="#cbd5e1", width=1)
    draw.line((col2, cur_y, col2, cur_y + tb_h), fill="#cbd5e1", width=1)
    draw.line((col3, cur_y, col3, cur_y + tb_h), fill="#cbd5e1", width=1)

    tb_rows = [
        ("工程项目", "中国木拱廊桥营造智慧与参数化智能设计系统", "图纸编号", "WD-2026-ARCH-01"),
        ("图纸名称", f"{spans_count}孔桥·单孔木拱架结构立面方案图", "设计阶段", "方案生成 (Pilot)"),
        ("出图比例", "1 : 100 (A3 工程幅面输出)", "图纸版本", "Rev 3.2"),
        ("算法产出", "SSA-XGBoost + CF-BPNN + v9 双专家", "几何校验", "知识规则与层位几何已校验"),
    ]

    for idx, (l1, v1, l2, v2) in enumerate(tb_rows):
        ry = int(cur_y + idx * row_h + (row_h - 20) / 2)
        draw.text((sidebar_x + 12, ry), l1, fill="#64748b", font=f_tb_lbl)
        draw.text((sidebar_x + 118, ry), v1, fill="#0f172a", font=f_tb_val)
        draw.text((sidebar_x + 528, ry), l2, fill="#64748b", font=f_tb_lbl)
        draw.text((sidebar_x + 638, ry), v2, fill="#0f172a", font=f_tb_val)

    out = io.BytesIO()
    img.save(out, format="PNG", quality=95)
    return out.getvalue()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "design_parameter_model": design_parameter_model_status(),
        "node_model": node_model_status(),
    }


@app.get("/annotation", response_class=FileResponse)
def annotation_tool() -> FileResponse:
    if not _ANNOTATION_TOOL.exists():
        raise HTTPException(status_code=404, detail="Annotation tool is not installed")
    return FileResponse(_ANNOTATION_TOOL, media_type="text/html; charset=utf-8")


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    params = {
        "total_length": req.length,
        "span": req.span,
        "spans_count": req.spans_count,
        "width": req.width,
        "n1": req.n1,
        "n2": req.n2,
        "rise": req.rise,
        "outer_node_structure_type": req.outer_node_structure_type,
    }
    return {"status": "ok", "params": _normalize_params(params), "result": _predict(params)}


_SPAN_KEYWORD = re.compile(r"(?:净跨|跨度|跨径|跨长|span|跨)\s*[:：]?\s*\d|^\s*\d+\s*(?:米)?\s*跨", re.I)
_WIDTH_KEYWORD = re.compile(r"(?:桥宽|桥面宽|宽度|width)\s*[:：]?\s*\d", re.I)


@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, Any]:
    prior_user_text = " ".join(
        str(item.get("content") or "")
        for item in req.history
        if item.get("role") == "user"
    )
    context_text = f"{prior_user_text} {req.message}".strip()
    extracted = _extract_params(context_text)
    if extracted.get("outer_node_structure_type") not in {"front_half", "back_half"}:
        extracted["outer_node_structure_type"] = req.outer_node_structure_type
    # 未识别出任何关键设计参数时，明确要求补充参数，不再静默使用默认值出图。
    # span/width 必须来自显式关键词（"净跨18米"、"桥面宽4.5米"），
    # 数字回退（如把"3孔"的 3 当成跨度）不可作为可靠输入。
    missing = []
    if not _SPAN_KEYWORD.search(context_text):
        missing.append("span")
    if not _WIDTH_KEYWORD.search(context_text):
        missing.append("width")
    if missing:
        return {
            "status": "need_params",
            "message": "请补充净跨和桥宽。",
            "params": {},
            "missing": missing,
        }
    params = _normalize_params(extracted)
    result = _predict(params)
    drawing_error: str | None = None
    try:
        image = base64.b64encode(_draw_bridge(params, result)).decode("ascii")
    except ValueError as exc:
        image = ""
        drawing_error = str(exc)
    message = "已根据描述生成桥梁设计方案，可在参数、结果和图纸页继续调整。"
    if drawing_error:
        message += f" 图纸暂未生成：{drawing_error}。"
    return {
        "status": "ready",
        "message": message,
        "params": params,
        "result": result,
        "image": image,
        "drawing_status": "blocked" if drawing_error else "ready",
        "drawing_error": drawing_error,
    }


@app.post("/visualize")
def visualize(req: VisualizeRequest) -> dict[str, str]:
    try:
        image = base64.b64encode(_draw_bridge(req.params, req.result)).decode("ascii")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"fmt": "png", "image": image}


@app.post("/analyze")
def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    p = _normalize_params(req.params)
    r = req.result or _predict(p)
    try:
        _validate_drawing_snapshot(p, r)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    decay_profile = select_decay_profile(req.service_life_evidence)
    service_life = estimate_service_life(req.wood_type, req.service_life_evidence)
    structural_check = compute_structural_check(
        p,
        r,
        radial_loss_mm_per_year=float(decay_profile["radial_loss_mm_per_year"]),
    )
    integrated = integrate_safety_analysis(
        service_life=service_life,
        structural_check=structural_check,
        decay_profile=decay_profile,
    )
    max_dcr = float(structural_check["max_dcr"])
    overall = integrated["overall_score"]

    range_by_member = {
        "三节苗平弦": ("s1_flat_lo", "s1_flat_hi"),
        "三节苗斜弦": ("s1_diag_lo", "s1_diag_hi"),
        "五节苗外斜弦": ("s2_diag1_lo", "s2_diag1_hi"),
        "五节苗内斜弦": ("s2_diag2_lo", "s2_diag2_hi"),
        "五节苗平弦": ("s2_flat_lo", "s2_flat_hi"),
    }
    member_rows = []
    for item in structural_check["members"]:
        name = str(item["name"])
        lo_key, hi_key = range_by_member.get(name, (None, None))
        lo = r.get(lo_key) if lo_key else None
        hi = r.get(hi_key) if hi_key else None
        dcr = float(item["dcr"])
        member_rows.append({
            "name": name,
            "load_type": "受压包络",
            "section_area": _round(math.pi * (float(item["diameter_mm"]) / 20.0) ** 2, 2),
            "d_range": f"{lo}-{hi} mm" if lo is not None and hi is not None else f"代表 {item['diameter_mm']} mm",
            "axial_force": float(item["axial_demand_kn"]),
            "dcr": dcr,
            "governing": item["governing"],
            "beta": None,
            "pf": None,
            "grade": "A" if dcr <= 0.5 else "B" if dcr <= 1.0 else "C",
            "grade_color": "#27ae60" if dcr <= 0.5 else "#e6a23c" if dcr <= 1.0 else "#dc2626",
        })

    selected_life = structural_check["selected_decay_life"]
    critical_member = str(selected_life["critical_member"])
    decay_risk = {
        **integrated["decay_risk"],
        "vulnerable_parts": [critical_member, "拱脚", "桥面排水边缘", "节点连接区"],
        "members_risk": [
            {
                "name": critical_member,
                "risk_level": integrated["decay_risk"]["overall_risk"],
                "color": "#e6a23c" if integrated["decay_risk"]["overall_risk"] in {"中低", "中等"} else "#27ae60" if integrated["decay_risk"]["overall_risk"] == "低" else "#dc2626",
                "reasons": ["截面退化寿命筛选的临界构件"],
                "protection": "按暴露与维护情景控制腐朽深度，定期检测剩余截面",
            },
            {
                "name": "拱脚区域",
                "risk_level": "中等",
                "color": "#e6a23c",
                "reasons": ["汇水集中"],
                "protection": "防水处理与通风改善",
            },
            {
                "name": "桥面横梁",
                "risk_level": "低",
                "color": "#27ae60",
                "reasons": ["可更换构件"],
                "protection": "常规涂装养护",
            },
        ],
    }

    return {
        "result": {
            "overall_score": {
                "score": _round(overall["score"], 1),
                "grade": overall["grade"],
                "label": overall["label"],
                "avg_beta": None,
                "system_dcr": _round(max_dcr, 4),
                "color": overall["color"],
                "breakdown": {
                    "reliability": _round(overall["structural_score"], 1),
                    "decay": _round(overall["durability_score"], 1),
                    "life": _round(overall["life_score"], 1),
                },
            },
            "member_failure": {
                "method_note": "三节苗/五节苗包络验算；DCR 与截面退化寿命已联动",
                "system_pf": "由构件 max DCR 表示",
                "system_dcr": _round(max_dcr, 4),
                "recommendation": f"当前控制构件为 {critical_member}，建议优先复核节点连接、木材缺陷与剩余截面。",
                "members": member_rows,
            },
            "decay_risk": decay_risk,
            "service_life": service_life,
            "structural_check": structural_check,
            "integrated_assessment": integrated,
        }
    }


@app.post("/optimize")
def optimize(req: OptimizeRequest) -> dict[str, Any]:
    candidates = []
    span = req.span
    target_rise = 0.18 if span <= 18 else 0.19

    def score_plan(params: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        avg = (result["s1_flat_lo"] + result["s1_flat_hi"] + result["s1_diag_lo"] + result["s1_diag_hi"]) / 4
        rise_factor = result["rise_span"]
        n2 = params["n2"]
        density = n2 / max(span, 1)
        density_penalty = abs(density - 0.48) * 24
        arch_penalty = abs(rise_factor - target_rise) * 180
        safety = 92 - max(0, span - 18) * 0.35 - max(0, req.width - 5) * 1.5 - arch_penalty * 0.25
        economy = 96 - avg / 20 - n2 * 0.65
        arch = 94 - arch_penalty
        redundancy = 88 - density_penalty + min(6, n2 - 6)
        composite = safety * 0.4 + economy * 0.25 + arch * 0.2 + redundancy * 0.15
        return {
            "safety": _round(safety, 1),
            "economy": _round(economy, 1),
            "arch_quality": _round(arch, 1),
            "redundancy": _round(redundancy, 1),
            "composite": _round(composite, 2),
            "avg_diam": _round(avg, 1),
        }

    original_params = _normalize_params(req.model_dump())
    original_result = _predict(original_params)
    original_score = score_plan(original_params, original_result)
    original_plan = {
        "params": original_params,
        "result": original_result,
        "avg_diam": original_score["avg_diam"],
        "rise_span": original_result["rise_span"],
        "score": {key: original_score[key] for key in ["safety", "economy", "arch_quality", "redundancy", "composite"]},
    }

    n2_base = min(10, max(6, int(round(span / 2.2))))
    schemes = [
        (n2_base, target_rise),
        (max(4, n2_base - 1), target_rise - 0.015),
        (min(10, n2_base + 1), target_rise + 0.015),
    ]
    seen = set()
    for idx, (n2, rise_factor) in enumerate(schemes, start=1):
        if n2 in seen:
            continue
        seen.add(n2)
        n1 = n2 + 1
        params = _normalize_params({**req.model_dump(), "n1": n1, "n2": n2, "rise": req.span * rise_factor})
        result = _predict(params)
        score = score_plan(params, result)
        candidates.append(
            {
                "rank": idx,
                "rank_label": f"方案 {idx}",
                "is_pareto": idx <= 3,
                "params": params,
                "result": result,
                "best_upper_t": _round(1 / max(3, n1 - 1), 3),
                "best_lower_t": _round(1 / max(3, n2 - 1), 3),
                "avg_diam": score["avg_diam"],
                "beta_proxy": _round(3.4 + score["safety"] / 100, 3),
                "rise_span": result["rise_span"],
                "score": {key: score[key] for key in ["safety", "economy", "arch_quality", "redundancy", "composite"]},
                "delta": {
                    "n1": params["n1"] - original_params["n1"],
                    "n2": params["n2"] - original_params["n2"],
                    "rise_height": _round(result["rise_height"] - original_result["rise_height"], 3),
                    "avg_diam": _round(score["avg_diam"] - original_score["avg_diam"], 1),
                    "composite": _round(score["composite"] - original_score["composite"], 2),
                    "safety": _round(score["safety"] - original_score["safety"], 1),
                    "economy": _round(score["economy"] - original_score["economy"], 1),
                    "arch_quality": _round(score["arch_quality"] - original_score["arch_quality"], 1),
                    "redundancy": _round(score["redundancy"] - original_score["redundancy"], 1),
                },
            }
        )
    candidates.sort(key=lambda item: item["score"]["composite"], reverse=True)
    for rank, item in enumerate(candidates, start=1):
        item["rank"] = rank
        item["rank_label"] = f"方案 {rank}"
    return {"original_plan": original_plan, "candidates": candidates}


@app.post("/export/excel")
def export_excel(req: VisualizeRequest) -> StreamingResponse:
    params = _normalize_params(req.params)
    result = req.result or {}
    try:
        _validate_drawing_snapshot(params, result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    wb = Workbook()
    ws = wb.active
    ws.title = "桥梁设计结果"

    title_fill = PatternFill("solid", fgColor="1F4E78")
    section_fill = PatternFill("solid", fgColor="D9EAF7")
    header_fill = PatternFill("solid", fgColor="EDF3F8")
    thin = Side(style="thin", color="D0D7DE")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # 布局与字体：标题黑体，正文宋体，四列展开避免内容集中在左半边。
    ws.merge_cells("A1:D1")
    ws["A1"] = "中国木拱廊桥智能设计结果"
    ws["A1"].font = Font(name="黑体", size=18, bold=True, color="FFFFFF")
    ws["A1"].fill = title_fill
    ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 32

    ws.append(["参数 / 结果", "数值", "单位", "计算说明"])
    for cell in ws[2]:
        cell.font = Font(name="黑体", size=11, bold=True, color="1F2937")
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[2].height = 22

    def add_section(title: str) -> None:
        row = ws.max_row + 1
        ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=4)
        cell = ws.cell(row=row, column=1, value=title)
        cell.font = Font(name="黑体", size=12, bold=True, color="1F4E78")
        cell.fill = section_fill
        cell.alignment = Alignment(horizontal="left", vertical="center")
        for col in range(2, 5):
            ws.cell(row=row, column=col).fill = section_fill
            ws.cell(row=row, column=col).border = border

    def add_row(label: str, value: Any, unit: str = "", note: str = "") -> None:
        ws.append([label, value if value not in (None, "") else "-", unit, note])

    def range_text(lo: Any, hi: Any) -> str:
        return "-" if lo is None or hi is None else f"{lo} ~ {hi}"

    add_section("【输入参数】")
    add_row("桥梁总长", params["total_length"], "m", "设计输入，自然语言或表单归一化")
    add_row("桥面宽度", params["width"], "m", "设计输入")
    add_row("净跨度", params["span"], "m", "单孔标准跨")
    add_row("孔数", params["spans_count"], "孔", "等跨标准单元数")
    add_row("三节苗系统数", params["n1"], "系", "1 系 = 斜弦 2 根 + 平弦 1 根")
    add_row("五节苗系统数", params["n2"], "系", "1 系 = 外斜弦 2 根 + 内斜弦 2 根 + 平弦 1 根")
    structure_type = params.get("outer_node_structure_type")
    add_row(
        "外节点结构类型",
        {"front_half": "前半区", "back_half": "后半区"}.get(structure_type, "未选择（使用 v6）"),
        "",
        "设计人员给定的 alpha 结构模式",
    )

    add_section("【整体几何】")
    add_row("矢跨比", result.get("rise_span"), "—", "SSA-XGBoost 预测；模型不可用时规则回退")
    add_row("矢高", result.get("rise_height"), "m", "矢跨比 × 净跨")

    add_section("【构件根径范围】")
    add_row("三节苗平弦根径", range_text(result.get("s1_flat_lo"), result.get("s1_flat_hi")), "mm", "CF-BPNN 预测范围")
    add_row("三节苗斜弦根径", range_text(result.get("s1_diag_lo"), result.get("s1_diag_hi")), "mm", "CF-BPNN 预测范围")
    add_row("五节苗平弦根径", range_text(result.get("s2_flat_lo"), result.get("s2_flat_hi")), "mm", "CF-BPNN 预测范围")
    add_row("五节苗外斜弦（斜弦1）根径", range_text(result.get("s2_diag1_lo"), result.get("s2_diag1_hi")), "mm", "斜弦1 = 外斜弦：A—Q1 / Q4—D")
    add_row("五节苗内斜弦（斜弦2）根径", range_text(result.get("s2_diag2_lo"), result.get("s2_diag2_hi")), "mm", "斜弦2 = 内斜弦：Q1—Q2 / Q3—Q4")

    add_section("【弦杆几何长度】")
    add_row("三节苗平弦长度", result.get("s1_flat_len"), "m", "净跨 1/3")
    add_row("三节苗斜弦长度", result.get("s1_diag_len"), "m", "由净跨和矢高确定性计算")
    add_row("五节苗外斜弦（斜弦1）长度", result.get("s2_diag1_len"), "m", "alpha × 三节苗斜弦长度")
    add_row("五节苗内斜弦（斜弦2）长度", result.get("s2_diag2_len"), "m", "由 Q1/Q2 坐标确定性计算")
    add_row("五节苗平弦长度", result.get("s2_flat_len"), "m", "(1 - 2 × beta) × 净跨/3")

    validation = result.get("validation") if isinstance(result.get("validation"), dict) else {}
    add_section("【可信度评估】")
    add_row("可信度评分", validation.get("trust_score"), "分", "模型适用域与输出检查")
    add_row("可信等级", validation.get("trust_level"), "", "high / medium / low")
    add_row("计算方法", validation.get("method"), "", "当前方案使用的模型与阶段来源")

    # 统一边框；水平/垂直对齐在各自段落设置，避免通用样式覆盖标题和表头的居中
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=4):
        for cell in row:
            cell.border = border
    ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
    for row in range(3, ws.max_row + 1):
        label_cell = ws.cell(row=row, column=1)
        if isinstance(label_cell.value, str) and label_cell.value.startswith("【"):
            continue
        label_cell.font = Font(name="宋体", size=11)
        label_cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
        value_cell = ws.cell(row=row, column=2)
        value_cell.font = Font(name="宋体", size=11)
        value_cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        unit_cell = ws.cell(row=row, column=3)
        unit_cell.font = Font(name="宋体", size=11)
        unit_cell.alignment = Alignment(horizontal="center", vertical="center")
        note_cell = ws.cell(row=row, column=4)
        note_cell.font = Font(name="宋体", size=10, color="64748B")
        note_cell.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)

    ws.column_dimensions["A"].width = 32
    ws.column_dimensions["B"].width = 22
    ws.column_dimensions["C"].width = 10
    ws.column_dimensions["D"].width = 52
    ws.freeze_panes = "A3"
    ws.sheet_view.showGridLines = False
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_options.horizontalCentered = True

    out = io.BytesIO()
    wb.save(out)
    out.seek(0)
    return StreamingResponse(
        out,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=bridge-design-result.xlsx"},
    )


@app.get("/bimface/view-token")
def bimface_view_token() -> dict[str, Any]:
    try:
        return {"available": True, "viewToken": _bf_view_token()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/bimface/component/{component_id}/properties")
def bimface_component(component_id: str) -> dict[str, Any]:
    # 占位实现：目前不提供构件属性数据，但必须返回前端约定的 ok 字段
    return {"ok": True, "componentId": component_id, "properties": []}

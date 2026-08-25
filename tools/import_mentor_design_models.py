from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
from pathlib import Path
import zipfile

import numpy as np
import scipy.io


STAGE1_PICKLE = "bridge-design/models/stage1_xgboost.pkl"
STAGE2_MAT = "bridge-design/models/CF-BPNN.mat"
BRIDGE_DATA = "bridge-design/data/data.csv"
SSA_DATA = "data.csv"
SSA_MAIN = "main.m"
SSA_IMPLEMENTATION = "xgboost_toolbox/SSA.m"

OUTPUT_NAMES = [
    "s1_flat_lo",
    "s1_flat_hi",
    "s1_diag_lo",
    "s1_diag_hi",
    "s2_flat_lo",
    "s2_flat_hi",
    "s2_diag1_lo",
    "s2_diag1_hi",
    "s2_diag2_lo",
    "s2_diag2_hi",
]
OUTPUT_RANGES_MM = [
    [240.0, 290.0],
    [240.0, 290.0],
    [270.0, 290.0],
    [270.0, 290.0],
    [190.0, 280.0],
    [190.0, 280.0],
    [170.0, 240.0],
    [170.0, 240.0],
    [200.0, 220.0],
    [200.0, 220.0],
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    try:
        return archive.read(name)
    except KeyError as exc:
        raise SystemExit(f"archive member is missing: {name}") from exc


def import_models(
    bridge_archive_path: Path,
    ssa_archive_path: Path,
    output_dir: Path,
    *,
    force: bool,
) -> dict[str, object]:
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise SystemExit(f"output directory is not empty: {output_dir} (use --force)")
    output_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(bridge_archive_path) as bridge_archive:
        stage1_pickle = _read_member(bridge_archive, STAGE1_PICKLE)
        stage2_mat = _read_member(bridge_archive, STAGE2_MAT)
        bridge_data = _read_member(bridge_archive, BRIDGE_DATA)

    with zipfile.ZipFile(ssa_archive_path) as ssa_archive:
        ssa_data = _read_member(ssa_archive, SSA_DATA)
        ssa_main = _read_member(ssa_archive, SSA_MAIN)
        ssa_implementation = _read_member(ssa_archive, SSA_IMPLEMENTATION)

    if bridge_data != ssa_data:
        raise SystemExit("the two source archives contain different data.csv files")

    # This archive is supplied by the project owner and explicitly trusted for
    # conversion. Runtime code never unpickles this legacy artifact.
    stage1_bundle = pickle.loads(stage1_pickle)
    model = stage1_bundle["model"]
    scaler = stage1_bundle["scaler"]
    feature_names = list(stage1_bundle["features"])
    if feature_names != ["length", "width", "span", "rise", "n1", "n2"]:
        raise SystemExit(f"unexpected stage-1 feature order: {feature_names}")

    stage1_path = output_dir / "stage1_xgboost.ubj"
    model.get_booster().save_model(stage1_path)

    mat = scipy.io.loadmat(io.BytesIO(stage2_mat), simplify_cells=True)
    net = mat["net"]
    inputs = net["inputs"]
    settings = inputs["processSettings"]
    xmeans = np.asarray(settings[0]["xmeans"], dtype=np.float64)

    iw = net["IW"]
    lw = net["LW"]
    biases = net["b"]
    w1 = np.asarray(iw[0], dtype=np.float64)
    w2 = np.asarray(iw[1], dtype=np.float64)
    wl = np.asarray(lw[1, 0], dtype=np.float64)
    b1 = np.asarray(biases[0], dtype=np.float64).reshape(-1)
    b2 = np.asarray(biases[1], dtype=np.float64).reshape(-1)

    expected_shapes = {
        "w1": (15, 14),
        "w2": (10, 14),
        "wl": (10, 15),
        "b1": (15,),
        "b2": (10,),
        "xmeans": (8,),
    }
    arrays = {"w1": w1, "w2": w2, "wl": wl, "b1": b1, "b2": b2, "xmeans": xmeans}
    actual_shapes = {name: value.shape for name, value in arrays.items()}
    if actual_shapes != expected_shapes:
        raise SystemExit(f"unexpected CF-BPNN shapes: {actual_shapes}")

    stage2_path = output_dir / "stage2_cfbpnn_weights.npz"
    np.savez_compressed(stage2_path, **arrays)

    manifest: dict[str, object] = {
        "schema_version": "mentor-design-parameter-model-v1",
        "model_version": "mentor-two-stage-2026-04-08",
        "source": {
            "bridge_archive": bridge_archive_path.name,
            "bridge_archive_sha256": _file_sha256(bridge_archive_path),
            "ssa_archive": ssa_archive_path.name,
            "ssa_archive_sha256": _file_sha256(ssa_archive_path),
            "training_data_sha256": _sha256(bridge_data),
            "ssa_matlab_main_sha256": _sha256(ssa_main),
            "ssa_matlab_implementation_sha256": _sha256(ssa_implementation),
            "legacy_stage1_pickle_sha256": _sha256(stage1_pickle),
            "legacy_stage2_mat_sha256": _sha256(stage2_mat),
        },
        "stage1": {
            "name": "SSA-XGBoost",
            "runtime_model": "XGBoost Booster",
            "artifact": stage1_path.name,
            "artifact_sha256": _file_sha256(stage1_path),
            "feature_names": feature_names,
            "feature_min": np.asarray(scaler.data_min_, dtype=float).tolist(),
            "feature_max": np.asarray(scaler.data_max_, dtype=float).tolist(),
            "output_bounds": [0.08, 0.60],
            "missing_rise_policy": {"name": "legacy_span_ratio_seed", "ratio": 0.17},
            "metrics_from_legacy_bundle": stage1_bundle.get("metrics", {}),
            "search_provenance": (
                "The source package contains the original MATLAB SSA implementation and a later "
                "Python candidate-search approximation. This runtime artifact is replayed unchanged; "
                "the manifest does not claim which search implementation produced the pickle."
            ),
        },
        "stage2": {
            "name": "CF-BPNN",
            "runtime_model": "direct cascade-forward matrix inference",
            "artifact": stage2_path.name,
            "artifact_sha256": _file_sha256(stage2_path),
            "network_shapes": {name: list(shape) for name, shape in expected_shapes.items()},
            "input_mapping": {
                "width": {"source_index": 1, "range": [3.2, 6.83]},
                "span": {"source_index": 2, "range": [5.0, 35.0]},
                "n1": {"source_index": 4, "range": [5.0, 9.0]},
                "n2": {"source_index": 5, "range": [4.0, 8.0]},
            },
            "preprocess": ["fixunknowns", "removeconstantrows", "mapminmax[-1,1]"],
            "hidden_activation": "tansig/tanh",
            "output_activation": "purelin",
            "output_names": OUTPUT_NAMES,
            "output_ranges_mm": OUTPUT_RANGES_MM,
        },
    }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert trusted mentor model archives into runtime artifacts")
    parser.add_argument("--bridge-design-zip", required=True, type=Path)
    parser.add_argument("--ssa-xgboost-zip", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = import_models(
        args.bridge_design_zip,
        args.ssa_xgboost_zip,
        args.output_dir,
        force=args.force,
    )
    print(json.dumps({"status": "ok", "model_version": manifest["model_version"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

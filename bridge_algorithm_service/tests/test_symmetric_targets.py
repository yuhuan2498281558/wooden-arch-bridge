from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ml_pipeline.prepare.symmetric_targets import (
    ZERO_INNER_FLAT_CHORD_RATIO_THRESHOLD,
    prepare_annotation,
    write_csv,
)

class SymmetricTargetPreparationTests(unittest.TestCase):
    def _write_annotation(self, directory: Path) -> Path:
        payload = {
            "schema_version": "five-miao-node-annotation-v1",
            "created_at": "2026-08-12T00:00:00Z",
            "source_image": {"file_name": "123.测试桥-纵剖.jpg", "width_px": 900, "height_px": 500},
            "bridge": {
                "bridge_id": "T-123",
                "bridge_name": "",
                "source_group_id": "survey-family-7",
                "span_count": 3,
                "span_m": 18.6,
                "drawing_url": "",
            },
            "span": {"index": 2, "count": 3, "clear_span_m": 18.6, "ordering": "left-to-right-v1"},
            "annotation": {
                "span_index": 2,
                "quality": "medium",
                "node_definition": "member-centerline-thickness-ignored-v1",
                "points": {
                    "A": {"x": 0, "y": 10},
                    "B": {"x": 300, "y": -90},
                    "C": {"x": 600, "y": -80},
                    "D": {"x": 900, "y": 20},
                    "Q1": {"x": 180, "y": -50},
                    "Q2": {"x": 360, "y": -88},
                    "Q3": {"x": 510, "y": -83},
                    "Q4": {"x": 690, "y": -50},
                },
                "ratios": {},
            },
        }
        path = directory / "123.测试桥-纵剖_five_miao_nodes.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_preserves_observations_and_builds_symmetric_mean_targets(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            row = prepare_annotation(self._write_annotation(Path(temporary_directory)))

        self.assertAlmostEqual(
            row["design_target_alpha"],
            (row["observed_alpha_left"] + row["observed_alpha_right"]) / 2,
        )
        self.assertAlmostEqual(
            row["design_target_beta"],
            (row["observed_beta_left"] + row["observed_beta_right"]) / 2,
        )
        self.assertTrue(row["design_target_is_symmetric"])
        self.assertEqual(row["ratio_definition"], "bridge-axis-parallel-chord-v2")
        self.assertEqual(row["bridge_name"], "123.测试桥-纵剖")
        self.assertEqual(row["span_m"], 18.6)
        self.assertEqual(row["span_count"], 3)
        self.assertEqual(row["span_index"], 2)
        self.assertEqual(row["sample_key"], "T-123:span-02-of-03")
        self.assertEqual(row["split_group_key"], "survey-family-7")
        self.assertFalse(row["is_edge_span"])
        self.assertNotEqual(row["middle_axis_angle_deg"], 0.0)
        self.assertEqual(row["span_center_distance"], 0.0)
        self.assertGreater(row["three_miao_rise_span_ratio"], 0.0)
        self.assertAlmostEqual(
            row["three_miao_rise_m"],
            row["span_m"] * row["three_miao_rise_span_ratio"],
        )
        self.assertAlmostEqual(
            row["three_miao_design_chord_angle_deg"],
            math.degrees(math.atan(3.0 * row["three_miao_rise_span_ratio"])),
        )

    def test_writes_excel_compatible_utf8_csv(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            row = prepare_annotation(self._write_annotation(directory))
            output = directory / "targets.csv"
            write_csv([row], output)
            raw = output.read_bytes()

        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        self.assertIn("design_target_alpha", raw.decode("utf-8-sig"))

    def test_historical_asymmetry_is_diagnostic_not_a_review_failure(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = self._write_annotation(Path(temporary_directory))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["annotation"]["points"]["Q4"] = {"x": 790, "y": -45}
            payload["annotation"]["points"]["Q3"] = {"x": 560, "y": -83}
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            row = prepare_annotation(path)

        self.assertTrue(row["has_observed_asymmetry"])
        self.assertTrue(row["observed_asymmetry_flags"])
        self.assertFalse(row["needs_review"])
        self.assertEqual(row["review_reasons"], "")
        self.assertAlmostEqual(
            row["design_target_alpha"],
            (row["observed_alpha_left"] + row["observed_alpha_right"]) / 2,
        )
        self.assertAlmostEqual(
            row["design_target_beta"],
            (row["observed_beta_left"] + row["observed_beta_right"]) / 2,
        )

    def test_zero_inner_flat_chord_is_retained_but_excluded_from_training_scope(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = self._write_annotation(Path(temporary_directory))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["annotation"]["points"]["Q2"] = {"x": 450, "y": -92}
            payload["annotation"]["points"]["Q3"] = {"x": 450, "y": -86}
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            row = prepare_annotation(path)

        self.assertLessEqual(
            row["inner_flat_chord_ratio"],
            ZERO_INNER_FLAT_CHORD_RATIO_THRESHOLD,
        )
        self.assertTrue(row["has_zero_inner_flat_chord"])
        self.assertTrue(row["training_scope_excluded"])
        self.assertEqual(row["scope_exclusion_reasons"], "zero_inner_flat_chord")
        self.assertFalse(row["needs_review"])

    def test_old_annotation_without_span_remains_supported(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = self._write_annotation(Path(temporary_directory))
            payload = json.loads(path.read_text(encoding="utf-8"))
            del payload["bridge"]["span_m"]
            del payload["bridge"]["span_count"]
            del payload["span"]
            del payload["annotation"]["span_index"]
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            row = prepare_annotation(path)

        self.assertEqual(row["span_m"], "")
        self.assertEqual(row["span_count"], 1)
        self.assertEqual(row["span_index"], 1)

    def test_rejects_span_index_outside_bridge_span_count(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = self._write_annotation(Path(temporary_directory))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["span"]["index"] = 4
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "must not exceed"):
                prepare_annotation(path)

    def test_source_image_is_used_when_bridge_id_is_missing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = self._write_annotation(Path(temporary_directory))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["bridge"]["bridge_id"] = ""
            payload["bridge"]["bridge_name"] = "上一座桥"
            payload["bridge"]["source_group_id"] = ""
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            row = prepare_annotation(path)

        self.assertEqual(row["bridge_key"], "123.测试桥-纵剖")
        self.assertEqual(row["split_group_key"], "123.测试桥-纵剖")

    def test_none_span_fields_fall_back_to_bridge_metadata(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = self._write_annotation(Path(temporary_directory))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["span"]["count"] = None
            payload["span"]["index"] = None
            payload["span"]["clear_span_m"] = None
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            row = prepare_annotation(path)

        # span.count/index/clear_span_m are None, so bridge.span_count,
        # annotation.span_index and bridge.span_m must be used instead of
        # silently treating the record as a single span.
        self.assertEqual(row["span_count"], 3)
        self.assertEqual(row["span_index"], 2)
        self.assertEqual(row["span_m"], 18.6)
        self.assertEqual(row["sample_key"], "T-123:span-02-of-03")
        self.assertFalse(row["is_edge_span"])

    def test_blank_source_group_id_falls_back_to_bridge_key(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = self._write_annotation(Path(temporary_directory))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["bridge"]["bridge_id"] = "T-123"
            payload["bridge"]["source_group_id"] = "   "
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            row = prepare_annotation(path)

        self.assertEqual(row["split_group_key"], "T-123")

    def test_defects_are_read_and_flagged_for_training_exclusion(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = self._write_annotation(Path(temporary_directory))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["annotation"]["defects"] = ["node_unclear", "drawing_distortion"]
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            row = prepare_annotation(path)

        self.assertTrue(row["has_defects"])
        self.assertEqual(row["defects"], "node_unclear;drawing_distortion")
        # 历史不对称仍只作诊断，缺陷本身不触发几何越界复核
        self.assertFalse(row["needs_review"])

    def test_defects_accept_semicolon_string_for_backward_compat(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = self._write_annotation(Path(temporary_directory))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["annotation"]["defects"] = "member_missing; scale_unknown "
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            row = prepare_annotation(path)

        self.assertTrue(row["has_defects"])
        self.assertEqual(row["defects"], "member_missing;scale_unknown")

    def test_missing_defects_field_is_treated_as_clean(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = self._write_annotation(Path(temporary_directory))
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["annotation"].pop("defects", None)
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            row = prepare_annotation(path)

        self.assertFalse(row["has_defects"])
        self.assertEqual(row["defects"], "")


if __name__ == "__main__":
    unittest.main()

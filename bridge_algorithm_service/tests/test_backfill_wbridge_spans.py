from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ml_pipeline.prepare.backfill_wbridge_spans import (
    apply_plans,
    build_drawing_index,
    plan_directory,
)


class WBridgeSpanBackfillTests(unittest.TestCase):
    def _write_annotation(self, directory: Path, *, source_file: str, span_m=None) -> Path:
        payload = {
            "schema_version": "five-miao-node-annotation-v1",
            "source_image": {"file_name": source_file},
            "bridge": {"bridge_name": "测试桥", "span_m": span_m},
            "span": {"index": 1, "count": 1, "clear_span_m": span_m},
            "annotation": {"points": {}},
        }
        path = directory / "测试桥_five_miao_nodes.json"
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def test_builds_index_from_exact_decoded_drawing_basename(self) -> None:
        detail = {
            "id": 79,
            "drawings": [{"url": "/draw/1722342623742.%E6%83%A0%E9%A3%8E%E6%A1%A5.jpg"}],
        }
        index = build_drawing_index([detail])
        self.assertEqual(index["1722342623742.惠风桥.jpg"], [detail])

    def test_plans_exact_drawing_match_and_description_approximation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source_file = "1722342115665.福建省寿宁县飞云桥-纵剖.jpg"
            self._write_annotation(directory, source_file=source_file)
            details = [{
                "id": 47,
                "name": "飞云桥",
                "max_span_length": None,
                "drawings": [{"url": f"/draw/{source_file}"}],
            }]

            plans, skipped, unresolved = plan_directory(directory, details)

            self.assertEqual(skipped, [])
            self.assertEqual(unresolved, [])
            self.assertEqual(len(plans), 1)
            self.assertEqual(plans[0].span_m, 19.0)
            self.assertTrue(plans[0].source_field.startswith("description_approximate"))

    def test_apply_updates_both_span_fields_and_preserves_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            directory = parent / "数据集"
            directory.mkdir()
            source_file = "1722342623742.福建省屏南县惠风桥-纵剖.jpg"
            path = self._write_annotation(directory, source_file=source_file)
            details = [{
                "id": 79,
                "name": "惠风桥",
                "max_span_length": 22.95,
                "drawings": [{"url": f"/draw/{source_file}"}],
            }]
            plans, _, unresolved = plan_directory(directory, details)
            self.assertEqual(unresolved, [])

            backup = apply_plans(directory, plans)

            updated = json.loads(path.read_text(encoding="utf-8"))
            original = json.loads((backup / path.name).read_text(encoding="utf-8"))
            self.assertEqual(updated["span"]["clear_span_m"], 22.95)
            self.assertEqual(updated["bridge"]["span_m"], 22.95)
            self.assertIsNone(original["span"]["clear_span_m"])
            self.assertTrue((backup / "span_backfill_audit.csv").is_file())

    def test_existing_span_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source_file = "already-filled.jpg"
            self._write_annotation(directory, source_file=source_file, span_m=18.6)
            plans, skipped, unresolved = plan_directory(directory, [])
            self.assertEqual(plans, [])
            self.assertEqual(len(skipped), 1)
            self.assertEqual(unresolved, [])

    def test_existing_legacy_bridge_span_is_synchronized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source_file = "legacy-span.jpg"
            path = self._write_annotation(directory, source_file=source_file)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["bridge"]["span_m"] = 26.29
            payload["span"]["clear_span_m"] = None
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            details = [{
                "id": 106,
                "name": "仙恩桥",
                "max_span_length": 26.29,
                "drawings": [{"url": f"/draw/{source_file}"}],
            }]

            plans, skipped, unresolved = plan_directory(directory, details)

            self.assertEqual(skipped, [])
            self.assertEqual(unresolved, [])
            self.assertEqual(plans[0].span_m, 26.29)
            self.assertEqual(plans[0].source_field, "existing_bridge.span_m")

    def test_conflicting_existing_span_fields_are_unresolved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source_file = "conflicting-span.jpg"
            path = self._write_annotation(directory, source_file=source_file, span_m=18.0)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["span"]["clear_span_m"] = 19.0
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            details = [{
                "id": 1,
                "name": "冲突桥",
                "max_span_length": 18.0,
                "drawings": [{"url": f"/draw/{source_file}"}],
            }]

            plans, skipped, unresolved = plan_directory(directory, details)

            self.assertEqual(plans, [])
            self.assertEqual(skipped, [])
            self.assertEqual(len(unresolved), 1)
            self.assertIn("conflicting existing span fields", unresolved[0])


if __name__ == "__main__":
    unittest.main()

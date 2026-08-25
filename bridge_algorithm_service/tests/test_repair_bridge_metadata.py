from __future__ import annotations

import unittest

from ml_pipeline.prepare.repair_bridge_metadata import (
    bridge_name_from_source,
    repair_payload,
    repaired_file_name,
)


class BridgeMetadataRepairTests(unittest.TestCase):
    def test_extracts_bridge_name_from_source_drawing(self) -> None:
        self.assertEqual(
            bridge_name_from_source("1722406715504.福建省屏南县百祥桥-纵剖.jpg"),
            "百祥桥",
        )
        self.assertEqual(bridge_name_from_source("1662823322679.万安桥_页面_05.jpg"), "万安桥")

    def test_repairs_stale_name_and_builds_multi_span_file_name(self) -> None:
        payload = {
            "schema_version": "five-miao-node-annotation-v1",
            "source_image": {"file_name": "1662823322679.万安桥_页面_05.jpg"},
            "bridge": {"bridge_id": "", "bridge_name": "千乘桥", "source_group_id": ""},
            "span": {"index": 2, "count": 6},
            "annotation": {"span_index": 2},
        }

        repair_payload(payload)

        self.assertEqual(payload["bridge"]["bridge_name"], "万安桥")
        self.assertEqual(payload["bridge"]["source_group_id"], "drawing:1662823322679")
        self.assertEqual(
            repaired_file_name(payload),
            "1662823322679.万安桥_页面_05_span_02_of_06_five_miao_nodes.json",
        )


if __name__ == "__main__":
    unittest.main()

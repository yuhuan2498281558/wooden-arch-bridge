from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from ml_pipeline.prepare.layered_targets import (
    build_layered_rows,
    load_normative_overrides,
    normative_review_template,
)


def _row(sample_key: str = "bridge-a:span-01-of-01") -> dict:
    return {
        "sample_key": sample_key,
        "bridge_name": "测试桥",
        "span_index": 1,
        "span_count": 1,
        "span_m": 18.0,
        "design_target_alpha": 0.61,
        "design_target_beta": 0.22,
        "needs_review": False,
        "quality": "high",
        "has_defects": False,
        "training_scope_excluded": False,
    }


class LayeredTargetTests(unittest.TestCase):
    def test_layering_preserves_proxy_and_does_not_invent_normative_target(self) -> None:
        rows = build_layered_rows([_row()])

        self.assertEqual(rows[0]["historical_observation_alpha"], 0.61)
        self.assertEqual(rows[0]["historical_observation_beta"], 0.22)
        self.assertEqual(rows[0]["normative_design_alpha"], "")
        self.assertEqual(rows[0]["normative_design_beta"], "")
        self.assertFalse(rows[0]["normative_design_training_eligible"])
        self.assertEqual(rows[0]["normative_design_exclusion_reason"], "awaiting_expert_normative_target")

    def test_confirmed_high_residual_is_retained_as_historical_variation(self) -> None:
        review = {
            "bridge-a:span-01-of-01": {
                "review_status": "confirmed_valid_historical_observation",
            }
        }

        row = build_layered_rows([_row()], historical_reviews=review)[0]

        self.assertTrue(row["historical_proxy_training_eligible"])
        self.assertEqual(row["historical_variation_status"], "confirmed_valid_historical_variation")
        self.assertEqual(row["historical_variation_cause"], "multiple_possible_causes_not_attributed")

    def test_approved_normative_target_requires_reviewer_and_evidence(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "overrides.csv"
            with path.open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "sample_key",
                    "normative_design_status",
                    "normative_design_alpha",
                    "normative_design_beta",
                    "reviewer",
                    "evidence",
                ])
                writer.writeheader()
                writer.writerow({
                    "sample_key": "bridge-a:span-01-of-01",
                    "normative_design_status": "approved",
                    "normative_design_alpha": "0.62",
                    "normative_design_beta": "0.21",
                    "reviewer": "",
                    "evidence": "专家复核",
                })

            with self.assertRaisesRegex(ValueError, "requires reviewer"):
                load_normative_overrides(path, expected_sample_keys={"bridge-a:span-01-of-01"})

    def test_approved_normative_target_becomes_trainable_without_overwriting_proxy(self) -> None:
        overrides = {
            "bridge-a:span-01-of-01": {
                "normative_design_status": "approved",
                "normative_design_alpha": 0.64,
                "normative_design_beta": 0.2,
                "normative_design_reviewer": "专家A",
                "normative_design_evidence": "设计图复核",
                "normative_design_notes": "",
            }
        }

        row = build_layered_rows([_row()], normative_overrides=overrides)[0]

        self.assertEqual(row["historical_observation_alpha"], 0.61)
        self.assertEqual(row["normative_design_alpha"], 0.64)
        self.assertTrue(row["normative_design_training_eligible"])

    def test_template_keeps_historical_and_normative_fields_separate(self) -> None:
        layered = build_layered_rows([_row()])

        template = normative_review_template(layered)

        self.assertEqual(template[0]["historical_observation_alpha"], 0.61)
        self.assertEqual(template[0]["normative_design_status"], "pending")
        self.assertEqual(template[0]["normative_design_alpha"], "")


if __name__ == "__main__":
    unittest.main()

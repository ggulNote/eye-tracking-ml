from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from gaze_pipeline.data.profile_side import preprocess_profile_side

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import preview_data_ver1_preprocessing as preview  # noqa: E402


def _write_participant(root: Path, subject: str, *, sharpness: tuple[float, float]) -> None:
    participant = root / subject
    (participant / "labels").mkdir(parents=True)
    (participant / "images/webcam").mkdir(parents=True)
    (participant / "images/phonecam").mkdir(parents=True)
    participant.joinpath("participant.json").write_text(
        json.dumps(
            {
                "screen": {"canvas_width_pixel": 1920, "canvas_height_pixel": 1080},
                "camera_config": [
                    {"role": "iphone_left", "position": "participant_left_30_45_deg"}
                ],
                "calibration_assets": {
                    "required_assets_valid": False,
                    "camera_intrinsics_valid": True,
                },
            }
        ),
        encoding="utf-8",
    )
    fields = [
        "sample",
        "pair",
        "webcam_image",
        "phonecam_image",
        "webcam_mediapipe_face_detected",
        "webcam_mediapipe_iris_detected",
        "phonecam_mediapipe_face_detected",
        "phonecam_mediapipe_iris_detected",
        "webcam_sharpness",
        "phonecam_sharpness",
        "x_px",
        "y_px",
        "x_centered",
        "y_centered",
    ]
    with participant.joinpath("labels/image_samples.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, score in enumerate(sharpness):
            sample = f"s{index:06d}"
            image = np.full((80, 120, 3), 80 + index, dtype=np.uint8)
            Image.fromarray(image).save(participant / f"images/webcam/{sample}.jpg")
            Image.fromarray(image).save(participant / f"images/phonecam/{sample}.jpg")
            writer.writerow(
                {
                    "sample": sample,
                    "pair": str(index),
                    "webcam_image": f"images/webcam/{sample}.jpg",
                    "phonecam_image": f"images/phonecam/{sample}.jpg",
                    "webcam_mediapipe_face_detected": "1",
                    "webcam_mediapipe_iris_detected": "1",
                    "phonecam_mediapipe_face_detected": "0",
                    "phonecam_mediapipe_iris_detected": "0",
                    "webcam_sharpness": str(score),
                    "phonecam_sharpness": str(score),
                    "x_px": "960",
                    "y_px": "540",
                    "x_centered": "0",
                    "y_centered": "0",
                }
            )


class PreviewDataVer1Tests(unittest.TestCase):
    def test_output_guard_allows_only_project_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            project = tmp_path / "project"
            raw = tmp_path / "data(ver1)"
            project.mkdir()
            raw.mkdir()

            preview.validate_preview_paths(raw, project / "outputs/preview", project)
            with self.assertRaisesRegex(ValueError, "project outputs"):
                preview.validate_preview_paths(raw, project / "preview", project)
            with self.assertRaisesRegex(ValueError, "project outputs|overlap"):
                preview.validate_preview_paths(raw, raw / "generated", project)

    def test_selection_is_quality_ranked_and_round_robin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            _write_participant(tmp_path, "p00", sharpness=(10.0, 90.0))
            _write_participant(tmp_path, "p03", sharpness=(20.0, 80.0))

            selected = preview.discover_selected_pairs(tmp_path, count=2)

            self.assertEqual(
                [(pair.subject_id, pair.sample_id) for pair in selected],
                [("p00", "s000001"), ("p03", "s000001")],
            )
            self.assertTrue(
                all(pair.front_path.is_file() and pair.side_path.is_file() for pair in selected)
            )

    def test_dummy_profile_annotation_cannot_be_training_label(self) -> None:
        image = np.full((720, 1280, 3), 127, dtype=np.uint8)

        annotation = preview._dummy_side_annotation(
            image,
            phone_position="participant_left_30_45_deg",
        )
        result = preprocess_profile_side(
            image,
            eye_bbox_xyxy=annotation["visible_eye_bbox_xyxy"],
            eyelid_keypoints_xy=annotation["visible_eye_keypoints_xy"],
            iris_center_xy=annotation["iris_center_xy"],
            head_origin_xy=annotation["profile_head_origin_xy"],
            head_forward_point_xy=annotation["profile_head_forward_xy"],
            output_size_hw=(128, 256),
            vertical_only=True,
        )

        self.assertIs(annotation["eye_annotation_valid"], False)
        self.assertIs(annotation["training_eligible"], False)
        self.assertIn("dummy", annotation["annotation_source"])
        self.assertEqual(result.patch.shape, (128, 256, 3))

    def test_preview_manifest_marks_side_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            _write_participant(tmp_path, "p00", sharpness=(10.0, 90.0))
            pair = preview.discover_selected_pairs(tmp_path, count=1)[0]

            rows = list(csv.DictReader(preview._manifest_csv([pair], tmp_path).splitlines()))

            self.assertEqual([row["view"] for row in rows], ["front", "side"])
            self.assertEqual(rows[1]["training_eligible"], "false")
            self.assertEqual(rows[1]["annotation_status"], "missing_strict_profile_annotation")
            self.assertFalse(Path(rows[1]["image_path"]).is_absolute())


if __name__ == "__main__":
    unittest.main()

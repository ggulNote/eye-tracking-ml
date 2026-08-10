from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from gaze_pipeline.config import load_config
from gaze_pipeline.data.transforms import OpenCVHaarFaceDetector, OrderedGazePreprocessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
BLAZEGAZE_PROFILE = PROJECT_ROOT / "configs" / "profiles" / "blazegaze.yaml"


def write_face_image(path: Path) -> np.ndarray:
    image = np.zeros((72, 128, 3), dtype=np.uint8)
    image[15:60, 30:98] = (180, 120, 90)
    Image.fromarray(image).save(path)
    return image


def landmarks() -> np.ndarray:
    return np.asarray(
        [[45, 30], [55, 30], [73, 30], [83, 30], [52, 50], [76, 50]],
        dtype=np.float32,
    )


def sample(path: Path) -> dict[str, object]:
    return {
        "sample_id": "p00:day01/0001.jpg",
        "subject_id": "p00",
        "view": "front",
        "image_path": str(path),
        "facial_landmarks_xy": landmarks(),
        "target_gaze_xy": torch.tensor([-0.25, 0.1]),
        "metadata": {},
    }


def test_base_preprocessing_returns_rgb_chw_tensor_and_transformed_landmarks(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "face.jpg"
    write_face_image(image_path)
    config = load_config(BASE_CONFIG)

    output = OrderedGazePreprocessor(config, split="validation")(sample(image_path))

    assert isinstance(output["image"], torch.Tensor)
    assert output["image"].shape == (3, 224, 224)
    assert output["image"].dtype == torch.float32
    assert 0 <= output["image"].min() <= output["image"].max() <= 1
    assert np.asarray(output["facial_landmarks_xy"]).shape == (6, 2)
    assert np.asarray(output["metadata"]["image_transform"]).shape == (3, 3)
    assert output["target_gaze_xy"].tolist() == pytest.approx([-0.25, 0.1])


def test_blazegaze_profile_warps_eye_region_to_128_by_512(tmp_path: Path) -> None:
    image_path = tmp_path / "face.jpg"
    write_face_image(image_path)
    config = load_config(BASE_CONFIG, profiles=(BLAZEGAZE_PROFILE,))

    points = np.tile(np.asarray([[64.0, 36.0]], dtype=np.float32), (478, 1))
    points[[103, 150, 379, 332]] = np.asarray(
        [[30, 15], [30, 60], [98, 60], [98, 15]], dtype=np.float32
    )
    points[4] = [64, 35]
    points[151] = [64, 25]
    points[195] = [64, 42]
    points[[362, 385, 387, 263, 373, 380]] = np.asarray(
        [[44, 31], [48, 28], [54, 28], [58, 31], [54, 34], [48, 34]],
        dtype=np.float32,
    )
    points[[133, 158, 160, 33, 144, 153]] = np.asarray(
        [[70, 31], [74, 28], [80, 28], [84, 31], [80, 34], [74, 34]],
        dtype=np.float32,
    )

    def detector(_image: np.ndarray, _view: str) -> dict[str, object]:
        return {
            "landmarks_xy": points,
            "face_rt": np.eye(4, dtype=np.float32),
            "face_bbox_xywh": np.asarray([30, 15, 69, 46], dtype=np.float32),
            "landmark_kind": "mediapipe_face_landmarker_478",
        }

    input_sample = sample(image_path)
    input_sample["metadata"] = {"face_center_3d": np.asarray([10.0, 20.0, 600.0], dtype=np.float32)}

    output = OrderedGazePreprocessor(
        config,
        split="validation",
        landmark_detector=detector,
    )(input_sample)

    assert output["image"].shape == (3, 128, 512)
    assert output["metadata"]["representation"] == "webeyetrack_eye_patch_v1"
    assert output["metadata"]["head_vector"].tolist() == pytest.approx([0.0, 0.0, -1.0])
    assert output["metadata"]["face_origin_3d"].tolist() == pytest.approx([1.0, 2.0, 60.0])
    assert output["metadata"]["gaze_valid"] is True


def test_horizontal_flip_changes_centered_screen_x_target(tmp_path: Path) -> None:
    image_path = tmp_path / "face.jpg"
    write_face_image(image_path)
    config = load_config(
        BASE_CONFIG,
        overrides=(
            "preprocessing.augment.enabled=true",
            "preprocessing.augment.horizontal_flip_probability=1.0",
        ),
    )

    output = OrderedGazePreprocessor(config, split="train")(sample(image_path))

    assert output["target_gaze_xy"].tolist() == pytest.approx([0.25, 0.1])


def test_horizontal_flip_mirrors_derived_front_3d_model_features(tmp_path: Path) -> None:
    image_path = tmp_path / "face.jpg"
    write_face_image(image_path)
    config = load_config(
        BASE_CONFIG,
        overrides=(
            "preprocessing.augment.enabled=true",
            "preprocessing.augment.horizontal_flip_probability=1.0",
        ),
    )
    input_sample = sample(image_path)
    input_sample["metadata"] = {
        "head_vector": np.asarray([0.2, -0.3, -0.93], dtype=np.float32),
        "face_origin_3d": np.asarray([4.0, 5.0, 60.0], dtype=np.float32),
        "head_orientation_valid": True,
    }

    output = OrderedGazePreprocessor(config, split="train")(input_sample)

    assert output["metadata"]["head_vector"].tolist() == pytest.approx([-0.2, -0.3, -0.93])
    assert output["metadata"]["face_origin_3d"].tolist() == pytest.approx([-4.0, 5.0, 60.0])
    assert output["metadata"]["head_orientation_valid"] is True


def test_metric_head_pose_exposes_orientation_validity_independently() -> None:
    preprocessor = OrderedGazePreprocessor({"stage_order": []}, split="validation")
    malformed = {
        "image": np.zeros((32, 32, 3), dtype=np.uint8),
        "metadata": {"face_rt": np.eye(3, dtype=np.float32)},
    }
    preprocessor._stage_metric_head_pose(malformed, {"on_failure": "mark_invalid"})

    assert malformed["metadata"]["head_orientation_valid"] is False
    assert malformed["metadata"]["head_pose_valid"] is False
    assert np.isnan(malformed["metadata"]["head_vector"]).all()

    orientation_only = {
        "image": np.zeros((32, 32, 3), dtype=np.uint8),
        "metadata": {"face_rt": np.eye(4, dtype=np.float32)},
    }
    preprocessor._stage_metric_head_pose(orientation_only, {"on_failure": "mark_invalid"})

    assert orientation_only["metadata"]["head_orientation_valid"] is True
    assert orientation_only["metadata"]["head_pose_valid"] is False
    assert np.isfinite(orientation_only["metadata"]["head_vector"]).all()


def test_front_and_side_haar_detectors_keep_branch_specific_configs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, int]] = []

    def fake_detect(
        detector: OpenCVHaarFaceDetector, _image: np.ndarray, view: str
    ) -> dict[str, object]:
        seen.append((view, int(detector.config["min_neighbors"])))
        return {
            "landmarks_xy": np.asarray(
                [[10, 10], [20, 10], [30, 10], [40, 10], [40, 40], [10, 40]],
                dtype=np.float32,
            ),
            "landmark_kind": "face_bbox_hull",
            "face_bbox_xywh": np.asarray([10, 10, 31, 31], dtype=np.float32),
            "mirrored_retry": False,
        }

    monkeypatch.setattr(OpenCVHaarFaceDetector, "detect", fake_detect)
    config = {
        "stage_order": ["decode", "face_landmarks"],
        "decode": {"enabled": True},
        "face_landmarks": {
            "enabled": True,
            "source": "annotation",
            "fallback_detector": "opencv_haar",
            "on_failure": "error",
        },
        "branch_overrides": {
            "front": {"face_landmarks": {"opencv_haar": {"min_neighbors": 5}}},
            "side": {"face_landmarks": {"opencv_haar": {"min_neighbors": 3}}},
        },
    }
    preprocessor = OrderedGazePreprocessor(config, split="validation")
    image = np.zeros((60, 80, 3), dtype=np.uint8)

    preprocessor({"image": image.copy(), "view": "front", "metadata": {}})
    preprocessor({"image": image.copy(), "view": "side", "metadata": {}})

    assert seen == [("front", 5), ("side", 3)]
    assert set(preprocessor._opencv_haar_detectors) == {"front", "side"}


def test_preserve_canvas_face_roi_masks_only_the_rectangular_roi() -> None:
    image = np.full((60, 80, 3), 25, dtype=np.uint8)
    image[10:41, 20:61] = (180, 120, 90)
    points = np.asarray(
        [[20, 10], [60, 10], [60, 40], [20, 40]],
        dtype=np.float32,
    )
    config = {
        "stage_order": ["decode", "face_landmarks", "face_roi", "background_mask"],
        "decode": {"enabled": True},
        "face_landmarks": {"enabled": True, "source": "annotation"},
        "face_roi": {
            "enabled": True,
            "mode": "preserve_canvas",
            "margin_ratio": 0.0,
        },
        "background_mask": {
            "enabled": True,
            "method": "face_roi_bbox",
            "fill_rgb": [0, 0, 0],
        },
    }

    output = OrderedGazePreprocessor(config, split="validation")(
        {
            "image": image.copy(),
            "view": "front",
            "facial_landmarks_xy": points,
            "metadata": {},
        }
    )

    result = np.asarray(output["image"])
    assert result.shape == image.shape
    assert output["metadata"]["face_roi_xyxy"].tolist() == [20, 10, 61, 41]
    assert np.all(result[:10] == 0)
    assert np.all(result[:, :20] == 0)
    assert np.array_equal(result[10:41, 20:61], image[10:41, 20:61])


def test_face_roi_mark_invalid_returns_fixed_black_model_tensor() -> None:
    image = np.full((60, 80, 3), 180, dtype=np.uint8)
    # Four annotations pass the landmark-count contract but cannot define a
    # non-zero face rectangle, exercising face_roi.on_failure itself.
    degenerate_landmarks = np.asarray([[30.0, 25.0]] * 4, dtype=np.float32)
    config = {
        "stage_order": ["decode", "face_landmarks", "face_roi", "resize", "normalize"],
        "decode": {"enabled": True},
        "face_landmarks": {"enabled": True, "source": "annotation"},
        "face_roi": {
            "enabled": True,
            "mode": "crop",
            "margin_ratio": 0.2,
            "on_failure": "mark_invalid",
        },
        "resize": {
            "enabled": True,
            "size_hw": [32, 48],
            "keep_aspect_ratio": False,
        },
        "normalize": {
            "enabled": True,
            "mode": "zero_one",
            "channel_order": "CHW",
        },
    }

    output = OrderedGazePreprocessor(config, split="validation")(
        {
            "image": image,
            "view": "front",
            "facial_landmarks_xy": degenerate_landmarks,
            "metadata": {},
        }
    )

    assert isinstance(output["image"], torch.Tensor)
    assert output["image"].shape == (3, 32, 48)
    assert output["image"].dtype == torch.float32
    assert torch.count_nonzero(output["image"]).item() == 0
    assert output["metadata"]["face_roi_valid"] is False
    assert output["metadata"]["gaze_valid"] is False
    assert output["metadata"]["representation"] == "invalid_face_roi"
    assert output["metadata"]["invalid_reasons"] == [
        "face_roi:landmarks do not span a valid face ROI"
    ]


def _profile90_config(*, on_failure: str = "error") -> dict[str, object]:
    return {
        "stage_order": ["decode", "eye_region_warp", "normalize"],
        "decode": {"enabled": True},
        "eye_region_warp": {
            "enabled": True,
            "method": "profile90_annotation",
            "size_hw": [128, 256],
            "crop_mode": "affine",
            "landmark_crop_scale_xy": [2.4, 1.2],
            "eyelid_tail_indices": [3, 2, 4],
            "feature_extraction": {
                "side_headpose": {"enabled": True, "source": "side_2d"},
                "side_eyeangle": {"enabled": True},
                "side_eyelidangle": {"enabled": True, "vertical_only": True},
            },
            "ear_threshold": 0.2,
            "on_closed": "mark_invalid",
            "on_failure": on_failure,
        },
        "normalize": {
            "enabled": True,
            "mode": "zero_one",
            "channel_order": "CHW",
        },
    }


def _profile90_sample(*, closed: bool = False) -> dict[str, object]:
    image = np.zeros((100, 160, 3), dtype=np.uint8)
    image[30:70, 50:130] = (180, 120, 90)
    upper_y = 49.0 if closed else 44.0
    lower_y = 51.0 if closed else 56.0
    return {
        "image": image,
        "view": "side",
        "metadata": {
            "visible_eye": "right",
            "visible_eye_bbox_xyxy": np.asarray([70, 35, 130, 65], dtype=np.float32),
            "visible_eye_keypoints_xy": np.asarray(
                [
                    [80, 50],
                    [90, upper_y],
                    [110, upper_y],
                    [120, 50],
                    [110, lower_y],
                    [90, lower_y],
                ],
                dtype=np.float32,
            ),
            "iris_center_xy": np.asarray([102, 46], dtype=np.float32),
            "profile_head_origin_xy": np.asarray([140, 70], dtype=np.float32),
            "profile_head_forward_xy": np.asarray([120, 50], dtype=np.float32),
            "eye_annotation_valid": True,
        },
    }


def test_profile90_annotation_stage_returns_eye_patch_and_selected_features() -> None:
    output = OrderedGazePreprocessor(_profile90_config(), split="validation")(_profile90_sample())

    assert output["image"].shape == (3, 128, 256)
    assert output["metadata"]["representation"] == "profile90_selectable_features_v3"
    assert output["metadata"]["selected_eye"] == "right"
    assert output["metadata"]["ear"].tolist() == pytest.approx([np.nan, 0.3], nan_ok=True)
    assert output["metadata"]["head_pose_2d"].tolist() == pytest.approx([-(2**-0.5), -(2**-0.5)])
    assert output["metadata"]["head_pose_2d_valid"] is True
    assert "head_pose_valid" not in output["metadata"]
    assert output["metadata"]["iris_pose_2d"].tolist() == pytest.approx([0.0, -1.0 / 3.0])
    np.testing.assert_allclose(
        output["metadata"]["eyelid_tail_vectors_px"],
        [[-10.0, -6.0], [-10.0, 6.0]],
    )
    expected_direction_degrees = float(np.degrees(np.arctan2(6.0, 10.0)))
    assert output["metadata"]["eye_angles"].shape == (2,)
    assert output["metadata"]["eye_angles"].tolist() == pytest.approx(
        [-expected_direction_degrees / 180.0, expected_direction_degrees / 180.0]
    )
    assert output["metadata"]["eyelid_included_angle_degrees"] == pytest.approx(
        2.0 * expected_direction_degrees
    )
    assert output["metadata"]["head_pitch_proxy_degrees"] == pytest.approx(45.0)
    assert output["metadata"]["gaze_valid"] is True


def test_profile90_closed_eye_is_kept_but_excluded_from_gaze_loss() -> None:
    output = OrderedGazePreprocessor(_profile90_config(), split="validation")(
        _profile90_sample(closed=True)
    )

    assert output["image"].shape == (3, 128, 256)
    assert output["metadata"]["gaze_state"] == "closed"
    assert output["metadata"]["gaze_valid"] is False
    assert "eye_state:selected_profile_eye_closed" in output["metadata"]["invalid_reasons"]


def test_profile90_horizontal_flip_mirrors_pose_and_eye_side() -> None:
    config = _profile90_config()
    config["stage_order"] = ["decode", "eye_region_warp", "normalize", "augment"]
    config["augment"] = {
        "enabled": True,
        "apply_to": "train_only",
        "horizontal_flip_probability": 1.0,
        "transform_target_gaze": False,
    }
    output = OrderedGazePreprocessor(config, split="train")(_profile90_sample())

    unit = 2**-0.5
    assert output["metadata"]["head_pose_2d"].tolist() == pytest.approx([unit, -unit])
    assert output["metadata"]["iris_pose_2d"].tolist() == pytest.approx([0.0, -1.0 / 3.0])
    expected_direction_degrees = float(np.degrees(np.arctan2(6.0, 10.0)))
    assert output["metadata"]["eye_angles"].tolist() == pytest.approx(
        [-expected_direction_degrees / 180.0, expected_direction_degrees / 180.0]
    )
    assert output["metadata"]["selected_eye"] == "left"
    assert output["metadata"]["visible_eye"] == "left"


def test_profile90_missing_annotation_can_return_fixed_invalid_patch() -> None:
    sample = _profile90_sample()
    del sample["metadata"]["iris_center_xy"]  # type: ignore[index]
    output = OrderedGazePreprocessor(
        _profile90_config(on_failure="mark_invalid"), split="validation"
    )(sample)

    assert output["image"].shape == (3, 128, 256)
    assert torch.count_nonzero(output["image"]).item() == 0
    assert output["metadata"]["eye_region_warp_valid"] is False
    assert output["metadata"]["gaze_valid"] is False
    assert output["metadata"]["representation"] == "invalid_eye_patch"
    assert np.isnan(output["metadata"]["iris_pose_2d"]).all()
    assert np.isnan(output["metadata"]["eye_angles"]).all()


def test_profile90_feature_toggles_do_not_disable_eye_roi() -> None:
    config = _profile90_config()
    feature_extraction = config["eye_region_warp"]["feature_extraction"]
    for feature in feature_extraction.values():
        feature["enabled"] = False
    sample = _profile90_sample()
    del sample["metadata"]["iris_center_xy"]  # type: ignore[index]
    del sample["metadata"]["profile_head_origin_xy"]  # type: ignore[index]
    del sample["metadata"]["profile_head_forward_xy"]  # type: ignore[index]

    output = OrderedGazePreprocessor(config, split="validation")(sample)

    assert output["image"].shape == (3, 128, 256)
    assert output["metadata"]["eye_region_warp_valid"] is True
    assert "head_pose_2d" not in output["metadata"]
    assert "eye_angles" not in output["metadata"]
    assert "iris_pose_2d" not in output["metadata"]


def test_profile90_front_3d_source_does_not_require_side_head_annotations() -> None:
    config = _profile90_config()
    config["eye_region_warp"]["feature_extraction"]["side_headpose"]["source"] = "front_3d"
    sample = _profile90_sample()
    del sample["metadata"]["profile_head_origin_xy"]  # type: ignore[index]
    del sample["metadata"]["profile_head_forward_xy"]  # type: ignore[index]

    output = OrderedGazePreprocessor(config, split="validation")(sample)

    assert output["image"].shape == (3, 128, 256)
    assert output["metadata"]["eye_region_warp_valid"] is True
    assert "head_pose_2d" not in output["metadata"]
    assert output["metadata"]["iris_pose_2d"].shape == (2,)
    assert output["metadata"]["eye_angles"].shape == (2,)

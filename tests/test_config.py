from datetime import UTC, datetime
from pathlib import Path

import pytest

from gaze_pipeline.config import (
    ConfigLoadError,
    ConfigValidationError,
    load_and_validate_config,
    resolve_model_forward_keys,
    select_model_forward_inputs,
    validate_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
PROFILE90_CONFIG = PROJECT_ROOT / "configs" / "profiles" / "side_profile_90.yaml"
MEASURED_PROFILE = PROJECT_ROOT / "configs" / "profiles" / "measured_head_down_neutral.yaml"
SIDE_ROI_PROFILE = PROJECT_ROOT / "configs" / "profiles" / "side_roi_only.yaml"
FRONT_MODEL_PROFILE = PROJECT_ROOT / "configs" / "models" / "front_webeyetrack.yaml"


@pytest.mark.parametrize(
    "profile_names",
    [
        (),
        ("blazegaze.yaml",),
        ("demo_dual_view_training.yaml",),
        ("blazegaze.yaml", "side_profile_90.yaml"),
        ("blazegaze.yaml", "side_profile_90.yaml", "side_roi_only.yaml"),
    ],
)
def test_execution_config_validation_accepts_shipped_profiles(
    profile_names: tuple[str, ...],
) -> None:
    profile_dir = PROJECT_ROOT / "configs" / "profiles"

    load_and_validate_config(
        BASE_CONFIG,
        profiles=tuple(profile_dir / name for name in profile_names),
    )


@pytest.mark.parametrize(
    ("side_model_name", "entrypoint", "expected_init_args"),
    [
        (
            "side_mobilenet_v4.yaml",
            "gaze_pipeline.models.side.mobilenet_v4:create_model",
            {
                "model_id",
                "pretrained",
                "embedding_dim",
                "auxiliary_dim",
                "normalize_imagenet",
                "image_mean",
                "image_std",
                "dropout",
            },
        ),
        (
            "side_blazegaze_transfer.yaml",
            "gaze_pipeline.models.side.blazegaze_transfer:create_model",
            {
                "encoder_weights_path",
                "image_feature_dim",
                "embedding_dim",
                "auxiliary_dim",
                "dropout",
            },
        ),
    ],
)
def test_measured_dataset_profile_composes_with_exactly_one_side_model(
    side_model_name: str,
    entrypoint: str,
    expected_init_args: set[str],
) -> None:
    config = load_and_validate_config(
        BASE_CONFIG,
        profiles=(
            PROJECT_ROOT / "configs" / "profiles" / "blazegaze.yaml",
            PROFILE90_CONFIG,
            SIDE_ROI_PROFILE,
            MEASURED_PROFILE,
            FRONT_MODEL_PROFILE,
            PROJECT_ROOT / "configs" / "models" / side_model_name,
        ),
    )

    front = config["model"]["front"]
    side = config["model"]["side"]
    assert front["entrypoint"] == "gaze_pipeline.models.webeyetrack_front:create_model"
    assert set(front["init_args"]) == {
        "weights_path",
        "expected_sha256",
        "unfreeze_encoder",
    }
    assert side["entrypoint"] == entrypoint
    assert set(side["init_args"]) == expected_init_args
    assert "image_key" not in side["init_args"]
    assert "feature_keys" not in side["init_args"]
    assert config["data"]["selection"]["session_ids"] == ["head_down", "neutral"]
    measured_front = config["preprocessing"]["branch_overrides"]["front"]
    assert measured_front["metric_head_pose"]["source"] == "precomputed_only"
    assert measured_front["metric_head_pose"]["on_failure"] == "error"
    side_warp = config["preprocessing"]["branch_overrides"]["side"]["eye_region_warp"]
    assert side_warp["crop_mode"] == "stretch"
    assert side_warp["bbox_scale_xy"] == [1.0, 1.0]
    assert resolve_model_forward_keys(config, "side") == ("side_image",)
    assert config["preprocessing"]["cache"]["enabled"] is True


def test_base_config_resolves_environment_references_and_overrides(tmp_path: Path) -> None:
    fixed_now = datetime(2026, 8, 7, 12, 34, 56, tzinfo=UTC)

    config = load_and_validate_config(
        BASE_CONFIG,
        environ={
            "GAZE_DATA_ROOT": str(tmp_path / "dataset"),
            "GAZE_OUTPUT_ROOT": str(tmp_path / "outputs"),
            "DUAL_VIEW_MANIFEST": str(tmp_path / "manifest.csv"),
        },
        now=fixed_now,
        overrides=("data.dataloader.batch_size=16",),
    )

    assert config["data"]["dataset_root"] == str(tmp_path / "dataset")
    assert config["data"]["reader"]["type"] == "generic_csv"
    assert config["data"]["reader"]["manifest_path"] == str(tmp_path / "manifest.csv")
    assert config["data"]["views"]["available"] == ["front", "side"]
    assert config["data"]["pairing"]["enabled"] is True
    assert config["data"]["dataloader"]["batch_size"] == 16
    assert config["experiment"]["run_name"].endswith("20260807_123456")
    assert config["checkpoint"]["save_best"]["monitor"].endswith("subject_macro_euclidean_cm")


def test_unknown_override_is_rejected() -> None:
    with pytest.raises(ConfigLoadError, match="config에 없습니다"):
        load_and_validate_config(BASE_CONFIG, overrides=("training.epohs=2",))


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("training", "max_epochs"), 0, "training.max_epochs"),
        (("training", "devices"), 2, "training.devices=1"),
        (("training", "precision"), 16, "training.precision=32"),
        (
            ("training", "gradient_accumulation_steps"),
            0,
            "training.gradient_accumulation_steps",
        ),
        (
            ("training", "validate_every_n_epochs"),
            0,
            "training.validate_every_n_epochs",
        ),
        (("optimizer", "name"), "SGD", "optimizer.name"),
        (("optimizer", "learning_rate"), 0.0, "optimizer.learning_rate"),
        (("loss", "primary", "name"), "cross_entropy", "loss.primary.name"),
        (("checkpoint", "save_best", "mode"), "median", "checkpoint.save_best.mode"),
        (("model_export", "format"), "onnx", "model_export.format"),
    ],
)
def test_execution_config_rejects_values_the_runner_cannot_honor(
    path: tuple[str, ...], value: object, message: str
) -> None:
    config = load_and_validate_config(BASE_CONFIG)
    target = config
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(ConfigValidationError, match=message):
        validate_config(config)


@pytest.mark.parametrize(
    ("scheduler", "message"),
    [
        ({"enabled": "yes"}, "scheduler.enabled"),
        ({"enabled": True, "name": "CosineAnnealingLR"}, "scheduler.name"),
        ({"enabled": True, "name": "ExponentialLR", "gamma": 0.0}, "scheduler.gamma"),
        (
            {
                "enabled": True,
                "name": "ReduceLROnPlateau",
                "mode": "median",
                "factor": 0.5,
                "patience_epochs": 5,
                "min_learning_rate": 0.0,
            },
            "scheduler.mode",
        ),
        (
            {
                "enabled": True,
                "name": "ReduceLROnPlateau",
                "mode": "min",
                "factor": 1.0,
                "patience_epochs": 5,
                "min_learning_rate": 0.0,
            },
            "scheduler.factor",
        ),
        (
            {
                "enabled": True,
                "name": "ReduceLROnPlateau",
                "mode": "min",
                "factor": 0.5,
                "patience_epochs": -1,
                "min_learning_rate": 0.0,
            },
            "scheduler.patience_epochs",
        ),
        (
            {
                "enabled": True,
                "name": "ReduceLROnPlateau",
                "mode": "min",
                "factor": 0.5,
                "patience_epochs": 5,
                "min_learning_rate": -1e-6,
            },
            "scheduler.min_learning_rate",
        ),
    ],
)
def test_scheduler_execution_contract_is_validated(
    scheduler: dict[str, object], message: str
) -> None:
    config = load_and_validate_config(BASE_CONFIG)
    config["scheduler"] = scheduler

    with pytest.raises(ConfigValidationError, match=message):
        validate_config(config)


def test_disabled_scheduler_and_profile_extension_namespaces_are_allowed() -> None:
    config = load_and_validate_config(BASE_CONFIG)
    config["scheduler"] = {
        "enabled": False,
        "profile_extension": {"custom_schedule": "unused_by_builtin_runner"},
    }
    config["training"]["profile_extension"] = {"custom_loop": True}
    config["optimizer"]["profile_extension"] = {"custom_parameter_group": True}
    config["loss"]["profile_extension"] = {"custom_objective": True}

    validate_config(config)


@pytest.mark.parametrize(
    "key",
    [
        "enabled",
        "log_system_metrics",
        "log_resolved_config",
        "log_dataset_manifest",
        "log_split_manifest",
        "log_predictions",
        "log_environment",
    ],
)
def test_mlflow_flags_must_be_booleans(key: str) -> None:
    config = load_and_validate_config(BASE_CONFIG)
    config["mlflow"][key] = "false"

    with pytest.raises(ConfigValidationError, match=rf"mlflow\.{key}"):
        validate_config(config)


def test_invalid_split_ratio_is_reported() -> None:
    with pytest.raises(ConfigValidationError, match="합은 1"):
        load_and_validate_config(
            BASE_CONFIG,
            overrides=(
                "data.split.ratios.train=0.8",
                "data.split.ratios.validation=0.15",
                "data.split.ratios.test=0.15",
            ),
        )


def test_session_selection_can_be_set_from_cli_override() -> None:
    config = load_and_validate_config(
        BASE_CONFIG,
        overrides=("data.selection.session_ids=[head_down,neutral]",),
    )

    assert config["data"]["selection"]["session_ids"] == ["head_down", "neutral"]


@pytest.mark.parametrize(
    "session_ids",
    ([], ["neutral", "neutral"], ["neutral", ""]),
)
def test_invalid_session_selection_is_reported(session_ids: list[str]) -> None:
    config = load_and_validate_config(BASE_CONFIG)
    config["data"]["selection"]["session_ids"] = session_ids

    with pytest.raises(ConfigValidationError, match=r"data\.selection\.session_ids"):
        validate_config(config)


def test_zero_validation_and_test_ratios_are_allowed_for_smoke_demo() -> None:
    config = load_and_validate_config(
        BASE_CONFIG,
        overrides=(
            "data.split.ratios.train=1.0",
            "data.split.ratios.validation=0.0",
            "data.split.ratios.test=0.0",
        ),
    )

    assert config["data"]["split"]["ratios"] == {
        "train": 1.0,
        "validation": 0.0,
        "test": 0.0,
    }


def test_train_split_ratio_must_be_positive() -> None:
    with pytest.raises(ConfigValidationError, match=r"ratios\.train은 0보다 커야"):
        load_and_validate_config(
            BASE_CONFIG,
            overrides=(
                "data.split.ratios.train=0.0",
                "data.split.ratios.validation=1.0",
                "data.split.ratios.test=0.0",
            ),
        )


@pytest.mark.parametrize("ratio", (-0.1, 1.1))
def test_each_split_ratio_must_be_between_zero_and_one(ratio: float) -> None:
    with pytest.raises(ConfigValidationError, match="0 이상 1 이하"):
        load_and_validate_config(
            BASE_CONFIG,
            overrides=(
                f"data.split.ratios.validation={ratio}",
                f"data.split.ratios.train={1.0 - ratio}",
            ),
        )


def test_fusion_requires_pairing_and_both_models() -> None:
    with pytest.raises(ConfigValidationError) as captured:
        load_and_validate_config(
            BASE_CONFIG,
            overrides=("data.pairing.enabled=false", "fusion.enabled=true"),
        )

    message = str(captured.value)
    assert "data.pairing.enabled" in message
    assert "front/side" in message


def test_y_axis_residual_requires_scalar_side_output_contract() -> None:
    config = _dual_view_config()
    config["fusion"]["enabled"] = True

    with pytest.raises(ConfigValidationError, match=r"delta_y_shape=\[B, 1\]"):
        config["model"]["side"]["output_contract"] = {
            "gaze_key": "gaze_xy",
            "gaze_shape": ["B", 2],
        }
        validate_config(config)


def test_y_axis_residual_rejects_side_policy_other_than_front_fallback() -> None:
    config = _dual_view_config()
    config["model"]["side"]["output_contract"] = {
        "gaze_key": None,
        "gaze_shape": None,
        "delta_y_key": "delta_y_side",
        "delta_y_shape": ["B", 1],
    }
    config["fusion"]["enabled"] = True
    config["fusion"]["missing_branch_policy"] = "drop"

    with pytest.raises(ConfigValidationError, match="use_available_branch"):
        validate_config(config)


def _dual_view_config() -> dict:
    config = load_and_validate_config(BASE_CONFIG)
    config["model"]["side"]["enabled"] = True
    config["preprocessing"]["branch_overrides"]["side"]["enabled"] = True
    return config


def test_side_branch_accepts_nested_stage_overrides() -> None:
    config = _dual_view_config()
    side = config["preprocessing"]["branch_overrides"]["side"]
    side.update(
        {
            "face_landmarks": {"enabled": True, "source": "detector"},
            "face_roi": {
                "enabled": True,
                "source": "landmarks_bbox",
                "margin_ratio": 0.25,
            },
            "background_mask": {
                "enabled": True,
                "method": "face_hull",
                "fill_rgb": [0, 0, 0],
            },
            "resize": {"enabled": True, "size_hw": [224, 224]},
        }
    )

    validate_config(config)


def test_profile90_annotation_crop_does_not_require_frontal_landmarks() -> None:
    config = load_and_validate_config(
        BASE_CONFIG,
        profiles=(PROJECT_ROOT / "configs" / "profiles" / "blazegaze.yaml", PROFILE90_CONFIG),
    )

    side = config["preprocessing"]["branch_overrides"]["side"]
    front = config["preprocessing"]["branch_overrides"]["front"]
    assert front["metric_head_pose"]["source"] == "precomputed_or_reconstruct"
    assert side["face_landmarks"]["enabled"] is False
    assert side["eye_region_warp"]["method"] == "profile90_annotation"
    assert side["eye_region_warp"]["eyelid_tail_indices"] == [3, 2, 4]
    contract = config["model"]["side"]["input_contract"]
    assert contract["shape"] == ["B", 3, 128, 256]
    assert contract["forward_keys"] == "auto"
    assert resolve_model_forward_keys(config, "side") == (
        "side_image",
        "side_head_pose_2d",
        "side_eye_angles",
        "side_iris_pose_2d",
    )
    assert set(contract["auxiliary_keys"]) == {
        "side_headpose",
        "side_eyeangle",
        "side_eyelidangle",
    }
    assert contract["auxiliary_keys"]["side_eyelidangle"]["vertical_only"] is True
    assert contract["auxiliary_keys"]["side_eyeangle"]["shape"] == ["B", 2]
    assert contract["auxiliary_keys"]["side_eyeangle"]["value_range"] == [-1.0, 1.0]
    assert contract["validity_key"] == "side_gaze_valid"
    assert contract["validity_role"] == "loss_metric_fusion_mask"
    assert contract["validity_passed_to_model"] is False


def test_profile90_feature_toggles_resolve_forward_keys_automatically() -> None:
    config = load_and_validate_config(
        BASE_CONFIG,
        profiles=(PROJECT_ROOT / "configs" / "profiles" / "blazegaze.yaml", PROFILE90_CONFIG),
        overrides=(
            "preprocessing.branch_overrides.side.eye_region_warp."
            "feature_extraction.side_eyeangle.enabled=false",
            "preprocessing.branch_overrides.side.eye_region_warp."
            "feature_extraction.side_eyelidangle.enabled=false",
        ),
    )

    assert resolve_model_forward_keys(config, "side") == (
        "side_image",
        "side_head_pose_2d",
    )


def test_profile90_all_disabled_features_resolve_to_image_only() -> None:
    config = load_and_validate_config(
        BASE_CONFIG,
        profiles=(PROJECT_ROOT / "configs" / "profiles" / "blazegaze.yaml", PROFILE90_CONFIG),
        overrides=(
            "preprocessing.branch_overrides.side.eye_region_warp."
            "feature_extraction.side_headpose.enabled=false",
            "preprocessing.branch_overrides.side.eye_region_warp."
            "feature_extraction.side_eyeangle.enabled=false",
            "preprocessing.branch_overrides.side.eye_region_warp."
            "feature_extraction.side_eyelidangle.enabled=false",
        ),
    )

    assert resolve_model_forward_keys(config, "side") == ("side_image",)


def test_front_3d_head_source_requires_pairing_and_resolves_front_key() -> None:
    config = load_and_validate_config(
        BASE_CONFIG,
        profiles=(PROJECT_ROOT / "configs" / "profiles" / "blazegaze.yaml", PROFILE90_CONFIG),
    )
    head_feature = config["preprocessing"]["branch_overrides"]["side"]["eye_region_warp"][
        "feature_extraction"
    ]["side_headpose"]
    head_feature["source"] = "front_3d"
    config["data"]["pairing"]["enabled"] = False

    with pytest.raises(ConfigValidationError, match=r"data\.pairing\.enabled=true"):
        validate_config(config)

    config["data"]["pairing"]["enabled"] = True
    validate_config(config)
    assert resolve_model_forward_keys(config, "side")[1] == "front_head_vector"
    assert head_feature["front_3d_validity_key"] == "front_head_orientation_valid"
    assert head_feature["front_3d_invalid_policy"] == "zero_fill_and_mask"


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("front_3d_key", "custom_head", "canonical key 'front_head_vector'"),
        (
            "front_3d_validity_key",
            "custom_valid",
            "canonical key 'front_head_orientation_valid'",
        ),
        ("front_3d_invalid_policy", "keep_nan", "front_3d_invalid_policy"),
    ],
)
def test_profile90_rejects_unproduced_front_3d_contract_keys(
    field: str, value: str, match: str
) -> None:
    config = load_and_validate_config(
        BASE_CONFIG,
        profiles=(PROJECT_ROOT / "configs" / "profiles" / "blazegaze.yaml", PROFILE90_CONFIG),
    )
    head_feature = config["preprocessing"]["branch_overrides"]["side"]["eye_region_warp"][
        "feature_extraction"
    ]["side_headpose"]
    head_feature[field] = value

    with pytest.raises(ConfigValidationError, match=match):
        validate_config(config)


@pytest.mark.parametrize(
    ("feature_name", "match"),
    [
        ("side_eyeangle", "canonical key 'side_eye_angles'"),
        ("side_eyelidangle", "canonical key 'side_iris_pose_2d'"),
    ],
)
def test_profile90_rejects_unproduced_eye_feature_keys(feature_name: str, match: str) -> None:
    config = load_and_validate_config(
        BASE_CONFIG,
        profiles=(PROJECT_ROOT / "configs" / "profiles" / "blazegaze.yaml", PROFILE90_CONFIG),
    )
    features = config["preprocessing"]["branch_overrides"]["side"]["eye_region_warp"][
        "feature_extraction"
    ]
    features[feature_name]["key"] = "custom_unproduced_key"

    with pytest.raises(ConfigValidationError, match=match):
        validate_config(config)


def test_standard_front_contract_is_resolved_and_selects_batch_inputs() -> None:
    config = load_and_validate_config(
        BASE_CONFIG,
        profiles=(PROJECT_ROOT / "configs" / "profiles" / "blazegaze.yaml",),
    )
    assert resolve_model_forward_keys(config, "front") == (
        "front_image",
        "front_head_vector",
        "front_face_origin_3d",
    )
    batch = {
        "front_image": object(),
        "front_head_vector": object(),
        "front_face_origin_3d": object(),
        "front_gaze_valid": object(),
    }

    selected = select_model_forward_inputs(batch, config, "front")

    assert tuple(selected) == resolve_model_forward_keys(config, "front")
    assert "front_gaze_valid" not in selected


def test_profile90_eye_roi_cannot_be_disabled_by_feature_ablation() -> None:
    config = load_and_validate_config(
        BASE_CONFIG,
        profiles=(PROJECT_ROOT / "configs" / "profiles" / "blazegaze.yaml", PROFILE90_CONFIG),
    )
    config["preprocessing"]["branch_overrides"]["side"]["eye_region_warp"]["enabled"] = False

    with pytest.raises(ConfigValidationError, match="eye_region_warp.enabled=true"):
        validate_config(config)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("crop_mode", "unknown", "crop_mode"),
        ("landmark_crop_scale_xy", [2.4, 0.0], "landmark_crop_scale_xy"),
        ("eyelid_tail_indices", [3, 1, 4], "eyelid_tail_indices"),
    ],
)
def test_profile90_geometry_options_are_validated(key: str, value: object, message: str) -> None:
    config = load_and_validate_config(
        BASE_CONFIG,
        profiles=(PROJECT_ROOT / "configs" / "profiles" / "blazegaze.yaml", PROFILE90_CONFIG),
    )
    config["preprocessing"]["branch_overrides"]["side"]["eye_region_warp"][key] = value

    with pytest.raises(ConfigValidationError, match=message):
        validate_config(config)


def test_profile90_vertical_only_is_validated_at_the_feature_toggle() -> None:
    config = load_and_validate_config(
        BASE_CONFIG,
        profiles=(PROJECT_ROOT / "configs" / "profiles" / "blazegaze.yaml", PROFILE90_CONFIG),
    )
    features = config["preprocessing"]["branch_overrides"]["side"]["eye_region_warp"][
        "feature_extraction"
    ]
    features["side_eyelidangle"]["vertical_only"] = "yes"

    with pytest.raises(ConfigValidationError, match="side_eyelidangle.vertical_only"):
        validate_config(config)


def test_profile90_rejects_removed_parent_vertical_only_key() -> None:
    config = load_and_validate_config(
        BASE_CONFIG,
        profiles=(PROJECT_ROOT / "configs" / "profiles" / "blazegaze.yaml", PROFILE90_CONFIG),
    )
    config["preprocessing"]["branch_overrides"]["side"]["eye_region_warp"][
        "eye_vector_vertical_only"
    ] = True

    with pytest.raises(ConfigValidationError, match="더 이상 지원하지 않습니다"):
        validate_config(config)


def test_blazegaze_face_crop_size_must_be_positive() -> None:
    config = load_and_validate_config(
        BASE_CONFIG,
        profiles=(PROJECT_ROOT / "configs" / "profiles" / "blazegaze.yaml",),
    )
    config["preprocessing"]["branch_overrides"]["front"]["eye_region_warp"]["face_crop_size"] = 0

    with pytest.raises(ConfigValidationError, match="face_crop_size"):
        validate_config(config)


@pytest.mark.parametrize(
    ("key", "value"),
    (("source", "unknown"), ("precomputed_face_origin_unit", "meter")),
)
def test_blazegaze_precomputed_pose_options_are_validated(key: str, value: object) -> None:
    config = load_and_validate_config(
        BASE_CONFIG,
        profiles=(PROJECT_ROOT / "configs" / "profiles" / "blazegaze.yaml",),
    )
    config["preprocessing"]["branch_overrides"]["front"]["metric_head_pose"][key] = value

    with pytest.raises(ConfigValidationError, match=key):
        validate_config(config)


def test_side_face_roi_requires_enabled_landmarks_first() -> None:
    config = _dual_view_config()
    side = config["preprocessing"]["branch_overrides"]["side"]
    side["face_landmarks"] = {"enabled": False}
    side["face_roi"] = {"enabled": True}

    with pytest.raises(ConfigValidationError) as captured:
        validate_config(config)

    message = str(captured.value)
    assert "side.face_roi.enabled=true" in message
    assert "side.face_landmarks" in message


def test_side_landmarks_must_precede_face_roi_in_stage_order() -> None:
    config = _dual_view_config()
    side = config["preprocessing"]["branch_overrides"]["side"]
    side["face_landmarks"] = {"enabled": True}
    side["face_roi"] = {"enabled": True}
    order = config["preprocessing"]["stage_order"]
    order.remove("face_landmarks")
    order.insert(order.index("face_roi") + 1, "face_landmarks")

    with pytest.raises(ConfigValidationError, match="face_landmarks가 face_roi보다"):
        validate_config(config)


@pytest.mark.parametrize(
    ("stage", "value", "message"),
    [
        ("face_roi", True, "stage override mapping"),
        ("resize", {"enabled": "yes"}, "resize.enabled는 true/false"),
    ],
)
def test_branch_stage_override_shape_is_validated(stage: str, value: object, message: str) -> None:
    config = load_and_validate_config(BASE_CONFIG)
    config["preprocessing"]["branch_overrides"]["side"][stage] = value

    with pytest.raises(ConfigValidationError, match=message):
        validate_config(config)

"""Geometry-only preprocessing for annotated 90-degree profile-eye images.

This module deliberately does not detect a face, eyelid, iris, or head pose.
It consumes image-space annotations produced by a profile-capable detector (or
by a human annotator) and turns them into a deterministic monocular model
input plus independently selectable geometric features.

The six eyelid points use this fixed contour order::

    [corner_a, upper_a, upper_b, corner_b, lower_b, lower_a]

Both pose vectors use image coordinates, where positive x points right and
positive y points down. ``head_pose_2d`` is a unit vector. ``eye_pose_2d`` is
an eye-size-normalized displacement and therefore is not a unit vector.
The optional eyelid-tail vectors are shape diagnostics, not gaze vectors.
Their two eye-local direction angles can be used as a model feature.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np


class ProfileSideGeometryError(ValueError):
    """Raised when a profile annotation cannot form a valid model input."""


@dataclass(frozen=True)
class ProfileSidePreprocessResult:
    """Outputs for one annotated 90-degree profile image.

    Attributes:
        patch: RGB, HWC monocular eye patch.
        source_to_patch: Homogeneous 3x3 transform from source-image pixels to
            patch pixels.
        head_pose_2d: Optional unit ``forward - origin`` vector in image
            coordinates.
        eye_pose_2d: Optional iris displacement in an eye-aligned coordinate
            frame, divided by the visible eye width and opening height.
        source_quad_xy: Source crop corners in TL, TR, BR, BL order.
        eyelid_center_xy: Eye anchor used for the eye displacement.
        iris_center_xy: Validated iris center in source-image coordinates.
        eye_size_xy: Width and opening height used to normalize eye pose.
        eyelid_tail_points_xy: ``[a0, a1, a2]`` where a0 is the temporal eye
            corner and a1/a2 are its adjacent upper/lower eyelid points.
        eyelid_tail_vectors_px: The two construction vectors
            ``[a0->a1, a0->a2]`` in source pixels. They are retained only for
            annotation audit and angle drawing, not as model features.
        eyelid_direction_angles_degrees: Direction of ``a0→a1`` and
            ``a0→a2`` in the semantic eye-local frame, ordered upper/lower.
        eyelid_direction_angles_normalized: The same two angles divided by
            pi, with mathematical range ``[-1,1]``. This is the selectable
            ``side_eye_angles`` model feature.
        eyelid_tail_angle_degrees: Smaller angle between ``a0→a1`` and
            ``a0→a2`` in degrees, or ``None`` without eyelid landmarks.
        eyelid_tail_angle_normalized: The same scalar divided by 180 degrees,
            with range ``[0,1]``. This legacy included-angle scalar is retained
            for annotation diagnostics only; v3 does not forward it to a model.
        head_pitch_proxy_degrees: Projected profile-head-anchor-to-nose elevation angle;
            positive is image-up. This is not calibrated 3D head pose.
    """

    patch: np.ndarray
    source_to_patch: np.ndarray
    head_pose_2d: np.ndarray | None
    eye_pose_2d: np.ndarray | None
    source_quad_xy: np.ndarray
    eyelid_center_xy: np.ndarray
    iris_center_xy: np.ndarray | None
    eye_size_xy: np.ndarray
    eyelid_tail_points_xy: np.ndarray | None
    eyelid_tail_vectors_px: np.ndarray | None
    eyelid_direction_angles_degrees: np.ndarray | None
    eyelid_direction_angles_normalized: np.ndarray | None
    eyelid_tail_angle_degrees: float | None
    eyelid_tail_angle_normalized: float | None
    head_pitch_proxy_degrees: float | None


def _as_point(value: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,):
        raise ProfileSideGeometryError(f"{name} must have shape [2], got {point.shape}")
    if not np.isfinite(point).all():
        raise ProfileSideGeometryError(f"{name} contains a non-finite coordinate")
    return point


def _require_point_in_image(
    point: np.ndarray,
    *,
    name: str,
    image_width: int,
    image_height: int,
) -> None:
    x, y = (float(point[0]), float(point[1]))
    if not (0.0 <= x <= image_width - 1 and 0.0 <= y <= image_height - 1):
        raise ProfileSideGeometryError(
            f"{name}={point.tolist()} lies outside the {image_width}x{image_height} image"
        )


def _as_positive_pair(value: Sequence[float], *, name: str) -> tuple[float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,) or not np.isfinite(array).all() or np.any(array <= 0):
        raise ProfileSideGeometryError(f"{name} must contain two finite positive values")
    return float(array[0]), float(array[1])


def _normalize_vector(vector: np.ndarray, *, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ProfileSideGeometryError(f"{name} has zero length")
    return vector / norm


def _validate_tail_indices(value: Sequence[int]) -> tuple[int, int, int]:
    indices = np.asarray(value)
    if indices.shape != (3,) or not np.issubdtype(indices.dtype, np.integer):
        raise ProfileSideGeometryError("eyelid_tail_indices must contain three integers")
    result = tuple(int(index) for index in indices)
    if len(set(result)) != 3 or any(index < 0 or index >= 6 for index in result):
        raise ProfileSideGeometryError(
            "eyelid_tail_indices must contain three distinct indices in [0,5]"
        )
    if result not in {(0, 1, 5), (3, 2, 4)}:
        raise ProfileSideGeometryError(
            "eyelid_tail_indices must select a corner followed by its upper/lower neighbors"
        )
    return result


def _eyelid_tail_geometry(
    points: np.ndarray,
    *,
    indices: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Return two eye-local vector directions plus their included angle."""

    a0, a1, a2 = (points[index] for index in indices)
    upper_vector = a1 - a0
    lower_vector = a2 - a0
    upper_unit = _normalize_vector(upper_vector, name="a0-to-a1 upper eyelid vector")
    lower_unit = _normalize_vector(lower_vector, name="a0-to-a2 lower eyelid vector")

    opposite_corner_index = 0 if indices[0] == 3 else 3
    local_x = _normalize_vector(
        points[opposite_corner_index] - a0,
        name="temporal-to-nasal eye-local horizontal axis",
    )
    upper_midpoint = (points[1] + points[2]) / 2.0
    lower_midpoint = (points[5] + points[4]) / 2.0
    downward_hint = lower_midpoint - upper_midpoint
    local_y = downward_hint - float(np.dot(downward_hint, local_x)) * local_x
    local_y = _normalize_vector(local_y, name="upper-to-lower eye-local vertical axis")
    direction_radians = np.asarray(
        [
            np.arctan2(np.dot(upper_vector, local_y), np.dot(upper_vector, local_x)),
            np.arctan2(np.dot(lower_vector, local_y), np.dot(lower_vector, local_x)),
        ],
        dtype=np.float64,
    )
    direction_degrees = np.degrees(direction_radians)
    direction_normalized = direction_radians / float(np.pi)

    cosine = float(np.dot(upper_unit, lower_unit))
    angle_radians = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
    angle_degrees = float(np.degrees(angle_radians))
    angle_normalized = angle_radians / float(np.pi)
    return (
        np.asarray([a0, a1, a2], dtype=np.float64),
        np.asarray([upper_vector, lower_vector], dtype=np.float64),
        direction_degrees,
        direction_normalized,
        angle_degrees,
        angle_normalized,
    )


def _validate_image(image: np.ndarray) -> np.ndarray:
    frame = np.asarray(image)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ProfileSideGeometryError(
            f"image must be an RGB HWC array with three channels, got {frame.shape}"
        )
    if frame.shape[0] < 2 or frame.shape[1] < 2:
        raise ProfileSideGeometryError("image height and width must each be at least 2")
    if not np.issubdtype(frame.dtype, np.number):
        raise ProfileSideGeometryError("image must have a numeric dtype")
    if np.issubdtype(frame.dtype, np.floating) and not np.isfinite(frame).all():
        raise ProfileSideGeometryError("image contains non-finite pixels")
    return np.ascontiguousarray(frame)


def _validate_bbox(
    eye_bbox_xyxy: Sequence[float] | np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    bbox = np.asarray(eye_bbox_xyxy, dtype=np.float64)
    if bbox.shape != (4,):
        raise ProfileSideGeometryError(f"eye_bbox_xyxy must have shape [4], got {bbox.shape}")
    if not np.isfinite(bbox).all():
        raise ProfileSideGeometryError("eye_bbox_xyxy contains a non-finite coordinate")
    x0, y0, x1, y1 = (float(value) for value in bbox)
    if x1 <= x0 or y1 <= y0:
        raise ProfileSideGeometryError("eye_bbox_xyxy must satisfy x1>x0 and y1>y0")
    # xyxy boxes conventionally use an exclusive right/bottom edge, so x1=W
    # and y1=H are valid while point annotations remain limited to W-1/H-1.
    if x0 < 0 or y0 < 0 or x1 > image_width or y1 > image_height:
        raise ProfileSideGeometryError(
            f"eye_bbox_xyxy={bbox.tolist()} lies outside the {image_width}x{image_height} image"
        )
    return bbox


def _validate_eyelid_points(
    eyelid_keypoints_xy: Sequence[Sequence[float]] | np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    points = np.asarray(eyelid_keypoints_xy, dtype=np.float64)
    if points.shape != (6, 2):
        raise ProfileSideGeometryError(
            f"eyelid_keypoints_xy must have shape [6,2], got {points.shape}"
        )
    if not np.isfinite(points).all():
        raise ProfileSideGeometryError("eyelid_keypoints_xy contains a non-finite coordinate")
    for index, point in enumerate(points):
        _require_point_in_image(
            point,
            name=f"eyelid_keypoints_xy[{index}]",
            image_width=image_width,
            image_height=image_height,
        )
    return points


def _eye_geometry_from_keypoints(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    corner_axis = points[3] - points[0]
    eye_width = float(np.linalg.norm(corner_axis))
    if eye_width <= 1e-8:
        raise ProfileSideGeometryError("selected eye has zero corner-to-corner width")

    axis_x = corner_axis / eye_width
    # Canonicalize the local horizontal axis so both annotated eye sides use
    # the same image-plane sign convention.
    if axis_x[0] < 0 or (abs(float(axis_x[0])) <= 1e-8 and axis_x[1] < 0):
        axis_x = -axis_x
    axis_y = np.asarray([-axis_x[1], axis_x[0]], dtype=np.float64)
    if axis_y[1] < 0:
        axis_y = -axis_y

    center = points.mean(axis=0)
    opening_a = abs(float(np.dot(points[1] - points[5], axis_y)))
    opening_b = abs(float(np.dot(points[2] - points[4], axis_y)))
    eye_height = (opening_a + opening_b) / 2.0
    if eye_height <= 1e-8:
        raise ProfileSideGeometryError("selected eye has zero eyelid opening height")
    return center, axis_x, axis_y, eye_width, eye_height


def _quad_from_center_axes(
    center: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
    width: float,
    height: float,
) -> np.ndarray:
    half_x = axis_x * (width / 2.0)
    half_y = axis_y * (height / 2.0)
    return np.asarray(
        [
            center - half_x - half_y,
            center + half_x - half_y,
            center + half_x + half_y,
            center - half_x + half_y,
        ],
        dtype=np.float64,
    )


def _scaled_bbox_quad(bbox: np.ndarray, scale_xy: tuple[float, float]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    center = np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float64)
    width = float(x1 - x0) * scale_xy[0]
    height = float(y1 - y0) * scale_xy[1]
    return _quad_from_center_axes(
        center,
        np.asarray([1.0, 0.0]),
        np.asarray([0.0, 1.0]),
        width,
        height,
    )


def _destination_quad(
    source_quad: np.ndarray,
    *,
    output_height: int,
    output_width: int,
    crop_mode: str,
) -> np.ndarray:
    if crop_mode == "affine":
        return np.asarray(
            [
                [0.0, 0.0],
                [output_width - 1.0, 0.0],
                [output_width - 1.0, output_height - 1.0],
                [0.0, output_height - 1.0],
            ],
            dtype=np.float64,
        )

    source_width = float(np.linalg.norm(source_quad[1] - source_quad[0]))
    source_height = float(np.linalg.norm(source_quad[3] - source_quad[0]))
    if source_width <= 1e-8 or source_height <= 1e-8:
        raise ProfileSideGeometryError("source crop quad is degenerate")
    scale = min(
        (output_width - 1.0) / source_width,
        (output_height - 1.0) / source_height,
    )
    mapped_width = source_width * scale
    mapped_height = source_height * scale
    x0 = (output_width - 1.0 - mapped_width) / 2.0
    y0 = (output_height - 1.0 - mapped_height) / 2.0
    return np.asarray(
        [
            [x0, y0],
            [x0 + mapped_width, y0],
            [x0 + mapped_width, y0 + mapped_height],
            [x0, y0 + mapped_height],
        ],
        dtype=np.float64,
    )


def preprocess_profile_side(
    image: np.ndarray,
    *,
    eye_bbox_xyxy: Sequence[float] | np.ndarray | None = None,
    eyelid_keypoints_xy: Sequence[Sequence[float]] | np.ndarray | None = None,
    iris_center_xy: Sequence[float] | np.ndarray | None = None,
    head_origin_xy: Sequence[float] | np.ndarray | None = None,
    head_forward_point_xy: Sequence[float] | np.ndarray | None = None,
    output_size_hw: Sequence[int] = (128, 256),
    crop_mode: Literal["affine", "letterbox", "stretch"] = "affine",
    vertical_only: bool = False,
    landmark_crop_scale_xy: Sequence[float] = (2.4, 1.2),
    bbox_crop_scale_xy: Sequence[float] = (1.0, 1.0),
    border_value_rgb: Sequence[float] = (0, 0, 0),
    eyelid_tail_indices: Sequence[int] = (3, 2, 4),
    extract_head_pose: bool = True,
    extract_iris_pose: bool = True,
    extract_eye_angles: bool = True,
) -> ProfileSidePreprocessResult:
    """Build one annotated strict-profile model input and its 2D geometry.

    ``affine`` uses a roll-aligned eyelid crop when six eyelid points exist.
    Its default crop is 2.4 eye-widths wide and 1.2 eye-widths high. If only a
    bbox is available, it maps the axis-aligned bbox to the full output.

    ``letterbox`` preserves the explicit bbox's aspect ratio and pads the
    remaining output with ``border_value_rgb``. Without an explicit bbox it
    letterboxes the axis-aligned bounds of the six eyelid points.

    ``stretch`` clips each sample's bbox to its source image, crops only that
    eye region, and directly resizes it to the requested output. It never adds
    letterbox padding, so variable person-specific bboxes still produce the
    same model tensor shape.

    When both annotations are supplied, eyelid points define eye-pose
    geometry. The bbox controls a letterbox/stretch crop; eyelid geometry
    controls an affine crop.
    """

    import cv2  # type: ignore

    frame = _validate_image(image)
    image_height, image_width = frame.shape[:2]
    if eye_bbox_xyxy is None and eyelid_keypoints_xy is None:
        raise ProfileSideGeometryError("provide eye_bbox_xyxy, eyelid_keypoints_xy, or both")

    output_pair = np.asarray(output_size_hw)
    if output_pair.shape != (2,) or not np.issubdtype(output_pair.dtype, np.number):
        raise ProfileSideGeometryError("output_size_hw must contain two integers")
    if not np.isfinite(output_pair.astype(np.float64)).all():
        raise ProfileSideGeometryError("output_size_hw contains a non-finite value")
    if np.any(output_pair != np.floor(output_pair)):
        raise ProfileSideGeometryError("output_size_hw values must be integers")
    output_height, output_width = (int(output_pair[0]), int(output_pair[1]))
    if output_height < 2 or output_width < 2:
        raise ProfileSideGeometryError("output height and width must each be at least 2")

    normalized_crop_mode = str(crop_mode).lower()
    if normalized_crop_mode not in {"affine", "letterbox", "stretch"}:
        raise ProfileSideGeometryError("crop_mode must be 'affine', 'letterbox', or 'stretch'")
    landmark_scale = _as_positive_pair(landmark_crop_scale_xy, name="landmark_crop_scale_xy")
    bbox_scale = _as_positive_pair(bbox_crop_scale_xy, name="bbox_crop_scale_xy")
    border = np.asarray(border_value_rgb, dtype=np.float64)
    if border.shape != (3,) or not np.isfinite(border).all():
        raise ProfileSideGeometryError("border_value_rgb must contain three finite values")

    bbox = (
        _validate_bbox(
            eye_bbox_xyxy,
            image_width=image_width,
            image_height=image_height,
        )
        if eye_bbox_xyxy is not None
        else None
    )
    eyelid_points = (
        _validate_eyelid_points(
            eyelid_keypoints_xy,
            image_width=image_width,
            image_height=image_height,
        )
        if eyelid_keypoints_xy is not None
        else None
    )
    iris_center = None
    if extract_iris_pose:
        if iris_center_xy is None:
            raise ProfileSideGeometryError("iris_center_xy is required when extract_iris_pose=true")
        iris_center = _as_point(iris_center_xy, name="iris_center_xy")
        _require_point_in_image(
            iris_center,
            name="iris_center_xy",
            image_width=image_width,
            image_height=image_height,
        )

    head_pose = None
    head_pitch_proxy_degrees = None
    if extract_head_pose:
        if head_origin_xy is None or head_forward_point_xy is None:
            raise ProfileSideGeometryError(
                "head_origin_xy and head_forward_point_xy are required when extract_head_pose=true"
            )
        head_origin = _as_point(head_origin_xy, name="head_origin_xy")
        head_forward = _as_point(head_forward_point_xy, name="head_forward_point_xy")
        for point, name in (
            (head_origin, "head_origin_xy"),
            (head_forward, "head_forward_point_xy"),
        ):
            _require_point_in_image(
                point,
                name=name,
                image_width=image_width,
                image_height=image_height,
            )
        head_pose = _normalize_vector(head_forward - head_origin, name="head pose vector")
        head_delta = head_forward - head_origin
        head_pitch_proxy_degrees = float(
            np.degrees(np.arctan2(-float(head_delta[1]), abs(float(head_delta[0]))))
        )

    if eyelid_points is not None:
        eye_center, axis_x, axis_y, eye_width, eye_height = _eye_geometry_from_keypoints(
            eyelid_points
        )
        if extract_eye_angles:
            tail_indices = _validate_tail_indices(eyelid_tail_indices)
            (
                eyelid_tail_points,
                eyelid_tail_vectors_px,
                eyelid_direction_angles_degrees,
                eyelid_direction_angles_normalized,
                eyelid_tail_angle_degrees,
                eyelid_tail_angle_normalized,
            ) = _eyelid_tail_geometry(
                eyelid_points,
                indices=tail_indices,
            )
        else:
            eyelid_tail_points = None
            eyelid_tail_vectors_px = None
            eyelid_direction_angles_degrees = None
            eyelid_direction_angles_normalized = None
            eyelid_tail_angle_degrees = None
            eyelid_tail_angle_normalized = None
    else:
        assert bbox is not None
        x0, y0, x1, y1 = bbox
        eye_center = np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
        axis_x = np.asarray([1.0, 0.0])
        axis_y = np.asarray([0.0, 1.0])
        eye_width = float(x1 - x0)
        eye_height = float(y1 - y0)
        eyelid_tail_points = None
        eyelid_tail_vectors_px = None
        eyelid_direction_angles_degrees = None
        eyelid_direction_angles_normalized = None
        eyelid_tail_angle_degrees = None
        eyelid_tail_angle_normalized = None

    if extract_eye_angles and eyelid_points is None:
        raise ProfileSideGeometryError(
            "eyelid_keypoints_xy is required when extract_eye_angles=true"
        )

    eye_pose = None
    if iris_center is not None:
        iris_delta = iris_center - eye_center
        eye_pose = np.asarray(
            [
                float(np.dot(iris_delta, axis_x)) / eye_width,
                float(np.dot(iris_delta, axis_y)) / eye_height,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(eye_pose).all():
            raise ProfileSideGeometryError("normalized eye pose is non-finite")
        if vertical_only:
            eye_pose[0] = 0.0

    if normalized_crop_mode == "affine" and eyelid_points is not None:
        source_quad = _quad_from_center_axes(
            eye_center,
            axis_x,
            axis_y,
            eye_width * landmark_scale[0],
            eye_width * landmark_scale[1],
        )
    else:
        if bbox is None:
            assert eyelid_points is not None
            minimum = eyelid_points.min(axis=0)
            maximum = eyelid_points.max(axis=0)
            bbox = np.asarray([*minimum, *maximum], dtype=np.float64)
        source_quad = _scaled_bbox_quad(bbox, bbox_scale)

    source_width = float(np.linalg.norm(source_quad[1] - source_quad[0]))
    source_height = float(np.linalg.norm(source_quad[3] - source_quad[0]))
    if source_width <= 1e-8 or source_height <= 1e-8:
        raise ProfileSideGeometryError("source crop quad is degenerate")
    if (
        float(source_quad[:, 0].max()) < 0
        or float(source_quad[:, 1].max()) < 0
        or float(source_quad[:, 0].min()) > image_width - 1
        or float(source_quad[:, 1].min()) > image_height - 1
    ):
        raise ProfileSideGeometryError("source crop quad does not intersect the image")

    if normalized_crop_mode == "stretch":
        crop_x0 = max(0, int(np.floor(float(source_quad[:, 0].min()))))
        crop_y0 = max(0, int(np.floor(float(source_quad[:, 1].min()))))
        crop_x1 = min(image_width, int(np.ceil(float(source_quad[:, 0].max()))))
        crop_y1 = min(image_height, int(np.ceil(float(source_quad[:, 1].max()))))
        if crop_x1 - crop_x0 < 2 or crop_y1 - crop_y0 < 2:
            raise ProfileSideGeometryError(
                "clipped stretch crop must be at least 2x2 source pixels"
            )
        source_quad = np.asarray(
            [
                [crop_x0, crop_y0],
                [crop_x1 - 1, crop_y0],
                [crop_x1 - 1, crop_y1 - 1],
                [crop_x0, crop_y1 - 1],
            ],
            dtype=np.float64,
        )

    destination_quad = _destination_quad(
        source_quad,
        output_height=output_height,
        output_width=output_width,
        crop_mode="affine" if normalized_crop_mode == "stretch" else normalized_crop_mode,
    )
    source_to_patch = cv2.getPerspectiveTransform(
        source_quad.astype(np.float32), destination_quad.astype(np.float32)
    )
    if not np.isfinite(source_to_patch).all():
        raise ProfileSideGeometryError("source-to-patch transform is non-finite")
    if abs(float(np.linalg.det(source_to_patch))) <= 1e-12:
        raise ProfileSideGeometryError("source-to-patch transform is singular")
    if normalized_crop_mode == "stretch":
        crop_x0, crop_y0 = np.rint(source_quad[0]).astype(int)
        crop_x1, crop_y1 = np.rint(source_quad[2]).astype(int) + 1
        source_crop = frame[crop_y0:crop_y1, crop_x0:crop_x1]
        patch = cv2.resize(
            source_crop,
            (output_width, output_height),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        patch = cv2.warpPerspective(
            frame,
            source_to_patch,
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=tuple(float(value) for value in border),
        )
    if normalized_crop_mode == "letterbox":
        # A homography is defined outside the four correspondences too, so a
        # direct full-frame warp would leak neighboring source pixels into the
        # intended letterbox bars. Mask them explicitly.
        content_mask = np.zeros((output_height, output_width), dtype=np.uint8)
        cv2.fillConvexPoly(
            content_mask,
            np.rint(destination_quad).astype(np.int32),
            255,
        )
        patch[content_mask == 0] = border.astype(patch.dtype, copy=False)
    if patch.shape != (output_height, output_width, 3):
        raise ProfileSideGeometryError(
            f"unexpected patch shape {patch.shape}; expected {(output_height, output_width, 3)}"
        )

    return ProfileSidePreprocessResult(
        patch=np.ascontiguousarray(patch),
        source_to_patch=source_to_patch.astype(np.float32),
        head_pose_2d=head_pose.astype(np.float32) if head_pose is not None else None,
        eye_pose_2d=eye_pose.astype(np.float32) if eye_pose is not None else None,
        source_quad_xy=source_quad.astype(np.float32),
        eyelid_center_xy=eye_center.astype(np.float32),
        iris_center_xy=iris_center.astype(np.float32) if iris_center is not None else None,
        eye_size_xy=np.asarray([eye_width, eye_height], dtype=np.float32),
        eyelid_tail_points_xy=(
            eyelid_tail_points.astype(np.float32) if eyelid_tail_points is not None else None
        ),
        eyelid_tail_vectors_px=(
            eyelid_tail_vectors_px.astype(np.float32)
            if eyelid_tail_vectors_px is not None
            else None
        ),
        eyelid_direction_angles_degrees=(
            eyelid_direction_angles_degrees.astype(np.float32)
            if eyelid_direction_angles_degrees is not None
            else None
        ),
        eyelid_direction_angles_normalized=(
            eyelid_direction_angles_normalized.astype(np.float32)
            if eyelid_direction_angles_normalized is not None
            else None
        ),
        eyelid_tail_angle_degrees=eyelid_tail_angle_degrees,
        eyelid_tail_angle_normalized=eyelid_tail_angle_normalized,
        head_pitch_proxy_degrees=head_pitch_proxy_degrees,
    )


__all__ = [
    "ProfileSideGeometryError",
    "ProfileSidePreprocessResult",
    "preprocess_profile_side",
]

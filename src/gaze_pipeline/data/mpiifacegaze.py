"""Strict reader for the original MPIIFaceGaze still-image annotations."""

from __future__ import annotations

import math
import re
import struct
import zlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .records import (
    CanonicalRecord,
    DataContractError,
    ScreenCalibration,
    ensure_unique_records,
)

MPIIFACEGAZE_TOKEN_COUNT = 28
_SUBJECT_RE = re.compile(r"p(?:0[0-9]|1[0-4])\Z")
_SESSION_RE = re.compile(r"day[0-9]+\Z")


class MPIIFaceGazeFormatError(DataContractError):
    """Raised for an invalid MPIIFaceGaze file, row, or directory layout."""


class ImageMetadataError(DataContractError):
    """Raised when image dimensions cannot be read safely."""


class MatFileError(DataContractError):
    """Raised when required MATLAB calibration scalars cannot be read."""


def read_mpiifacegaze_dataset(
    dataset_root: str | Path,
    reader_config: Mapping[str, Any] | None = None,
) -> list[CanonicalRecord]:
    """Read all discovered p00..p14 annotation files.

    The returned list has one record per annotation row.  No temporal or
    left/right-eye pairing is inferred because the source is a still-image
    annotation subset.
    """

    config = dict(reader_config or {})
    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise MPIIFaceGazeFormatError(f"MPIIFaceGaze dataset_root is not a directory: {root}")

    subject_glob = str(config.get("subject_glob", "p??"))
    subject_dirs = sorted(path for path in root.glob(subject_glob) if path.is_dir())
    if not subject_dirs:
        raise MPIIFaceGazeFormatError(
            f"no subject directories matched {subject_glob!r} below {root}"
        )
    invalid_subjects = [path.name for path in subject_dirs if not _SUBJECT_RE.fullmatch(path.name)]
    if invalid_subjects:
        raise MPIIFaceGazeFormatError(
            "MPIIFaceGaze subject directories must be p00..p14; invalid: "
            + ", ".join(invalid_subjects)
        )

    expected = config.get("expected_subject_ids")
    if expected is not None:
        expected_ids = {str(item) for item in expected}
        actual_ids = {path.name for path in subject_dirs}
        if actual_ids != expected_ids:
            missing = sorted(expected_ids - actual_ids)
            extra = sorted(actual_ids - expected_ids)
            raise MPIIFaceGazeFormatError(f"subject set mismatch; missing={missing}, extra={extra}")

    annotation_template = str(
        config.get("annotation_file_template", "{subject_id}/{subject_id}.txt")
    )
    screen_template = str(
        _nested_get(
            config,
            ("calibration", "screen_size_file_template"),
            "{subject_id}/Calibration/screenSize.mat",
        )
    )
    allowed_extensions = {
        str(extension).lower()
        for extension in config.get("image_extensions", (".jpg", ".jpeg", ".png"))
    }
    verify_image_exists = bool(config.get("verify_image_exists", True))
    verify_image_shape = bool(config.get("verify_image_shape", True))
    if verify_image_shape:
        verify_image_exists = True
    target_bounds_policy = str(config.get("target_bounds_policy", "keep_flagged")).strip().lower()
    if target_bounds_policy not in {"error", "keep_flagged"}:
        raise MPIIFaceGazeFormatError(
            f"target_bounds_policy must be 'error' or 'keep_flagged', got {target_bounds_policy!r}"
        )

    records: list[CanonicalRecord] = []
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        annotation_path = _resolve_template_path(root, annotation_template, subject_id=subject_id)
        screen_path = _resolve_template_path(root, screen_template, subject_id=subject_id)
        calibration = load_screen_calibration(screen_path)
        records.extend(
            parse_mpiifacegaze_annotation(
                annotation_path,
                subject_dir=subject_dir,
                subject_id=subject_id,
                screen=calibration,
                allowed_extensions=allowed_extensions,
                verify_image_exists=verify_image_exists,
                verify_image_shape=verify_image_shape,
                allow_out_of_bounds_targets=(target_bounds_policy == "keep_flagged"),
            )
        )

    ensure_unique_records(records)
    return sorted(records, key=lambda record: record.sample_id)


def parse_mpiifacegaze_annotation(
    annotation_path: str | Path,
    *,
    subject_dir: str | Path,
    subject_id: str,
    screen: ScreenCalibration,
    allowed_extensions: Iterable[str] = (".jpg", ".jpeg", ".png"),
    verify_image_exists: bool = True,
    verify_image_shape: bool = True,
    allow_out_of_bounds_targets: bool = False,
) -> list[CanonicalRecord]:
    """Parse one headerless, whitespace-delimited 28-token annotation file."""

    path = Path(annotation_path)
    subject_root = Path(subject_dir).resolve()
    if not path.is_file():
        raise MPIIFaceGazeFormatError(f"annotation file does not exist: {path}")
    if not _SUBJECT_RE.fullmatch(subject_id):
        raise MPIIFaceGazeFormatError(f"subject_id must be p00..p14, got {subject_id!r}")

    extensions = {str(extension).lower() for extension in allowed_extensions}
    records: list[CanonicalRecord] = []
    with path.open("r", encoding="utf-8-sig", newline=None) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                raise MPIIFaceGazeFormatError(f"{path}:{line_number}: blank rows are not allowed")
            tokens = raw_line.split()
            if len(tokens) != MPIIFACEGAZE_TOKEN_COUNT:
                raise MPIIFaceGazeFormatError(
                    f"{path}:{line_number}: expected exactly "
                    f"{MPIIFACEGAZE_TOKEN_COUNT} whitespace-delimited tokens, "
                    f"got {len(tokens)}"
                )
            try:
                record = _parse_row(
                    tokens,
                    annotation_path=path,
                    line_number=line_number,
                    subject_root=subject_root,
                    subject_id=subject_id,
                    screen=screen,
                    allowed_extensions=extensions,
                    verify_image_exists=verify_image_exists,
                    verify_image_shape=verify_image_shape,
                    allow_out_of_bounds_targets=allow_out_of_bounds_targets,
                )
            except DataContractError as exc:
                if str(exc).startswith(f"{path}:{line_number}:"):
                    raise
                raise MPIIFaceGazeFormatError(f"{path}:{line_number}: {exc}") from exc
            records.append(record)

    if not records:
        raise MPIIFaceGazeFormatError(f"annotation file is empty: {path}")
    return records


def _parse_row(
    tokens: Sequence[str],
    *,
    annotation_path: Path,
    line_number: int,
    subject_root: Path,
    subject_id: str,
    screen: ScreenCalibration,
    allowed_extensions: set[str],
    verify_image_exists: bool,
    verify_image_shape: bool,
    allow_out_of_bounds_targets: bool,
) -> CanonicalRecord:
    image_relative = _strict_relative_path(tokens[0])
    if image_relative.suffix.lower() not in allowed_extensions:
        raise MPIIFaceGazeFormatError(
            f"unsupported image extension {image_relative.suffix!r} for {image_relative.as_posix()}"
        )
    if len(image_relative.parts) < 2:
        raise MPIIFaceGazeFormatError(
            f"image path must include a day directory: {image_relative.as_posix()}"
        )
    session_id = image_relative.parts[0]
    if not _SESSION_RE.fullmatch(session_id):
        raise MPIIFaceGazeFormatError(f"image path session must match dayNN, got {session_id!r}")

    image_path = (subject_root / Path(*image_relative.parts)).resolve()
    if not image_path.is_relative_to(subject_root):
        raise MPIIFaceGazeFormatError(
            f"image path escapes subject directory: {image_relative.as_posix()}"
        )
    if verify_image_exists and not image_path.is_file():
        raise MPIIFaceGazeFormatError(f"referenced image does not exist: {image_path}")
    if verify_image_shape:
        width, height, channels = read_image_shape(image_path)
    else:
        # Shape verification is part of the normal preparation contract.  The
        # explicit opt-out exists only for metadata-only inspection.
        width, height, channels = 1, 1, None

    numeric = [
        _parse_finite_float(token, column=index + 2) for index, token in enumerate(tokens[1:27])
    ]
    landmarks_flat = numeric[2:14]
    landmarks = tuple(
        (landmarks_flat[index], landmarks_flat[index + 1]) for index in range(0, 12, 2)
    )
    evaluation_eye = tokens[27].strip().lower()
    if evaluation_eye not in {"left", "right"}:
        raise MPIIFaceGazeFormatError(
            f"column 28 evaluation eye must be left/right, got {tokens[27]!r}"
        )

    return CanonicalRecord(
        sample_id=f"{subject_id}:{image_relative.as_posix()}",
        subject_id=subject_id,
        session_id=session_id,
        view="front",
        pair_id=None,
        image_path=image_path,
        image_relative_path=f"{subject_id}/{image_relative.as_posix()}",
        image_width_px=width,
        image_height_px=height,
        image_channels=channels,
        gaze_screen_xy_px=(numeric[0], numeric[1]),
        screen=screen,
        facial_landmarks_xy=landmarks,
        head_rotation_3d=tuple(numeric[14:17]),  # type: ignore[arg-type]
        head_translation_3d=tuple(numeric[17:20]),  # type: ignore[arg-type]
        face_center_3d=tuple(numeric[20:23]),  # type: ignore[arg-type]
        gaze_target_3d=tuple(numeric[23:26]),  # type: ignore[arg-type]
        evaluation_eye=evaluation_eye,
        source_dataset="MPIIFaceGaze",
        annotation_path=annotation_path.resolve(),
        annotation_line=line_number,
        allow_out_of_screen_target=allow_out_of_bounds_targets,
    )


def _parse_finite_float(token: str, *, column: int) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise MPIIFaceGazeFormatError(f"column {column} must be numeric, got {token!r}") from exc
    if not math.isfinite(value):
        raise MPIIFaceGazeFormatError(f"column {column} must be finite, got {token!r}")
    return value


def _strict_relative_path(value: str) -> PurePosixPath:
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise MPIIFaceGazeFormatError(f"image path must be a safe relative path, got {value!r}")
    if any(part in {"", "."} for part in path.parts):
        raise MPIIFaceGazeFormatError(f"invalid relative image path: {value!r}")
    return path


def _resolve_template_path(root: Path, template: str, *, subject_id: str) -> Path:
    rendered = template.format(subject_id=subject_id)
    if "${" in rendered:
        raise MPIIFaceGazeFormatError(
            f"unresolved config interpolation in path template: {rendered!r}"
        )
    candidate = Path(rendered).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _nested_get(mapping: Mapping[str, Any], keys: Sequence[str], default: Any) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def load_screen_calibration(path: str | Path) -> ScreenCalibration:
    """Read required screenSize.mat scalars, using stdlib before SciPy.

    Original MPIIFaceGaze files are MATLAB v5 files containing compressed
    scalar numeric matrices.  The small built-in reader handles that format so
    manifest preparation does not require importing the scientific stack.  An
    installed SciPy is used only as a fallback for other MATLAB-v5 encodings.
    """

    mat_path = Path(path).expanduser().resolve()
    if not mat_path.is_file():
        raise MatFileError(f"screen calibration file does not exist: {mat_path}")

    try:
        values = _read_mat_v5_numeric_scalars(mat_path)
    except (MatFileError, OSError, struct.error, zlib.error) as builtin_error:
        try:
            from scipy.io import loadmat  # type: ignore[import-not-found]
        except ModuleNotFoundError as exc:
            raise MatFileError(
                f"could not read {mat_path} with the built-in MATLAB-v5 scalar "
                "reader, and optional dependency 'scipy' is not installed; "
                f"install scipy for this MAT encoding. Cause: {builtin_error}"
            ) from exc
        try:
            loaded = loadmat(mat_path)
            values = {
                name: _single_numeric_value(value, name=name)
                for name, value in loaded.items()
                if not name.startswith("__")
            }
        except Exception as exc:  # scipy raises several format-specific types
            raise MatFileError(f"failed to read screen calibration {mat_path}: {exc}") from exc

    required = ("width_pixel", "height_pixel", "width_mm", "height_mm")
    missing = [name for name in required if name not in values]
    if missing:
        raise MatFileError(f"{mat_path} is missing required scalar(s): {', '.join(missing)}")
    width_px = _positive_integral_scalar(values["width_pixel"], "width_pixel")
    height_px = _positive_integral_scalar(values["height_pixel"], "height_pixel")
    return ScreenCalibration(
        width_px=width_px,
        height_px=height_px,
        width_mm=float(values["width_mm"]),
        height_mm=float(values["height_mm"]),
        source_path=mat_path,
    )


def _single_numeric_value(value: Any, *, name: str) -> float:
    """Extract one scalar from a SciPy/numpy value without importing numpy."""

    current = value
    while hasattr(current, "shape") and getattr(current, "size", None) == 1:
        try:
            current = current.item()
        except (AttributeError, ValueError):
            current = current[0]
    try:
        result = float(current)
    except (TypeError, ValueError) as exc:
        raise MatFileError(f"MAT variable {name!r} is not a numeric scalar") from exc
    if not math.isfinite(result):
        raise MatFileError(f"MAT variable {name!r} is not finite")
    return result


def _positive_integral_scalar(value: float, name: str) -> int:
    numeric = float(value)
    rounded = round(numeric)
    if not math.isfinite(numeric) or numeric <= 0 or not math.isclose(numeric, rounded):
        raise MatFileError(f"{name} must be a positive integer scalar, got {value!r}")
    return int(rounded)


def _read_mat_v5_numeric_scalars(path: Path) -> dict[str, float]:
    payload = path.read_bytes()
    if len(payload) < 128 or not payload.startswith(b"MATLAB 5.0 MAT-file"):
        raise MatFileError(f"{path} is not a supported MATLAB v5 MAT-file")
    endian_marker = payload[126:128]
    if endian_marker == b"IM":
        endian = "<"
    elif endian_marker == b"MI":
        endian = ">"
    else:
        raise MatFileError(f"{path} has an invalid MAT endian marker")

    result: dict[str, float] = {}
    for data_type, data in _iter_mat_elements(payload[128:], endian, top_level=True):
        if data_type == 15:  # miCOMPRESSED
            decompressed = zlib.decompress(data)
            nested = list(_iter_mat_elements(decompressed, endian))
            if len(nested) != 1:
                raise MatFileError("compressed MAT element must contain one matrix")
            data_type, data = nested[0]
        if data_type != 14:  # miMATRIX
            continue
        name, value = _parse_scalar_matrix(data, endian)
        result[name] = value
    if not result:
        raise MatFileError(f"{path} contains no readable numeric scalar matrices")
    return result


def _iter_mat_elements(
    payload: bytes, endian: str, *, top_level: bool = False
) -> Iterator[tuple[int, bytes]]:
    offset = 0
    payload_size = len(payload)
    while offset + 8 <= payload_size:
        tag = struct.unpack_from(f"{endian}I", payload, offset)[0]
        packed_size = tag >> 16
        if packed_size:
            data_type = tag & 0xFFFF
            if packed_size > 4:
                raise MatFileError("invalid small-data MAT element")
            data = payload[offset + 4 : offset + 4 + packed_size]
            offset += 8
        else:
            data_type = tag
            size = struct.unpack_from(f"{endian}I", payload, offset + 4)[0]
            data_start = offset + 8
            data_end = data_start + size
            if data_end > payload_size:
                raise MatFileError("truncated MAT data element")
            data = payload[data_start:data_end]
            unaligned_next = data_end
            aligned_next = data_start + ((size + 7) // 8) * 8
            # The original MPIIFaceGaze files place consecutive compressed
            # top-level elements without the nominal 8-byte padding.  Prefer
            # the unaligned position when it starts with a plausible MAT type.
            if (
                top_level
                and data_type == 15
                and _plausible_mat_tag(payload, unaligned_next, endian)
            ):
                offset = unaligned_next
            else:
                offset = aligned_next
        if not 1 <= data_type <= 18:
            raise MatFileError(f"invalid MAT data type {data_type}")
        yield data_type, data


def _plausible_mat_tag(payload: bytes, offset: int, endian: str) -> bool:
    if offset == len(payload):
        return True
    if offset + 4 > len(payload):
        return False
    tag = struct.unpack_from(f"{endian}I", payload, offset)[0]
    data_type = (tag & 0xFFFF) if (tag >> 16) else tag
    return 1 <= data_type <= 18


def _parse_scalar_matrix(payload: bytes, endian: str) -> tuple[str, float]:
    parts = list(_iter_mat_elements(payload, endian))
    if len(parts) < 4:
        raise MatFileError("MAT matrix is missing required sub-elements")

    # MATLAB v5 matrix order is flags, dimensions, name, real data.
    name_type, name_bytes = parts[2]
    if name_type not in {1, 2, 16, 17, 18}:
        raise MatFileError("MAT matrix name has an unsupported encoding")
    try:
        name = name_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise MatFileError("MAT matrix name is not ASCII") from exc
    if not name:
        raise MatFileError("MAT matrix has an empty name")

    data_type, data = parts[3]
    numeric_formats: dict[int, str] = {
        1: "b",  # miINT8
        2: "B",  # miUINT8
        3: "h",  # miINT16
        4: "H",  # miUINT16
        5: "i",  # miINT32
        6: "I",  # miUINT32
        7: "f",  # miSINGLE
        9: "d",  # miDOUBLE
        12: "q",  # miINT64
        13: "Q",  # miUINT64
    }
    number_format = numeric_formats.get(data_type)
    if number_format is None:
        raise MatFileError(f"MAT scalar {name!r} uses unsupported numeric type {data_type}")
    item_size = struct.calcsize(number_format)
    if len(data) != item_size:
        raise MatFileError(f"MAT variable {name!r} is not a scalar")
    value = float(struct.unpack(f"{endian}{number_format}", data)[0])
    if not math.isfinite(value):
        raise MatFileError(f"MAT variable {name!r} is not finite")
    return name, value


def read_image_shape(path: str | Path) -> tuple[int, int, int | None]:
    """Return ``(width, height, channels)`` without decoding image pixels.

    JPEG and PNG metadata are handled with the standard library.  Other image
    types fall back to Pillow and raise an actionable dependency error when it
    is unavailable.
    """

    image_path = Path(path)
    if not image_path.is_file():
        raise ImageMetadataError(f"image does not exist: {image_path}")
    try:
        with image_path.open("rb") as handle:
            signature = handle.read(24)
            handle.seek(0)
            if signature.startswith(b"\x89PNG\r\n\x1a\n"):
                return _read_png_shape(handle.read(33), image_path)
            if signature.startswith(b"\xff\xd8"):
                return _read_jpeg_shape(handle, image_path)
    except OSError as exc:
        raise ImageMetadataError(f"cannot read image {image_path}: {exc}") from exc

    try:
        from PIL import Image, UnidentifiedImageError  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ImageMetadataError(
            f"{image_path} is not JPEG/PNG; install optional dependency 'Pillow' "
            "to inspect this image format"
        ) from exc
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            channels = len(image.getbands())
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise ImageMetadataError(f"invalid image {image_path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ImageMetadataError(f"image has invalid dimensions: {image_path}")
    return int(width), int(height), int(channels)


def _read_png_shape(header: bytes, path: Path) -> tuple[int, int, int | None]:
    if len(header) < 33 or header[12:16] != b"IHDR":
        raise ImageMetadataError(f"invalid or truncated PNG header: {path}")
    width, height = struct.unpack(">II", header[16:24])
    color_type = header[25]
    channels_by_color_type = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    channels = channels_by_color_type.get(color_type)
    if width <= 0 or height <= 0 or channels is None:
        raise ImageMetadataError(f"invalid PNG metadata: {path}")
    return width, height, channels


def _read_jpeg_shape(handle: Any, path: Path) -> tuple[int, int, int | None]:
    if handle.read(2) != b"\xff\xd8":
        raise ImageMetadataError(f"invalid JPEG start marker: {path}")
    start_of_frame_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    standalone_markers = {0x01, *range(0xD0, 0xDA)}
    while True:
        byte = handle.read(1)
        if not byte:
            raise ImageMetadataError(f"JPEG has no start-of-frame marker: {path}")
        if byte != b"\xff":
            continue
        while byte == b"\xff":
            byte = handle.read(1)
            if not byte:
                raise ImageMetadataError(f"truncated JPEG marker: {path}")
        marker = byte[0]
        if marker in standalone_markers:
            continue
        length_bytes = handle.read(2)
        if len(length_bytes) != 2:
            raise ImageMetadataError(f"truncated JPEG segment: {path}")
        segment_length = struct.unpack(">H", length_bytes)[0]
        if segment_length < 2:
            raise ImageMetadataError(f"invalid JPEG segment length: {path}")
        if marker in start_of_frame_markers:
            frame = handle.read(6)
            if len(frame) != 6:
                raise ImageMetadataError(f"truncated JPEG frame header: {path}")
            height, width = struct.unpack(">HH", frame[1:5])
            channels = frame[5]
            if width <= 0 or height <= 0 or channels <= 0:
                raise ImageMetadataError(f"invalid JPEG dimensions: {path}")
            return width, height, channels
        handle.seek(segment_length - 2, 1)


# Friendly aliases for callers that use loader terminology.
load_mpiifacegaze_records = read_mpiifacegaze_dataset

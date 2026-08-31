"""Trusted-local, content-addressed cache for deterministic preprocessing.

The cache stores one pickle shard per source sample.  Pickle is intentionally
not treated as a safe interchange format: callers must explicitly acknowledge
that the cache directory is local and trusted.  A SHA-256 sidecar detects
accidental corruption, but it is not a signature and cannot make an
attacker-controlled pickle safe.

Cache identities include the schema and implementation versions, the resolved
preprocessing configuration, the manifest row, view, resolved source path, and
source-image bytes.  Consequently, changing any of those inputs selects a new
shard instead of silently reusing stale tensors.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import pickle
import re
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

CACHE_MAGIC = "gaze-pipeline-preprocessing-cache"
CACHE_SCHEMA_VERSION = 1
DEFAULT_IMPLEMENTATION_ID = "ordered-gaze-preprocessor-v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class PreprocessingCacheError(RuntimeError):
    """Base error for preprocessing-cache failures."""


class PreprocessingCacheSecurityError(PreprocessingCacheError):
    """Raised when a cache path or trust boundary is unsafe."""


class PreprocessingCacheCorruptionError(PreprocessingCacheError):
    """Raised when a shard fails its hash or envelope validation."""


class PreprocessingCacheConfigurationError(PreprocessingCacheError, ValueError):
    """Raised when cache identity inputs cannot be canonicalized."""


class PreprocessingCacheSourceChangedError(PreprocessingCacheError):
    """Raised when a source image changes while an entry is being produced."""


@dataclass(frozen=True, slots=True)
class PreprocessingCacheIdentity:
    """Immutable content identity for one preprocessed source sample."""

    key: str
    schema_version: int
    implementation_id: str
    config_sha256: str
    row_sha256: str
    view: str
    source_path: str
    source_size_bytes: int
    source_sha256: str


@dataclass(frozen=True, slots=True)
class PreprocessingCacheEntryPaths:
    """Managed shard and checksum-sidecar paths for one identity."""

    shard: Path
    checksum: Path


class TrustedLocalPreprocessingCache:
    """Content-addressed per-sample pickle cache beneath a private local root.

    Args:
        root: Directory owned by this cache. Managed descendants may not be
            symbolic links.
        config: Resolved deterministic preprocessing configuration. Passing the
            full resolved config is also valid and makes invalidation broader.
        trusted_local: Required explicit acknowledgement that pickle files in
            ``root`` are not supplied or modified by an untrusted party.
        schema_version: Payload-envelope schema. Bump this after an incompatible
            cache format change.
        implementation_id: Preprocessor semantic version. Bump this whenever
            preprocessing behavior changes without a corresponding config
            change.
        max_entry_bytes: Upper bound checked before reading a shard into memory.

    This class does not cache stochastic augmentation. Integrations should store
    the deterministic result immediately before the ``augment`` stage and apply
    augmentation after every cache hit with the current epoch's random context.
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        config: Mapping[str, Any],
        *,
        trusted_local: bool,
        schema_version: int = CACHE_SCHEMA_VERSION,
        implementation_id: str = DEFAULT_IMPLEMENTATION_ID,
        max_entry_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        if trusted_local is not True:
            raise PreprocessingCacheSecurityError(
                "pickle cache loading requires trusted_local=true; SHA-256 is not a signature"
            )
        if isinstance(schema_version, bool) or not isinstance(schema_version, int):
            raise PreprocessingCacheConfigurationError("schema_version must be an integer")
        if schema_version <= 0:
            raise PreprocessingCacheConfigurationError("schema_version must be positive")
        implementation_id = str(implementation_id).strip()
        if not implementation_id:
            raise PreprocessingCacheConfigurationError("implementation_id must not be empty")
        if isinstance(max_entry_bytes, bool) or not isinstance(max_entry_bytes, int):
            raise PreprocessingCacheConfigurationError("max_entry_bytes must be an integer")
        if max_entry_bytes <= 0:
            raise PreprocessingCacheConfigurationError("max_entry_bytes must be positive")
        if not isinstance(config, Mapping):
            raise PreprocessingCacheConfigurationError("config must be a mapping")

        self.schema_version = schema_version
        self.implementation_id = implementation_id
        self.max_entry_bytes = max_entry_bytes
        self.config_sha256 = _canonical_sha256(config, name="config")
        self.root = _create_private_root(root)

    def identity(
        self,
        *,
        row: Mapping[str, Any],
        view: str,
        image_path: str | os.PathLike[str],
    ) -> PreprocessingCacheIdentity:
        """Hash one row, view, and current source image into a shard identity."""

        if not isinstance(row, Mapping):
            raise PreprocessingCacheConfigurationError("row must be a mapping")
        normalized_view = str(view).strip().lower()
        if normalized_view not in {"front", "side"}:
            raise PreprocessingCacheConfigurationError("view must be 'front' or 'side'")
        source = Path(image_path).expanduser().resolve(strict=True)
        source_stat = source.stat()
        if not stat.S_ISREG(source_stat.st_mode):
            raise PreprocessingCacheConfigurationError(
                f"source image must be a regular file: {source}"
            )
        source_sha256 = _sha256_file(source)
        row_sha256 = _canonical_sha256(row, name="row")
        material = {
            "magic": CACHE_MAGIC,
            "schema_version": self.schema_version,
            "implementation_id": self.implementation_id,
            "config_sha256": self.config_sha256,
            "row_sha256": row_sha256,
            "view": normalized_view,
            "source_path": str(source),
            "source_size_bytes": int(source_stat.st_size),
            "source_sha256": source_sha256,
        }
        key = _canonical_sha256(material, name="cache identity")
        return PreprocessingCacheIdentity(
            key=key,
            schema_version=self.schema_version,
            implementation_id=self.implementation_id,
            config_sha256=self.config_sha256,
            row_sha256=row_sha256,
            view=normalized_view,
            source_path=str(source),
            source_size_bytes=int(source_stat.st_size),
            source_sha256=source_sha256,
        )

    def entry_paths(self, identity: PreprocessingCacheIdentity) -> PreprocessingCacheEntryPaths:
        """Return safe managed paths without creating the shard directory."""

        self._validate_identity(identity)
        namespace = self.config_sha256[:16]
        directory = self.root / namespace / identity.key[:2]
        self._assert_managed_path(directory)
        return PreprocessingCacheEntryPaths(
            shard=directory / f"{identity.key}.pkl",
            checksum=directory / f"{identity.key}.sha256",
        )

    def load(self, identity: PreprocessingCacheIdentity) -> Any | None:
        """Return a verified cached payload, or ``None`` when no entry exists.

        The checksum is verified before ``pickle.loads``. This detects damage,
        not malicious replacement by an actor who can also rewrite the sidecar.
        """

        paths = self.entry_paths(identity)
        shard_exists = _lexists(paths.shard)
        checksum_exists = _lexists(paths.checksum)
        if not shard_exists and not checksum_exists:
            return None
        if shard_exists != checksum_exists:
            raise PreprocessingCacheCorruptionError(
                f"cache entry is incomplete for key {identity.key}"
            )
        shard_bytes = self._read_managed_regular_file(paths.shard, max_bytes=self.max_entry_bytes)
        checksum_bytes = self._read_managed_regular_file(paths.checksum, max_bytes=1024)
        try:
            declared_sha256 = checksum_bytes.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise PreprocessingCacheCorruptionError(
                f"cache checksum is not ASCII for key {identity.key}"
            ) from exc
        if not _SHA256_PATTERN.fullmatch(declared_sha256):
            raise PreprocessingCacheCorruptionError(
                f"cache checksum has an invalid format for key {identity.key}"
            )
        actual_sha256 = hashlib.sha256(shard_bytes).hexdigest()
        if not hmac.compare_digest(declared_sha256, actual_sha256):
            raise PreprocessingCacheCorruptionError(
                f"cache SHA-256 mismatch for key {identity.key}"
            )
        try:
            envelope = pickle.loads(shard_bytes)  # noqa: S301 - explicit trusted-local boundary
        except Exception as exc:
            raise PreprocessingCacheCorruptionError(
                f"cache pickle cannot be decoded for key {identity.key}"
            ) from exc
        return self._validated_payload(envelope, identity)

    def store(self, identity: PreprocessingCacheIdentity, payload: Any) -> Path:
        """Atomically write a verified shard and checksum sidecar.

        The source image is hashed again just before serialization so a file
        changed between identity creation and preprocessing cannot be stored
        under the old content identity.
        """

        paths = self.entry_paths(identity)
        self._verify_source_unchanged(identity)
        _validate_payload(payload)
        envelope = {
            "magic": CACHE_MAGIC,
            "schema_version": self.schema_version,
            "implementation_id": self.implementation_id,
            "cache_key": identity.key,
            "config_sha256": identity.config_sha256,
            "row_sha256": identity.row_sha256,
            "view": identity.view,
            "source_path": identity.source_path,
            "source_size_bytes": identity.source_size_bytes,
            "source_sha256": identity.source_sha256,
            "payload": payload,
        }
        shard_bytes = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
        if len(shard_bytes) > self.max_entry_bytes:
            raise PreprocessingCacheConfigurationError(
                f"serialized cache entry exceeds max_entry_bytes={self.max_entry_bytes}"
            )
        shard_sha256 = hashlib.sha256(shard_bytes).hexdigest()
        self._create_managed_directory(paths.shard.parent)
        self._atomic_replace(paths.shard, shard_bytes)
        self._atomic_replace(paths.checksum, f"{shard_sha256}\n".encode("ascii"))
        return paths.shard

    def _validate_identity(self, identity: PreprocessingCacheIdentity) -> None:
        if not isinstance(identity, PreprocessingCacheIdentity):
            raise TypeError("identity must be a PreprocessingCacheIdentity")
        for name in ("key", "config_sha256", "row_sha256", "source_sha256"):
            value = str(getattr(identity, name))
            if not _SHA256_PATTERN.fullmatch(value):
                raise PreprocessingCacheSecurityError(f"identity {name} is not a SHA-256 digest")
        if identity.schema_version != self.schema_version:
            raise PreprocessingCacheSecurityError(
                "identity schema_version belongs to another cache"
            )
        if identity.implementation_id != self.implementation_id:
            raise PreprocessingCacheSecurityError(
                "identity implementation_id belongs to another cache"
            )
        if identity.config_sha256 != self.config_sha256:
            raise PreprocessingCacheSecurityError("identity config digest belongs to another cache")
        if identity.view not in {"front", "side"}:
            raise PreprocessingCacheSecurityError("identity view is invalid")

    def _verify_source_unchanged(self, identity: PreprocessingCacheIdentity) -> None:
        source = Path(identity.source_path)
        try:
            source_stat = source.stat()
        except OSError as exc:
            raise PreprocessingCacheSourceChangedError(
                f"source image is no longer readable: {source}"
            ) from exc
        if not stat.S_ISREG(source_stat.st_mode):
            raise PreprocessingCacheSourceChangedError(
                f"source image is no longer a regular file: {source}"
            )
        if int(source_stat.st_size) != identity.source_size_bytes:
            raise PreprocessingCacheSourceChangedError(
                f"source image size changed while preprocessing: {source}"
            )
        if not hmac.compare_digest(_sha256_file(source), identity.source_sha256):
            raise PreprocessingCacheSourceChangedError(
                f"source image content changed while preprocessing: {source}"
            )

    def _validated_payload(self, envelope: Any, identity: PreprocessingCacheIdentity) -> Any:
        if not isinstance(envelope, Mapping):
            raise PreprocessingCacheCorruptionError("cache envelope must be a mapping")
        expected = {
            "magic": CACHE_MAGIC,
            "schema_version": identity.schema_version,
            "implementation_id": identity.implementation_id,
            "cache_key": identity.key,
            "config_sha256": identity.config_sha256,
            "row_sha256": identity.row_sha256,
            "view": identity.view,
            "source_path": identity.source_path,
            "source_size_bytes": identity.source_size_bytes,
            "source_sha256": identity.source_sha256,
        }
        for name, expected_value in expected.items():
            if envelope.get(name) != expected_value:
                raise PreprocessingCacheCorruptionError(
                    f"cache envelope field {name!r} does not match identity {identity.key}"
                )
        if "payload" not in envelope:
            raise PreprocessingCacheCorruptionError("cache envelope has no payload")
        payload = envelope["payload"]
        _validate_payload(payload)
        return payload

    def _create_managed_directory(self, directory: Path) -> None:
        self._assert_managed_path(directory)
        relative = directory.relative_to(self.root)
        current = self.root
        for component in relative.parts:
            current = current / component
            try:
                mode = current.lstat().st_mode
            except FileNotFoundError:
                try:
                    current.mkdir(mode=0o700)
                except FileExistsError:
                    pass
                mode = current.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise PreprocessingCacheSecurityError(
                    f"cache managed directory is not a real directory: {current}"
                )
            self._assert_managed_path(current)

    def _assert_managed_path(self, path: Path) -> None:
        absolute = path if path.is_absolute() else self.root / path
        try:
            absolute.relative_to(self.root)
        except ValueError as exc:
            raise PreprocessingCacheSecurityError(
                f"cache path escapes managed root {self.root}: {absolute}"
            ) from exc
        # Resolve only the existing parent. Resolving the final entry could
        # silently follow precisely the symlink that this check must reject.
        parent = absolute if absolute == self.root else absolute.parent
        try:
            resolved_parent = parent.resolve(strict=parent.exists())
        except OSError as exc:
            raise PreprocessingCacheSecurityError(
                f"cache path parent cannot be resolved safely: {parent}"
            ) from exc
        try:
            resolved_parent.relative_to(self.root)
        except ValueError as exc:
            raise PreprocessingCacheSecurityError(
                f"cache path parent escapes managed root {self.root}: {resolved_parent}"
            ) from exc

    def _read_managed_regular_file(self, path: Path, *, max_bytes: int) -> bytes:
        self._assert_managed_path(path)
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError as exc:
            raise PreprocessingCacheCorruptionError(f"cache file disappeared: {path}") from exc
        if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
            raise PreprocessingCacheSecurityError(
                f"cache entry must be a non-symlink regular file: {path}"
            )
        flags = os.O_RDONLY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise PreprocessingCacheSecurityError(
                f"cache file could not be opened without following links: {path}"
            ) from exc
        try:
            file_stat = os.fstat(descriptor)
            if not stat.S_ISREG(file_stat.st_mode):
                raise PreprocessingCacheSecurityError(
                    f"cache entry changed type while opening: {path}"
                )
            if file_stat.st_size > max_bytes:
                raise PreprocessingCacheCorruptionError(
                    f"cache file exceeds allowed size {max_bytes}: {path}"
                )
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                return stream.read(max_bytes + 1)
        finally:
            os.close(descriptor)

    def _atomic_replace(self, destination: Path, content: bytes) -> None:
        self._assert_managed_path(destination)
        self._create_managed_directory(destination.parent)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(temporary_name)
        try:
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            else:
                # Windows has no fchmod; chmod still applies its supported
                # read/write protection to this just-created private file.
                os.chmod(temporary, stat.S_IREAD | stat.S_IWRITE)
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.close(descriptor)
            descriptor = -1
            self._assert_managed_path(temporary)
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _create_private_root(root: str | os.PathLike[str]) -> Path:
    requested = Path(root).expanduser().absolute()
    if _lexists(requested):
        mode = requested.lstat().st_mode
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
            raise PreprocessingCacheSecurityError(
                f"cache root must be a non-symlink directory: {requested}"
            )
    else:
        requested.mkdir(parents=True, mode=0o700)
    resolved = requested.resolve(strict=True)
    mode = resolved.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise PreprocessingCacheSecurityError(
            f"cache root must resolve to a real directory: {requested}"
        )
    return resolved


def _canonical_sha256(value: Any, *, name: str) -> str:
    canonical = _canonicalize(value, name=name)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonicalize(value: Any, *, name: str) -> Any:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PreprocessingCacheConfigurationError(f"{name} contains a non-finite float")
        return value
    if isinstance(value, np.generic):
        return _canonicalize(value.item(), name=name)
    if isinstance(value, np.ndarray):
        return _canonicalize(value.tolist(), name=name)
    if isinstance(value, os.PathLike):
        return str(Path(value))
    if isinstance(value, bytes):
        return {"__bytes_hex__": value.hex()}
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PreprocessingCacheConfigurationError(f"{name} mapping keys must be strings")
            result[key] = _canonicalize(item, name=f"{name}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [_canonicalize(item, name=name) for item in value]
    raise PreprocessingCacheConfigurationError(
        f"{name} contains unsupported value type {type(value).__name__}"
    )


def _validate_payload(value: Any, *, path: str = "payload") -> None:
    """Restrict locally-written payloads to ordinary data containers/tensors."""

    if value is None or isinstance(value, bool | int | float | str | bytes):
        return
    if isinstance(value, np.ndarray | np.generic | torch.Tensor):
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise PreprocessingCacheConfigurationError(f"{path} mapping keys must be strings")
            _validate_payload(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for index, item in enumerate(value):
            _validate_payload(item, path=f"{path}[{index}]")
        return
    raise PreprocessingCacheConfigurationError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lexists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "CACHE_MAGIC",
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_IMPLEMENTATION_ID",
    "PreprocessingCacheConfigurationError",
    "PreprocessingCacheCorruptionError",
    "PreprocessingCacheEntryPaths",
    "PreprocessingCacheError",
    "PreprocessingCacheIdentity",
    "PreprocessingCacheSecurityError",
    "PreprocessingCacheSourceChangedError",
    "TrustedLocalPreprocessingCache",
]

"""Release manifest: versions, sha256 digests and the manager compatibility gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from roboscout_manager import (
    APP_ASSET_NAME,
    APP_NAME,
    LEGACY_APP_ASSET_NAME,
    MANAGER_ASSET_NAME,
    MANAGER_VERSION,
    SETUP_ASSET_NAME,
)

SCHEMA_VERSION = 1
_CHUNK = 1024 * 1024


class ManifestError(ValueError):
    """Manifest is missing, malformed or does not describe a usable release."""


class VerificationError(RuntimeError):
    """A downloaded artifact does not match the manifest digest/size."""


# --- versions -------------------------------------------------------------------


def normalize_version(text: str) -> str:
    value = (text or "").strip()
    if value[:1] in {"v", "V"}:
        value = value[1:]
    return value


def version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in re.split(r"[^\d]+", normalize_version(version)):
        if chunk.isdigit():
            parts.append(int(chunk))
    return tuple(parts or [0])


def compare_versions(a: str, b: str) -> int:
    ta, tb = version_tuple(a), version_tuple(b)
    width = max(len(ta), len(tb))
    ta += (0,) * (width - len(ta))
    tb += (0,) * (width - len(tb))
    if ta > tb:
        return 1
    if ta < tb:
        return -1
    return 0


def is_newer(candidate: str, current: str) -> bool:
    return compare_versions(candidate, current) > 0


def looks_like_version(text: str) -> bool:
    return bool(re.match(r"^\d+(\.\d+)*", normalize_version(text)))


# --- model ----------------------------------------------------------------------


def infer_kind(name: str) -> str:
    lower = name.lower()
    if lower == SETUP_ASSET_NAME.lower() or "setup" in lower:
        return "setup"
    if lower == APP_ASSET_NAME.lower() or "-app" in lower:
        return "app"
    if lower == MANAGER_ASSET_NAME.lower() or "manager" in lower:
        return "manager"
    if lower == LEGACY_APP_ASSET_NAME.lower():
        return "legacy"
    if lower.endswith(".zip") or lower.endswith(".tar.gz"):
        return "archive"
    return "other"


def infer_platform(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".exe"):
        return "windows-x64"
    if "macos-arm64" in lower:
        return "macos-arm64"
    if "macos" in lower:
        return "macos-x64"
    if "linux" in lower:
        return "linux-x86_64"
    return "any"


@dataclass
class Artifact:
    name: str
    sha256: str
    size: int
    kind: str = "other"
    platform: str = "any"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("name")
        return data


@dataclass
class Manifest:
    version: str
    channel: str = ""
    min_manager_version: str = "0.0.0"
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    commit: str = ""
    created_at: str = ""
    product: str = APP_NAME
    schema: int = SCHEMA_VERSION
    notes: str = ""

    def artifact(self, name: str) -> Artifact | None:
        return self.artifacts.get(name)

    def _by_kind(self, kind: str, preferred: str) -> Artifact | None:
        hit = self.artifacts.get(preferred)
        if hit is not None:
            return hit
        for art in self.artifacts.values():
            if art.kind == kind:
                return art
        return None

    def app_artifact(self) -> Artifact | None:
        return self._by_kind("app", APP_ASSET_NAME)

    def manager_artifact(self) -> Artifact | None:
        return self._by_kind("manager", MANAGER_ASSET_NAME)

    def setup_artifact(self) -> Artifact | None:
        return self._by_kind("setup", SETUP_ASSET_NAME)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "product": self.product,
            "version": self.version,
            "channel": self.channel,
            "commit": self.commit,
            "created_at": self.created_at,
            "min_manager_version": self.min_manager_version,
            "notes": self.notes,
            "artifacts": {name: art.to_dict() for name, art in self.artifacts.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n"


def parse_manifest(source: str | bytes | dict[str, Any]) -> Manifest:
    if isinstance(source, (str, bytes)):
        try:
            data = json.loads(source)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"manifest is not valid JSON: {exc}") from exc
    else:
        data = source
    if not isinstance(data, dict):
        raise ManifestError("manifest must be a JSON object")
    version = normalize_version(str(data.get("version") or ""))
    if not looks_like_version(version):
        raise ManifestError(f"manifest version is missing or invalid: {data.get('version')!r}")
    schema = int(data.get("schema") or SCHEMA_VERSION)
    if schema > SCHEMA_VERSION:
        raise ManifestError(
            f"manifest schema {schema} is newer than this manager understands ({SCHEMA_VERSION}); update the manager first"
        )
    raw_arts = data.get("artifacts") or {}
    if not isinstance(raw_arts, dict):
        raise ManifestError("manifest artifacts must be an object keyed by file name")
    artifacts: dict[str, Artifact] = {}
    for name, raw in raw_arts.items():
        if not isinstance(raw, dict):
            raise ManifestError(f"artifact {name!r} must be an object")
        sha = str(raw.get("sha256") or "").strip().lower()
        if not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ManifestError(f"artifact {name!r} has an invalid sha256")
        try:
            size = int(raw.get("size") or 0)
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"artifact {name!r} has an invalid size") from exc
        artifacts[str(name)] = Artifact(
            name=str(name),
            sha256=sha,
            size=size,
            kind=str(raw.get("kind") or infer_kind(str(name))),
            platform=str(raw.get("platform") or infer_platform(str(name))),
        )
    return Manifest(
        version=version,
        channel=str(data.get("channel") or ""),
        min_manager_version=normalize_version(str(data.get("min_manager_version") or "0.0.0")) or "0.0.0",
        artifacts=artifacts,
        commit=str(data.get("commit") or ""),
        created_at=str(data.get("created_at") or ""),
        product=str(data.get("product") or APP_NAME),
        schema=schema,
        notes=str(data.get("notes") or ""),
    )


def load_manifest(path: Path) -> Manifest:
    try:
        return parse_manifest(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc


def manager_satisfies(manifest: Manifest, manager_version: str = MANAGER_VERSION) -> bool:
    """True when this manager is new enough to install the manifest's app."""
    return compare_versions(manager_version, manifest.min_manager_version) >= 0


# --- hashing --------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(_CHUNK)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, artifact: Artifact | None = None, *, sha256: str = "", size: int | None = None) -> str:
    """Verify ``path`` against an artifact (or explicit digest); return the digest."""
    path = Path(path)
    if not path.is_file():
        raise VerificationError(f"{path.name} is missing")
    expected_sha = (artifact.sha256 if artifact else sha256 or "").lower()
    expected_size = artifact.size if artifact else size
    actual_size = path.stat().st_size
    if expected_size and actual_size != int(expected_size):
        raise VerificationError(f"{path.name} size {actual_size} != expected {expected_size}")
    actual = sha256_file(path)
    if expected_sha and actual != expected_sha:
        raise VerificationError(f"{path.name} sha256 mismatch: {actual[:12]}… != {expected_sha[:12]}…")
    return actual


def build_manifest(
    files: list[Path],
    *,
    version: str,
    channel: str = "",
    min_manager_version: str = MANAGER_VERSION,
    commit: str = "",
    notes: str = "",
    created_at: str | None = None,
) -> Manifest:
    artifacts: dict[str, Artifact] = {}
    for raw in files:
        path = Path(raw)
        if not path.is_file():
            raise ManifestError(f"artifact file missing: {path}")
        artifacts[path.name] = Artifact(
            name=path.name,
            sha256=sha256_file(path),
            size=path.stat().st_size,
            kind=infer_kind(path.name),
            platform=infer_platform(path.name),
        )
    return Manifest(
        version=normalize_version(version),
        channel=channel,
        min_manager_version=normalize_version(min_manager_version) or "0.0.0",
        artifacts=artifacts,
        commit=commit,
        created_at=created_at if created_at is not None else datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=notes,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roboscout_manager.manifest", description="Build or verify a release manifest")
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="write manifest.json for the given artifact files")
    build.add_argument("--version", required=True)
    build.add_argument("--channel", default="")
    build.add_argument("--min-manager-version", default=MANAGER_VERSION)
    build.add_argument("--commit", default="")
    build.add_argument("--notes", default="")
    build.add_argument("--out", required=True)
    build.add_argument("files", nargs="+")
    verify = sub.add_parser("verify", help="verify files against a manifest")
    verify.add_argument("manifest")
    verify.add_argument("files", nargs="+")
    args = parser.parse_args(argv)

    if args.command == "build":
        manifest = build_manifest(
            [Path(f) for f in args.files],
            version=args.version,
            channel=args.channel,
            min_manager_version=args.min_manager_version,
            commit=args.commit,
            notes=args.notes,
        )
        Path(args.out).write_text(manifest.to_json(), encoding="utf-8")
        print(f"wrote {args.out} ({len(manifest.artifacts)} artifacts, version {manifest.version})")
        return 0

    manifest = load_manifest(Path(args.manifest))
    failures = 0
    for raw in args.files:
        path = Path(raw)
        art = manifest.artifact(path.name)
        if art is None:
            print(f"SKIP {path.name}: not in manifest")
            continue
        try:
            verify_file(path, art)
            print(f"OK   {path.name}")
        except VerificationError as exc:
            failures += 1
            print(f"FAIL {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

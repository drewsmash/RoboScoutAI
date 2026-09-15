from pathlib import Path

import pytest

from ramscout import updater


def test_normalize_version_strips_v_prefix():
    assert updater.normalize_version("v1.2.3") == "1.2.3"
    assert updater.normalize_version("0.4.0") == "0.4.0"


def test_is_newer_compares_semver_ish():
    assert updater.is_newer("0.4.1", "0.4.0")
    assert updater.is_newer("1.0.0", "0.9.9")
    assert not updater.is_newer("0.4.0", "0.4.0")
    assert not updater.is_newer("0.3.9", "0.4.0")


def test_version_tuple_ignores_junk():
    assert updater.version_tuple("v0.4.0-beta") == (0, 4, 0)


def test_preferred_assets_include_windows_or_mac(monkeypatch):
    monkeypatch.setattr(updater, "platform_key", lambda: "windows")
    assert "RamScoutAI-windows-x64.exe" in updater.preferred_asset_names()
    monkeypatch.setattr(updater, "platform_key", lambda: "macos-arm64")
    assert "RamScoutAI-macos-arm64.zip" in updater.preferred_asset_names()


def test_git_remote_env_and_legacy_repo(monkeypatch):
    monkeypatch.setenv("RAMSCOUT_GIT_REMOTE", "https://example.com/ram.git")
    monkeypatch.delenv("RAMSCOUT_GITHUB_REPO", raising=False)
    assert updater.git_remote() == "https://example.com/ram.git"
    monkeypatch.delenv("RAMSCOUT_GIT_REMOTE", raising=False)
    monkeypatch.setenv("RAMSCOUT_GITHUB_REPO", "acme/RamScoutAI")
    assert updater.git_remote().endswith("acme/RamScoutAI.git")


def test_check_source_up_to_date(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()

    def fake_git(args, *, cwd=None, timeout=60.0, check=True, env=None):
        if args[:2] == ["rev-parse", "HEAD"]:
            return "abc123"
        if args[:2] == ["rev-parse", "origin/main"]:
            return "abc123"
        if args[0] == "fetch":
            return ""
        if args[0] == "remote":
            return ""
        if args[0] == "show":
            return '__version__ = "0.4.2"\n'
        return ""

    monkeypatch.setattr(updater, "source_git_root", lambda: root)
    monkeypatch.setattr(updater, "is_frozen", lambda: False)
    monkeypatch.setattr(updater, "_run_git", fake_git)
    monkeypatch.setattr(updater.shutil, "which", lambda _cmd: "/usr/bin/git")
    info = updater.check_for_update("0.4.2")
    assert info.available is False
    assert info.message == "up to date"
    assert info.mode == "source"
    assert info.local_sha == "abc123"


def test_check_source_update_available(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()

    def fake_git(args, *, cwd=None, timeout=60.0, check=True, env=None):
        if args[:2] == ["rev-parse", "HEAD"]:
            return "aaa111"
        if args[:2] == ["rev-parse", "origin/main"]:
            return "bbb222"
        if args[0] == "fetch":
            return ""
        if args[0] == "remote":
            return ""
        if args[0] == "show":
            return '__version__ = "0.4.3"\n'
        return ""

    monkeypatch.setattr(updater, "source_git_root", lambda: root)
    monkeypatch.setattr(updater, "is_frozen", lambda: False)
    monkeypatch.setattr(updater, "_run_git", fake_git)
    monkeypatch.setattr(updater.shutil, "which", lambda _cmd: "/usr/bin/git")
    info = updater.check_for_update("0.4.2")
    assert info.available is True
    assert info.message == "update available from git"
    assert info.can_apply is True
    assert info.remote_sha == "bbb222"
    assert "api.github.com" not in (info.error or "")
    assert "releases" not in (info.release_url or "").lower() or "github.com" in info.release_url


def test_check_git_remote_unreachable(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()

    def boom(*_a, **_k):
        raise RuntimeError("could not resolve host")

    monkeypatch.setattr(updater, "source_git_root", lambda: root)
    monkeypatch.setattr(updater, "is_frozen", lambda: False)
    monkeypatch.setattr(updater, "_run_git", boom)
    monkeypatch.setattr(updater.shutil, "which", lambda _cmd: "/usr/bin/git")
    info = updater.check_for_update("0.4.2")
    assert info.available is False
    assert "git remote unreachable" in (info.error or "").lower()
    assert info.message == "git remote unreachable"


def test_frozen_update_finds_artifact(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    mirror = cache / "mirror"
    mirror.mkdir(parents=True)
    (mirror / ".git").mkdir()
    artifact_dir = mirror / "desktop-downloads"
    artifact_dir.mkdir()
    exe = artifact_dir / "RamScoutAI-windows-x64.exe"
    exe.write_bytes(b"MZ-fake")

    def fake_git(args, *, cwd=None, timeout=60.0, check=True, env=None):
        if args[:1] == ["rev-parse"]:
            return "ccc333"
        if args[0] == "fetch":
            return ""
        if args[0] == "remote":
            return ""
        if args[0] == "show":
            return '__version__ = "0.4.9"\n'
        if args[0] == "ls-tree":
            return "desktop-downloads/RamScoutAI-windows-x64.exe"
        if args[0] == "cat-file":
            return ""
        if args[0] == "clone":
            return ""
        return ""

    monkeypatch.setattr(updater, "source_git_root", lambda: None)
    monkeypatch.setattr(updater, "is_frozen", lambda: True)
    monkeypatch.setattr(updater, "update_cache_dir", lambda: cache)
    monkeypatch.setattr(updater, "platform_key", lambda: "windows")
    monkeypatch.setattr(updater, "_read_installed_sha", lambda: "oldsha")
    monkeypatch.setattr(updater, "_ensure_mirror", lambda *a, **k: None)
    monkeypatch.setattr(updater, "_run_git", fake_git)
    monkeypatch.setattr(updater.shutil, "which", lambda _cmd: "/usr/bin/git")
    info = updater.check_for_update("0.4.2")
    assert info.available is True
    assert info.mode == "frozen"
    assert info.message == "update available from git"
    assert info.asset_name == "RamScoutAI-windows-x64.exe"
    assert info.asset_url.startswith("git:")


def test_no_releases_api_usage():
    source = Path(updater.__file__).read_text(encoding="utf-8")
    assert "api.github.com" not in source
    assert "/releases" not in source or "GitHub Releases" not in source
    assert "RAMSCOUT_GIT_REMOTE" in source
    assert "release-artifacts/" in source


def test_artifact_paths_prefer_release_artifacts():
    names = {Path(p).name for p in updater._ARTIFACT_REL_PATHS}
    assert "RamScoutAI-windows-x64.exe" in names
    assert updater._ARTIFACT_REL_PATHS[0].startswith("release-artifacts/")


def test_git_branch_bundled_channel(monkeypatch, tmp_path):
    channel = tmp_path / "update-channel.txt"
    channel.write_text("cursor/feature-e0ef\n", encoding="utf-8")
    monkeypatch.delenv("RAMSCOUT_GIT_BRANCH", raising=False)
    monkeypatch.setattr(updater, "app_dir", lambda: tmp_path)
    monkeypatch.setattr("ramscout.paths.bundle_root", lambda: tmp_path / "missing")
    assert updater.git_branch() == "cursor/feature-e0ef"
    monkeypatch.setenv("RAMSCOUT_GIT_BRANCH", "main")
    assert updater.git_branch() == "main"

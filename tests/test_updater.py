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
    assert "RoboScoutAI-windows-x64.exe" in updater.preferred_asset_names()
    assert all("RamScout" not in n for n in updater.preferred_asset_names())
    monkeypatch.setattr(updater, "platform_key", lambda: "macos-arm64")
    assert "RoboScoutAI-macos-arm64.zip" in updater.preferred_asset_names()


def test_git_remote_env_and_legacy_repo(monkeypatch):
    monkeypatch.setenv("RAMSCOUT_GIT_REMOTE", "https://example.com/ram.git")
    monkeypatch.delenv("RAMSCOUT_GITHUB_REPO", raising=False)
    assert updater.git_remote() == "https://example.com/ram.git"
    monkeypatch.delenv("RAMSCOUT_GIT_REMOTE", raising=False)
    monkeypatch.setenv("RAMSCOUT_GITHUB_REPO", "acme/RoboScoutAI")
    assert updater.git_remote().endswith("acme/RoboScoutAI.git")


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
    artifact_dir = mirror / "release-artifacts"
    artifact_dir.mkdir()
    exe = artifact_dir / "RoboScoutAI-windows-x64.exe"
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
            return "release-artifacts/RoboScoutAI-windows-x64.exe"
        if args[0] == "cat-file":
            rel = args[-1].split(":", 1)[-1]
            if "RoboScoutAI-windows-x64.exe" in rel:
                return ""
            raise RuntimeError("missing")
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
    assert info.asset_name == "RoboScoutAI-windows-x64.exe"
    assert info.asset_url.startswith("git:")
    # Browse URL only — Releases download pins are emergency-only on apply failure.
    assert "releases/download" not in (info.release_url or "").lower()


def test_no_releases_api_usage():
    source = Path(updater.__file__).read_text(encoding="utf-8")
    # Git updater must not call the GitHub Releases REST API; browser download
    # URLs for Access Denied fallback are fine.
    assert "api.github.com" not in source
    assert "httpx" not in source
    assert "RAMSCOUT_GIT_REMOTE" in source
    assert "release-artifacts/" in source


def test_artifact_paths_prefer_release_artifacts():
    names = {Path(p).name for p in updater._ARTIFACT_REL_PATHS}
    assert "RoboScoutAI-windows-x64.exe" in names
    assert "RoboScoutAI-windows-x64.exe" in names
    assert all("RamScout" not in n for n in names)
    assert updater._ARTIFACT_REL_PATHS[0].startswith("release-artifacts/")


def test_manual_download_url_points_at_github_cdn(monkeypatch):
    monkeypatch.setattr(updater, "platform_key", lambda: "windows")
    monkeypatch.setattr(updater, "github_repo", lambda: "drewsmash/RoboScoutAI")
    url = updater.manual_download_url(version="0.5.1", asset_name="RoboScoutAI-windows-x64.exe")
    assert url == (
        "https://github.com/drewsmash/RoboScoutAI/releases/download/"
        "v0.5.1/RoboScoutAI-windows-x64.exe"
    )
    # Empty / non-semver must not invent a stale pin — use /releases/latest.
    assert updater.manual_download_url(version="") == (
        "https://github.com/drewsmash/RoboScoutAI/releases/latest"
    )
    assert updater.manual_download_url(version="deadbeef") == (
        "https://github.com/drewsmash/RoboScoutAI/releases/latest"
    )


def test_should_open_manual_fallback_contract():
    assert updater.should_open_manual_fallback({"ok": False, "open_url": "https://x"}) is True
    assert updater.should_open_manual_fallback({"ok": True, "open_url": "https://x"}) is False
    assert updater.should_open_manual_fallback({"ok": True, "restarting": True, "open_url": "https://x"}) is False
    assert updater.should_open_manual_fallback({"ok": False}) is False
    assert updater.should_open_manual_fallback({"ok": False, "open_url": ""}) is False


def test_apply_update_now_success_omits_open_url(monkeypatch, tmp_path):
    info = updater.UpdateInfo(
        available=True,
        current_version="0.5.2",
        latest_version="0.5.6",
        remote_sha="newsha",
        local_sha="oldsha",
        can_apply=True,
        mode="frozen",
        frozen=True,
        asset_name="RoboScoutAI-windows-x64.exe",
        release_url="https://github.com/drewsmash/RoboScoutAI/releases/download/v0.5.6/RoboScoutAI-windows-x64.exe",
    )
    pkg = tmp_path / "RoboScoutAI-windows-x64.exe"
    pkg.write_bytes(b"MZ")

    monkeypatch.setattr(updater, "check_for_update", lambda: info)
    monkeypatch.setattr(updater, "download_update", lambda _info: pkg)
    monkeypatch.setattr(
        updater,
        "apply_downloaded_update",
        lambda _pkg: "Updating from git — installing into %LOCALAPPDATA%\\RoboScoutAI\\RoboScoutAI.exe.",
    )
    monkeypatch.setattr(updater, "is_frozen", lambda: True)
    monkeypatch.setattr(updater, "last_check", lambda: info.as_dict())

    result = updater.apply_update_now()
    assert result["ok"] is True
    assert result["restarting"] is True
    assert "open_url" not in result
    assert "git" in result["message"].lower() or "LOCALAPPDATA" in result["message"]
    assert updater.should_open_manual_fallback(result) is False


def test_apply_update_now_failure_prefers_local_package(monkeypatch, tmp_path):
    info = updater.UpdateInfo(
        available=True,
        current_version="0.5.2",
        latest_version="0.5.6",
        remote_sha="newsha",
        can_apply=True,
        mode="frozen",
        frozen=True,
        asset_name="RoboScoutAI-windows-x64.exe",
    )
    pkg = tmp_path / "packages" / "RoboScoutAI-windows-x64.exe"
    pkg.parent.mkdir(parents=True)
    pkg.write_bytes(b"MZ")
    revealed = {}

    monkeypatch.setattr(updater, "check_for_update", lambda: info)
    monkeypatch.setattr(updater, "download_update", lambda _info: pkg)
    monkeypatch.setattr(
        updater,
        "apply_downloaded_update",
        lambda _pkg: (_ for _ in ()).throw(RuntimeError("Access is denied")),
    )
    monkeypatch.setattr(updater, "_reveal_path", lambda path: revealed.setdefault("path", path) or True)
    monkeypatch.setattr(updater, "update_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(updater, "_read_pending_update", lambda: None)

    result = updater.apply_update_now()
    assert result["ok"] is False
    assert result["restarting"] is False
    assert result.get("open_url", "").startswith("file:")
    assert revealed.get("path") == pkg
    assert updater.should_open_manual_fallback(result) is True
    # Must not fall back to an outdated Releases pin when a local package exists.
    assert "releases/download/v0.5.0" not in (result.get("open_url") or "")


def test_apply_update_now_up_to_date_no_open_url(monkeypatch):
    info = updater.UpdateInfo(
        available=False,
        current_version="0.5.6",
        latest_version="0.5.6",
        message="up to date",
        mode="frozen",
    )
    monkeypatch.setattr(updater, "check_for_update", lambda: info)
    result = updater.apply_update_now()
    assert result["ok"] is True
    assert result["restarting"] is False
    assert "open_url" not in result


def test_windows_user_install_dir_uses_localappdata(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    path = updater._windows_user_install_dir()
    assert path == tmp_path / "Local" / "RoboScoutAI"


def test_classify_rejects_downgrade():
    assert (
        updater.classify_update(
            current_version="0.4.4",
            remote_version="0.4.2",
            local_sha="",
            remote_sha="bbbbbbb",
        )
        == "ahead"
    )
    assert (
        updater.classify_update(
            current_version="0.4.4",
            remote_version="0.4.4",
            local_sha="",
            remote_sha="bbbbbbb",
        )
        == "up_to_date"
    )
    assert (
        updater.classify_update(
            current_version="0.4.2",
            remote_version="0.4.4",
            local_sha="old",
            remote_sha="new",
        )
        == "available"
    )


def test_classify_semver_beats_matching_sha():
    # Failed Windows apply used to pin remote SHA while still on the old exe.
    assert (
        updater.classify_update(
            current_version="0.5.2",
            remote_version="0.5.4",
            local_sha="samesha",
            remote_sha="samesha",
        )
        == "available"
    )


def test_apply_windows_does_not_write_sha_immediately(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    pkg = tmp_path / "RoboScoutAI-windows-x64.exe"
    pkg.write_bytes(b"MZ-fake-exe")
    written = {}

    monkeypatch.setattr(updater.platform, "system", lambda: "Windows")
    monkeypatch.setattr(updater, "is_frozen", lambda: True)
    monkeypatch.setattr(updater, "last_check", lambda: {"remote_sha": "deadbeef"})
    monkeypatch.setattr(updater, "_write_installed_sha", lambda sha: written.setdefault("sha", sha))
    monkeypatch.setattr(updater.subprocess, "Popen", lambda *a, **k: None)
    monkeypatch.setattr(updater.threading, "Timer", lambda *a, **k: type("T", (), {"start": lambda self: None})())
    monkeypatch.setattr(updater.sys, "executable", str(tmp_path / "old" / "RoboScoutAI.exe"))

    msg = updater.apply_downloaded_update(pkg)
    assert "SmartScreen" in msg or "unsigned" in msg.lower()
    assert "sha" not in written
    pending = updater._read_pending_update()
    assert pending and pending["sha"] == "deadbeef"


def test_windows_update_bat_hardening_invariants():
    """Bat must not use premature self-delete / move-on-locked-exe patterns."""
    lines = updater._windows_update_bat_lines(
        pid=4242,
        src=r"C:\Temp\pkg.exe",
        dst=r"C:\Users\x\AppData\Local\RoboScoutAI\RoboScoutAI.exe",
        portable=r"C:\Users\x\Downloads\RoboScoutAI.exe",
        sha_path=r"C:\Users\x\AppData\Local\RoboScoutAI\update-git-sha.txt",
        log_path=r"C:\Users\x\AppData\Local\RoboScoutAI\update.log",
        sha="abc123",
        install_dir=r"C:\Users\x\AppData\Local\RoboScoutAI",
    )
    text = "\r\n".join(lines)
    lower = text.lower()

    # Package install uses copy / Copy-Item — never move package→target.
    assert "Copy-Item" in text
    assert "copy /y" in lower
    assert "move /y" not in lower
    assert "move " not in lower

    # Reliable PID wait + settle; no flaky tasklist/findstr loop.
    assert "Wait-Process" in text
    assert "tasklist" not in lower
    assert "findstr" not in lower

    # Logging + SmartScreen MOTW unblock + locked-target .new replace.
    assert "update.log" in text
    assert "Unblock-File" in text
    assert ".new" in text
    assert "TARGET_LOCKED" in text
    assert "COPY_FAILED" in text
    assert 'explorer.exe /select,"%SRC%"' in text

    # SHA only after successful copy block (appears after COPY_OK / replace).
    sha_idx = text.index('>"%SHAFILE%" echo %SHA%')
    copy_idx = text.index("Copy-Item")
    assert copy_idx < sha_idx

    # Self-delete must be last actionable line via safe goto pattern.
    non_empty = [ln for ln in lines if ln.strip()]
    assert non_empty[-1] == '(goto) 2>nul & del "%~f0"'
    # No bare mid-script del of self before work finishes.
    premature = [
        i
        for i, ln in enumerate(lines)
        if 'del "%~f0"' in ln and "(goto)" not in ln
    ]
    assert premature == []


def test_apply_windows_writes_hardened_bat(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    pkg = tmp_path / "RoboScoutAI-windows-x64.exe"
    pkg.write_bytes(b"MZ-fake-exe")
    captured: dict = {}

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["creationflags"] = kwargs.get("creationflags", 0)
        return None

    monkeypatch.setattr(updater.platform, "system", lambda: "Windows")
    monkeypatch.setattr(updater, "is_frozen", lambda: True)
    monkeypatch.setattr(updater, "last_check", lambda: {"remote_sha": "cafebabe"})
    monkeypatch.setattr(updater.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(updater.threading, "Timer", lambda *a, **k: type("T", (), {"start": lambda self: None})())
    monkeypatch.setattr(updater.sys, "executable", str(tmp_path / "old" / "RoboScoutAI.exe"))
    monkeypatch.setattr(updater.tempfile, "gettempdir", lambda: str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()

    updater.apply_downloaded_update(pkg)
    bat = tmp_path / "tmp" / "roboscout_update.bat"
    assert bat.is_file()
    body = bat.read_text(encoding="utf-8")
    assert "Wait-Process" in body
    assert "Copy-Item" in body
    assert "Unblock-File" in body
    assert "update.log" in body
    assert body.rstrip().endswith('(goto) 2>nul & del "%~f0"')
    assert "move /Y" not in body
    # CREATE_NO_WINDOW (0x08000000) so success path stays silent.
    assert captured["creationflags"] & 0x08000000
    assert "cmd.exe" in captured["args"][0]


def test_frozen_does_not_offer_older_main(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    mirror = cache / "mirror"
    mirror.mkdir(parents=True)
    (mirror / ".git").mkdir()
    written = {}

    def fake_git(args, *, cwd=None, timeout=60.0, check=True, env=None):
        if args[:1] == ["rev-parse"]:
            return "mainsha0"
        if args[0] == "show":
            return '__version__ = "0.4.2"\n'
        return ""

    monkeypatch.setattr(updater, "source_git_root", lambda: None)
    monkeypatch.setattr(updater, "is_frozen", lambda: True)
    monkeypatch.setattr(updater, "update_cache_dir", lambda: cache)
    monkeypatch.setattr(updater, "_read_installed_sha", lambda: "")
    monkeypatch.setattr(updater, "_write_installed_sha", lambda sha: written.setdefault("sha", sha))
    monkeypatch.setattr(updater, "_ensure_mirror", lambda *a, **k: None)
    monkeypatch.setattr(updater, "_run_git", fake_git)
    monkeypatch.setattr(updater.shutil, "which", lambda _cmd: "/usr/bin/git")
    info = updater.check_for_update("0.4.4")
    assert info.available is False
    assert info.message == "up to date"
    assert info.can_apply is False
    assert info.latest_version == "0.4.2"
    assert written.get("sha") == "mainsha0"


def test_frozen_no_binary_not_nagging(monkeypatch, tmp_path):
    cache = tmp_path / "cache"
    mirror = cache / "mirror"
    mirror.mkdir(parents=True)
    (mirror / ".git").mkdir()

    def fake_git(args, *, cwd=None, timeout=60.0, check=True, env=None):
        if args[:1] == ["rev-parse"]:
            return "newsha99"
        if args[0] == "show":
            return '__version__ = "0.4.5"\n'
        if args[0] == "ls-tree":
            return ""
        if args[0] == "cat-file":
            raise RuntimeError("missing")
        return ""

    monkeypatch.setattr(updater, "source_git_root", lambda: None)
    monkeypatch.setattr(updater, "is_frozen", lambda: True)
    monkeypatch.setattr(updater, "update_cache_dir", lambda: cache)
    monkeypatch.setattr(updater, "platform_key", lambda: "windows")
    monkeypatch.setattr(updater, "_read_installed_sha", lambda: "oldsha")
    monkeypatch.setattr(updater, "_ensure_mirror", lambda *a, **k: None)
    monkeypatch.setattr(updater, "_find_remote_artifact", lambda *a, **k: None)
    monkeypatch.setattr(updater, "_run_git", fake_git)
    monkeypatch.setattr(updater.shutil, "which", lambda _cmd: "/usr/bin/git")
    info = updater.check_for_update("0.4.4")
    assert info.available is False
    assert info.can_apply is False
    assert "release-artifacts" in (info.error or "")


def test_git_branch_bundled_channel(monkeypatch, tmp_path):
    channel = tmp_path / "update-channel.txt"
    channel.write_text("cursor/feature-e0ef\n", encoding="utf-8")
    monkeypatch.delenv("RAMSCOUT_GIT_BRANCH", raising=False)
    monkeypatch.setattr(updater, "app_dir", lambda: tmp_path)
    monkeypatch.setattr("ramscout.paths.bundle_root", lambda: tmp_path / "missing")
    assert updater.git_branch() == "cursor/feature-e0ef"
    monkeypatch.setenv("RAMSCOUT_GIT_BRANCH", "main")
    assert updater.git_branch() == "main"

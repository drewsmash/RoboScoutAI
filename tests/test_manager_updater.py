import hashlib
import json
import os
from pathlib import Path

import pytest

from roboscout_manager import MANAGER_VERSION, install, ipc, state, updater
from roboscout_manager.manifest import Manifest, build_manifest


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOSCOUT_INSTALL_ROOT", str(tmp_path / "RoboScoutAI"))
    monkeypatch.delenv("ROBOSCOUT_CHANNEL", raising=False)
    r = state.default_install_root()
    r.mkdir(parents=True)
    return r


class FakeChannel:
    """Stands in for GitChannel / ReleasesChannel."""

    name = "fake"

    def __init__(self, manifest: Manifest | None, files: dict[str, bytes], *, sha="deadbeef", fail_fetch=False, browse="https://example/tree/x"):
        self.manifest = manifest
        self.files = files
        self.sha = sha
        self.fail_fetch = fail_fetch
        self.downloads: list[str] = []
        self._browse = browse

    def fetch(self, **_kw):
        if self.fail_fetch:
            raise updater.ChannelError("offline")
        return self.sha

    def read_manifest(self):
        return self.manifest

    def read_source_version(self):
        return "0.9.9"

    def download(self, name, dest, progress=lambda *_a: None, *, expected_size=0):
        self.downloads.append(name)
        if name not in self.files:
            raise updater.ChannelError(f"no {name}")
        Path(dest).write_bytes(self.files[name])
        progress("downloading", 50, "half")
        return Path(dest)

    def browse_url(self):
        return self._browse


def _release(tmp_path, version, *, min_manager=MANAGER_VERSION, with_manager=True, channel="main"):
    d = tmp_path / f"rel-{version}"
    d.mkdir()
    app = d / "RoboScoutAI-app-windows-x64.exe"
    app.write_bytes(f"APP {version} ".encode() + os.urandom(128))
    files = [app]
    if with_manager:
        mgr = d / "RoboScoutAI.exe"
        mgr.write_bytes(f"MGR {version} ".encode() + os.urandom(32))
        files.append(mgr)
    manifest = build_manifest(files, version=version, channel=channel, min_manager_version=min_manager)
    blobs = {p.name: p.read_bytes() for p in files}
    return manifest, blobs


def _install(root, tmp_path, version):
    p = tmp_path / f"seed-{version}.exe"
    p.write_bytes(f"SEED {version}".encode())
    return install.install_app_payload(root, p, version, sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def test_check_uses_git_channel_first(root, tmp_path):
    _install(root, tmp_path, "0.6.0")
    manifest, blobs = _release(tmp_path, "0.6.1")
    git = FakeChannel(manifest, blobs)
    rel = FakeChannel(None, {}, fail_fetch=True)
    check = updater.check_for_update(root, git_factory=lambda c, r: git, releases_factory=lambda c: rel)
    assert check.available and check.can_apply
    assert check.source == "git"
    assert check.latest_version == "0.6.1"
    assert check.current_version == "0.6.0"
    assert check.remote_sha == "deadbeef"
    assert not check.needs_manager_update
    d = check.as_dict()
    assert d["mode"] == "managed" and d["asset_name"] == "RoboScoutAI-app-windows-x64.exe"
    assert d["branch"] == state.read_config(root).channel


def test_check_falls_back_to_releases_when_git_unreachable(root, tmp_path):
    _install(root, tmp_path, "0.6.0")
    manifest, blobs = _release(tmp_path, "0.6.2")
    git = FakeChannel(None, {}, fail_fetch=True)
    rel = FakeChannel(manifest, blobs, sha="rel-sha", browse="https://github.com/x/y/releases/tag/v0.6.2")
    check = updater.check_for_update(root, git_factory=lambda c, r: git, releases_factory=lambda c: rel)
    assert check.available and check.source == "releases"
    assert check.errors and check.errors[0].startswith("git:")
    assert check.release_url.endswith("v0.6.2")


def test_check_git_branch_without_manifest_reports_error_then_falls_back(root, tmp_path):
    _install(root, tmp_path, "0.6.0")
    git = FakeChannel(None, {})  # fetch ok, no manifest on branch
    rel = FakeChannel(None, {}, fail_fetch=True)
    check = updater.check_for_update(root, git_factory=lambda c, r: git, releases_factory=lambda c: rel)
    assert not check.available
    assert check.manifest is None
    assert "no release-artifacts/manifest.json" in check.error
    assert "source version 0.9.9" in check.error
    assert check.message == "update channel unreachable"


def test_check_respects_releases_fallback_flag(root, tmp_path):
    _install(root, tmp_path, "0.6.0")
    cfg = state.read_config(root)
    cfg.releases_fallback = False
    state.write_config(root, cfg)
    manifest, blobs = _release(tmp_path, "0.6.2")
    called = []

    def rel_factory(c):
        called.append(1)
        return FakeChannel(manifest, blobs)

    check = updater.check_for_update(root, git_factory=lambda c, r: FakeChannel(None, {}, fail_fetch=True), releases_factory=rel_factory)
    assert not called and check.manifest is None


def test_check_up_to_date_and_ahead(root, tmp_path):
    _install(root, tmp_path, "0.6.1")
    manifest, blobs = _release(tmp_path, "0.6.1")
    check = updater.check_for_update(root, git_factory=lambda c, r: FakeChannel(manifest, blobs))
    assert not check.available and check.message == "up to date"
    older, blobs2 = _release(tmp_path, "0.6.0")
    check = updater.check_for_update(root, git_factory=lambda c, r: FakeChannel(older, blobs2))
    assert not check.available and "newer than channel" in check.message


def test_check_flags_manager_gate(root, tmp_path):
    _install(root, tmp_path, "0.6.0")
    manifest, blobs = _release(tmp_path, "0.7.0", min_manager="9.0.0")
    check = updater.check_for_update(root, git_factory=lambda c, r: FakeChannel(manifest, blobs))
    assert check.available and check.needs_manager_update
    assert "manager" in check.message
    assert check.as_dict()["min_manager_version"] == "9.0.0"


def test_perform_update_installs_side_by_side_and_prunes(root, tmp_path):
    _install(root, tmp_path, "0.5.9")
    _install(root, tmp_path, "0.6.0")
    manifest, blobs = _release(tmp_path, "0.6.1")
    chan = FakeChannel(manifest, blobs)
    events = []
    check = updater.check_for_update(root, git_factory=lambda c, r: chan)
    result = updater.perform_update(root, check, progress=lambda s, p, m: events.append((s, p)))
    assert result.ok and result.installed_version == "0.6.1"
    cur = state.read_current(root)
    assert cur.version == "0.6.1" and cur.previous_version == "0.6.0" and cur.source == "git"
    assert state.app_exe_path(root, "0.6.1").read_bytes() == blobs["RoboScoutAI-app-windows-x64.exe"]
    assert state.app_exe_path(root, "0.6.0").is_file()  # kept for rollback
    assert not state.app_exe_path(root, "0.5.9").exists()  # pruned (older than 2)
    assert chan.downloads == ["RoboScoutAI-app-windows-x64.exe"]
    states = [s for s, _ in events]
    assert states[-1] == "done"
    assert "downloading" in states and "verifying" in states and "installing" in states
    status = ipc.read_status(root)
    assert status.state == "done" and status.progress == 100 and status.version == "0.6.1"
    assert not list((root / "app").rglob("*.part"))


def test_perform_update_rejects_corrupt_download(root, tmp_path):
    _install(root, tmp_path, "0.6.0")
    manifest, blobs = _release(tmp_path, "0.6.1")
    blobs["RoboScoutAI-app-windows-x64.exe"] = b"corrupted bytes"
    check = updater.check_for_update(root, git_factory=lambda c, r: FakeChannel(manifest, blobs))
    result = updater.perform_update(root, check)
    assert not result.ok and "mismatch" in result.error or "size" in result.error
    assert state.read_current(root).version == "0.6.0"
    assert not state.app_exe_path(root, "0.6.1").exists()
    assert ipc.read_status(root).state == "error"
    assert not list((root / "app").rglob("*.part"))


def test_perform_update_noop_when_up_to_date(root, tmp_path):
    _install(root, tmp_path, "0.6.1")
    manifest, blobs = _release(tmp_path, "0.6.1")
    chan = FakeChannel(manifest, blobs)
    check = updater.check_for_update(root, git_factory=lambda c, r: chan)
    result = updater.perform_update(root, check)
    assert result.ok and chan.downloads == []
    assert ipc.read_status(root).state == "done"


def test_manager_gate_stages_manager_binary_and_swaps_when_not_locked(root, tmp_path):
    _install(root, tmp_path, "0.6.0")
    (root / "RoboScoutAI.exe").write_bytes(b"old manager")
    manifest, blobs = _release(tmp_path, "0.7.0", min_manager="9.0.0")
    chan = FakeChannel(manifest, blobs)
    check = updater.check_for_update(root, git_factory=lambda c, r: chan)
    result = updater.perform_update(root, check, spawn_swap=False)
    # In tests we are not running from root/RoboScoutAI.exe, so the swap is immediate.
    assert result.ok and not result.manager_update_scheduled
    assert chan.downloads == ["RoboScoutAI.exe"]
    assert (root / "RoboScoutAI.exe").read_bytes() == blobs["RoboScoutAI.exe"]
    assert not (root / "RoboScoutAI.exe.new").exists()
    assert state.read_current(root).version == "0.6.0"  # app untouched until manager restarts


def test_manager_gate_without_manager_artifact_errors(root, tmp_path):
    _install(root, tmp_path, "0.6.0")
    manifest, blobs = _release(tmp_path, "0.7.0", min_manager="9.0.0", with_manager=False)
    check = updater.check_for_update(root, git_factory=lambda c, r: FakeChannel(manifest, blobs))
    result = updater.perform_update(root, check, spawn_swap=False)
    assert not result.ok and "Setup" in result.message


def test_manager_gate_schedules_swap_when_running_from_install(root, tmp_path, monkeypatch):
    _install(root, tmp_path, "0.6.0")
    target = root / "RoboScoutAI.exe"
    target.write_bytes(b"running manager")
    monkeypatch.setattr(updater, "running_exe", lambda: target)
    manifest, blobs = _release(tmp_path, "0.7.0", min_manager="9.0.0")
    check = updater.check_for_update(root, git_factory=lambda c, r: FakeChannel(manifest, blobs))
    result = updater.perform_update(root, check, spawn_swap=False, relaunch_after_manager_swap=["--update", "--launch-after"])
    assert result.ok and result.manager_update_scheduled
    assert (root / "RoboScoutAI.exe.new").read_bytes() == blobs["RoboScoutAI.exe"]
    assert target.read_bytes() == b"running manager"
    assert ipc.read_status(root).state == "manager-update"


def test_manager_update_bat_lines():
    lines = updater.manager_update_bat_lines(
        pid=77, staged=r"C:\x\RoboScoutAI.exe.new", target=r"C:\x\RoboScoutAI.exe", log_path=r"C:\x\update.log", relaunch_args=["--update", "--launch-after"]
    )
    text = "\n".join(lines)
    assert "set PID=77" in text and "Wait-Process" in text
    assert 'copy /Y "%SRC%" "%DST%"' in text
    assert ":retry" in text
    assert 'start "" "%DST%" --update --launch-after' in text
    assert lines[-2] == '(goto) 2>nul & del "%~f0"'


def test_swap_staged_manager_applies_when_not_running_from_target(root):
    (root / "RoboScoutAI.exe").write_bytes(b"old")
    (root / "RoboScoutAI.exe.new").write_bytes(b"new")
    assert install.swap_staged_manager(root)
    assert (root / "RoboScoutAI.exe").read_bytes() == b"new"
    assert not (root / "RoboScoutAI.exe.new").exists()


def test_channel_switch_persists_and_validates(root):
    cfg = state.set_channel(root, "release/2027")
    assert cfg.channel == "release/2027"
    assert state.read_config(root).channel == "release/2027"
    assert json.loads(state.manager_json_path(root).read_text())["channel"] == "release/2027"
    state.set_channel(root, "v0.7.0")
    assert state.read_config(root).channel == "v0.7.0"
    for bad in ("", "  ", "bad name", "-flag", "a:b", "x^y"):
        with pytest.raises(ValueError):
            state.set_channel(root, bad)


def test_channel_env_override_and_bundled_default(root, monkeypatch):
    cfg = state.read_config(root)
    assert cfg.channel == state.bundled_channel()  # release-artifacts/update-channel.txt in the repo
    monkeypatch.setenv("ROBOSCOUT_CHANNEL", "feature/x")
    assert state.read_config(root).channel == "feature/x"


def test_repair_reinstalls_channel_build_even_when_same_version(root, tmp_path, monkeypatch):
    _install(root, tmp_path, "0.6.1")
    manifest, blobs = _release(tmp_path, "0.6.1")
    state.app_exe_path(root, "0.6.1").write_bytes(b"damaged")
    assert install.verify_current(root)[1] != ""
    monkeypatch.setattr(updater, "GitChannel", lambda remote, ref, mirror: FakeChannel(manifest, blobs))
    result = updater.repair(root, cfg=state.read_config(root))
    assert result.ok and result.installed_version == "0.6.1"
    assert state.app_exe_path(root, "0.6.1").read_bytes() == blobs["RoboScoutAI-app-windows-x64.exe"]
    assert install.verify_current(root)[1] == ""


def test_git_channel_fetch_uses_filter_then_falls_back(tmp_path):
    calls = []

    def fake_git(args, *, cwd=None, timeout=0, check=True, stdout_file=None):
        calls.append(args)
        if args[0] == "fetch" and "--filter=blob:none" in args:
            raise updater.ChannelError("filter unsupported")
        if args[0] == "fetch" and "refs/heads/" in args[-1]:
            return ""
        if args[0] == "rev-parse":
            return "abc123"
        if args[0] == "init":
            (tmp_path / "mirror" / ".git").mkdir(parents=True, exist_ok=True)
            return ""
        if args[0] == "cat-file" and "-p" in args:
            if args[-1].endswith("manifest.json"):
                return json.dumps({"version": "0.6.5", "artifacts": {}})
            return '__version__ = "0.6.5"'
        return ""

    chan = updater.GitChannel("https://example/repo.git", "main", tmp_path / "mirror", git=fake_git)
    assert chan.fetch() == "abc123"
    fetches = [c for c in calls if c[0] == "fetch"]
    assert "--filter=blob:none" in fetches[0]
    assert "--filter=blob:none" not in fetches[1]
    assert fetches[1][-1] == "+refs/heads/main:refs/channel/main"
    m = chan.read_manifest()
    assert m.version == "0.6.5"
    assert chan.read_source_version() == "0.6.5"
    assert chan.browse_url() == "https://example/repo/tree/main"


def test_git_channel_tries_tags_after_heads(tmp_path):
    calls = []

    def fake_git(args, *, cwd=None, timeout=0, check=True, stdout_file=None):
        calls.append(args)
        if args[0] == "fetch" and "refs/heads/" in args[-1]:
            raise updater.ChannelError("couldn't find remote ref")
        if args[0] == "fetch":
            return ""
        if args[0] == "rev-parse":
            return "tagsha"
        if args[0] == "init":
            (tmp_path / "m" / ".git").mkdir(parents=True, exist_ok=True)
        return ""

    chan = updater.GitChannel("r", "v0.6.0", tmp_path / "m", git=fake_git)
    assert chan.fetch() == "tagsha"
    assert any("+refs/tags/v0.6.0:refs/channel/v0.6.0" in c for c in calls if c[0] == "fetch")


def test_releases_channel_picks_newest_semver_with_manifest(tmp_path):
    releases = [
        {"tag_name": "v0.6.2", "draft": True, "assets": [{"name": "manifest.json", "browser_download_url": "u"}]},
        {"tag_name": "v0.6.1", "prerelease": False, "html_url": "https://gh/r/v0.6.1", "target_commitish": "c1", "assets": [
            {"name": "manifest.json", "browser_download_url": "https://dl/manifest.json"},
            {"name": "RoboScoutAI-app-windows-x64.exe", "browser_download_url": "https://dl/app.exe"},
        ]},
        {"tag_name": "v0.5.7", "assets": [{"name": "RoboScoutAI-windows-x64.exe", "browser_download_url": "x"}]},
        {"tag_name": "v0.6.9", "prerelease": True, "assets": [{"name": "manifest.json", "browser_download_url": "pre"}]},
    ]
    app_bytes = b"APP"
    manifest = {"version": "0.6.1", "artifacts": {"RoboScoutAI-app-windows-x64.exe": {"sha256": hashlib.sha256(app_bytes).hexdigest(), "size": 3}}}

    def get(url, *, headers=None, timeout=0):
        if url.startswith("https://api.github.com/repos/o/r/releases"):
            return json.dumps(releases).encode()
        if url == "https://dl/manifest.json":
            return json.dumps(manifest).encode()
        raise AssertionError(url)

    def download(url, dest, *, headers=None, progress=None, timeout=0):
        assert url == "https://dl/app.exe"
        Path(dest).write_bytes(app_bytes)
        return Path(dest)

    chan = updater.ReleasesChannel("o/r", get=get, download=download, token="")
    assert chan.fetch() == "c1"
    assert chan.release["tag_name"] == "v0.6.1"
    m = chan.read_manifest()
    assert m.version == "0.6.1"
    out = chan.download("RoboScoutAI-app-windows-x64.exe", tmp_path / "a.exe")
    assert out.read_bytes() == app_bytes
    assert chan.browse_url() == "https://gh/r/v0.6.1"
    # Exact-tag channels (e.g. "v0.6.9") may select prereleases and specific tags.
    pinned = updater.ReleasesChannel("o/r", tag="v0.6.9", get=get, download=download, token="")
    pinned.fetch()
    assert pinned.release["tag_name"] == "v0.6.9"
    with pytest.raises(updater.ChannelError):
        updater.ReleasesChannel("o/r", tag="v0.5.7", get=get, download=download, token="").fetch()


def test_releases_channel_uses_api_asset_url_with_token():
    releases = [{"tag_name": "v1.0.0", "assets": [{"name": "manifest.json", "url": "https://api/assets/1", "browser_download_url": "https://dl/m"}]}]
    seen = {}

    def get(url, *, headers=None, timeout=0):
        seen[url] = headers
        if "releases" in url:
            return json.dumps(releases).encode()
        return json.dumps({"version": "1.0.0", "artifacts": {}}).encode()

    chan = updater.ReleasesChannel("o/r", get=get, token="tok")
    chan.fetch()
    chan.read_manifest()
    assert seen["https://api/assets/1"]["Accept"] == "application/octet-stream"
    assert seen["https://api/assets/1"]["Authorization"] == "Bearer tok"


def test_download_artifact_falls_back_to_url_then_releases(tmp_path):
    payload = b"APP-FROM-RELEASES" * 20
    from roboscout_manager.manifest import Artifact

    art = Artifact(
        name="RoboScoutAI-app-windows-x64.exe",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        kind="app",
        platform="windows-x64",
        url="https://example.invalid/missing.exe",
    )

    class BrokenGit:
        name = "git"

        def download(self, name, dest, progress=lambda *_a: None, *, expected_size=0):
            raise updater.ChannelError("blob too large / missing")

    class FakeRel:
        name = "releases"

        def __init__(self, repo, tag):
            self.repo = repo
            self.tag = tag
            self.fetched = False

        def fetch(self):
            self.fetched = True
            return "abc"

        def download(self, name, dest, progress=lambda *_a: None, *, expected_size=0):
            assert self.fetched and name == art.name
            Path(dest).write_bytes(payload)
            return Path(dest)

    # URL fails, Releases succeeds.
    out = tmp_path / "app.exe"
    path = updater.download_artifact(
        BrokenGit(),
        art,
        out,
        github_repo="drewsmash/RoboScoutAI",
        release_tag="0.6.2",
        releases_factory=lambda repo, tag: FakeRel(repo, tag),
    )
    assert path.read_bytes() == payload

    # Direct URL succeeds without touching Releases.
    good = Artifact(
        name=art.name,
        sha256=art.sha256,
        size=art.size,
        kind="app",
        url="https://cdn.example/app.exe",
    )
    seen = {}

    def fake_http(url, dest, *, headers=None, progress=None, timeout=0):
        seen["url"] = url
        Path(dest).write_bytes(payload)
        return Path(dest)

    monkey_out = tmp_path / "via-url.exe"
    import roboscout_manager.updater as U

    original = U.http_download
    U.http_download = fake_http
    try:
        updater.download_artifact(BrokenGit(), good, monkey_out, github_repo="x/y", release_tag="0.6.2")
    finally:
        U.http_download = original
    assert seen["url"] == "https://cdn.example/app.exe"
    assert monkey_out.read_bytes() == payload


def test_perform_update_uses_releases_when_git_blob_missing(root, tmp_path, monkeypatch):
    _install(root, tmp_path, "0.6.0")
    payload = b"NEWAPP" * 40
    files = {
        "RoboScoutAI-app-windows-x64.exe": payload,
        "RoboScoutAI.exe": b"MGR" * 10,
    }
    # Manifest lists the app, but the git channel has no blob for it.
    app = tmp_path / "RoboScoutAI-app-windows-x64.exe"
    app.write_bytes(payload)
    mgr = tmp_path / "RoboScoutAI.exe"
    mgr.write_bytes(files["RoboScoutAI.exe"])
    manifest = build_manifest([app, mgr], version="0.6.2", channel="main", min_manager_version="0.6.0")
    git = FakeChannel(manifest, {"RoboScoutAI.exe": files["RoboScoutAI.exe"]})  # no app blob

    class Rel:
        name = "releases"

        def __init__(self, *a, **k):
            pass

        def fetch(self):
            return "sha"

        def download(self, name, dest, progress=lambda *_a: None, *, expected_size=0):
            Path(dest).write_bytes(payload)
            return Path(dest)

    monkeypatch.setattr(
        updater,
        "download_artifact",
        lambda channel, artifact, dest, **kw: (
            Path(dest).write_bytes(payload) or Path(dest)
        ),
    )
    check = updater.check_for_update(root, git_factory=lambda c, r: git)
    assert check.available and check.latest_version == "0.6.2"
    result = updater.perform_update(root, check)
    assert result.ok and result.installed_version == "0.6.2"
    assert state.app_exe_path(root, "0.6.2").read_bytes() == payload

import hashlib
import json
import os

import pytest

from roboscout_manager import INSTALLED_APP_EXE_NAME, install, rollback, state


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOSCOUT_INSTALL_ROOT", str(tmp_path / "RoboScoutAI"))
    monkeypatch.delenv("ROBOSCOUT_CHANNEL", raising=False)
    return state.default_install_root()


def _payload(tmp_path, version, body=None):
    data = body if body is not None else f"APP {version} ".encode() + os.urandom(64)
    path = tmp_path / f"payload-{version}.exe"
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest(), len(data)


def test_install_root_honours_override(root, tmp_path):
    assert root == tmp_path / "RoboScoutAI"
    assert state.app_exe_path(root, "0.6.0") == root / "app" / "0.6.0" / INSTALLED_APP_EXE_NAME


def test_install_side_by_side_and_atomic_switch(root, tmp_path):
    p1, sha1, size1 = _payload(tmp_path, "0.6.0")
    s1 = install.install_app_payload(root, p1, "0.6.0", sha256=sha1, size=size1, channel="main", source="setup")
    assert s1.exe_path(root).is_file()
    assert s1.exe == "app/0.6.0/RoboScoutAI-app.exe"
    assert s1.sha256 == sha1 and s1.previous_version == ""
    assert p1.is_file()  # copied, not moved

    p2, sha2, size2 = _payload(tmp_path, "0.6.1")
    s2 = install.install_app_payload(root, p2, "0.6.1", sha256=sha2, size=size2, source="git", move=True)
    assert not p2.is_file()  # moved into place
    assert s2.previous_version == "0.6.0" and s2.previous_sha256 == sha1
    current = json.loads(state.current_json_path(root).read_text())
    assert current["version"] == "0.6.1"
    assert current["exe"] == "app/0.6.1/RoboScoutAI-app.exe"
    # Both versions live side-by-side; the old exe is untouched.
    assert state.app_exe_path(root, "0.6.0").is_file()
    assert state.app_exe_path(root, "0.6.1").is_file()
    # No temp files left behind from the atomic write.
    assert not list(root.glob(".current.json.*"))
    assert not list((root / "app").rglob("*.part"))


def test_install_rejects_bad_digest_and_unversioned(root, tmp_path):
    p, sha, size = _payload(tmp_path, "0.6.0")
    with pytest.raises(Exception):
        install.install_app_payload(root, p, "0.6.0", sha256="0" * 64, size=size)
    assert state.read_current(root) is None
    with pytest.raises(install.InstallError):
        install.install_app_payload(root, p, "latest")


def test_reinstall_same_version_same_bytes_is_noop(root, tmp_path):
    p, sha, size = _payload(tmp_path, "0.6.0")
    install.install_app_payload(root, p, "0.6.0", sha256=sha)
    before = state.app_exe_path(root, "0.6.0").stat().st_mtime_ns
    install.install_app_payload(root, p, "0.6.0", sha256=sha)
    assert state.app_exe_path(root, "0.6.0").stat().st_mtime_ns == before


def test_write_current_is_atomic_on_failure(root, monkeypatch):
    cur = state.CurrentState(version="0.6.0", exe="app/0.6.0/RoboScoutAI-app.exe")
    state.write_current(root, cur)
    original = state.current_json_path(root).read_text()

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(state.os, "replace", boom)
    with pytest.raises(OSError):
        state.write_current(root, state.CurrentState(version="0.6.1", exe="app/0.6.1/RoboScoutAI-app.exe"))
    assert state.current_json_path(root).read_text() == original
    assert not list(root.glob(".current.json.*"))


def test_rollback_switches_to_previous_and_back(root, tmp_path):
    p1, sha1, _ = _payload(tmp_path, "0.6.0")
    p2, sha2, _ = _payload(tmp_path, "0.6.1")
    install.install_app_payload(root, p1, "0.6.0", sha256=sha1)
    install.install_app_payload(root, p2, "0.6.1", sha256=sha2)
    assert rollback.rollback_candidates(root) == ["0.6.0"]
    back = rollback.rollback(root)
    assert back.version == "0.6.0" and back.sha256 == sha1
    assert back.previous_version == "0.6.1"
    assert state.read_current(root).version == "0.6.0"
    # Rolling "back" again flips to 0.6.1 (it is the recorded previous now).
    fwd = rollback.rollback(root)
    assert fwd.version == "0.6.1"


def test_rollback_without_previous_fails(root, tmp_path):
    p1, sha1, _ = _payload(tmp_path, "0.6.0")
    install.install_app_payload(root, p1, "0.6.0", sha256=sha1)
    with pytest.raises(rollback.RollbackError):
        rollback.rollback(root)
    with pytest.raises(rollback.RollbackError):
        rollback.rollback(root, "9.9.9")


def test_rollback_refuses_tampered_previous(root, tmp_path):
    p1, sha1, _ = _payload(tmp_path, "0.6.0")
    p2, sha2, _ = _payload(tmp_path, "0.6.1")
    install.install_app_payload(root, p1, "0.6.0", sha256=sha1)
    install.install_app_payload(root, p2, "0.6.1", sha256=sha2)
    state.app_exe_path(root, "0.6.0").write_bytes(b"tampered")
    with pytest.raises(rollback.RollbackError):
        rollback.rollback(root)
    assert state.read_current(root).version == "0.6.1"


def test_prune_keeps_current_and_previous_only(root, tmp_path):
    for ver in ("0.5.7", "0.5.8", "0.6.0", "0.6.1"):
        p, sha, _ = _payload(tmp_path, ver)
        install.install_app_payload(root, p, ver, sha256=sha)
    assert install.installed_versions(root) == ["0.6.1", "0.6.0", "0.5.8", "0.5.7"]
    removed = install.prune_versions(root, keep=2)
    assert sorted(removed) == ["0.5.7", "0.5.8"]
    assert install.installed_versions(root) == ["0.6.1", "0.6.0"]
    assert state.read_current(root).version == "0.6.1"
    # The rollback target survives even if it is not among the newest N.
    rollback.rollback(root)  # current 0.6.0, previous 0.6.1
    p, sha, _ = _payload(tmp_path, "0.7.0")
    install.install_app_payload(root, p, "0.7.0", sha256=sha)  # current 0.7.0, previous 0.6.0
    p, sha, _ = _payload(tmp_path, "0.7.1")
    install.install_app_payload(root, p, "0.7.1", sha256=sha)  # current 0.7.1, previous 0.7.0
    install.prune_versions(root, keep=2)
    assert install.installed_versions(root) == ["0.7.1", "0.7.0"]


def test_verify_current_detects_missing_or_corrupt(root, tmp_path):
    assert install.verify_current(root) == (None, "current.json missing or invalid")
    p, sha, _ = _payload(tmp_path, "0.6.0")
    install.install_app_payload(root, p, "0.6.0", sha256=sha)
    st, problem = install.verify_current(root)
    assert st.version == "0.6.0" and problem == ""
    state.app_exe_path(root, "0.6.0").write_bytes(b"corrupt")
    _, problem = install.verify_current(root)
    assert "mismatch" in problem or "size" in problem
    state.app_exe_path(root, "0.6.0").unlink()
    _, problem = install.verify_current(root)
    assert "missing" in problem


def test_run_setup_from_payload_dir(root, tmp_path):
    payload = tmp_path / "payload"
    payload.mkdir()
    app_bytes = b"APP" * 50
    mgr_bytes = b"MGR" * 5
    (payload / "RoboScoutAI-app-windows-x64.exe").write_bytes(app_bytes)
    (payload / "RoboScoutAI.exe").write_bytes(mgr_bytes)
    from roboscout_manager.manifest import build_manifest

    m = build_manifest(
        [payload / "RoboScoutAI-app-windows-x64.exe", payload / "RoboScoutAI.exe"],
        version="0.6.0",
        channel="release/test",
    )
    (payload / "manifest.json").write_text(m.to_json())
    calls = {}

    def fake_shortcuts(r, *, desktop, start_menu):
        calls["shortcuts"] = (desktop, start_menu)
        return [r / "fake.lnk"]

    def fake_register(r, version):
        calls["register"] = version
        return True

    result = install.run_setup(
        install.SetupOptions(root=root, payload_dir=payload, desktop_shortcut=True),
        shortcut_fn=fake_shortcuts,
        register_fn=fake_register,
    )
    assert result.version == "0.6.0"
    assert result.registered and calls["register"] == "0.6.0"
    assert calls["shortcuts"] == (True, True)
    assert (root / "RoboScoutAI.exe").read_bytes() == mgr_bytes
    assert state.app_exe_path(root, "0.6.0").read_bytes() == app_bytes
    cfg = state.read_config(root)
    assert cfg.channel == "release/test"  # channel seeded from the payload manifest
    assert state.read_current(root).source == "setup"
    assert not result.needs_download


def test_run_setup_without_payload_defers_download(root, tmp_path):
    mgr = tmp_path / "RoboScoutAI.exe"
    mgr.write_bytes(b"MGR")
    result = install.run_setup(
        install.SetupOptions(root=root, payload_dir=None, register=False, start_menu_shortcut=False, manager_source=mgr),
    )
    assert result.needs_download
    assert result.version == ""
    assert (root / "RoboScoutAI.exe").is_file()
    assert state.manager_json_path(root).is_file()


def test_bootstrap_install_writes_config_once(root):
    msgs = install.bootstrap_install(root, frozen=False)
    assert any("manager.json" in m for m in msgs)
    assert state.manager_json_path(root).is_file()
    assert install.bootstrap_install(root, frozen=False) == []


def test_uninstall_bat_waits_for_pid_and_keeps_data_by_default():
    lines = install.uninstall_bat_lines(pid=4242, root=r"C:\Users\u\AppData\Local\RoboScoutAI", purge=False, log_path=r"C:\t\u.log")
    text = "\n".join(lines)
    assert "set PID=4242" in text
    assert "Wait-Process" in text
    assert 'rmdir /S /Q "%ROOT%\\app"' in text
    assert 'del /F /Q "%ROOT%\\RoboScoutAI.exe"' in text
    assert "%ROOT%\\data" not in text
    assert lines[-2] == '(goto) 2>nul & del "%~f0"'
    purge = "\n".join(install.uninstall_bat_lines(pid=1, root="R", purge=True, log_path="L"))
    assert 'rmdir /S /Q "%ROOT%\\data"' in purge


def test_uninstall_non_windows_removes_files_but_keeps_data(root, tmp_path, monkeypatch):
    p, sha, _ = _payload(tmp_path, "0.6.0")
    install.install_app_payload(root, p, "0.6.0", sha256=sha)
    (root / "RoboScoutAI.exe").write_bytes(b"MGR")
    (root / "data").mkdir()
    (root / "data" / "jobs.json").write_text("{}")
    monkeypatch.setattr(install.sys, "platform", "linux")
    script = install.uninstall(root, purge=False, unregister_fn=lambda: True, remove_shortcuts_fn=lambda: [])
    assert script.is_file()
    assert not (root / "app").exists()
    assert not (root / "RoboScoutAI.exe").exists()
    assert not state.current_json_path(root).exists()
    assert (root / "data" / "jobs.json").is_file()

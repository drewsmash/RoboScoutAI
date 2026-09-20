"""CLI, IPC token/status, launcher relaunch loop and the managed FastAPI endpoints."""

import hashlib
import json
import os
import sys

import pytest

from roboscout_manager import RELAUNCH_EXIT_CODE, cli, install, ipc, launcher, setup_entry, state
from roboscout_manager.manifest import build_manifest


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOSCOUT_INSTALL_ROOT", str(tmp_path / "RoboScoutAI"))
    monkeypatch.delenv("ROBOSCOUT_CHANNEL", raising=False)
    r = state.default_install_root()
    r.mkdir(parents=True)
    return r


def _install(root, tmp_path, version):
    p = tmp_path / f"seed-{version}.exe"
    p.write_bytes(f"SEED {version}".encode())
    return install.install_app_payload(root, p, version, sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


# --- IPC -------------------------------------------------------------------------


def test_ipc_token_issue_and_validate(root):
    token = ipc.issue_token(root, manager_pid=123, port=8123)
    assert len(token) >= 32
    data = ipc.read_ipc(root)
    assert data["manager_pid"] == 123 and data["port"] == 8123
    assert ipc.validate_token(root, token)
    assert not ipc.validate_token(root, token + "x")
    assert not ipc.validate_token(root, "")
    assert not ipc.validate_token(root, None)
    with pytest.raises(ipc.IpcError):
        ipc.require_token(root, "nope")
    ipc.require_token(root, token)
    # Re-issuing rotates the token.
    token2 = ipc.issue_token(root)
    assert token2 != token and not ipc.validate_token(root, token)


def test_ipc_status_roundtrip(root):
    assert ipc.read_status(root).state == "idle"
    st = ipc.write_status(root, "downloading", progress=42.7, message="half", version="0.6.1", current_version="0.6.0")
    assert st.progress == 42
    back = ipc.read_status(root)
    assert back.state == "downloading" and back.progress == 42 and back.version == "0.6.1"
    assert back.relaunch_exit_code == RELAUNCH_EXIT_CODE == 75
    assert not back.finished
    ipc.write_status(root, "done", progress=100)
    assert ipc.read_status(root).finished
    with pytest.raises(ValueError):
        ipc.write_status(root, "bogus")
    ipc.clear_status(root)
    assert ipc.read_status(root).state == "idle"


# --- CLI ---------------------------------------------------------------------------


def test_cli_version_and_list(root, tmp_path):
    _install(root, tmp_path, "0.6.0")
    out = tmp_path / "v.json"
    assert cli.main(["--root", str(root), "--version", "--json", "--out", str(out)]) == 0
    data = _read(out)
    assert data["app_version"] == "0.6.0" and data["manager_version"] == "0.6.0"
    assert data["installed_versions"] == ["0.6.0"]
    assert cli.main(["--root", str(root), "--list", "--json", "--out", str(out)]) == 0
    assert _read(out)["installed_versions"] == ["0.6.0"]


def test_cli_channel_switch(root, tmp_path):
    out = tmp_path / "c.json"
    assert cli.main(["--root", str(root), "--channel", "release/next", "--json", "--out", str(out)]) == 0
    assert _read(out) == {"ok": True, "channel": "release/next"}
    assert state.read_config(root).channel == "release/next"
    assert cli.main(["--root", str(root), "--channel", "bad name"]) == 2


def test_cli_rollback(root, tmp_path):
    _install(root, tmp_path, "0.6.0")
    _install(root, tmp_path, "0.6.1")
    out = tmp_path / "r.json"
    assert cli.main(["--root", str(root), "--rollback", "--json", "--out", str(out)]) == 0
    assert _read(out)["version"] == "0.6.0"
    assert cli.main(["--root", str(root), "--rollback", "0.6.1", "--json", "--out", str(out)]) == 0
    assert state.read_current(root).version == "0.6.1"
    _install(root, tmp_path, "0.6.2")
    assert cli.main(["--root", str(root), "--rollback", "9.9.9"]) == 1


def test_cli_status(root, tmp_path, capsys):
    ipc.write_status(root, "installing", progress=90, message="almost")
    assert cli.main(["--root", str(root), "--status"]) == 0
    out = capsys.readouterr().out
    assert json.loads(out)["state"] == "installing"


def test_cli_install_from_payload_silent(root, tmp_path, monkeypatch):
    payload = tmp_path / "payload"
    payload.mkdir()
    app = payload / "RoboScoutAI-app-windows-x64.exe"
    app.write_bytes(b"APP" * 40)
    mgr = payload / "RoboScoutAI.exe"
    mgr.write_bytes(b"MGR" * 4)
    (payload / "manifest.json").write_text(build_manifest([app, mgr], version="0.6.0", channel="main").to_json())
    monkeypatch.setattr("roboscout_manager.shortcuts.create_shortcuts", lambda r, **k: [])
    monkeypatch.setattr("roboscout_manager.registry.register_uninstall", lambda r, v: True)
    out = tmp_path / "s.json"
    code = setup_entry.main(["--root", str(root), "--payload", str(payload), "--silent", "--no-launch", "--json", "--out", str(out)])
    assert code == 0
    data = _read(out)
    assert data["ok"] and data["version"] == "0.6.0" and data["registered"]
    assert state.read_current(root).version == "0.6.0"
    assert (root / "RoboScoutAI.exe").read_bytes() == b"MGR" * 4
    # Setup exe forwards management commands without forcing --install.
    assert setup_entry.main(["--root", str(root), "--version", "--json", "--out", str(out)]) == 0
    assert _read(out)["app_version"] == "0.6.0"


def test_cli_update_rejects_bad_token(root, tmp_path):
    ipc.issue_token(root)
    out = tmp_path / "u.json"
    assert cli.main(["--root", str(root), "--update", "--token", "wrong", "--json", "--out", str(out)]) == 3
    assert "token" in _read(out)["error"]


def test_cli_update_refuses_concurrent_run(root, tmp_path):
    ipc.write_status(root, "downloading", progress=10, pid=os.getppid())  # another live process
    out = tmp_path / "u.json"
    assert cli.main(["--root", str(root), "--update", "--json", "--out", str(out)]) == 4
    assert "already running" in _read(out)["error"]


def test_cli_uninstall_non_windows(root, tmp_path, monkeypatch):
    _install(root, tmp_path, "0.6.0")
    (root / "RoboScoutAI.exe").write_bytes(b"MGR")
    monkeypatch.setattr("roboscout_manager.registry.unregister_uninstall", lambda: True)
    monkeypatch.setattr("roboscout_manager.shortcuts.remove_shortcuts", lambda: [])
    monkeypatch.setattr(install.sys, "platform", "linux")
    out = tmp_path / "un.json"
    assert cli.main(["--root", str(root), "--uninstall", "--silent", "--json", "--out", str(out)]) == 0
    assert _read(out)["ok"]
    assert not (root / "app").exists()


# --- launcher -----------------------------------------------------------------------


def test_build_app_command(root):
    cmd = launcher.build_app_command(root / "app" / "0.6.0" / "RoboScoutAI-app.exe", port=8123, token="tok", root=root, extra=["--browser"])
    assert cmd[0].endswith("RoboScoutAI-app.exe")
    assert "--managed" in cmd and cmd[cmd.index("--port") + 1] == "8123"
    assert cmd[cmd.index("--manager-token") + 1] == "tok"
    assert cmd[cmd.index("--manager-root") + 1] == str(root)
    assert "--skip-update-check" in cmd and cmd[-1] == "--browser"


def test_ensure_launchable_rolls_back_when_current_damaged(root, tmp_path):
    _install(root, tmp_path, "0.6.0")
    _install(root, tmp_path, "0.6.1")
    state.app_exe_path(root, "0.6.1").write_bytes(b"corrupt")
    st = launcher.ensure_launchable(root)
    assert st.version == "0.6.0"
    assert state.read_current(root).version == "0.6.0"


def test_ensure_launchable_repairs_when_nothing_else(root, tmp_path, monkeypatch):
    _install(root, tmp_path, "0.6.0")
    state.app_exe_path(root, "0.6.0").unlink()
    fixed = {}

    def fake_repair(r, *, cfg=None, progress=None):
        _install(r, tmp_path, "0.6.0")
        fixed["yes"] = True
        from roboscout_manager.updater import UpdateResult

        return UpdateResult(ok=True, message="repaired", installed_version="0.6.0")

    monkeypatch.setattr(launcher, "repair", fake_repair)
    st = launcher.ensure_launchable(root)
    assert fixed and st.version == "0.6.0"


class FakeChild:
    def __init__(self, code):
        self._code = code
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self):
        self.returncode = self._code
        return self._code

    def terminate(self):
        pass


def test_run_launcher_relaunches_on_exit_code_and_picks_new_version(root, tmp_path, monkeypatch):
    _install(root, tmp_path, "0.6.0")
    spawned = []
    codes = iter([RELAUNCH_EXIT_CODE, 0])

    def fake_popen(cmd, **kwargs):
        spawned.append((cmd, kwargs))
        if len(spawned) == 1:
            # Simulate the manager --update having installed 0.6.1 while the app ran.
            _install(root, tmp_path, "0.6.1")
        return FakeChild(next(codes))

    monkeypatch.setattr(launcher, "wait_for_health", lambda port, **k: True)
    code = launcher.run_launcher(root, port=8765, splash=False, background_check=False, popen=fake_popen)
    assert code == 0
    assert len(spawned) == 2
    assert spawned[0][0][0].endswith("app/0.6.0/RoboScoutAI-app.exe")
    assert spawned[1][0][0].endswith("app/0.6.1/RoboScoutAI-app.exe")
    # Same port is preferred on relaunch so an open UI window can reload.
    assert spawned[0][0][spawned[0][0].index("--port") + 1] == spawned[1][0][spawned[1][0].index("--port") + 1]
    env = spawned[1][1]["env"]
    assert env["ROBOSCOUT_MANAGED"] == "1" and env["ROBOSCOUT_MANAGER_ROOT"] == str(root)
    assert ipc.validate_token(root, env["ROBOSCOUT_MANAGER_TOKEN"])
    # First-run bootstrap wrote manager.json.
    assert state.manager_json_path(root).is_file()


def test_run_launcher_stops_when_manager_swap_pending(root, tmp_path, monkeypatch):
    _install(root, tmp_path, "0.6.0")

    def fake_popen(cmd, **kwargs):
        ipc.write_status(root, "manager-update", progress=100)
        return FakeChild(RELAUNCH_EXIT_CODE)

    monkeypatch.setattr(launcher, "wait_for_health", lambda port, **k: True)
    assert launcher.run_launcher(root, splash=False, background_check=False, popen=fake_popen) == 0


# --- managed app endpoints -----------------------------------------------------------------


def test_managed_endpoints_delegate_to_manager(root, tmp_path, monkeypatch):
    _install(root, tmp_path, "0.6.0")
    monkeypatch.setenv("ROBOSCOUT_MANAGED", "1")
    monkeypatch.setenv("ROBOSCOUT_MANAGER_ROOT", str(root))
    monkeypatch.setenv("ROBOSCOUT_MANAGER_TOKEN", ipc.issue_token(root))
    monkeypatch.setenv("ROBOSCOUT_MANAGER_VERSION", "0.6.0")
    from ramscout import managed

    monkeypatch.setattr(managed, "run_manager", lambda args, timeout=0: {"available": True, "latest_version": "0.6.1", "message": "update available", "can_apply": True})
    spawned = []
    monkeypatch.setattr(managed.subprocess, "Popen", lambda cmd, **kw: spawned.append(cmd))
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)
    version = client.get("/api/version").json()
    assert version["managed"] is True and version["manager"]["manager_version"] == "0.6.0"
    check = client.get("/api/updates/check").json()
    assert check["available"] and check["mode"] == "managed" and check["current_version"] == "0.6.0"
    started = client.post("/api/updates/download").json()
    assert started["ok"] and started["managed"] and started["status_url"] == "/api/updates/status"
    updates = [c for c in spawned if "--update" in c]
    assert updates and "--token" in updates[0] and "--root" in updates[0]
    assert ipc.read_status(root).state == "checking"
    ipc.write_status(root, "done", progress=100, version="0.6.1")
    status = client.get("/api/updates/status").json()
    assert status["state"] == "done" and status["version"] == "0.6.1" and status["managed"]
    exits = []
    monkeypatch.setattr(managed.os, "_exit", lambda code: exits.append(code))
    monkeypatch.setattr(managed.threading, "Timer", lambda delay, fn: type("T", (), {"start": lambda self: fn()})())
    relaunch = client.post("/api/updates/relaunch").json()
    assert relaunch["ok"] and relaunch["exit_code"] == RELAUNCH_EXIT_CODE
    assert exits == [RELAUNCH_EXIT_CODE]


def test_unmanaged_relaunch_is_rejected(monkeypatch):
    monkeypatch.delenv("ROBOSCOUT_MANAGED", raising=False)
    from fastapi.testclient import TestClient
    from app import app

    client = TestClient(app)
    assert client.post("/api/updates/relaunch").status_code == 400
    assert client.get("/api/version").json()["managed"] is False


def test_managed_run_manager_uses_python_module_in_dev(root, monkeypatch):
    monkeypatch.setenv("ROBOSCOUT_MANAGED", "1")
    monkeypatch.setenv("ROBOSCOUT_MANAGER_ROOT", str(root))
    monkeypatch.delenv("ROBOSCOUT_MANAGER_EXE", raising=False)
    from ramscout import managed

    cmd = managed._manager_command(["--check"])
    assert cmd[:3] == [sys.executable, "-m", "roboscout_manager"]
    assert cmd[-2:] == ["--root", str(root)]
    result = managed.run_manager(["--version"])
    assert result["manager_version"] == "0.6.0" and result["root"] == str(root)


def test_legacy_updater_refuses_when_managed_install_present(root, tmp_path, monkeypatch):
    _install(root, tmp_path, "0.6.0")
    from ramscout import updater as legacy

    monkeypatch.setattr(legacy, "source_git_root", lambda: None)
    monkeypatch.setattr(legacy, "is_frozen", lambda: True)
    monkeypatch.setattr(legacy.shutil, "which", lambda _c: "/usr/bin/git")
    info = legacy.check_for_update("0.5.7")
    assert not info.available
    assert "manager" in (info.error or "").lower()
    assert info.message == "managed install"

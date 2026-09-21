"""Setup detects an existing install and reports upgrade mode."""

from __future__ import annotations

import json

import pytest

from roboscout_manager import install, state
from roboscout_manager.cli import cmd_install
from roboscout_manager.manifest import build_manifest
from roboscout_manager.ui import SetupChoice, setup_dialog


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOSCOUT_INSTALL_ROOT", str(tmp_path / "RoboScoutAI"))
    monkeypatch.delenv("ROBOSCOUT_CHANNEL", raising=False)
    r = state.default_install_root()
    r.mkdir(parents=True)
    return r


def test_setup_dialog_silent_upgrade_mode():
    choice = setup_dialog(r"C:\Users\x\AppData\Local\RoboScoutAI", version="0.7.0", existing_version="0.6.2", silent=True)
    assert choice.proceed and choice.mode == "upgrade"


def test_setup_dialog_silent_reinstall_same_version():
    choice = setup_dialog("/tmp/rs", version="0.7.0", existing_version="0.7.0", silent=True)
    assert choice.mode == "reinstall"


def test_setup_dialog_silent_fresh_install():
    choice = setup_dialog("/tmp/rs", version="0.7.0", existing_version="", silent=True)
    assert choice.mode == "install"


def test_cmd_install_upgrades_existing(root, tmp_path, monkeypatch):
    # Seed an older install.
    app_old = tmp_path / "old.exe"
    app_old.write_bytes(b"OLDAPP" * 20)
    install.install_app_payload(root, app_old, "0.6.2", sha256="", source="setup")
    assert state.read_current(root).version == "0.6.2"

    payload = tmp_path / "payload"
    payload.mkdir()
    app = payload / "RoboScoutAI-app-windows-x64.exe"
    app.write_bytes(b"NEWAPP" * 40)
    mgr = payload / "RoboScoutAI.exe"
    mgr.write_bytes(b"NEWMGR" * 4)
    (payload / "manifest.json").write_text(
        build_manifest([app, mgr], version="0.7.0", channel="main").to_json()
    )

    monkeypatch.setattr(
        "roboscout_manager.ui.setup_dialog",
        lambda *a, **k: SetupChoice(proceed=True, desktop_shortcut=False, mode="upgrade"),
    )
    monkeypatch.setattr("roboscout_manager.shortcuts.create_shortcuts", lambda r, **k: [])
    monkeypatch.setattr("roboscout_manager.registry.register_uninstall", lambda r, v: True)

    out_path = tmp_path / "out.json"
    payload_path = str(payload)

    class Args:
        desktop_shortcut = False
        silent = True
        no_launch = True
        channel = ""
        json = True

    args = Args()
    args.payload = payload_path
    args.out = str(out_path)

    assert cmd_install(args, root) == 0
    assert state.read_current(root).version == "0.7.0"
    data = json.loads(out_path.read_text())
    assert data["version"] == "0.7.0"
    assert data["previous_version"] == "0.6.2"
    assert data["mode"] == "upgrade"

"""Windows-only side effects (registry, .lnk shortcuts) exercised through mocks on Linux."""

from pathlib import Path

import pytest

from roboscout_manager import registry, shortcuts, state


class FakeWinreg:
    HKEY_CURRENT_USER = "HKCU"
    KEY_SET_VALUE = 2
    KEY_READ = 1
    REG_SZ = "REG_SZ"
    REG_DWORD = "REG_DWORD"

    def __init__(self):
        self.keys: dict[str, dict[str, tuple[str, object]]] = {}
        self.closed = 0

    def CreateKeyEx(self, hive, path, reserved, access):
        assert hive == "HKCU"
        return self.keys.setdefault(path, {})

    def OpenKey(self, hive, path, reserved, access):
        if path not in self.keys:
            raise OSError("not found")
        return self.keys[path]

    def SetValueEx(self, key, name, reserved, kind, value):
        key[name] = (kind, value)

    def QueryValueEx(self, key, name):
        kind, value = key[name]
        return value, kind

    def DeleteKey(self, hive, path):
        if path not in self.keys:
            raise OSError("not found")
        del self.keys[path]

    def CloseKey(self, key):
        self.closed += 1


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("ROBOSCOUT_INSTALL_ROOT", str(tmp_path / "RoboScoutAI"))
    r = state.default_install_root()
    r.mkdir()
    (r / "RoboScoutAI.exe").write_bytes(b"MGR" * 1024)
    return r


def test_uninstall_entry_payload(root):
    values = registry.uninstall_entry(root, "0.6.0", manager_version="0.6.0", size_kb=1234)
    exe = str(root / "RoboScoutAI.exe")
    assert values["DisplayName"] == ("REG_SZ", "RoboScoutAI")
    assert values["DisplayVersion"] == ("REG_SZ", "0.6.0")
    assert values["Publisher"] == ("REG_SZ", "RoboScoutAI")
    assert values["InstallLocation"] == ("REG_SZ", str(root))
    assert values["DisplayIcon"] == ("REG_SZ", f"{exe},0")
    assert values["UninstallString"] == ("REG_SZ", f'"{exe}" --uninstall')
    assert values["QuietUninstallString"] == ("REG_SZ", f'"{exe}" --uninstall --silent')
    assert values["EstimatedSize"] == ("REG_DWORD", 1234)
    assert values["NoModify"] == ("REG_DWORD", 0)
    assert "RamScout" not in repr(values)


def test_estimated_size_skips_user_data(root):
    (root / "data").mkdir()
    (root / "data" / "big.bin").write_bytes(b"0" * 1_000_000)
    kb = registry.estimated_size_kb(root)
    assert 1 <= kb < 100


def test_register_and_unregister_with_mock_winreg(root):
    reg = FakeWinreg()
    assert registry.register_uninstall(root, "0.6.0", winreg=reg)
    entry = reg.keys[registry.UNINSTALL_KEY]
    assert entry["DisplayVersion"] == ("REG_SZ", "0.6.0")
    assert entry["UninstallString"][1].endswith('RoboScoutAI.exe" --uninstall')
    assert reg.keys[registry.APP_PATHS_KEY][None] == ("REG_SZ", str(root / "RoboScoutAI.exe"))
    assert registry.registered_version(winreg=reg) == "0.6.0"
    assert reg.closed >= 2
    assert registry.unregister_uninstall(winreg=reg)
    assert registry.UNINSTALL_KEY not in reg.keys
    assert registry.registered_version(winreg=reg) is None
    assert not registry.unregister_uninstall(winreg=reg)


def test_register_without_winreg_is_noop(root, monkeypatch):
    monkeypatch.setattr(registry, "_winreg", lambda: None)
    assert registry.register_uninstall(root, "0.6.0") is False
    assert registry.unregister_uninstall() is False


def test_shortcut_paths_use_start_menu_and_desktop(monkeypatch):
    monkeypatch.setenv("APPDATA", r"C:\Users\u\AppData\Roaming")
    monkeypatch.setenv("USERPROFILE", r"C:\Users\u")
    paths = shortcuts.shortcut_paths(desktop=True, start_menu=True)
    assert [p.name for p in paths] == ["RoboScoutAI.lnk", "RoboScoutAI.lnk"]
    assert "Start Menu" in str(paths[0]) and "Programs" in str(paths[0])
    assert str(paths[1]).endswith("Desktop/RoboScoutAI.lnk") or str(paths[1]).endswith("Desktop\\RoboScoutAI.lnk")
    assert shortcuts.shortcut_paths(desktop=False) == [paths[0]]


def test_shortcut_script_generation_escapes_and_targets_manager(root):
    target = root / "RoboScoutAI.exe"
    link = Path(r"C:\Users\o'neil\Desktop\RoboScoutAI.lnk")
    script = shortcuts.shortcut_script(target, [link], description="RoboScoutAI — scout")
    assert "WScript.Shell" in script
    assert f"$target = '{target}'" in script
    assert "'C:\\Users\\o''neil\\Desktop\\RoboScoutAI.lnk'" in script  # single quotes doubled
    assert "$lnk.TargetPath = $target" in script
    assert f"$lnk.WorkingDirectory = '{target.parent}'" in script
    assert f"$lnk.IconLocation = '{target},0'" in script
    assert "$lnk.Save()" in script
    assert script.count("CreateShortcut") == 1
    assert "SHORTCUTS_OK" in script


def test_create_and_remove_shortcuts_use_runner(root, monkeypatch):
    monkeypatch.setenv("APPDATA", str(root / "roaming"))
    monkeypatch.setenv("USERPROFILE", str(root / "profile"))
    ran = []
    links = shortcuts.create_shortcuts(root, desktop=True, runner=lambda s: ran.append(s) or "SHORTCUTS_OK")
    assert len(links) == 2 and len(ran) == 1
    assert ran[0].count("CreateShortcut") == 2
    removed = shortcuts.remove_shortcuts(runner=lambda s: ran.append(s) or "ok")
    assert len(removed) == 2
    assert "Remove-Item" in ran[1]


def test_run_powershell_refuses_off_windows(monkeypatch):
    monkeypatch.setattr(shortcuts.sys, "platform", "linux")
    with pytest.raises(OSError):
        shortcuts.run_powershell("Write-Output hi")

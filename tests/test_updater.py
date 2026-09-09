from ramscout.updater import is_newer, normalize_version, preferred_asset_names, version_tuple


def test_normalize_version_strips_v_prefix():
    assert normalize_version("v1.2.3") == "1.2.3"
    assert normalize_version("0.4.0") == "0.4.0"


def test_is_newer_compares_semver_ish():
    assert is_newer("0.4.1", "0.4.0")
    assert is_newer("1.0.0", "0.9.9")
    assert not is_newer("0.4.0", "0.4.0")
    assert not is_newer("0.3.9", "0.4.0")


def test_version_tuple_ignores_junk():
    assert version_tuple("v0.4.0-beta") == (0, 4, 0)


def test_preferred_assets_include_windows_or_mac(monkeypatch):
    monkeypatch.setattr("ramscout.updater.platform_key", lambda: "windows")
    assert "RamScoutAI-windows-x64.exe" in preferred_asset_names()
    monkeypatch.setattr("ramscout.updater.platform_key", lambda: "macos-arm64")
    assert "RamScoutAI-macos-arm64.zip" in preferred_asset_names()

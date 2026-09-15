from ramscout.updater import check_for_update, is_newer, normalize_version, preferred_asset_names, version_tuple


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
    names = preferred_asset_names()
    assert "RamScoutAI-windows-x64.exe" in names
    assert "RamScoutAI-windows-x64-signed.exe" in names
    monkeypatch.setattr("ramscout.updater.platform_key", lambda: "macos-arm64")
    assert "RamScoutAI-macos-arm64.zip" in preferred_asset_names()


def test_missing_release_explains_next_step(monkeypatch):
    class FakeResp:
        status_code = 404

        def raise_for_status(self):
            return None

        def json(self):
            return {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            return FakeResp()

    monkeypatch.setattr("ramscout.updater.httpx.Client", FakeClient)
    monkeypatch.setattr("ramscout.updater.github_token", lambda: "")
    info = check_for_update("0.4.0")
    assert info.available is False
    assert info.error
    soft = info.error.lower()
    assert "no updates" in soft or "published" in soft or "private" in soft or "token" in soft
    assert "releases" in (info.release_url or "")


def test_missing_release_with_token_is_soft(monkeypatch):
    class FakeResp:
        status_code = 404

        def raise_for_status(self):
            return None

        def json(self):
            return {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            return FakeResp()

    monkeypatch.setattr("ramscout.updater.httpx.Client", FakeClient)
    monkeypatch.setattr("ramscout.updater.github_token", lambda: "ghs_test")
    info = check_for_update("0.4.3")
    assert info.available is False
    assert "published" in (info.error or "").lower() or "not" in (info.error or "").lower()


def test_picks_signed_windows_asset_as_fallback(monkeypatch):
    from ramscout.updater import _pick_asset

    monkeypatch.setattr("ramscout.updater.platform_key", lambda: "windows")
    assets = [
        {"name": "notes.txt", "browser_download_url": "https://example/notes"},
        {
            "name": "RamScoutAI-windows-x64-signed.exe",
            "browser_download_url": "https://example/signed",
            "url": "https://api.example/signed",
        },
    ]
    picked = _pick_asset(assets)
    assert picked is not None
    assert picked["name"] == "RamScoutAI-windows-x64-signed.exe"


def test_finds_newer_release_asset(monkeypatch):
    payload = {
        "tag_name": "v0.9.0",
        "html_url": "https://github.com/drewsmash/RamScoutAI/releases/tag/v0.9.0",
        "body": "desktop",
        "assets": [
            {
                "name": "RamScoutAI-windows-x64.exe",
                "browser_download_url": "https://example/RamScoutAI-windows-x64.exe",
                "url": "https://api.example/asset",
            }
        ],
    }

    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return payload

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            return FakeResp()

    monkeypatch.setattr("ramscout.updater.httpx.Client", FakeClient)
    monkeypatch.setattr("ramscout.updater.platform_key", lambda: "windows")
    monkeypatch.setattr("ramscout.updater.github_token", lambda: "")
    info = check_for_update("0.4.3")
    assert info.available is True
    assert info.latest_version == "0.9.0"
    assert info.asset_name == "RamScoutAI-windows-x64.exe"
    assert "windows-x64.exe" in info.asset_url

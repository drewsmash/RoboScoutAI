from ramscout.updater import (
    _pick_asset,
    _pick_newest_release,
    check_for_update,
    is_newer,
    normalize_version,
    preferred_asset_names,
    version_tuple,
)


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


def test_preferred_assets_prefer_roboscout_then_legacy(monkeypatch):
    monkeypatch.setattr("ramscout.updater.platform_key", lambda: "windows")
    names = preferred_asset_names()
    assert names[0] == "RoboScoutAI-windows-x64.exe"
    assert "RamScoutAI-windows-x64.exe" in names
    assert names.index("RoboScoutAI-windows-x64.exe") < names.index("RamScoutAI-windows-x64.exe")

    monkeypatch.setattr("ramscout.updater.platform_key", lambda: "macos-arm64")
    mac = preferred_asset_names()
    assert mac[0] == "RoboScoutAI-macos-arm64.zip"
    assert "RamScoutAI-macos-arm64.zip" in mac


def test_pick_asset_accepts_legacy_and_signed(monkeypatch):
    monkeypatch.setattr("ramscout.updater.platform_key", lambda: "windows")
    legacy = _pick_asset(
        [
            {"name": "RamScoutAI-windows-x64.exe", "browser_download_url": "https://example/legacy"},
        ]
    )
    assert legacy["name"] == "RamScoutAI-windows-x64.exe"

    signed = _pick_asset(
        [
            {"name": "notes.txt"},
            {"name": "RamScoutAI-windows-x64-signed.exe", "browser_download_url": "https://example/signed"},
        ]
    )
    assert signed["name"] == "RamScoutAI-windows-x64-signed.exe"

    monkeypatch.setattr("ramscout.updater.platform_key", lambda: "macos-arm64")
    seven = _pick_asset([{"name": "RamScoutAI-macos-arm64.7z", "browser_download_url": "https://example/7z"}])
    assert seven["name"].endswith(".7z")


def test_pick_newest_release_ignores_mislabelled_latest():
    releases = [
        {"tag_name": "v0.1.4", "draft": False, "prerelease": False, "assets": []},
        {"tag_name": "v0.4.2", "draft": False, "prerelease": False, "assets": [{"name": "x"}]},
        {"tag_name": "v0.4.1", "draft": False, "prerelease": False, "assets": []},
    ]
    best = _pick_newest_release(releases)
    assert best["tag_name"] == "v0.4.2"


def test_check_for_update_uses_newest_semver(monkeypatch):
    class FakeResp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return [
                {
                    "tag_name": "v0.1.4",
                    "draft": False,
                    "prerelease": False,
                    "html_url": "https://github.com/example/releases/tag/v0.1.4",
                    "body": "old",
                    "assets": [
                        {
                            "name": "RamScoutAI-windows-x64.exe",
                            "browser_download_url": "https://example/old.exe",
                            "url": "https://api.example/old",
                        }
                    ],
                },
                {
                    "tag_name": "v0.4.2",
                    "draft": False,
                    "prerelease": False,
                    "html_url": "https://github.com/example/releases/tag/v0.4.2",
                    "body": "new",
                    "assets": [
                        {
                            "name": "RamScoutAI-windows-x64.exe",
                            "browser_download_url": "https://example/new.exe",
                            "url": "https://api.example/new",
                        }
                    ],
                },
            ]

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, params=None):
            return FakeResp()

    monkeypatch.setattr("ramscout.updater.httpx.Client", FakeClient)
    monkeypatch.setattr("ramscout.updater.github_token", lambda: "")
    monkeypatch.setattr("ramscout.updater.platform_key", lambda: "windows")
    info = check_for_update("0.4.0")
    assert info.available is True
    assert info.latest_version == "0.4.2"
    assert info.asset_url.endswith("new.exe")


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

        def get(self, url, params=None):
            return FakeResp()

    monkeypatch.setattr("ramscout.updater.httpx.Client", FakeClient)
    monkeypatch.setattr("ramscout.updater.github_token", lambda: "")
    info = check_for_update("0.4.0")
    assert info.available is False
    assert "private" in (info.error or "").lower() or "releases" in (info.error or "").lower()
    assert "releases" in (info.release_url or "")

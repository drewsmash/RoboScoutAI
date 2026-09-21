import hashlib
import json

import pytest

import roboscout_manager
from ramscout import __version__ as app_version
from roboscout_manager import manifest as mf


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_manager_version_tracks_app_version():
    # Setup/manager and app ship from one tag; keep the two constants in lockstep.
    assert roboscout_manager.__version__ == app_version == "0.7.0"


def test_version_helpers():
    assert mf.normalize_version("v1.2.3") == "1.2.3"
    assert mf.version_tuple("0.6.0-beta") == (0, 6, 0)
    assert mf.compare_versions("0.6.0", "0.6") == 0
    assert mf.compare_versions("0.10.0", "0.9.9") > 0
    assert mf.is_newer("1.0.0", "0.9.9")
    assert not mf.is_newer("0.6.0", "0.6.0")
    assert mf.looks_like_version("v0.6.1")
    assert not mf.looks_like_version("main")


def test_build_and_parse_manifest_roundtrip(tmp_path):
    app = _write(tmp_path, "RoboScoutAI-app-windows-x64.exe", b"app" * 100)
    mgr = _write(tmp_path, "RoboScoutAI.exe", b"mgr" * 10)
    setup = _write(tmp_path, "RoboScoutAI-Setup.exe", b"setup")
    built = mf.build_manifest([app, mgr, setup], version="v0.6.0", channel="main", commit="abc", created_at="now")
    parsed = mf.parse_manifest(built.to_json())
    assert parsed.version == "0.6.0"
    assert parsed.channel == "main"
    assert parsed.commit == "abc"
    assert parsed.min_manager_version == roboscout_manager.MANAGER_VERSION
    assert parsed.app_artifact().name == "RoboScoutAI-app-windows-x64.exe"
    assert parsed.app_artifact().kind == "app"
    assert parsed.manager_artifact().kind == "manager"
    assert parsed.setup_artifact().kind == "setup"
    assert parsed.app_artifact().sha256 == hashlib.sha256(b"app" * 100).hexdigest()
    assert parsed.app_artifact().size == 300
    assert parsed.app_artifact().platform == "windows-x64"


def test_parse_manifest_rejects_bad_input():
    with pytest.raises(mf.ManifestError):
        mf.parse_manifest("not json")
    with pytest.raises(mf.ManifestError):
        mf.parse_manifest({"version": "main"})
    with pytest.raises(mf.ManifestError):
        mf.parse_manifest({"version": "0.6.0", "artifacts": {"x.exe": {"sha256": "zz", "size": 1}}})
    with pytest.raises(mf.ManifestError):
        mf.parse_manifest({"version": "0.6.0", "schema": 99})


def test_verify_file_checks_sha_and_size(tmp_path):
    path = _write(tmp_path, "RoboScoutAI-app-windows-x64.exe", b"payload")
    art = mf.Artifact(name=path.name, sha256=hashlib.sha256(b"payload").hexdigest(), size=7)
    assert mf.verify_file(path, art) == art.sha256
    bad = mf.Artifact(name=path.name, sha256="0" * 64, size=7)
    with pytest.raises(mf.VerificationError):
        mf.verify_file(path, bad)
    wrong_size = mf.Artifact(name=path.name, sha256=art.sha256, size=8)
    with pytest.raises(mf.VerificationError):
        mf.verify_file(path, wrong_size)
    with pytest.raises(mf.VerificationError):
        mf.verify_file(tmp_path / "missing.exe", art)


def test_min_manager_version_gate():
    m = mf.parse_manifest({"version": "0.7.0", "min_manager_version": "0.7.0", "artifacts": {}})
    assert not mf.manager_satisfies(m, "0.6.0")
    assert mf.manager_satisfies(m, "0.7.0")
    assert mf.manager_satisfies(m, "1.0.0")
    m2 = mf.parse_manifest({"version": "0.7.0", "artifacts": {}})
    assert mf.manager_satisfies(m2, "0.0.1")


def test_manifest_cli_build_and_verify(tmp_path, capsys):
    app = _write(tmp_path, "RoboScoutAI-app-windows-x64.exe", b"x" * 10)
    out = tmp_path / "manifest.json"
    assert mf.main(["build", "--version", "0.6.0", "--channel", "main", "--out", str(out), str(app)]) == 0
    data = json.loads(out.read_text())
    assert data["version"] == "0.6.0"
    assert "RoboScoutAI-app-windows-x64.exe" in data["artifacts"]
    assert mf.main(["verify", str(out), str(app)]) == 0
    app.write_bytes(b"y" * 10)
    assert mf.main(["verify", str(out), str(app)]) == 1
    assert "FAIL" in capsys.readouterr().out

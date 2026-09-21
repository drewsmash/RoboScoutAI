"""Ensure required runtime packages are declared and importable."""

from __future__ import annotations

from pathlib import Path

from ramscout.deps import REQUIRED, check_deps


ROOT = Path(__file__).resolve().parents[1]


def test_required_deps_import():
    report = check_deps()
    assert report["ok"] is True, f"Missing required packages: {report['missing_required']}"
    assert not report["missing_required"]


def test_onnxruntime_listed_as_required():
    keys = {s.key for s in REQUIRED}
    assert "onnxruntime" in keys
    assert "opencv" in keys
    assert "fastapi" in keys


def test_requirements_files_mention_onnxruntime():
    full = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    desktop = (ROOT / "requirements-desktop.txt").read_text(encoding="utf-8")
    assert "onnxruntime" in full
    assert "onnxruntime" in desktop
    # Desktop freeze must stay slim: no ultralytics pin (comment mention is fine).
    assert not any(line.strip().startswith("ultralytics") for line in desktop.splitlines())
    assert any(line.strip().startswith("ultralytics") for line in full.splitlines())
    assert any(line.strip().startswith("onnxruntime") for line in full.splitlines())
    assert any(line.strip().startswith("onnxruntime") for line in desktop.splitlines())

def test_health_reports_deps(client=None):
    from fastapi.testclient import TestClient

    from app import app

    res = TestClient(app).get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert "deps_ok" in body
    assert body["deps_ok"] is True
    assert isinstance(body.get("missing_required"), list)

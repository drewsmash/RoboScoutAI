import time

from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_index_serves_dark_ui():
    res = client.get("/")
    assert res.status_code == 200
    assert "RamScoutAI" in res.text
    assert "color-scheme" in res.text


def test_demo_job_returns_scout_cards():
    created = client.post("/api/jobs", json={"demo": True})
    assert created.status_code == 200
    job_id = created.json()["id"]
    job = None
    for _ in range(50):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in {"ready", "error"}:
            break
        time.sleep(0.05)
    assert job is not None
    assert job["status"] == "ready"
    assert len(job["cards"]) == 6
    assert job["match"]["key"] == "2026nhdur_qm12"
    assert any(event["type"] == "hub_score_candidate" for event in job["events"])
    csv = client.get(f"/api/jobs/{job_id}/export.csv")
    assert csv.status_code == 200
    assert "hub_score_candidates" in csv.text


def test_analyze_requires_url():
    res = client.post("/api/jobs", json={"url": ""})
    assert res.status_code == 400

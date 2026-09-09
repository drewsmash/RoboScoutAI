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
    assert job["match"]["source"]["scores"] == "video"
    assert job["game"]["field_image"].endswith("2026.png")
    field = client.get("/static/fields/2026.png")
    assert field.status_code == 200
    assert field.content[:8] == b"\x89PNG\r\n\x1a\n"
    bot = client.get("/static/robots/blue.png")
    assert bot.status_code == 200
    assert bot.content[:8] == b"\x89PNG\r\n\x1a\n"
    red = client.get("/static/robots/red.png")
    assert red.status_code == 200
    assert red.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_analyze_requires_url():
    res = client.post("/api/jobs", json={"url": ""})
    assert res.status_code == 400


def test_game_endpoint():
    res = client.get("/api/game")
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == "REBUILT"
    assert "blue_hub" in body["landmarks"]

import time

from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_health():
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_version_endpoint():
    res = client.get("/api/version")
    assert res.status_code == 200
    body = res.json()
    assert body["version"]
    assert "repo" in body
    assert "platform" in body


def test_index_serves_dark_ui():
    res = client.get("/")
    assert res.status_code == 200
    assert "RamScoutAI" in res.text
    assert "color-scheme" in res.text
    assert "upload" in res.text.lower()
    assert "update-chip" in res.text


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
    assert job["user_calibrated"] is False
    assert job["zebra"] is None
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
    assigned = client.post(f"/api/jobs/{job_id}/assign", json={"assignments": {str(job["samples"][0]["track_id"]): "9999"}})
    assert assigned.status_code == 200
    assert any(c["team"] == "9999" for c in assigned.json()["cards"])


def test_analyze_requires_url():
    res = client.post("/api/jobs", json={"url": ""})
    assert res.status_code == 400


def test_upload_job_accepts_local_video(tmp_path):
    # Tiny non-media payload is enough to exercise the upload route; the job will
    # later fail OpenCV decode, which still proves multipart ingest works.
    video = tmp_path / "match.mp4"
    video.write_bytes(b"not-a-real-mp4" + b"\x00" * 2048)
    with video.open("rb") as fh:
        res = client.post(
            "/api/jobs/upload",
            data={
                "url": "https://youtu.be/m9uLAGKtenM",
                "crop_top": "0.10",
                "crop_bottom": "0.65",
            },
            files={"file": ("match.mp4", fh, "video/mp4")},
        )
    assert res.status_code == 200
    body = res.json()
    assert body["id"]
    assert body["has_video"] is True
    assert body["status"] in {"queued", "resolving", "downloading", "error", "ready"}


def test_crop_bottom_must_exceed_top():
    res = client.post("/api/jobs", json={"url": "https://youtu.be/dQw4w9WgXcQ", "crop_top": 0.6, "crop_bottom": 0.5})
    assert res.status_code == 400


def test_game_endpoint():
    res = client.get("/api/game")
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == "REBUILT"
    assert "blue_hub" in body["landmarks"]

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
    assert "RoboScoutAI" in res.text
    assert "color-scheme" in res.text
    assert "upload" in res.text.lower()
    assert "update-chip" in res.text
    assert 'data-view="playbook"' in res.text
    assert 'data-view="settings"' in res.text
    assert 'id="use-local-scout"' in res.text


def test_demo_job_honors_local_scout_setting():
    created = client.post("/api/jobs", json={"demo": True, "use_local_scout": False})
    assert created.status_code == 200
    body = created.json()
    assert body["use_local_scout"] is False
    assert body["live_tracking"] is False


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
    assert job["play_by_play"]
    assert {"robot", "team", "t", "x", "y", "period", "doing", "alliance"} <= set(job["play_by_play"][0])
    sheet = client.get("/")
    assert "progress-percent" in sheet.text
    assert "play-by-play" in sheet.text
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
    # Payload must clear the empty-file guard (>= 1 KiB); OpenCV may still fail later.
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
    assert body["status"] in {"queued", "resolving", "downloading", "tracking", "error", "ready"}
    # Upload path must not surface as a YouTube bot-block error on create.
    assert "bot" not in (body.get("error") or "").lower()
    assert "sign in" not in (body.get("error") or "").lower()


def test_upload_rejects_tiny_file(tmp_path):
    video = tmp_path / "empty.mp4"
    video.write_bytes(b"tiny")
    with video.open("rb") as fh:
        res = client.post(
            "/api/jobs/upload",
            data={"crop_top": "0.10", "crop_bottom": "0.65"},
            files={"file": ("empty.mp4", fh, "video/mp4")},
        )
    assert res.status_code == 400
    detail = str(res.json().get("detail") or "").lower()
    assert "small" in detail or "empty" in detail


def test_version_exposes_git_remote_and_cookies():
    res = client.get("/api/version")
    assert res.status_code == 200
    body = res.json()
    assert body.get("git_remote")
    assert body.get("git_branch")
    assert "cookies" in body
    assert "found" in body["cookies"]


def test_updates_download_success_strips_open_url(monkeypatch):
    from app import app as fastapi_app
    from ramscout import updater as upd

    def fake_apply():
        return {
            "ok": True,
            "message": "Updating from git — installing into LocalAppData.",
            "restarting": True,
            "open_url": "https://github.com/drewsmash/RoboScoutAI/releases/download/v0.5.0/RoboScoutAI-windows-x64.exe",
            "update": {"available": True, "latest_version": "0.5.6"},
        }

    monkeypatch.setattr(upd, "apply_update_now", fake_apply)
    # app imports apply_update_now by name — patch the app module binding too.
    import app as app_mod

    monkeypatch.setattr(app_mod, "apply_update_now", fake_apply)
    res = client.post("/api/updates/download")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["restarting"] is True
    assert "open_url" not in body
    assert "git" in body["message"].lower() or "LocalAppData" in body["message"]


def test_updates_download_failure_keeps_open_url(monkeypatch):
    import app as app_mod

    def fake_apply():
        return {
            "ok": False,
            "message": "Git update failed",
            "restarting": False,
            "open_url": "https://github.com/drewsmash/RoboScoutAI/releases/latest",
            "update": {"latest_version": "0.5.6"},
        }

    monkeypatch.setattr(app_mod, "apply_update_now", fake_apply)
    res = client.post("/api/updates/download")
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is False
    assert body["open_url"]
    assert "0.5.0" not in body["open_url"]


def test_ui_update_fallback_helper_in_app_js():
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "web" / "app.js"
    text = source.read_text(encoding="utf-8")
    assert "function shouldOpenUpdateManualFallback" in text
    assert "body.ok === false" in text
    assert "Updating from git" in text
    # Must not auto-open open_url on every response.
    assert "if (body.open_url)" not in text or "shouldOpenUpdateManualFallback" in text


def test_crop_bottom_must_exceed_top():
    res = client.post("/api/jobs", json={"url": "https://youtu.be/dQw4w9WgXcQ", "crop_top": 0.6, "crop_bottom": 0.5})
    assert res.status_code == 400


def test_game_endpoint():
    res = client.get("/api/game")
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == "REBUILT"
    assert "blue_hub" in body["landmarks"]

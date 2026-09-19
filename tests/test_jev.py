"""Tests for Vercel AI Gateway Jev evaluation helpers."""

from __future__ import annotations

from unittest.mock import patch

from ramscout import jev


def test_gateway_api_key_prefers_explicit(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "env-key")
    assert jev.gateway_api_key("job-key") == "job-key"
    assert jev.gateway_api_key("") == "env-key"
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    assert jev.gateway_api_key(None) == ""


def test_boolean_probability_accepts_noul_and_probability():
    assert jev.boolean_probability({"type": "boolean", "probability": 0.91}) == 0.91
    assert jev.boolean_probability({"type": "noul", "noul": 0.12}) == 0.12
    assert jev.boolean_probability({"type": "choice", "choice": "a"}) is None


def test_verify_scout_events_noops_without_key(monkeypatch):
    monkeypatch.delenv("AI_GATEWAY_API_KEY", raising=False)
    events = [{"type": "hub_score_candidate", "team": "195", "t": 12.0, "confidence": 0.4}]
    out, notes = jev.verify_scout_events(events, api_key="")
    assert out == events
    assert notes == []


def test_verify_scout_events_boosts_and_rejects(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "test-key")
    events = [
        {"type": "hub_score_candidate", "team": "195", "t": 10.0, "zone": "blue_hub", "confidence": 0.48, "detail": "dwell"},
        {"type": "defense", "team": "230", "t": 40.0, "zone": "red_half", "confidence": 0.5, "detail": "block"},
        {"type": "climb_attempt", "team": "177", "t": 140.0, "zone": "blue_tower", "confidence": 0.55, "detail": "park"},
    ]

    fake = {
        "model": jev.JEV_MODEL,
        "answers": {
            "e0": {"type": "boolean", "probability": 0.92},
            "e1": {"type": "boolean", "probability": 0.1},
            "e2": {"type": "boolean", "probability": 0.5},
        },
    }
    with patch.object(jev, "evaluate", return_value=fake) as mocked:
        out, notes = jev.verify_scout_events(events, match={"key": "qm1"}, api_key="test-key")
    assert mocked.called
    assert len(notes) == 1 and "Jev" in notes[0]
    by_team = {e["team"]: e for e in out}
    assert by_team["195"]["confidence"] > 0.48
    assert by_team["195"]["jev_probability"] == 0.92
    assert by_team["230"].get("jev_rejected") is True
    assert by_team["230"]["confidence"] <= 0.22
    assert abs(by_team["177"]["confidence"] - (0.5 * 0.55 + 0.5 * 0.5)) < 1e-6


def _fake_client(status_code: int, body: dict | str, captured: dict):
    class FakeResp:
        def __init__(self):
            self.status_code = status_code
            self.text = body if isinstance(body, str) else ""
            self._body = body

        def raise_for_status(self):
            if self.status_code >= 400:
                raise jev.httpx.HTTPStatusError("err", request=None, response=self)
            return None

        def json(self):
            if isinstance(self._body, dict):
                return self._body
            raise ValueError("not json")

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, headers=None, json=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeResp()

    return FakeClient


def test_evaluate_posts_to_ai_gateway(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw-secret")
    monkeypatch.delenv("AI_GATEWAY_ONLY", raising=False)
    monkeypatch.delenv("JEV_GATEWAY_ONLY", raising=False)
    monkeypatch.delenv("AI_GATEWAY_ZERO_DATA_RETENTION", raising=False)
    monkeypatch.delenv("JEV_ZERO_DATA_RETENTION", raising=False)
    captured: dict = {}

    with patch.object(
        jev.httpx,
        "Client",
        _fake_client(
            200,
            {"model": jev.JEV_MODEL, "answers": {"ok": {"type": "boolean", "probability": 1.0}}},
            captured,
        ),
    ):
        data = jev.evaluate(
            {"hello": "world"},
            {"ok": {"type": "boolean", "instructions": "yes?"}},
        )
    assert data["answers"]["ok"]["probability"] == 1.0
    assert captured["url"] == jev.EVALUATE_URL
    assert captured["json"]["model"] == "typesafe-ai/jev"
    assert captured["json"]["state"] == {"hello": "world"}
    assert captured["json"]["questions"]["ok"]["type"] == "boolean"
    assert "providerOptions" not in captured["json"]
    assert captured["headers"]["Authorization"] == "Bearer gw-secret"


def test_evaluate_provider_options_env_gated(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw")
    monkeypatch.setenv("AI_GATEWAY_ONLY", "typesafe-ai")
    monkeypatch.setenv("AI_GATEWAY_ZERO_DATA_RETENTION", "1")
    captured: dict = {}

    with patch.object(
        jev.httpx,
        "Client",
        _fake_client(200, {"model": jev.JEV_MODEL, "answers": {}}, captured),
    ):
        jev.evaluate({"s": 1}, {"q": {"type": "boolean", "instructions": "?"}})
    opts = captured["json"]["providerOptions"]["gateway"]
    assert opts["only"] == ["typesafe-ai"]
    assert opts["zeroDataRetention"] is True


def test_evaluate_403_raises_actionable_message(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "bad-key")
    captured: dict = {}

    with patch.object(
        jev.httpx,
        "Client",
        _fake_client(403, {"error": {"message": "Forbidden: no access"}}, captured),
    ):
        try:
            jev.evaluate({"s": 1}, {"q": {"type": "boolean", "instructions": "?"}})
            raise AssertionError("expected RuntimeError")
        except RuntimeError as exc:
            msg = str(exc)
            assert "403" in msg
            assert "AI_GATEWAY_API_KEY" in msg
            assert "ONLY" in msg or "ZDR" in msg or "ZERO_DATA" in msg
            assert "Forbidden" in msg


def test_verify_scout_events_soft_fails_on_403(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "bad-key")
    events = [
        {"type": "hub_score_candidate", "team": "195", "t": 10.0, "confidence": 0.4, "detail": "dwell"},
    ]
    captured: dict = {}

    with patch.object(
        jev.httpx,
        "Client",
        _fake_client(403, {"message": "denied"}, captured),
    ):
        out, notes = jev.verify_scout_events(events, api_key="bad-key")
    assert out == events
    assert len(notes) == 1
    assert "403" in notes[0]
    assert "AI_GATEWAY_API_KEY" in notes[0]


def test_classify_camera_layout(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "gw")
    fake = {
        "answers": {
            "layout": {
                "type": "choice",
                "choice": "stacked_sides",
                "probabilities": {"single": 0.05, "stacked_top": 0.1, "stacked_sides": 0.85},
            }
        }
    }
    with patch.object(jev, "evaluate", return_value=fake):
        mode, conf, notes = jev.classify_camera_layout({"classical_mode": "single"})
    assert mode == "stacked_sides"
    assert conf == 0.85
    assert notes == []


def test_classify_camera_layout_soft_fails_on_403(monkeypatch):
    monkeypatch.setenv("AI_GATEWAY_API_KEY", "bad")
    captured: dict = {}
    with patch.object(
        jev.httpx,
        "Client",
        _fake_client(403, {"error": "nope"}, captured),
    ):
        mode, conf, notes = jev.classify_camera_layout({"classical_mode": "single"})
    assert mode is None
    assert conf == 0.0
    assert notes and "403" in notes[0]

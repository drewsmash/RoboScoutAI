"""OpenAI vision sparse sampling, stride, and 429 backoff."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ramscout.trackers.ensemble import EnsembleTracker
from ramscout.trackers.openai_vision import (
    DEFAULT_FRAME_STRIDE,
    DEFAULT_INTERVAL_S,
    DEFAULT_MAX_CALLS,
    OpenAIVisionTracker,
)
from ramscout.trackers.types import Detection, TrackerContext


def _ctx(t: float = 0.0, frame_index: int = 0, key: str = "sk-test") -> TrackerContext:
    return TrackerContext(
        frame_w=320,
        frame_h=180,
        crop_w=320,
        crop_h=120,
        t=t,
        frame_index=frame_index,
        openai_key=key,
    )


def _frame() -> np.ndarray:
    return np.zeros((120, 320, 3), dtype=np.uint8)


def test_openai_defaults_are_thrifty():
    tracker = OpenAIVisionTracker()
    assert tracker.interval_s == DEFAULT_INTERVAL_S
    assert tracker.frame_stride == DEFAULT_FRAME_STRIDE
    assert tracker.max_calls == DEFAULT_MAX_CALLS
    assert tracker.interval_s >= 1.0
    assert tracker.frame_stride >= 15
    assert tracker.max_calls <= 48


def test_openai_holds_boxes_between_time_samples():
    tracker = OpenAIVisionTracker(interval_s=2.0, frame_stride=1, max_calls=10)
    calls = {"n": 0}

    def fake_query(cropped, api_key, model):
        calls["n"] += 1
        return [{"x1": 0.1, "y1": 0.1, "x2": 0.3, "y2": 0.4, "alliance": "blue", "team": "254"}]

    with patch.object(tracker, "_query", side_effect=fake_query):
        d0 = tracker.detect(_frame(), _ctx(t=0.0, frame_index=0))
        d1 = tracker.detect(_frame(), _ctx(t=0.5, frame_index=1))
        d2 = tracker.detect(_frame(), _ctx(t=1.5, frame_index=2))
        d3 = tracker.detect(_frame(), _ctx(t=2.1, frame_index=3))

    assert calls["n"] == 2
    assert len(d0) == 1 and d0[0].source == "openai"
    assert d1 == d0  # hold
    assert d2 == d0  # still inside interval
    assert len(d3) == 1


def test_openai_respects_frame_stride_even_when_time_allows():
    tracker = OpenAIVisionTracker(interval_s=0.01, frame_stride=30, max_calls=10)
    calls = {"n": 0}

    def fake_query(cropped, api_key, model):
        calls["n"] += 1
        return [{"x1": 0.2, "y1": 0.2, "x2": 0.4, "y2": 0.5, "alliance": "red", "team": ""}]

    with patch.object(tracker, "_query", side_effect=fake_query):
        tracker.detect(_frame(), _ctx(t=0.0, frame_index=0))
        tracker.detect(_frame(), _ctx(t=1.0, frame_index=10))
        tracker.detect(_frame(), _ctx(t=2.0, frame_index=20))
        tracker.detect(_frame(), _ctx(t=3.0, frame_index=30))

    assert calls["n"] == 2  # frames 0 and 30 only


def test_openai_empty_result_still_paces_calls():
    """Regression: empty cache used to bypass interval and call every frame."""
    tracker = OpenAIVisionTracker(interval_s=2.0, frame_stride=15, max_calls=20)
    calls = {"n": 0}

    def fake_query(cropped, api_key, model):
        calls["n"] += 1
        return []  # no robots — must NOT re-query next frame

    with patch.object(tracker, "_query", side_effect=fake_query):
        for i in range(10):
            tracker.detect(_frame(), _ctx(t=i * 0.1, frame_index=i))

    assert calls["n"] == 1


def test_openai_caps_max_calls_per_match():
    tracker = OpenAIVisionTracker(interval_s=0.01, frame_stride=1, max_calls=3)
    calls = {"n": 0}

    def fake_query(cropped, api_key, model):
        calls["n"] += 1
        return [{"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2, "alliance": "blue", "team": ""}]

    with patch.object(tracker, "_query", side_effect=fake_query):
        for i in range(8):
            tracker.detect(_frame(), _ctx(t=float(i), frame_index=i * 40))

    assert calls["n"] == 3
    assert any("max 3" in w for w in tracker.warnings)


def test_openai_429_backs_off_without_model_cascade():
    import urllib.error

    tracker = OpenAIVisionTracker(interval_s=0.01, frame_stride=1, max_calls=10)
    attempted: list[str] = []

    def fake_urlopen(req, timeout=45):
        # Capture model from body is hard; count attempts instead.
        attempted.append("hit")
        raise urllib.error.HTTPError(
            url="https://api.openai.com/v1/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=None,
        )

    with (
        patch("ramscout.trackers.openai_vision.encode_jpeg_bgr", return_value=b"fake"),
        patch("urllib.request.urlopen", side_effect=fake_urlopen),
        patch(
            "ramscout.trackers.openai_vision.openai_model_candidates",
            return_value=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"],
        ),
    ):
        dets = tracker.detect(_frame(), _ctx(t=0.0, frame_index=0))

    # Only one HTTP attempt — do not walk the full model list on 429.
    assert len(attempted) == 1
    assert dets == []
    assert tracker._fail_until > 0
    assert any("429" in w or "rate-limited" in w for w in tracker.warnings)

    # While cooling down, further frames must not call again.
    with patch.object(tracker, "_query") as q:
        tracker.detect(_frame(), _ctx(t=0.5, frame_index=1))
        q.assert_not_called()


def test_ensemble_skips_openai_when_gemini_has_boxes():
    class FakeGemini:
        name = "gemini"
        warnings: list[str] = []

        def detect(self, cropped, ctx):
            return [Detection(6001, [10, 10, 40, 40], "gemini", alliance="blue")]

    class CountingOpenAI:
        name = "openai"
        warnings: list[str] = []
        calls = 0

        def detect(self, cropped, ctx):
            self.calls += 1
            return [Detection(5001, [12, 12, 42, 42], "openai", alliance="red")]

    openai = CountingOpenAI()
    ens = EnsembleTracker([FakeGemini(), openai])
    dets = ens.detect(_frame(), _ctx())
    assert openai.calls == 0
    assert any(d.source == "gemini" for d in dets)

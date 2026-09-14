"""Cloud vision model discovery helpers."""

from __future__ import annotations

from ramscout.trackers.cloud_models import (
    GEMINI_STATIC_FALLBACKS,
    gemini_model_candidates,
    openai_model_candidates,
)


def test_gemini_candidates_prefer_requested_then_fallbacks():
    cands = gemini_model_candidates("", preferred="gemini-2.0-flash")
    assert cands[0] == "gemini-2.0-flash"
    assert "gemini-flash-latest" in cands
    assert cands.index("gemini-flash-latest") < cands.index("gemini-3.5-flash")


def test_openai_candidates_include_4o_mini():
    cands = openai_model_candidates(preferred="gpt-old-model")
    assert cands[0] == "gpt-old-model"
    assert "gpt-4o-mini" in cands
    assert set(GEMINI_STATIC_FALLBACKS).issubset(set(gemini_model_candidates("")))

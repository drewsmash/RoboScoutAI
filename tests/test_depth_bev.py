"""Tests for depth estimation, BEV calibration, and multi-view sectioning."""

from __future__ import annotations

import numpy as np

from ramscout.bev import calibrate_bev, project_detections_bev, warp_crop_to_bev
from ramscout.depth import estimate_depth, refine_foot_point, sample_depth
from ramscout.field import FIELD_LENGTH, FIELD_WIDTH
from ramscout.multicam import analyze_frame, apply_layout
from ramscout.multiview import SideCue, merge_side_cues_into_events, _thin_cues


def _fieldish_frame(h=240, w=420):
    frame = np.full((h, w, 3), 40, dtype=np.uint8)
    # Brighter carpet toward the bottom (closer to camera).
    for y in range(h):
        frame[y, :, 1] = int(40 + 140 * (y / max(h - 1, 1)))
    # Dark horizontal gutter mid-frame for stacked layout tests.
    return frame


def test_classical_depth_estimates_pitch():
    frame = _fieldish_frame()
    result = estimate_depth(frame, prefer_neural=False)
    assert result.source == "classical"
    assert result.depth.shape[:2] == frame.shape[:2]
    assert 10.0 <= result.pitch_deg <= 60.0
    assert 0.0 <= result.tilt_strength <= 1.0
    assert sample_depth(result.depth, 10, 10) >= 0.0


def test_missing_pil_falls_back_to_classical(monkeypatch, caplog):
    """Depth Anything path should soft-fail with a Pillow install hint."""
    import builtins
    import logging

    import ramscout.depth as depth_mod

    monkeypatch.setattr(depth_mod, "_PIPE", None)
    monkeypatch.setattr(depth_mod, "_PIPE_FAILED", False)
    # Exercise the torch path: pretend the ONNX backend is not present.
    monkeypatch.setattr(depth_mod, "_get_onnx_session", lambda: None)

    real_import = builtins.__import__

    def _block_pil(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("No module named 'PIL'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block_pil)
    with caplog.at_level(logging.INFO, logger="ramscout.depth"):
        result = estimate_depth(_fieldish_frame(), prefer_neural=True)
    assert result.source == "classical"
    assert any("pip install Pillow" in r.message for r in caplog.records)


def test_refine_foot_point_blends_toward_deep_band():
    depth = np.zeros((40, 40), dtype=np.float32)
    depth[30:40, 10:30] = 1.0
    fx, fy = refine_foot_point([10, 5, 30, 35], depth, crop_y0=0)
    assert 10 <= fx <= 30
    assert fy >= 20


def test_calibrate_bev_returns_four_corners():
    frame = _fieldish_frame(360, 640)
    cal, depth = calibrate_bev(frame, crop_top=0.1, crop_bottom=0.7, prefer_neural=False)
    assert len(cal.src_points) == 4
    assert cal.used_depth is True
    assert depth.source == "classical"
    mapped = project_detections_bev(
        [cal.src_points[0], cal.src_points[2]],
        cal.src_points,
    )
    assert mapped.shape == (2, 2)
    assert 0 <= mapped[0][0] <= FIELD_LENGTH
    assert 0 <= mapped[0][1] <= FIELD_WIDTH
    warped, H = warp_crop_to_bev(frame[36:252, :])
    assert warped.shape[0] > 0 and warped.shape[1] > 0
    assert H.shape == (3, 3)


def test_multicam_sections_side_panes_when_split():
    h, w = 400, 640
    frame = np.random.randint(60, 200, (h, w, 3), dtype=np.uint8)
    frame[190:210, :, :] = 4
    # Vertical gutter in lower half.
    frame[220:390, 310:330, :] = 4
    layout = analyze_frame(frame)
    assert layout.mode in {"stacked_top", "stacked_sides", "single"}
    assert layout.panes
    overview = layout.overview_pane()
    assert overview is not None
    assert overview.role == "overview"
    top, bottom, chosen = apply_layout(layout, user_crop_top=0.10, user_crop_bottom=0.65, auto=True)
    assert top < bottom
    if chosen.mode == "stacked_sides":
        roles = {p.role for p in chosen.panes}
        assert "blue_side" in roles and "red_side" in roles


def test_merge_side_cues_boosts_hub_events():
    events = [
        {
            "team": "59",
            "type": "hub_score_candidate",
            "t": 40.0,
            "zone": "blue_hub",
            "confidence": 0.5,
            "detail": "dwell",
        }
    ]
    cues = [
        {
            "t": 40.5,
            "alliance": "blue",
            "kind": "hub_activity",
            "confidence": 0.8,
            "detail": "side dwell",
            "role": "blue_side",
        }
    ]
    cards = [{"team": "59", "alliance": "blue"}]
    out = merge_side_cues_into_events(events, cues, cards=cards)
    assert out[0]["confidence"] > 0.5
    assert out[0].get("side_confirmed") is True


def test_thin_cues_keeps_best_per_window():
    cues = [
        SideCue(10.0, "blue", "hub_activity", 0.4, "a", "blue_side"),
        SideCue(10.5, "blue", "hub_activity", 0.9, "b", "blue_side"),
        SideCue(12.1, "blue", "hub_activity", 0.5, "c", "blue_side"),
        SideCue(10.2, "blue", "motion", 0.9, "ignore", "blue_side"),
    ]
    thin = _thin_cues(cues, window_s=2.0)
    assert len(thin) == 2
    assert thin[0].confidence == 0.9


# --------------------------------------------------------------- backend chain


def _no_download(monkeypatch):
    import ramscout.depth as depth_mod

    monkeypatch.setenv("ROBOSCOUT_NO_DOWNLOAD", "1")
    depth_mod.reset_backends()
    return depth_mod


def test_depth_backends_reports_missing_modules(monkeypatch):
    import builtins

    depth_mod = _no_download(monkeypatch)
    real_import = builtins.__import__

    def _block(name, *args, **kwargs):
        if name in {"onnxruntime", "transformers", "torch"}:
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _block)
    status = depth_mod.depth_backends()
    assert status["backends"]["onnx"]["importable"] is False
    assert "onnxruntime" in status["backends"]["onnx"]["missing"]
    assert status["backends"]["torch"]["importable"] is False
    assert "torch" in status["backends"]["torch"]["missing"] or "transformers" in status["backends"]["torch"]["missing"]
    assert status["backends"]["classical"]["importable"] is True
    assert status["preferred_order"] == ["onnx", "torch", "classical"]


def test_onnx_backend_used_when_session_available(monkeypatch, tmp_path):
    depth_mod = _no_download(monkeypatch)

    class _Sess:
        def run(self, _names, feeds):
            x = feeds["pixel_values"]
            assert x.shape == (1, 3, 518, 518)
            # Fake "closer at the bottom" depth ramp.
            ramp = np.linspace(0.0, 1.0, 518, dtype=np.float32).reshape(1, 518, 1)
            return [np.repeat(ramp, 518, axis=2)]

    monkeypatch.setattr(depth_mod, "_get_onnx_session", lambda: _Sess())
    result = depth_mod.estimate_depth(_fieldish_frame(120, 200), prefer_neural=True)
    assert result.source == "depth_anything_v2_onnx"
    assert result.neural is True
    assert result.depth.shape == (120, 200)
    assert result.depth.min() >= 0.0 and result.depth.max() <= 1.0
    assert depth_mod.active_backend() == "onnx"
    # Bottom rows closer → tilt should be substantial.
    assert result.tilt_strength > 0.3


def test_onnx_failure_falls_through_to_torch_then_classical(monkeypatch):
    depth_mod = _no_download(monkeypatch)
    monkeypatch.setattr(depth_mod, "_get_onnx_session", lambda: None)
    calls = {"torch": 0}

    def _fake_torch(frame):
        calls["torch"] += 1
        return None

    monkeypatch.setattr(depth_mod, "_try_depth_anything", _fake_torch)
    result = depth_mod.estimate_depth(_fieldish_frame(), prefer_neural=True)
    assert calls["torch"] == 1
    assert result.source == "classical"
    assert depth_mod.active_backend() == "classical"


def test_torch_backend_used_when_onnx_unavailable(monkeypatch):
    depth_mod = _no_download(monkeypatch)
    monkeypatch.setattr(depth_mod, "_get_onnx_session", lambda: None)

    def _fake_torch(frame):
        h, w = frame.shape[:2]
        return np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1).repeat(w, axis=1)

    monkeypatch.setattr(depth_mod, "_try_depth_anything", _fake_torch)
    result = depth_mod.estimate_depth(_fieldish_frame(), prefer_neural=True)
    assert result.source == "depth_anything_v2"
    assert depth_mod.active_backend() == "torch"


def test_ensure_onnx_model_respects_no_download(monkeypatch, tmp_path):
    depth_mod = _no_download(monkeypatch)
    monkeypatch.setenv("ROBOSCOUT_MODELS", str(tmp_path))
    assert depth_mod.onnx_model_cached() is False
    assert depth_mod.ensure_onnx_model() is None
    # A too-small (corrupt) cached file is not accepted either.
    (tmp_path / "depth-anything-v2-small.onnx").write_bytes(b"x" * 100)
    assert depth_mod.onnx_model_cached() is False

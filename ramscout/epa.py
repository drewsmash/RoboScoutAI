"""Statbotics-style EPA fetch (best-effort, optional)."""

from __future__ import annotations

import logging
from typing import Any

import httpx

log = logging.getLogger(__name__)

STATBOTICS_BASE = "https://api.statbotics.io/v3"


def fetch_team_epa(team: int, year: int = 2026, timeout: float = 12.0) -> dict[str, Any] | None:
    try:
        with httpx.Client(timeout=timeout, headers={"User-Agent": "RamScoutAI/0.4"}) as client:
            response = client.get(f"{STATBOTICS_BASE}/team_year/{int(team)}/{int(year)}")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            data = response.json()
    except Exception as exc:  # noqa: BLE001
        log.info("Statbotics EPA lookup failed for %s: %s", team, exc)
        return None
    epa = data.get("epa") or {}
    return {
        "team": int(team),
        "year": int(year),
        "epa": float(epa.get("total_points") or epa.get("end") or epa.get("norm") or 0),
        "unitless": float(epa.get("unitless") or 0),
        "rank": (data.get("epa") or {}).get("ranks") or {},
        "record": data.get("record") or {},
        "source": "statbotics",
    }


def fetch_epa_map(teams: list[int], year: int = 2026) -> dict[str, float]:
    out: dict[str, float] = {}
    for team in teams:
        info = fetch_team_epa(int(team), year=year)
        if info and info.get("epa") is not None:
            out[str(int(team))] = float(info["epa"])
    return out

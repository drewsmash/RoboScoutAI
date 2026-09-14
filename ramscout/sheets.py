"""Spreadsheet-friendly exports for common scout workflows."""

from __future__ import annotations

import csv
import io
from typing import Any


def cards_to_sheets_csv(cards: list[dict[str, Any]]) -> str:
    """CSV columns compatible with typical alliance-selection Google Sheets."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "Team",
            "Nickname",
            "Matches",
            "Hub Candidates",
            "Climb Attempt",
            "Climb Rate",
            "Defense Seconds",
            "Collection Seconds",
            "Path Length in",
            "Max Speed in/s",
            "Avg Speed in/s",
            "Fouls",
            "Do Not Pick",
            "Notes",
            "EPA",
            "Role Score",
        ]
    )
    for card in cards or []:
        writer.writerow(
            [
                card.get("team"),
                card.get("nickname") or "",
                card.get("matches") or 1,
                card.get("hub_score_candidates") or 0,
                "yes" if card.get("climb_attempt") else "no",
                card.get("climb_rate") if card.get("climb_rate") is not None else "",
                card.get("defense_time_s") or 0,
                card.get("collection_time_s") or 0,
                card.get("path_length_in") or 0,
                card.get("max_speed_in_s") or 0,
                card.get("avg_speed_in_s") or 0,
                card.get("foul_count") or 0,
                "yes" if card.get("do_not_pick") else "",
                (card.get("notes") or "").replace("\n", " "),
                card.get("epa") or "",
                card.get("role_score") or card.get("score") or "",
            ]
        )
    return buffer.getvalue()


def picklist_to_sheets_csv(ranked: list[dict[str, Any]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Rank", "Team", "Nickname", "Score", "Role", "Reasons"])
    for row in ranked or []:
        writer.writerow(
            [
                row.get("rank") or row.get("rank"),
                row.get("team"),
                row.get("nickname") or "",
                row.get("score"),
                row.get("role") or "balanced",
                "; ".join(row.get("reasons") or row.get("fit_reasons") or []),
            ]
        )
    return buffer.getvalue()

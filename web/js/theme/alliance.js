/** Canonical alliance / identity colors for CSS + canvas. Never coerce unknown→blue. */

export const ALLIANCE = Object.freeze({
  red: "red",
  blue: "blue",
  unknown: "unknown",
});

export const ALLIANCE_HEX = Object.freeze({
  red: "#ff6b6b",
  blue: "#5b9cff",
  unknown: "#94a3b8",
});

export const ALLIANCE_HEX_SOFT = Object.freeze({
  red: "rgba(255,107,107,0.85)",
  blue: "rgba(91,156,255,0.85)",
  unknown: "rgba(148,163,184,0.85)",
});

/** Resolve alliance to red | blue | unknown (never invent blue). */
export function resolveAlliance(value) {
  const v = String(value || "").toLowerCase().trim();
  if (v === "red") return "red";
  if (v === "blue") return "blue";
  return "unknown";
}

export function allianceColor(value, { soft = false } = {}) {
  const key = resolveAlliance(value);
  return soft ? ALLIANCE_HEX_SOFT[key] : ALLIANCE_HEX[key];
}

export function allianceLabel(value) {
  const key = resolveAlliance(value);
  if (key === "red") return "Red";
  if (key === "blue") return "Blue";
  return "Unknown";
}

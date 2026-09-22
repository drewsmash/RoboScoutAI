const FIELD = { L: 651.2, W: 317.7, ALLIANCE: 158.6 };
const AUTO_END = 20;
const ENDGAME_START = 130;
const MATCH_END = 160;

import { runBrowserPotato, shouldRunBrowserPotato } from "./potato.js";
import { allianceColor, allianceLabel, resolveAlliance } from "./js/theme/alliance.js";

const $ = (id) => document.getElementById(id);

const state = {
  job: null,
  poll: null,
  playing: false,
  t: 0,
  lastTs: 0,
  calPoints: [],
  fieldImg: null,
  bots: { blue: null, red: null, unknown: null },
  landmarks: {},
  calMode: false,
  calJob: null,
  selectedSeed: null,
  picklist: null,
  potatoJobs: new Set(),
  potatoRunning: false,
  view: "home",
  cropLocked: false,
  selectedRobot: null,
};

function applyGame(game) {
  if (!game) return;
  if (game.length_in) FIELD.L = game.length_in;
  if (game.width_in) FIELD.W = game.width_in;
  if (game.alliance_depth_in) FIELD.ALLIANCE = game.alliance_depth_in;
  state.landmarks = game.landmarks || {};
  const tag = $("game-tagline");
  if (tag && game.name) {
    tag.textContent = `FRC ${game.year} ${game.name} · auto-scout from match video`;
  }
  const yearEl = $("field-year");
  if (yearEl && game.name) yearEl.textContent = `${game.year} ${game.name}`;
  const slider = $("time-slider");
  if (slider && game.match_end_s) slider.max = String(game.match_end_s);
  loadAsset("field", game.field_image, (img) => {
    state.fieldImg = img;
    drawField();
  });
  const icons = game.robot_icons || {};
  loadAsset("blueBot", icons.blue || "/static/robots/blue.png", (img) => {
    state.bots.blue = img;
    drawField();
  });
  loadAsset("redBot", icons.red || "/static/robots/red.png", (img) => {
    state.bots.red = img;
    drawField();
  });
  // Unknown stays gray — never fall back to the blue sprite.
  state.bots.unknown = null;
}

function loadAsset(key, src, onload) {
  if (!src) return;
  if (state[`_src_${key}`] === src && state[`_img_${key}`]) {
    onload(state[`_img_${key}`]);
    return;
  }
  const img = new Image();
  img.onload = () => {
    state[`_src_${key}`] = src;
    state[`_img_${key}`] = img;
    onload(img);
  };
  img.onerror = () => {
    state[`_src_${key}`] = src;
    state[`_img_${key}`] = null;
  };
  img.src = src;
}

function storageGet(key, fallback = null) {
  const modern = `roboscout.${key}`;
  const legacy = `ramscout.${key}`;
  const value = localStorage.getItem(modern);
  if (value != null) return value;
  const old = localStorage.getItem(legacy);
  if (old != null) {
    localStorage.setItem(modern, old);
    return old;
  }
  return fallback;
}

function storageSet(key, value) {
  localStorage.setItem(`roboscout.${key}`, value);
}

function loadTbaKey() {
  const saved = storageGet("tbaKey", "") || "";
  $("tba-key").value = saved;
  const openai = storageGet("openaiKey", "") || "";
  const google = storageGet("googleKey", "") || "";
  const gateway = storageGet("aiGatewayKey", "") || "";
  if ($("openai-key")) $("openai-key").value = openai;
  if ($("google-key")) $("google-key").value = google;
  if ($("ai-gateway-key")) $("ai-gateway-key").value = gateway;
}

$("google-key")?.addEventListener("change", () => {
  saveScoutKeys();
});

function saveScoutKeys() {
  storageSet("tbaKey", $("tba-key").value.trim());
  if ($("openai-key")) storageSet("openaiKey", $("openai-key").value.trim());
  if ($("google-key")) storageSet("googleKey", $("google-key").value.trim());
  if ($("ai-gateway-key")) storageSet("aiGatewayKey", $("ai-gateway-key").value.trim());
}

function trackerPayload() {
  return {
    // Always the best available method. Manual tracker selection was removed.
    tracker_mode: "auto",
    auto_multicam: $("auto-multicam")?.checked !== false,
    top_overview_only: $("top-overview-only")?.checked !== false,
    crop_locked: Boolean(state.cropLocked),
    openai_key: $("openai-key")?.value.trim() || "",
    google_key: $("google-key")?.value.trim() || "",
    ai_gateway_key: $("ai-gateway-key")?.value.trim() || "",
  };
}

function robotThumb(alliance) {
  const key = resolveAlliance(alliance);
  if (key === "red") return "/static/robots/red.png";
  if (key === "blue") return "/static/robots/blue.png";
  return ""; // unknown: CSS swatch, never blue
}

$("start-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  saveScoutKeys();
  const fileInput = $("video-file");
  const file = fileInput?.files?.[0] || null;
  const url = $("url").value.trim();
  if (!file && !url) {
    alert("Upload an MP4/MKV match VOD, or paste a YouTube URL.");
    return;
  }
  if (file && file.size < 1024) {
    alert("That file looks empty. Choose a real downloaded MP4/MKV match video.");
    return;
  }
  await createJob({
    url,
    tba_key: $("tba-key").value.trim(),
    event_key: $("event-key").value.trim(),
    match_key: $("match-key").value.trim(),
    crop_top: Number($("crop-top").value || 0.1),
    crop_bottom: Number($("crop-bottom").value || 0.65),
    ...trackerPayload(),
    demo: false,
    file,
  });
});

$("demo-btn").addEventListener("click", async () => {
  saveScoutKeys();
  await createJob({ url: "", tba_key: $("tba-key").value.trim(), demo: true });
});

$("play-btn").addEventListener("click", () => {
  state.playing = !state.playing;
  $("play-btn").textContent = state.playing ? "❚❚" : "▶";
  if (state.playing) {
    state.lastTs = performance.now();
    requestAnimationFrame(tick);
  }
});

$("time-slider").addEventListener("input", (event) => {
  state.t = Number(event.target.value);
  $("time-label").textContent = `${state.t.toFixed(1)}s`;
  drawField();
});

$("cal-reset").addEventListener("click", () => {
  state.calPoints = [];
  state.selectedSeed = null;
  drawCal();
  updateCalHelp();
});

$("cal-toggle")?.addEventListener("click", () => {
  state.calMode = true;
  if (state.job) renderJob(state.job);
});
$("video-toggle")?.addEventListener("click", () => {
  state.calMode = false;
  if (state.job) renderJob(state.job);
});

$("assign-team")?.addEventListener("change", async () => {
  if (!state.job || !state.selectedSeed) return;
  const team = $("assign-team").value.trim();
  if (!team) return;
  await postAssign({ [String(state.selectedSeed.track_id)]: team });
});

$("cal-canvas").addEventListener("click", async (event) => {
  const canvas = $("cal-canvas");
  const rect = canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
  const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
  if (state.calPoints.length < 4) {
    state.calPoints.push([x, y]);
    drawCal();
    updateCalHelp();
    if (state.calPoints.length === 4 && state.job) {
      const res = await fetch(`/api/jobs/${state.job.id}/calibrate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ src_points: state.calPoints }),
      });
      state.job = await res.json();
      renderJob(state.job);
    }
    return;
  }
  const seed = hitSeed(x, y, state.job?.seeds || []);
  if (!seed || !state.job) return;
  state.selectedSeed = seed;
  const next = cycleTeam(state.job, seed);
  fillAssignSelect(state.job, next);
  await postAssign({ [String(seed.track_id)]: next });
});

async function createJob(payload) {
  let res;
  if (payload.file) {
    const form = new FormData();
    form.append("file", payload.file);
    form.append("url", payload.url || "");
    form.append("tba_key", payload.tba_key || "");
    form.append("event_key", payload.event_key || "");
    form.append("match_key", payload.match_key || "");
    form.append("crop_top", String(payload.crop_top ?? 0.1));
    form.append("crop_bottom", String(payload.crop_bottom ?? 0.65));
    form.append("tracker_mode", payload.tracker_mode || "auto");
    form.append("openai_key", payload.openai_key || "");
    form.append("google_key", payload.google_key || "");
    form.append("ai_gateway_key", payload.ai_gateway_key || "");
    form.append("auto_multicam", payload.auto_multicam === false ? "false" : "true");
    form.append("top_overview_only", payload.top_overview_only === false ? "false" : "true");
    form.append("crop_locked", payload.crop_locked ? "true" : "false");
    res = await fetch("/api/jobs/upload", { method: "POST", body: form });
  } else {
    const { file: _file, ...body } = payload;
    res = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  }
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Could not start job." }));
    const detail = err.detail;
    const message = Array.isArray(detail)
      ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
      : (detail || "Could not start job.");
    alert(message);
    return;
  }
  const job = await res.json();
  state.job = job;
  navigate("studio");
  renderJob(job);
  if (state.poll) clearInterval(state.poll);
  state.poll = setInterval(refreshJob, 700);
}

async function refreshJob() {
  if (!state.job) return;
  const res = await fetch(`/api/jobs/${state.job.id}`);
  if (!res.ok) return;
  const job = await res.json();
  state.job = job;
  renderJob(job);
  if (job.status === "ready" || job.status === "error" || job.status === "cancelled") {
    clearInterval(state.poll);
    state.poll = null;
  }
}

function renderJob(job) {
  window.__roboscoutJobId = job.id;
  window.dispatchEvent(new CustomEvent("roboscout:job", { detail: job }));
  const ytHelp = $("yt-help");
  const uploadHelp = $("upload-help");
  const errText = `${job.error || ""} ${job.message || ""} ${(job.warnings || []).join(" ")}`;
  const errLower = errText.toLowerCase();
  const failed = job.status === "error";
  const hasVideo = Boolean(job.has_video);
  const isUpload = job.media_source === "upload" || (hasVideo && !String(job.url || "").includes("youtu"));
  // Never show YouTube-blocked help for local uploads — even if an error mentions "bot".
  const youtubeBlocked =
    failed
    && !isUpload
    && !hasVideo
    && (
      errLower.includes("bot")
      || errLower.includes("sign in")
      || (errLower.includes("youtube") && (errLower.includes("block") || errLower.includes("cookies")))
    );
  const uploadFailed =
    failed
    && (isUpload || hasVideo)
    && !youtubeBlocked
    && (
      isUpload
      || errLower.includes("upload")
      || errLower.includes("missing on disk")
      || errLower.includes("empty")
      || errLower.includes("opencv")
      || errLower.includes("could not")
      || errLower.includes("decode")
      || errLower.includes("video")
    );
  if (ytHelp) ytHelp.hidden = !youtubeBlocked;
  if (uploadHelp) uploadHelp.hidden = !(uploadFailed || (failed && isUpload));

  const pct = Math.max(0, Math.min(100, Number(job.progress || 0)));
  $("progress-status").textContent = labelStatus(job.status);
  $("progress-message").textContent = job.error || job.message || "";
  const percentEl = $("progress-percent");
  if (percentEl) percentEl.textContent = `${Math.round(pct)}%`;
  const track = $("progress-track");
  if (track) track.setAttribute("aria-valuenow", String(Math.round(pct)));
  $("progress-fill").style.width = `${pct}%`;
  renderProgressClock(job);
  const cancelBtn = $("cancel-job");
  if (cancelBtn) cancelBtn.hidden = ["ready", "error", "cancelled"].includes(job.status);
  $("progress-panel").hidden = job.status === "ready";
  $("progress-panel")?.classList.toggle("thinking", job.status !== "ready" && job.status !== "error" && job.status !== "cancelled");
  renderThinkStages(job);
  renderViews(job);

  const match = job.match;
  const info = job.video_info;
  applyGame(job.game);
  $("match-title").textContent = info?.title || match?.source?.channels?.[0] || "Match video";
  if (match) {
    $("scoreboard").hidden = false;
    $("match-key-label").textContent = match.key || "Match";
    $("blue-score").textContent = match.alliances?.blue?.score ?? "—";
    $("red-score").textContent = match.alliances?.red?.score ?? "—";
    $("winner-label").textContent = match.winning_alliance
      ? `${match.winning_alliance.toUpperCase()} alliance win`
      : "";
    const src = match.source || {};
    const scoreSrc = (
      src.scores === "tba" ? "Scores from TBA"
      : src.scores === "firstevents" ? "Scores from FIRST Events"
      : "Scores from video"
    );
    $("score-source").textContent = scoreSrc;
    const overlayEl = $("overlay-source");
    const channels = src.channels || job.overlay?.sources || [];
    if (overlayEl) {
      const trackBits = [];
      if (job.tracker_mode) trackBits.push(`Track: ${job.tracker_mode}`);
      if (job.camera?.mode) trackBits.push(`Cam: ${job.camera.mode}`);
      if (job.bev?.depth_source) trackBits.push(`Depth: ${job.bev.depth_source}`);
      if (job.bev?.pitch_deg != null) trackBits.push(`Pitch≈${Math.round(job.bev.pitch_deg)}°`);
      const hits = job.source_hits || {};
      const hitNames = Object.keys(hits).filter((k) => hits[k] > 0);
      if (hitNames.length) trackBits.push(hitNames.join("+"));
      const parts = [...channels.map(labelChannel), ...trackBits];
      if (parts.length) {
        overlayEl.hidden = false;
        overlayEl.textContent = parts.join(" · ");
      } else {
        overlayEl.hidden = true;
      }
    }
    renderTeams("blue-teams", match, "blue");
    renderTeams("red-teams", match, "red");
  }

  const video = $("match-video");
  const empty = $("video-empty");
  const cal = $("cal-canvas");
  if (state.calJob !== job.id) {
    state.calJob = job.id;
    state.calPoints = (job.src_points && job.src_points.length === 4) ? job.src_points.map((p) => [...p]) : [];
    state.calMode = Boolean(job.has_frame && !job.user_calibrated && !job.demo);
    state.selectedSeed = null;
  }
  const showCal = Boolean(job.has_frame && (state.calMode || (!job.has_video && !job.demo)));
  $("cal-toggle").hidden = !(job.has_frame && job.has_video && !showCal);
  $("video-toggle").hidden = !(job.has_video && showCal);
  if (job.has_video && !showCal) {
    video.hidden = false;
    empty.hidden = true;
    cal.hidden = true;
    $("cal-bar").hidden = true;
    if (video.dataset.src !== job.id) {
      video.src = `/api/jobs/${job.id}/video`;
      video.dataset.src = job.id;
    }
    $("video-chip").textContent = "Match VOD";
  } else if (showCal) {
    video.hidden = true;
    empty.hidden = true;
    cal.hidden = false;
    $("cal-bar").hidden = false;
    $("video-chip").textContent = "Calibrate field";
    fillAssignSelect(job);
    updateCalHelp();
    loadCalFrame(job.id);
  } else {
    video.hidden = true;
    empty.hidden = false;
    cal.hidden = true;
    $("cal-bar").hidden = true;
    $("video-chip").textContent = job.demo ? "Sample" : "Waiting";
  }

  $("export-json").href = `/api/jobs/${job.id}/export.json`;
  $("export-csv").href = `/api/jobs/${job.id}/export.csv`;
  $("export-json").setAttribute("download", `${job.id}.json`);
  $("export-csv").setAttribute("download", `${job.id}.csv`);

  renderRobots(job);
  renderPlayByPlay(job);
  renderTimeline(job);
  renderWarnings(job);
  drawField();
  drawTrackOverlay();
  if (job.status === "ready") {
    updateScoutbookMeta();
    refreshPicklist();
    maybeRunBrowserPotato(job);
  }
}

function renderTeams(elId, match, color) {
  const root = $(elId);
  root.innerHTML = "";
  const keys = match.alliances?.[color]?.team_keys || [];
  for (const key of keys) {
    const info = match.teams?.[key] || {};
    const chip = document.createElement("span");
    chip.className = `chip ${color}`;
    chip.textContent = `${info.team_number || key.replace("frc", "")} ${info.nickname || ""}`.trim();
    root.appendChild(chip);
  }
}

function formatClock(seconds) {
  const s = Math.max(0, Math.round(Number(seconds) || 0));
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (m <= 0) return `${r}s`;
  return `${m}m ${String(r).padStart(2, "0")}s`;
}

function renderProgressClock(job) {
  const el = $("progress-eta");
  if (!el) return;
  if (["ready", "error", "cancelled"].includes(job.status)) {
    el.textContent = "";
    return;
  }
  const pct = Number(job.progress || 0);
  const start = Date.parse(job.created_at || "");
  if (!Number.isFinite(start)) {
    el.textContent = "";
    return;
  }
  const elapsed = Math.max(0, (Date.now() - start) / 1000);
  if (pct < 3) {
    el.textContent = `${formatClock(elapsed)} elapsed`;
    return;
  }
  const remain = elapsed * (100 - pct) / pct;
  el.textContent = `${formatClock(elapsed)} elapsed · ${formatClock(remain)} left`;
}

function renderPlayByPlay(job) {
  const root = $("play-by-play");
  if (!root) return;
  const rows = job.play_by_play || [];
  if (!rows.length) {
    const waiting = job.status && !["ready", "error", "cancelled"].includes(job.status);
    root.innerHTML = `<li><span>${waiting ? "Play-by-play appears when the overview track finishes." : "No play-by-play yet."}</span></li>`;
    return;
  }
  root.innerHTML = rows.map((row) => {
    const who = row.team ? `${row.robot} · ${row.team}` : row.robot;
    const pos = row.x == null ? "" : `${Number(row.x).toFixed(0)}, ${Number(row.y).toFixed(0)}`;
    return `<li>
      <span class="t">${Number(row.t).toFixed(1)}s</span>
      <span>${escapeHtml(who)}</span>
      <span class="doing">${escapeHtml(row.doing || "")}</span>
      <span class="chip ghost">${escapeHtml(row.period || "")}${pos ? ` · ${escapeHtml(pos)}` : ""}</span>
    </li>`;
  }).join("");
}

function renderRobots(job) {
  const grid = $("robot-grid");
  grid.innerHTML = "";
  for (const card of job.cards || []) {
    const el = document.createElement("article");
    el.className = "robot-card";
    el.dataset.team = String(card.team || "");
    if (state.selectedRobot && String(state.selectedRobot) === String(card.team)) el.classList.add("is-selected");
    const trust = card.trust_carpet == null ? "" : `<span class="chip ghost" title="Share of this path on the carpet, not the wall">Carpet ${Math.round(Number(card.trust_carpet) * 100)}%</span>`;
    const role = card.scout_role ? `<span class="chip">${escapeHtml(card.scout_role)}</span>` : "";
    const climb = card.scout_climb ? `<span class="chip ghost">Climb ${escapeHtml(card.scout_climb)}</span>` : "";
    const note = card.scout_note ? `<p class="tool-meta">${escapeHtml(card.scout_note)}</p>` : "";
    const why = card.why ? `<p class="tool-meta">${escapeHtml(card.why)}</p>` : "";
    el.innerHTML = `
      <header>
        <div class="robot-ident">
          <div class="bot-thumb-wrap">
            ${robotThumb(card.alliance)
              ? `<img class="bot-thumb" alt="" src="${robotThumb(card.alliance)}" />`
              : `<span class="bot-thumb bot-thumb-unknown" aria-hidden="true"></span>`}
            <span class="bot-num">${escapeHtml(card.team)}</span>
          </div>
          <div>
            <h3>${escapeHtml(card.team)}</h3>
            <div class="nick">${escapeHtml(card.nickname || "")}</div>
          </div>
        </div>
        <span class="chip ${resolveAlliance(card.alliance)}">${allianceLabel(card.alliance)}</span>
      </header>
      ${why}
      ${note}
      <div class="robot-meta-row">
        ${trust}
        ${role}
        ${climb}
        <span class="chip ghost">${escapeHtml(card.identity_state || card.visibility || "observed")}</span>
        ${card.alliance_conf != null ? `<span class="chip ghost" title="Alliance confidence">${Math.round(Number(card.alliance_conf) * 100)}%</span>` : ""}
        ${card.assignment_source === "start_pose_unverified" ? `<span class="chip warn">Unverified team</span>` : ""}
      </div>
      <div class="stats">
        <div class="stat"><b>${card.hub_score_candidates}</b><span>Hub dwells</span></div>
        <div class="stat"><b>${card.climb_attempt ? "Yes" : "No"}</b><span>Climb attempt</span></div>
        <div class="stat"><b>${Number(card.defense_time_s || 0).toFixed(0)}s</b><span>Defense</span></div>
        <div class="stat"><b>${Number(card.collection_time_s || 0).toFixed(0)}s</b><span>Collection</span></div>
        <div class="stat"><b>${Number(card.path_length_in || 0).toFixed(0)} in</b><span>Path length</span></div>
        ${card.tba ? `<div class="stat"><b>${card.tba.tba_teleop_points ?? "—"}</b><span>TBA teleop</span></div>` : ""}
      </div>
      <div class="pick-actions">
        <button type="button" data-pick="${escapeHtml(card.team)}">Add to pick list</button>
        <button type="button" data-watch="${escapeHtml(card.team)}">Watch</button>
        <button type="button" data-note="${escapeHtml(card.team)}">Note</button>
      </div>
      <label class="field">
        <span>Assign team number</span>
        <input class="team-input" data-track="${escapeHtml(card.team)}" value="${escapeHtml(card.team)}" inputmode="numeric" />
      </label>
    `;
    grid.appendChild(el);
  }
  grid.querySelectorAll("[data-pick]").forEach((btn) => {
    btn.addEventListener("click", () => {
      addCurrentMatchToBook(false);
      refreshPicklist();
      $("cmp-a").value = String(btn.dataset.pick);
      $("scout-tools")?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
  grid.querySelectorAll("[data-watch]").forEach((btn) => {
    btn.addEventListener("click", () => toggleWatch(Number(btn.dataset.watch)));
  });
  grid.querySelectorAll("[data-note]").forEach((btn) => {
    btn.addEventListener("click", () => {
      $("note-team").value = String(btn.dataset.note);
      $("note-text").focus();
      $("scout-tools")?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });
  grid.querySelectorAll(".team-input").forEach((input) => {
    input.addEventListener("change", async () => {
      if (!state.job) return;
      const from = input.dataset.track;
      const to = input.value.trim();
      if (!to) return;
      const assignments = {};
      for (const sample of state.job.samples || []) {
        if (String(sample.team) === String(from)) assignments[String(sample.track_id)] = to;
      }
      const res = await fetch(`/api/jobs/${state.job.id}/assign`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ assignments }),
      });
      state.job = await res.json();
      renderJob(state.job);
    });
  });
}

function renderTimeline(job) {
  const root = $("timeline");
  root.innerHTML = "";
  const events = job.events || [];
  if (!events.length) {
    root.innerHTML = `<div class="event-row"><span>No motion events yet. They appear after tracking (or immediately on the sample match).</span></div>`;
    return;
  }
  for (const event of events) {
    const row = document.createElement("div");
    row.className = "event-row";
    row.innerHTML = `
      <span class="t">${Number(event.t).toFixed(1)}s</span>
      <span class="chip">${escapeHtml(event.type)}</span>
      <span>${escapeHtml(event.team)}</span>
      <span>${escapeHtml(event.detail || event.zone)}</span>
      <span class="confidence">${Math.round((event.confidence || 0) * 100)}%</span>
    `;
    root.appendChild(row);
  }
}

function renderWarnings(job) {
  const root = $("warnings");
  root.innerHTML = "";
  for (const warning of job.warnings || []) {
    const li = document.createElement("li");
    li.textContent = warning;
    root.appendChild(li);
  }
}

function labelChannel(value) {
  if (String(value).includes("frc-events.firstinspires.org")) return "FIRST Event Web";
  return ({
    youtube_title: "YouTube title",
    youtube_description: "description",
    video_scorebug: "scorebug",
    overlay_unreadable: "overlay unread",
    firstevents: "FIRST Events",
  })[value] || value;
}

function labelStatus(status) {
  return ({
    queued: "Queued",
    resolving: "Resolving match",
    downloading: "Downloading video",
    views: "Sectioning camera views",
    tracking: "Tracking robots",
    tryout: "Tracker tryout",
    playbyplay: "Play-by-play",
    side_views: "Side cameras · scoring & climbs",
    scouting: "Reading the play-by-play",
    ready: "Ready",
    cancelled: "Cancelled",
    error: "Error",
  })[status] || status;
}

const STAGE_ORDER = ["resolving", "downloading", "views", "tryout", "tracking", "playbyplay", "side_views", "scouting", "ready"];
const STAGE_LABELS = {
  resolving: "Resolve",
  downloading: "Download",
  views: "Layout",
  tryout: "Tryout",
  tracking: "Overview",
  playbyplay: "Play-by-play",
  side_views: "Side cams",
  scouting: "Scout model",
  ready: "Done",
};

function renderThinkStages(job) {
  const el = $("think-stages");
  if (!el) return;
  const current = job.status || "queued";
  const seen = new Set((job.thinking_stages || []).map((s) => s.status));
  seen.add(current);
  const curIdx = STAGE_ORDER.indexOf(current);
  el.innerHTML = STAGE_ORDER.map((key) => {
    const idx = STAGE_ORDER.indexOf(key);
    const done = (curIdx >= 0 && idx < curIdx) || current === "ready";
    const active = key === current && current !== "ready" && current !== "error";
    const cls = active ? "active" : done && seen.has(key) ? "done" : "";
    return `<li class="${cls}">${STAGE_LABELS[key] || key}</li>`;
  }).join("");
}

function renderViews(job) {
  const panel = $("views-panel");
  const grid = $("views-grid");
  const bevMeta = $("bev-meta");
  const lede = $("views-lede");
  if (!panel || !grid) return;
  const panes = job.views?.panes || job.camera?.panes || [];
  const hasBev = job.bev && (job.bev.depth_source || job.bev.pitch_deg != null);
  if (!panes.length && !hasBev) {
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  if (lede) {
    const mode = job.views?.mode || job.camera?.mode || "single";
    const detail = job.views?.detail || job.camera?.detail || "";
    const nPane = Number(job.views?.pane_detections || job.pane_detections?.length || 0);
    const matchRate = job.views?.view_match_rate ?? job.view_correlation?.match_rate;
    const corr =
      nPane > 0
        ? ` Detected separately on each pane (${nPane} hits${
            matchRate != null ? `, ${Math.round(Number(matchRate) * 100)}% correlated to overview tracks` : ""
          }).`
        : "";
    const base =
      mode === "stacked_sides"
        ? "Top wide-angle for movement · bottom-left blue scoring/climb · bottom-right red scoring/climb."
        : mode === "stacked_top"
          ? "Top wide-angle for the top-down map · lower pane for scoring & climb cues."
          : mode === "side_by_side" || mode === "grid"
            ? "Multi-pane broadcast — each camera is cropped and detected on its own, then linked."
            : "Single overview camera — BEV adjusts for camera angle on the field map.";
    lede.textContent = `${detail ? `${base} ${detail}` : base}${corr}`;
  }
  const roleTitle = {
    overview: "Overview",
    overview_alt: "Overview (alt angle)",
    blue_side: "Blue side",
    red_side: "Red side",
    sideline: "Sideline",
    graphics: "Graphics / scorebug",
    other: "Other",
  };
  const frameUrl = job.has_frame ? `/api/jobs/${job.id}/frame?ts=${job.status === "ready" ? "r" : Date.now()}` : "";
  grid.innerHTML = panes
    .map((pane) => {
      const role = pane.role || "overview";
      const x0 = Number(pane.crop_left ?? 0);
      const x1 = Number(pane.crop_right ?? 1);
      const y0 = Number(pane.crop_top ?? 0);
      const y1 = Number(pane.crop_bottom ?? 1);
      const crop =
        `y ${Math.round(y0 * 100)}–${Math.round(y1 * 100)}% · x ${Math.round(x0 * 100)}–${Math.round(x1 * 100)}%`;
      const conf = pane.confidence != null ? `${Math.round(Number(pane.confidence) * 100)}%` : "";
      // Show the actual pane pixels: the calibration frame, CSS-cropped to
      // this pane's box (background-size scales the full frame so the box
      // fills the thumbnail).
      const w = Math.max(x1 - x0, 0.02);
      const h = Math.max(y1 - y0, 0.02);
      const thumb = frameUrl
        ? `<div class="view-thumb" style="aspect-ratio:${(16 * w) / (9 * h)};background-image:url('${frameUrl}');background-size:${(100 / w).toFixed(2)}% ${(100 / h).toFixed(2)}%;background-position:${((x0 / (1 - w)) * 100 || 0).toFixed(2)}% ${((y0 / (1 - h)) * 100 || 0).toFixed(2)}%;"></div>`
        : "";
      const bug = pane.scorebug ? `<span class="chip ghost tiny">scorebug inside</span>` : "";
      return `<article class="view-card ${role}">
        ${thumb}
        <div class="view-role">${roleTitle[role] || role}${conf ? ` <span class="view-conf">${conf}</span>` : ""}</div>
        <div class="view-purpose">${pane.purpose || ""}</div>
        <div class="view-crop">${crop} ${bug}</div>
      </article>`;
    })
    .join("");
  if (bevMeta) {
    const bits = [];
    if (hasBev) {
      if (job.bev.depth_source) bits.push(`Depth: ${job.bev.depth_source}`);
      if (job.bev.method) bits.push(`Homography: ${String(job.bev.method).replaceAll("_", " ")}`);
      if (job.bev.pitch_deg != null) bits.push(`Camera pitch ≈ ${Math.round(job.bev.pitch_deg)}°`);
      if (job.bev.tilt_strength != null) bits.push(`Tilt ${Number(job.bev.tilt_strength).toFixed(2)}`);
      if (job.bev.orientation && job.bev.orientation.blue_left === false) bits.push("Blue wall on the right (mirrored)");
    }
    if (job.depth_backends?.active) bits.push(`Depth backend: ${job.depth_backends.active}`);
    const missing = job.depth_backends?.missing || {};
    for (const [backend, mods] of Object.entries(missing)) {
      if (mods && mods.length) bits.push(`${backend} unavailable (missing ${mods.join(", ")})`);
    }
    const views = job.views || {};
    if (views.method === "decomposition") bits.push(`Layout: ${views.mode} (${Math.round((views.confidence || 0) * 100)}%)`);
    if (views.timeline?.length > 1) bits.push(`${views.timeline.length - 1} layout switch(es)`);
    if (job.gaps?.length) bits.push(`${job.gaps.length} overview gap(s)`);
    const sel = job.tracker_selection;
    if (sel?.chosen) {
      const ranking = (sel.ranking || []).slice(0, 4).map((m) => `${m} ${Number(sel.scores?.[m]?.total ?? 0).toFixed(2)}`);
      bits.push(`Auto picked ${sel.chosen} (${ranking.join(" · ")})`);
    }
    if (job.side_cues?.length) bits.push(`${job.side_cues.length} side cues`);
    if (bits.length) {
      bevMeta.hidden = false;
      bevMeta.innerHTML = bits.map((b) => `<span class="chip ghost">${escapeHtml(b)}</span>`).join("");
    } else {
      bevMeta.hidden = true;
      bevMeta.innerHTML = "";
    }
  }
}

function samplesNearTime(samples, t, window = 0.35) {
  const hits = [];
  for (const sample of samples || []) {
    if (Math.abs(Number(sample.t) - t) <= window) hits.push(sample);
  }
  return hits;
}

function drawTrackOverlay() {
  const canvas = $("track-overlay");
  const video = $("match-video");
  const toggle = $("overlay-toggle");
  if (!canvas || !video) return;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const w = Math.max(1, Math.round(video.clientWidth || 0));
  const h = Math.max(1, Math.round(video.clientHeight || 0));
  if (w < 2 || h < 2) return;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  ctx.clearRect(0, 0, w, h);
  if (toggle && !toggle.checked) return;
  const job = state.job;
  if (!job || !job.samples?.length) return;
  const t = Number.isFinite(video.currentTime) && video.currentTime > 0.05 ? video.currentTime : state.t;
  const fw = Number(job.frame_size?.[0] || video.videoWidth || w);
  const fh = Number(job.frame_size?.[1] || video.videoHeight || h);
  const sx = w / fw;
  const sy = h / fh;

  for (const pane of job.views?.panes || []) {
    const x0 = (pane.crop_left || 0) * w;
    const x1 = (pane.crop_right ?? 1) * w;
    const y0 = (pane.crop_top || 0) * h;
    const y1 = (pane.crop_bottom || 1) * h;
    ctx.strokeStyle =
      pane.role === "blue_side" ? "rgba(138,180,248,0.35)"
      : pane.role === "red_side" ? "rgba(242,139,130,0.35)"
      : "rgba(168,199,250,0.25)";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 5]);
    ctx.strokeRect(x0 + 1, y0 + 1, Math.max(0, x1 - x0 - 2), Math.max(0, y1 - y0 - 2));
    ctx.setLineDash([]);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = "600 11px Outfit, Roboto, sans-serif";
    ctx.fillText(String(pane.role || "view"), x0 + 8, y0 + 16);
  }

  const near = samplesNearTime(job.samples, t, 0.4);
  const byTrack = new Map();
  for (const sample of near) byTrack.set(sample.track_id, sample);
  for (const sample of byTrack.values()) {
    const alliance = resolveAlliance(sample.alliance);
    const color = allianceColor(alliance);
    const box = sample.bbox || [];
    if (box.length >= 4) {
      const [x1, y1, x2, y2] = box;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x1 * sx, y1 * sy, (x2 - x1) * sx, (y2 - y1) * sy);
      ctx.fillStyle = color;
      ctx.font = "700 13px IBM Plex Sans, sans-serif";
      ctx.fillText(String(sample.team || sample.track_id), x1 * sx + 4, Math.max(14, y1 * sy - 6));
    } else if (sample.px != null && sample.py != null) {
      const px = sample.px * sx;
      const py = sample.py * sy;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(px, py, 8, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = "700 12px IBM Plex Sans, sans-serif";
      ctx.fillText(String(sample.team || sample.track_id), px + 10, py - 4);
    }
  }

  for (const cue of job.side_cues || []) {
    if (Math.abs(Number(cue.t) - t) > 0.6) continue;
    const color = allianceColor(cue.alliance, { soft: true });
    ctx.fillStyle = color;
    ctx.font = "650 12px IBM Plex Sans, sans-serif";
    const label = cue.kind === "climb_activity" ? "CLIMB" : cue.kind === "hub_activity" ? "SCORE" : "SIDE";
    ctx.fillText(label, 12, h - 14);
  }
}

function bindTrackOverlay() {
  const video = $("match-video");
  const toggle = $("overlay-toggle");
  if (!video) return;
  const tickOverlay = () => {
    drawTrackOverlay();
    if (!video.paused) requestAnimationFrame(tickOverlay);
  };
  video.addEventListener("play", () => requestAnimationFrame(tickOverlay));
  video.addEventListener("seeked", drawTrackOverlay);
  video.addEventListener("loadedmetadata", drawTrackOverlay);
  video.addEventListener("timeupdate", drawTrackOverlay);
  toggle?.addEventListener("change", drawTrackOverlay);
  $("time-slider")?.addEventListener("input", () => {
    if (video.paused) drawTrackOverlay();
  });
}

function tick(now) {
  if (!state.playing) return;
  const dt = (now - state.lastTs) / 1000;
  state.lastTs = now;
  const end = Number(state.job?.game?.match_end_s || MATCH_END);
  state.t = Math.min(end, state.t + dt * 4);
  $("time-slider").value = String(state.t);
  $("time-label").textContent = `${state.t.toFixed(1)}s`;
  drawField();
  drawTrackOverlay();
  if (state.t < end) requestAnimationFrame(tick);
  else {
    state.playing = false;
    $("play-btn").textContent = "▶";
  }
}

function drawField() {
  const canvas = $("field-canvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const sx = w / FIELD.L;
  const sy = h / FIELD.W;
  const X = (x) => x * sx;
  const Y = (y) => y * sy;

  if (state.fieldImg) {
    ctx.drawImage(state.fieldImg, 0, 0, w, h);
  } else {
    ctx.fillStyle = "#102018";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "rgba(138,180,248,0.16)";
    ctx.fillRect(0, 0, X(FIELD.ALLIANCE), h);
    ctx.fillStyle = "rgba(242,139,130,0.16)";
    ctx.fillRect(X(FIELD.L - FIELD.ALLIANCE), 0, X(FIELD.ALLIANCE), h);
  }
  const job = state.job;
  if (!job) return;
  drawLandmarks(ctx, X, Y);
  drawZebra(ctx, X, Y, job.zebra);

  const autoEnd = Number(job.game?.auto_end_s || AUTO_END);
  const endgameStart = Number(job.game?.endgame_start_s || ENDGAME_START);
  // Period bands on the field edge — path strokes stay alliance-stable.
  drawPeriodBands(ctx, w, h, state.t, autoEnd, endgameStart);
  const byTeam = {};
  for (const sample of job.samples || []) {
    const team = sample.team || `T${sample.track_id}`;
    (byTeam[team] ||= []).push(sample);
  }
  for (const [team, samples] of Object.entries(byTeam)) {
    samples.sort((a, b) => a.t - b.t);
    const alliance = resolveAlliance(samples[0]?.alliance);
    const color = allianceColor(alliance);
    drawPath(ctx, X, Y, samples, state.t, color);
    const now = lastAt(samples, state.t);
    if (!now) continue;
    const predicted = Boolean(now.predicted || now.is_prediction);
    drawBot(ctx, X(now.x), Y(now.y), alliance, team, { predicted });
  }
}

function drawPeriodBands(ctx, w, h, t, autoEnd, endgameStart) {
  const matchEnd = Number(state.job?.game?.match_end_s || MATCH_END);
  const bands = [
    { a: 0, b: autoEnd / matchEnd, color: "#fbbf24", label: "AUTO" },
    { a: autoEnd / matchEnd, b: endgameStart / matchEnd, color: "#8ab4f8", label: "TELEOP" },
    { a: endgameStart / matchEnd, b: 1, color: "#34d399", label: "ENDGAME" },
  ];
  const barH = 16;
  const y = h - barH;
  ctx.save();
  for (const band of bands) {
    ctx.globalAlpha = 0.9;
    ctx.fillStyle = band.color;
    ctx.fillRect(w * band.a, y, Math.max(0, w * (band.b - band.a)), barH);
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#0e0e10";
    ctx.font = "700 10px IBM Plex Sans, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(band.label, w * (band.a + band.b) / 2, y + barH / 2);
  }
  ctx.fillStyle = "#f8fafc";
  ctx.fillRect(w * Math.max(0, Math.min(1, t / matchEnd)) - 1, y, 2, barH);
  ctx.restore();
}

function drawZebra(ctx, X, Y, zebra) {
  if (!zebra?.alliances) return;
  const inch = 39.3701;
  ctx.save();
  ctx.setLineDash([5, 5]);
  ctx.globalAlpha = 0.4;
  ctx.lineWidth = 2;
  for (const [color, robots] of Object.entries(zebra.alliances)) {
    ctx.strokeStyle = color === "red" ? "#f28b82" : "#e8eaed";
    for (const robot of robots || []) {
      const xs = robot.xs || [];
      const ys = robot.ys || [];
      const peak = Math.max(0, ...xs.filter((v) => v != null));
      const scale = peak > 0 && peak < 30 ? inch : 1;
      ctx.beginPath();
      let started = false;
      for (let i = 0; i < xs.length; i++) {
        if (xs[i] == null || ys[i] == null) continue;
        const px = X(xs[i] * scale);
        const py = Y(ys[i] * scale);
        if (!started) {
          ctx.moveTo(px, py);
          started = true;
        } else ctx.lineTo(px, py);
      }
      if (started) ctx.stroke();
    }
  }
  ctx.restore();
}

function drawLandmarks(ctx, X, Y) {
  const marks = state.landmarks || {};
  ctx.save();
  ctx.font = "600 11px Outfit, Roboto, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (const mark of Object.values(marks)) {
    if (mark == null || mark.x == null || mark.y == null) continue;
    const px = X(mark.x);
    const py = Y(mark.y);
    ctx.strokeStyle = "rgba(232, 234, 237, 0.28)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(px, py, 18, 0, Math.PI * 2);
    ctx.stroke();
    if (mark.label) {
      ctx.fillStyle = "rgba(232, 234, 237, 0.72)";
      ctx.fillText(String(mark.label), px, py - 26);
    }
  }
  ctx.restore();
}

function drawBot(ctx, x, y, alliance, team, { predicted = false } = {}) {
  const key = resolveAlliance(alliance);
  const img = state.bots[key];
  const size = 58;
  if (img) {
    ctx.save();
    if (predicted) ctx.globalAlpha = 0.55;
    ctx.drawImage(img, x - size / 2, y - size / 2, size, size);
    ctx.restore();
  } else {
    ctx.fillStyle = allianceColor(key);
    ctx.beginPath();
    if (typeof ctx.roundRect === "function") ctx.roundRect(x - 18, y - 18, 36, 36, 6);
    else ctx.rect(x - 18, y - 18, 36, 36);
    ctx.fill();
    if (predicted) {
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(x - 12, y - 16, 24, 8);
    ctx.fillRect(x - 12, y + 8, 24, 8);
  }
  const label = String(team);
  ctx.font = `800 ${label.length > 3 ? 9 : 11}px IBM Plex Sans, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(248, 250, 252, 0.95)";
  ctx.fillStyle = "#0e0e10";
  const plateY = y - size * 0.36;
  ctx.strokeText(label, x, plateY);
  ctx.fillText(label, x, plateY);
}

/** Stable alliance path color. Period is shown via timeline / band — never recolors the trail. */
function drawPath(ctx, X, Y, samples, t, color) {
  if (samples.length < 2) return;
  ctx.lineWidth = 3;
  ctx.lineJoin = "round";
  ctx.beginPath();
  let started = false;
  let lastT = null;
  let lastX = null;
  let lastY = null;
  const breakStroke = () => {
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.9;
    if (started) ctx.stroke();
    ctx.beginPath();
    started = false;
    lastX = null;
    lastY = null;
  };
  const drawable = (sample) => {
    if (!sample) return false;
    if (sample.gap || sample.camera_cut) return false;
    if (sample.field_valid === false) return false;
    if (sample.x == null || sample.y == null) return false;
    const x = Number(sample.x);
    const y = Number(sample.y);
    return Number.isFinite(x) && Number.isFinite(y);
  };
  for (const sample of samples) {
    if (sample.t > t) break;
    // Break path across camera cuts / long gaps — do not interpolate through invalid intervals.
    if (lastT != null && sample.t - lastT > 1.25) {
      breakStroke();
    }
    if (!drawable(sample)) {
      breakStroke();
      lastT = sample.t;
      continue;
    }
    const px = X(sample.x);
    const py = Y(sample.y);
    // Teleport guard: physically impossible jumps are gap artifacts, not motion.
    if (started && lastX != null && lastY != null) {
      const dx = sample.x - lastX;
      const dy = sample.y - lastY;
      const dist = Math.hypot(dx, dy);
      const dt = Math.max(sample.t - (lastT ?? sample.t), 1e-3);
      const speed = dist / dt;
      if (dist > 96 || speed > 200) {
        breakStroke();
      }
    }
    if (!started) {
      ctx.moveTo(px, py);
      started = true;
    } else ctx.lineTo(px, py);
    lastT = sample.t;
    lastX = sample.x;
    lastY = sample.y;
  }
  if (started) {
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.9;
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function lastAt(samples, t) {
  let found = null;
  for (const sample of samples) {
    if (sample.t > t) break;
    if (sample.field_valid === false) continue;
    if (sample.x == null || sample.y == null) continue;
    if (!Number.isFinite(Number(sample.x)) || !Number.isFinite(Number(sample.y))) continue;
    if (sample.gap || sample.camera_cut) continue;
    found = sample;
  }
  return found;
}

let calImage = null;
let calJobId = null;
function loadCalFrame(jobId) {
  if (calJobId === jobId && calImage) {
    drawCal();
    return;
  }
  const img = new Image();
  img.onload = () => {
    calImage = img;
    calJobId = jobId;
    const canvas = $("cal-canvas");
    canvas.width = img.width;
    canvas.height = img.height;
    drawCal();
  };
  img.src = `/api/jobs/${jobId}/frame?ts=${Date.now()}`;
}

function drawCal() {
  const canvas = $("cal-canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (calImage) ctx.drawImage(calImage, 0, 0);
  const seeds = state.job?.seeds || [];
  seeds.forEach((seed) => {
    const box = seed.bbox || [];
    if (box.length < 4) return;
    const [x1, y1, x2, y2] = box;
    const selected = state.selectedSeed && String(state.selectedSeed.track_id) === String(seed.track_id);
    ctx.strokeStyle = allianceColor(seed.alliance);
    ctx.lineWidth = selected ? 4 : 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = "700 14px IBM Plex Sans, sans-serif";
    ctx.fillText(String(seed.team || seed.track_id), x1 + 4, Math.max(14, y1 - 6));
  });
  ctx.fillStyle = "#22d3ee";
  ctx.strokeStyle = "#22d3ee";
  ctx.lineWidth = 3;
  state.calPoints.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(p[0], p[1], 8, 0, Math.PI * 2);
    ctx.fill();
    if (i > 0) {
      ctx.beginPath();
      ctx.moveTo(state.calPoints[i - 1][0], state.calPoints[i - 1][1]);
      ctx.lineTo(p[0], p[1]);
      ctx.stroke();
    }
  });
}

function hitSeed(x, y, seeds) {
  for (const seed of seeds) {
    const [x1, y1, x2, y2] = seed.bbox || [];
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2) return seed;
  }
  return null;
}

function matchTeams(job, alliance) {
  const keys = job?.match?.alliances?.[alliance]?.team_keys || [];
  return keys.map((key) => String(key).replace("frc", ""));
}

function cycleTeam(job, seed) {
  const alliance = resolveAlliance(seed.alliance);
  // Unknown: cycle through the full roster without inventing an alliance color.
  const teams =
    alliance === "unknown"
      ? [...matchTeams(job, "blue"), ...matchTeams(job, "red")]
      : matchTeams(job, alliance);
  if (!teams.length) return String(seed.team || "");
  const idx = teams.indexOf(String(seed.team || ""));
  return teams[(idx + 1) % teams.length];
}

function fillAssignSelect(job, selected) {
  const sel = $("assign-team");
  const wrap = $("assign-wrap");
  if (!sel || !wrap) return;
  const teams = [...matchTeams(job, "blue"), ...matchTeams(job, "red")];
  wrap.hidden = teams.length === 0 || state.calPoints.length < 4;
  sel.innerHTML = teams.map((team) => `<option value="${team}">${team}</option>`).join("");
  const value = selected || state.selectedSeed?.team;
  if (value) sel.value = value;
}

function updateCalHelp() {
  const help = $("cal-help");
  if (!help) return;
  if (state.calPoints.length < 4) {
    help.textContent = `Click the four field corners (${state.calPoints.length}/4): top-left, top-right, bottom-right, bottom-left.`;
  } else {
    help.textContent = "Corners set. Click a boxed robot to cycle its team number.";
  }
  const wrap = $("assign-wrap");
  if (wrap) wrap.hidden = state.calPoints.length < 4;
}

async function postAssign(assignments) {
  if (!state.job) return;
  const res = await fetch(`/api/jobs/${state.job.id}/assign`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ assignments }),
  });
  state.job = await res.json();
  renderJob(state.job);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

loadTbaKey();

async function loadTrackerAvailability() {
  const note = $("tracker-availability");
  if (!note) return;
  try {
    const res = await fetch("/api/trackers");
    if (!res.ok) return;
    const data = await res.json();
    const ready = (data.strategies || []).filter((s) => s.available).map((s) => s.name);
    const depth = data.depth?.active ? `depth ${data.depth.active}` : "";
    note.hidden = false;
    note.textContent = `Using best available: ${ready.join(", ") || "OpenCV"}${depth ? ` · ${depth}` : ""}.`;
  } catch (_err) {
    /* offline UI still works */
  }
}
loadTrackerAvailability();

fetch("/api/game").then((res) => res.json()).then(applyGame).catch(() => applyGame({
  year: 2026,
  name: "REBUILT",
  field_image: "/static/fields/2026.png",
  robot_icons: { blue: "/static/robots/blue.png", red: "/static/robots/red.png" },
}));

/* ---- Scout book / pick list (localStorage) ---- */

const BOOK_KEY = "roboscout.scoutBook";
const NOTES_KEY = "roboscout.teamNotes";
const WATCH_KEY = "roboscout.watchlist";
const EXCLUDE_KEY = "roboscout.pickedExclude";
const LEGACY_KEYS = {
  [BOOK_KEY]: "ramscout.scoutBook",
  [NOTES_KEY]: "ramscout.teamNotes",
  [WATCH_KEY]: "ramscout.watchlist",
  [EXCLUDE_KEY]: "ramscout.pickedExclude",
};

function loadJson(key, fallback) {
  try {
    let raw = localStorage.getItem(key);
    if (raw == null && LEGACY_KEYS[key]) {
      raw = localStorage.getItem(LEGACY_KEYS[key]);
      if (raw != null) localStorage.setItem(key, raw);
    }
    return raw ? JSON.parse(raw) : fallback;
  } catch (_err) {
    return fallback;
  }
}

function saveJson(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

function loadBook() {
  return loadJson(BOOK_KEY, { matches: [], cards: [] });
}

function saveBook(book) {
  saveJson(BOOK_KEY, book);
}

function loadNotes() {
  return loadJson(NOTES_KEY, {});
}

function loadWatch() {
  return new Set(loadJson(WATCH_KEY, []));
}

function saveWatch(set) {
  saveJson(WATCH_KEY, [...set]);
}

function loadExcluded() {
  return loadJson(EXCLUDE_KEY, []);
}

function excludeTeam(remove, team) {
  const list = loadExcluded().filter((t) => Number(t) !== Number(team));
  if (!remove) list.push(Number(team));
  saveJson(EXCLUDE_KEY, list);
}

function addCurrentMatchToBook(announce = true) {
  const job = state.job;
  if (!job || job.status !== "ready" || !(job.cards || []).length) {
    if (announce) alert("Analyze a match first, then add it to the scout book.");
    return false;
  }
  const book = loadBook();
  const matchKey = job.match?.key || job.id;
  if (book.matches.includes(matchKey)) {
    if (announce) alert(`Match ${matchKey} is already in the scout book.`);
    return false;
  }
  book.matches.push(matchKey);
  book.play_by_play = book.play_by_play || [];
  for (const row of job.play_by_play || []) {
    book.play_by_play.push({ ...row, _match: matchKey });
  }
  for (const card of job.cards) {
    book.cards.push({
      ...card,
      _match: matchKey,
      _job: job.id,
    });
  }
  saveBook(book);
  if (announce) {
    updateScoutbookMeta();
    refreshPicklist();
  }
  return true;
}

function updateScoutbookMeta() {
  const el = $("scoutbook-meta");
  if (!el) return;
  const book = loadBook();
  const teams = new Set(book.cards.map((c) => String(c.team)));
  if (!book.matches.length) {
    el.textContent = "No matches in scout book yet. Analyze a match, then Add match to scout book.";
    return;
  }
  el.textContent = `${book.matches.length} match${book.matches.length === 1 ? "" : "es"} · ${teams.size} teams scouted`;
}

async function refreshPicklist() {
  const book = loadBook();
  const cards = book.cards.length ? book.cards : (state.job?.cards || []);
  const excluded = loadExcluded();
  const res = await fetch("/api/picklist", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cards, already_picked: excluded, limit: 24 }),
  });
  if (!res.ok) return;
  state.picklist = await res.json();
  renderPicklist(state.picklist);
  renderSuggestedAlliance(state.picklist);
  updateScoutbookMeta();
}

function renderSuggestedAlliance(data) {
  const el = $("suggested-alliance");
  if (!el) return;
  const rows = data?.suggested_alliance || data?.first_round || [];
  if (!rows.length) {
    el.textContent = "";
    return;
  }
  const bits = rows.map((row) => `${row.team}${row.why ? ` — ${row.why}` : ""}`);
  el.textContent = `Suggested alliance: ${bits.join(" · ")}`;
}

function renderPicklist(data) {
  const rounds = $("pick-rounds");
  const ranked = $("pick-ranked");
  if (!rounds || !ranked) return;
  const watch = loadWatch();
  const sections = [
    ["1st round", data.first_round || []],
    ["2nd round", data.second_round || []],
    ["3rd round", data.third_round || []],
  ];
  rounds.innerHTML = sections.map(([label, rows]) => `
    <div class="pick-round">
      <div class="round-label">${label}</div>
      ${rows.length ? rows.map((row) => `
        <div class="pick-chip">
          <b>${escapeHtml(row.team)}</b>
          <span class="why">${escapeHtml((row.reasons || [])[0] || "")}</span>
        </div>
      `).join("") : `<div class="pick-chip"><span class="why">Need more scout data</span></div>`}
    </div>
  `).join("");

  ranked.innerHTML = (data.ranked || []).map((row) => `
    <li class="${watch.has(row.team) ? "watched" : ""}">
      <span class="rank">#${row.rank}</span>
      <span class="team-n">${escapeHtml(row.team)}</span>
      <span>${escapeHtml(row.why || (row.reasons || [])[0] || "")}</span>
      <span class="score">${Number(row.score).toFixed(1)}</span>
      <button type="button" data-exclude="${row.team}">Mark taken</button>
    </li>
  `).join("") || `<li><span>Analyze matches and add them to the scout book to build a draft board.</span></li>`;

  ranked.querySelectorAll("[data-exclude]").forEach((btn) => {
    btn.addEventListener("click", () => {
      excludeTeam(false, Number(btn.dataset.exclude));
      refreshPicklist();
    });
  });
}

function renderNotes() {
  const root = $("notes-list");
  if (!root) return;
  const notes = loadNotes();
  const watch = loadWatch();
  const teams = [...new Set([...Object.keys(notes), ...watch].map(String))].sort((a, b) => Number(a) - Number(b));
  if (!teams.length) {
    root.innerHTML = `<li><div class="note-team">No notes yet</div><div>Save a note or watchlist a team from a robot card.</div></li>`;
    return;
  }
  root.innerHTML = teams.map((team) => `
    <li>
      <div class="note-team">
        <span>${escapeHtml(team)}</span>
        ${watch.has(Number(team)) ? `<span class="watched-dot" title="On watchlist">★</span>` : ""}
      </div>
      <div>${escapeHtml(notes[team] || "On watchlist")}</div>
    </li>
  `).join("");
}

function toggleWatch(team) {
  const set = loadWatch();
  const n = Number(team);
  if (set.has(n)) set.delete(n);
  else set.add(n);
  saveWatch(set);
  renderNotes();
  refreshPicklist();
}

async function runCompare() {
  const teams = [$("cmp-a")?.value, $("cmp-b")?.value, $("cmp-c")?.value]
    .map((v) => Number(String(v || "").trim()))
    .filter((n) => Number.isFinite(n) && n > 0);
  const grid = $("compare-grid");
  const summary = $("alliance-summary");
  if (!teams.length) {
    if (grid) grid.innerHTML = "";
    if (summary) summary.textContent = "Enter at least one team number.";
    return;
  }
  const book = loadBook();
  const cards = book.cards.length ? book.cards : (state.job?.cards || []);
  const res = await fetch("/api/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cards, teams }),
  });
  if (!res.ok) {
    if (summary) summary.textContent = "Compare failed.";
    return;
  }
  const body = await res.json();
  if (grid) {
    grid.innerHTML = (body.compare?.teams || []).map((row) => {
      if (!row.found) {
        return `<div class="compare-card"><div class="team-n">${escapeHtml(row.team)}</div><div class="nick">Not in scout book</div></div>`;
      }
      return `
        <div class="compare-card">
          <div class="team-n">${escapeHtml(row.team)}</div>
          <div class="nick">${escapeHtml(row.nickname || "")}</div>
          <div class="row"><span>Hubs / match</span><b>${row.hubs_per_match}</b></div>
          <div class="row"><span>Climb rate</span><b>${Math.round(row.climb_rate * 100)}%</b></div>
          <div class="row"><span>Defense</span><b>${row.defense_s_per_match}s</b></div>
          <div class="row"><span>Draft score</span><b>${row.score}</b></div>
        </div>
      `;
    }).join("");
  }
  const a = body.alliance || {};
  if (summary) {
    summary.innerHTML = a.complete
      ? `Alliance totals · <strong>${a.combined_hubs_per_match}</strong> hubs/match · avg climb <strong>${Math.round((a.avg_climb_rate || 0) * 100)}%</strong>`
      : `Missing scout data for: ${(a.missing || []).join(", ") || "—"}`;
  }
}

function exportPicklist() {
  const data = state.picklist;
  if (!data?.ranked?.length) {
    alert("No ranked teams yet.");
    return;
  }
  const lines = ["rank,team,score,hubs_per_match,climb_rate,reasons"];
  for (const row of data.ranked) {
    lines.push([
      row.rank,
      row.team,
      row.score,
      row.hubs_per_match,
      row.climb_rate,
      `"${(row.reasons || []).join("; ").replaceAll('"', "'")}"`,
    ].join(","));
  }
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "roboscout-picklist.csv";
  a.click();
  URL.revokeObjectURL(url);
}

function initScoutTools() {
  $("nav-scoutbook")?.addEventListener("click", () => {
    navigate("picklist");
  });
  $("add-match-book")?.addEventListener("click", () => addCurrentMatchToBook(true));
  $("refresh-picks")?.addEventListener("click", () => refreshPicklist());
  $("export-picklist")?.addEventListener("click", () => exportPicklist());
  $("clear-scoutbook")?.addEventListener("click", () => {
    if (!window.confirm("Clear scout book, exclusions, and keep notes/watchlist?")) return;
    saveBook({ matches: [], cards: [] });
    saveJson(EXCLUDE_KEY, []);
    refreshPicklist();
  });
  $("run-compare")?.addEventListener("click", () => runCompare());
  $("save-note")?.addEventListener("click", () => {
    const team = String($("note-team")?.value || "").trim();
    const text = String($("note-text")?.value || "").trim();
    if (!team) return;
    const notes = loadNotes();
    if (text) notes[team] = text;
    else delete notes[team];
    saveJson(NOTES_KEY, notes);
    $("note-text").value = "";
    renderNotes();
  });
  $("watch-team")?.addEventListener("click", () => {
    const team = Number($("note-team")?.value || 0);
    if (!team) return;
    toggleWatch(team);
  });
  updateScoutbookMeta();
  renderNotes();
  refreshPicklist();
}

const VIEW_TITLES = {
  home: "Analyze a match",
  matches: "Match library",
  studio: "Analysis Studio",
  teams: "Teams",
  compare: "Compare",
  picklist: "Pick List",
  settings: "Settings",
};

function navigate(view) {
  const next = VIEW_TITLES[view] ? view : "home";
  state.view = next;
  document.querySelectorAll(".nav-item[data-nav]").forEach((el) => {
    el.classList.toggle("is-active", el.dataset.nav === next);
  });
  document.querySelectorAll(".view-panel").forEach((panel) => {
    const match = panel.dataset.view === next;
    panel.classList.toggle("is-active", match);
    if (panel.id === "workspace") {
      panel.hidden = next !== "studio" && next !== "picklist" && next !== "compare";
      if (next === "picklist" || next === "compare") {
        const more = $("more-tools");
        if (more) more.open = true;
        $("scout-tools")?.scrollIntoView({ behavior: "smooth", block: "start" });
        if (next === "compare") $("cmp-a")?.focus();
      } else if (next === "teams") {
        renderTeamPage();
      } else if (next === "studio" && state.job) {
        panel.hidden = false;
        panel.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    } else if (panel.dataset.view) {
      panel.hidden = !match;
    }
  });
  const title = $("topbar-title");
  if (title) title.textContent = VIEW_TITLES[next] || "RoboScoutAI";
  if (next === "home") {
    $("composer")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }
  if (next === "matches") renderMatchLibrary();
  if (next === "settings") renderSettingsPanel();
}

function renderMatchLibrary() {
  const root = $("match-library");
  if (!root) return;
  const book = loadBook();
  const matches = book.matches || [];
  if (!matches.length) {
    root.innerHTML = `<div class="empty-state"><h3>No matches yet</h3><p>Analyze a VOD from Home, or try the sample match.</p><button type="button" class="btn primary" id="lib-analyze">Analyze a match</button></div>`;
    $("lib-analyze")?.addEventListener("click", () => navigate("home"));
    return;
  }
  root.innerHTML = matches
    .slice()
    .reverse()
    .map(
      (m) => `<article class="match-card">
      <header><strong>${escapeHtml(m.match_key || m.id || "Match")}</strong>
      <span class="chip ghost">${escapeHtml(m.event_key || "")}</span></header>
      <p class="tool-meta">${(m.teams || []).join(" · ") || "Teams pending"}</p>
      <span class="chip ${(m.quality || "unknown") === "good" ? "ok" : "warn"}">${escapeHtml(m.quality || "tracked")}</span>
    </article>`
    )
    .join("");
}

function renderSettingsPanel() {
  const depth = $("settings-depth");
  const depsEl = $("settings-deps");
  if (depth) {
    depth.textContent = "Depth & detector status load from /api/trackers when available.";
  }
    if (depsEl) depsEl.textContent = "Checking installed packages…";
    const scoutEl = $("settings-scout-model");
    if (scoutEl) scoutEl.textContent = "Local scout model: checking…";
  Promise.all([
    fetch("/api/trackers").then((r) => r.json()).catch(() => ({})),
    fetch("/api/health").then((r) => r.json()).catch(() => ({})),
  ]).then(([trackers, health]) => {
    const d = trackers.depth || {};
    const active = d.active || d.selected || "unknown";
    if (depth) {
      depth.textContent = `Depth backend: ${active}. ONNX: ${d.onnx?.importable || trackers.detector?.onnxruntime ? "available" : "not installed"}. Detector FRC-ready: ${trackers.detector?.frc_ready ? "yes" : "no"}.`;
    }
    if (depsEl) {
      const missing = health.missing_required || trackers.deps?.missing_required || [];
      const optional = health.missing_optional || trackers.deps?.missing_optional || [];
      if (health.deps_ok === false || missing.length) {
        depsEl.textContent = `Missing required: ${missing.join(", ") || "unknown"}. Run: ${(health.install || trackers.deps?.install || ["pip install -r requirements.txt"])[0]}`;
        depsEl.classList.add("warn-text");
      } else {
        depsEl.textContent = `Required packages OK. Optional not installed: ${optional.length ? optional.join(", ") : "none"}.`;
        depsEl.classList.remove("warn-text");
      }
    }
    const model = health.scout_model || trackers.scout_model || {};
    const scoutLine = $("settings-scout-model");
    if (scoutLine) {
      const stateLabel = model.state || (model.installed ? "installed" : "not installed");
      const cmd = model.install_command || "pip install -r requirements-laya.txt";
      scoutLine.textContent = `Local scout model: ${stateLabel}. ${model.installed ? "Weights load on first scout." : `Install: ${cmd}`}`;
    }
  });
}

function renderTeamPage() {
  const root = $("team-page");
  if (!root) return;
  const book = loadBook();
  const notes = loadNotes();
  const groups = new Map();
  for (const card of book.cards || []) {
    const team = String(card.team || "");
    if (!team) continue;
    if (!groups.has(team)) groups.set(team, []);
    groups.get(team).push(card);
  }
  if (!groups.size) {
    root.innerHTML = `<div class="empty-state"><h3>No teams in the book</h3><p>Add a match from Analysis Studio. This page lists every robot across those matches.</p></div>`;
    return;
  }
  const plays = book.play_by_play || [];
  root.innerHTML = [...groups.keys()].sort((a, b) => Number(a) - Number(b)).map((team) => {
    const cards = groups.get(team);
    const climbs = cards.filter((c) => c.climb_attempt || c.scout_climb).length;
    const path = cards.reduce((sum, c) => sum + Number(c.path_length_in || 0), 0);
    const role = cards.map((c) => c.scout_role).filter(Boolean).pop() || "";
    const matches = [...new Set(cards.map((c) => c._match).filter(Boolean))];
    const actions = plays.filter((row) => String(row.team) === team || String(row.robot) === team).slice(0, 6);
    const actionLine = actions.map((row) => `${Number(row.t).toFixed(0)}s ${row.doing}`).join(" · ");
    return `<article class="team-card-page">
      <h3>${escapeHtml(team)} ${role ? `<span class="chip">${escapeHtml(role)}</span>` : ""}</h3>
      <p class="tool-meta">${matches.length || cards.length} match${(matches.length || cards.length) === 1 ? "" : "es"} · climbs ${climbs} · path ${path.toFixed(0)} in</p>
      ${notes[team] ? `<p>${escapeHtml(notes[team])}</p>` : ""}
      ${actionLine ? `<p class="tool-meta">${escapeHtml(actionLine)}</p>` : ""}
    </article>`;
  }).join("");
}

function printMatchSheet() {
  const job = state.job;
  if (!job) return;
  const match = job.match || {};
  const cards = (job.cards || []).slice(0, 6).map((card) => `
    <section>
      <h2>${card.team} · ${card.alliance || ""}</h2>
      <p>${card.why || card.scout_note || ""}</p>
      <p>Hub ${card.hub_score_candidates ?? 0} · climb ${card.climb_attempt ? "yes" : "no"} · ${card.scout_role || ""} ${card.scout_climb || ""}</p>
    </section>`).join("");
  const plays = (job.play_by_play || []).slice(0, 24).map((row) =>
    `<li>${Number(row.t).toFixed(1)}s ${row.team || row.robot} ${row.doing} (${row.period})</li>`
  ).join("");
  const html = `<!DOCTYPE html><html><head><title>Match sheet</title>
    <style>
      body { font-family: sans-serif; color: #111; margin: 24px; }
      h1 { font-size: 20px; }
      .cards { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; }
      section { border: 1px solid #ccc; padding: 8px; }
      @media print { button { display: none; } }
    </style></head><body>
    <h1>${match.key || "Match"} · Blue ${match.alliances?.blue?.score ?? "—"} · Red ${match.alliances?.red?.score ?? "—"}</h1>
    <div class="cards">${cards}</div>
    <h2>Play-by-play</h2>
    <ol>${plays}</ol>
    </body></html>`;
  const frame = document.createElement("iframe");
  frame.style.position = "fixed";
  frame.style.right = "0";
  frame.style.bottom = "0";
  frame.style.width = "0";
  frame.style.height = "0";
  frame.style.border = "0";
  document.body.appendChild(frame);
  const doc = frame.contentDocument;
  doc.open();
  doc.write(html);
  doc.close();
  frame.contentWindow.focus();
  frame.contentWindow.print();
  setTimeout(() => frame.remove(), 1000);
}

function selectRobot(delta) {
  const cards = [...document.querySelectorAll("#robot-grid .robot-card")];
  if (!cards.length) return;
  const cur = cards.findIndex((card) => card.classList.contains("is-selected"));
  const next = cur < 0 ? 0 : (cur + delta + cards.length) % cards.length;
  cards.forEach((card, index) => card.classList.toggle("is-selected", index === next));
  state.selectedRobot = cards[next].dataset.team || null;
  cards[next].scrollIntoView({ block: "nearest" });
}

function initShell() {
  document.querySelectorAll(".nav-item[data-nav]").forEach((btn) => {
    btn.addEventListener("click", () => navigate(btn.dataset.nav));
  });
  $("nav-home")?.addEventListener("click", () => navigate("home"));
  $("sidebar-toggle")?.addEventListener("click", () => {
    $("app-shell")?.classList.toggle("is-collapsed");
  });
  $("cta-analyze")?.addEventListener("click", () => navigate("home"));
  $("crop-lock-btn")?.addEventListener("click", () => {
    state.cropLocked = !state.cropLocked;
    const status = $("crop-lock-status");
    const btn = $("crop-lock-btn");
    if (status) status.textContent = state.cropLocked ? "Locked · top overview" : "Auto · unlocked";
    if (btn) btn.textContent = state.cropLocked ? "Unlock crop" : "Lock crop";
  });
  $("motion-chip")?.addEventListener("click", () => {
    const order = ["full", "reduced", "off"];
    const cur = document.documentElement.dataset.motion || "full";
    const next = order[(order.indexOf(cur) + 1) % order.length];
    document.documentElement.dataset.motion = next;
    storageSet("motion", next);
    $("motion-chip").textContent = `Motion: ${next}`;
  });
  const savedMotion = storageGet("motion", "full");
  document.documentElement.dataset.motion = savedMotion;
  if ($("motion-chip")) $("motion-chip").textContent = `Motion: ${savedMotion}`;
  $("next-uncertain")?.addEventListener("click", () => {
    const job = state.job;
    if (!job) return;
    const t = state.t;
    const gaps = [...(job.gaps || []), ...(job.layout_switches || [])]
      .map((g) => Number(g.t ?? g.t0 ?? NaN))
      .filter((v) => Number.isFinite(v) && v > t + 0.2)
      .sort((a, b) => a - b);
    const conflict = (job.samples || [])
      .filter((s) => s.alliance_conflict || s.flags?.includes?.("alliance_conflict"))
      .map((s) => Number(s.t))
      .filter((v) => v > t + 0.2)
      .sort((a, b) => a - b);
    const next = [...gaps, ...conflict].sort((a, b) => a - b)[0];
    if (next == null) {
      alert("No further uncertain moments marked in this match.");
      return;
    }
    state.t = next;
    if ($("time-slider")) $("time-slider").value = String(next);
    if ($("time-label")) $("time-label").textContent = `${next.toFixed(1)}s`;
    const video = $("match-video");
    if (video && Number.isFinite(video.duration)) video.currentTime = next;
    drawField();
    drawTrackOverlay();
  });
  $("print-sheet")?.addEventListener("click", () => printMatchSheet());
  $("cancel-job")?.addEventListener("click", async () => {
    if (!state.job) return;
    await fetch(`/api/jobs/${state.job.id}/cancel`, { method: "POST" });
    refreshJob();
  });
  const dropForm = $("start-form");
  dropForm?.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropForm.classList.add("is-drop");
  });
  dropForm?.addEventListener("dragleave", () => dropForm.classList.remove("is-drop"));
  dropForm?.addEventListener("drop", (event) => {
    event.preventDefault();
    dropForm.classList.remove("is-drop");
    const file = event.dataTransfer?.files?.[0];
    const input = $("video-file");
    if (!file || !input || typeof DataTransfer === "undefined") return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    input.files = transfer.files;
    input.dispatchEvent(new Event("change"));
  });
  document.addEventListener("keydown", (event) => {
    if (state.view !== "studio") return;
    const tag = document.activeElement?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
    if (event.code === "Space") {
      event.preventDefault();
      $("play-btn")?.click();
    } else if (event.key === "n" || event.key === "N") {
      $("next-uncertain")?.click();
    } else if (event.key === "ArrowRight") {
      selectRobot(1);
    } else if (event.key === "ArrowLeft") {
      selectRobot(-1);
    }
  });
  navigate("home");
}

/** Only open a URL when git apply failed and the API attached a manual fallback. */
function shouldOpenUpdateManualFallback(body) {
  return Boolean(body) && body.ok === false && Boolean(body.open_url);
}

/** Label for the update chip while the manager downloads/installs side-by-side. */
function managedUpdateLabel(status) {
  if (!status) return "Updating…";
  const pct = Number.isFinite(status.progress) ? `${status.progress}%` : "";
  switch (status.state) {
    case "checking":
      return "Checking…";
    case "downloading":
      return `Downloading ${pct}`.trim();
    case "verifying":
      return "Verifying…";
    case "installing":
      return "Installing…";
    case "manager-update":
      return `Updating manager ${pct}`.trim();
    case "done":
      return status.version ? `Restart to ${status.version}` : "Restart to finish";
    case "error":
      return "Update failed";
    default:
      return "Updating…";
  }
}

/** Poll /api/updates/status until the manager reports done or error. */
async function waitForManagedUpdate(onProgress, { intervalMs = 1000, timeoutMs = 30 * 60 * 1000 } = {}) {
  const started = Date.now();
  let lastState = "";
  while (Date.now() - started < timeoutMs) {
    let status = null;
    try {
      const res = await fetch("/api/updates/status", { cache: "no-store" });
      status = await res.json();
    } catch (_err) {
      status = null;
    }
    if (status) {
      if (status.state !== lastState || status.state === "downloading") onProgress(status);
      lastState = status.state || "";
      if (status.state === "done" || status.state === "error") return status;
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  return { state: "error", error: "Timed out waiting for the update to finish." };
}

/** After a managed relaunch, wait for the new app process and reload the page. */
async function waitForRelaunchThenReload(updateChip) {
  updateChip.textContent = "Restarting…";
  const started = Date.now();
  // Give the old process time to exit before we start treating a healthy reply as the new one.
  await new Promise((resolve) => setTimeout(resolve, 3000));
  while (Date.now() - started < 3 * 60 * 1000) {
    try {
      const res = await fetch("/api/health", { cache: "no-store" });
      if (res.ok) {
        window.location.reload();
        return;
      }
    } catch (_err) {
      // still restarting
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  updateChip.textContent = "Restart RoboScoutAI";
  updateChip.disabled = false;
}

async function initUpdater() {
  const versionChip = $("version-chip");
  const updateChip = $("update-chip");
  if (!versionChip || !updateChip) return;

  async function refresh(force = false) {
    try {
      const res = await fetch(force ? "/api/updates/check" : "/api/version");
      const data = await res.json();
      const version = data.version || data.current_version || "?";
      versionChip.textContent = `v${version}`;
      const update = data.update || data;
      const available = Boolean(update?.available);
      const canApply = Boolean(update?.can_apply) || Boolean(update?.mode === "source" && available);
      if (updateChip.dataset.pendingRestart === "1" || updateChip.disabled) {
        // A managed update is in flight or installed; leave the chip alone.
      } else if (available && canApply) {
        updateChip.hidden = false;
        const label = update.latest_version || (update.remote_sha || "").slice(0, 7) || "git";
        updateChip.textContent = `Update ${label}`;
        updateChip.title = update.message || "Update available from git";
        updateChip.dataset.releaseUrl = update.release_url || update.remote || "";
        updateChip.dataset.canApply = "1";
      } else {
        updateChip.hidden = true;
        updateChip.dataset.canApply = "0";
      }
      const cookiesEl = $("cookies-status");
      if (cookiesEl && data.cookies) {
        if (data.cookies.found) {
          cookiesEl.hidden = false;
          cookiesEl.textContent = `YouTube cookies found (${data.cookies.path || "cookies.txt"}). Downloads can use them; upload is still the most reliable escape hatch.`;
        } else {
          cookiesEl.hidden = false;
          cookiesEl.textContent =
            "No cookies.txt yet. If YouTube blocks downloads: upload an MP4/MKV, or place cookies.txt next to the EXE / in %APPDATA%\\RoboScoutAI\\ / set YTDLP_COOKIES.";
        }
      }
    } catch (_err) {
      versionChip.textContent = "v?";
    }
  }

  versionChip.addEventListener("click", async () => {
    await refresh(true);
    try {
      const res = await fetch("/api/updates/check");
      const update = await res.json();
      if (update.available && (update.can_apply || update.mode === "source")) {
        updateChip.hidden = false;
        updateChip.dataset.canApply = "1";
        const label = update.latest_version || (update.remote_sha || "").slice(0, 7) || "git";
        updateChip.textContent = `Update ${label}`;
        alert(
          `Update available: ${update.current_version} → ${label}\n${update.message || "Click Update to apply from git."}`
        );
        return;
      }
      if (update.available && !update.can_apply) {
        updateChip.hidden = true;
        alert(
          update.error
          || "A newer git revision exists, but no desktop binary is published on that branch yet."
        );
        return;
      }
      if (update.error) {
        const soft = /up to date|git remote unreachable|no desktop binary|unreachable|nothing to apply/i.test(
          update.error
        );
        alert(update.error);
        if (!soft && update.release_url && window.confirm(`Open git remote page?\n${update.release_url}`)) {
          window.open(update.release_url, "_blank", "noopener");
        }
      } else {
        const body = update.body ? `\n${update.body}` : "";
        alert(
          update.message === "up to date" || !update.message
            ? `RoboScoutAI ${update.current_version || versionChip.textContent} is up to date.${body}`
            : `${update.message}${body}`
        );
      }
    } catch (_err) {
      alert("Could not check the git remote for updates.");
    }
  });
  async function relaunchManaged() {
    updateChip.disabled = true;
    updateChip.dataset.pendingRestart = "0";
    await fetch("/api/updates/relaunch", { method: "POST" }).catch(() => {});
    await waitForRelaunchThenReload(updateChip);
  }

  updateChip.addEventListener("click", async () => {
    const releaseUrl = updateChip.dataset.releaseUrl || "";
    if (updateChip.dataset.pendingRestart === "1") {
      await relaunchManaged();
      return;
    }
    if (updateChip.dataset.canApply === "1") {
      updateChip.textContent = "Updating from git…";
      updateChip.disabled = true;
      try {
        const res = await fetch("/api/updates/download", { method: "POST" });
        const body = await res.json().catch(() => ({}));
        // Never open a URL on success/restarting — that was downloading an old Release exe.
        if (shouldOpenUpdateManualFallback(body)) {
          window.open(body.open_url, "_blank", "noopener");
        }
        if (!res.ok || body.ok === false) {
          alert(
            body.message ||
              body.detail ||
              "Git update failed. If a fallback link opened, install the newest exe manually."
          );
          updateChip.disabled = false;
          updateChip.textContent = "Update available";
          return;
        }
        if (body.managed) {
          // Manager downloads side-by-side; we only show progress and then ask to restart.
          updateChip.textContent = managedUpdateLabel({ state: "checking" });
          const status = await waitForManagedUpdate((s) => {
            updateChip.textContent = managedUpdateLabel(s);
            updateChip.title = s.message || "";
          });
          if (status.state === "error") {
            alert(
              `Update failed: ${status.error || status.message || "unknown error"}\n\n` +
                "Check %LOCALAPPDATA%\\RoboScoutAI\\update.log, or run RoboScoutAI.exe --repair."
            );
            updateChip.disabled = false;
            updateChip.textContent = "Update available";
            return;
          }
          updateChip.disabled = false;
          updateChip.textContent = managedUpdateLabel(status);
          const restartNow = window.confirm(
            `RoboScoutAI ${status.version || ""} is installed side-by-side.\n\nRestart now to switch to it? (The previous version is kept for rollback.)`
          );
          if (!restartNow) {
            updateChip.dataset.pendingRestart = "1";
            updateChip.title = "Click to restart into the new version";
            return;
          }
          await relaunchManaged();
          return;
        }
        alert(
          (body.message ||
            "Updating from git — installing into %LOCALAPPDATA%\\RoboScoutAI\\RoboScoutAI.exe.") +
            "\n\nBuilds are unsigned for now — if SmartScreen appears: More info → Run anyway." +
            "\nAfter update, run from %LOCALAPPDATA%\\RoboScoutAI\\RoboScoutAI.exe"
        );
        if (!body.restarting) {
          updateChip.disabled = false;
          updateChip.hidden = true;
          await refresh(true);
        }
      } catch (_err) {
        // Network/parse failure only — last-resort manual fallback.
        if (releaseUrl && /releases\//i.test(releaseUrl)) {
          window.open(releaseUrl, "_blank", "noopener");
        }
        alert(
          "Git update request failed. Install the newest RoboScoutAI-windows-x64.exe manually if needed."
        );
        updateChip.disabled = false;
        updateChip.textContent = "Update available";
      }
      return;
    }
    alert("No applyable desktop update is available right now. Updates install from git into LocalAppData.");
  });

  refresh(false);
  setInterval(() => refresh(false), 30 * 60 * 1000);
}

initUpdater();
initScoutTools();
initShell();
bindTrackOverlay();
drawField();
drawTrackOverlay();

async function maybeRunBrowserPotato(job) {
  if (!shouldRunBrowserPotato(job)) return;
  if (state.potatoJobs.has(job.id) || state.potatoRunning) return;
  const video = $("match-video");
  if (!video || video.hidden || !job.has_video) return;
  // Ensure video URL is set before seeking.
  if (video.dataset.src !== job.id) {
    video.src = `/api/jobs/${job.id}/video`;
    video.dataset.src = job.id;
  }
  state.potatoJobs.add(job.id);
  state.potatoRunning = true;
  const msg = $("progress-message");
  const panel = $("progress-panel");
  const status = $("progress-status");
  const fill = $("progress-fill");
  if (panel) panel.hidden = false;
  if (status) status.textContent = "Browser potato";
  try {
    const samples = await runBrowserPotato(video, {
      cropTop: job.crop_top ?? 0.1,
      cropBottom: job.crop_bottom ?? 0.65,
      durationHint: (job.game && job.game.match_end_s) || 150,
      onProgress: (pct, text) => {
        if (fill) fill.style.width = `${Math.max(4, pct)}%`;
        if (msg) msg.textContent = text;
      },
    });
    if (!samples.length) {
      if (msg) msg.textContent = "Browser potato found no motion blobs.";
      return;
    }
    const replace = (job.tracker_mode || "").toLowerCase() === "potato" && (job.samples || []).length < 8;
    const res = await fetch(`/api/jobs/${job.id}/browser-tracks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ samples, replace }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      if (msg) msg.textContent = err.detail || "Browser potato upload failed.";
      return;
    }
    const updated = await res.json();
    state.job = updated;
    renderJob(updated);
  } catch (err) {
    console.warn("browser potato failed", err);
    if (msg) msg.textContent = "Browser potato skipped (video not ready).";
  } finally {
    state.potatoRunning = false;
    if (panel && state.job?.status === "ready") panel.hidden = true;
  }
}

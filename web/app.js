const FIELD = { L: 651.2, W: 317.7, ALLIANCE: 158.6 };
const AUTO_END = 20;
const ENDGAME_START = 130;
const MATCH_END = 160;

import { runBrowserPotato, shouldRunBrowserPotato } from "./potato.js";

const $ = (id) => document.getElementById(id);

const state = {
  job: null,
  poll: null,
  playing: false,
  t: 0,
  lastTs: 0,
  calPoints: [],
  fieldImg: null,
  bots: { blue: null, red: null },
  landmarks: {},
  calMode: false,
  calJob: null,
  selectedSeed: null,
  picklist: null,
  potatoJobs: new Set(),
  potatoRunning: false,
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

function loadTbaKey() {
  const saved = localStorage.getItem("ramscout.tbaKey") || "";
  $("tba-key").value = saved;
  const openai = localStorage.getItem("ramscout.openaiKey") || "";
  const google = localStorage.getItem("ramscout.googleKey") || "";
  const mode = localStorage.getItem("ramscout.trackerMode") || (google ? "gemini" : "auto");
  if ($("openai-key")) $("openai-key").value = openai;
  if ($("google-key")) $("google-key").value = google;
  if ($("tracker-mode") && [...$("tracker-mode").options].some((o) => o.value === mode)) {
    $("tracker-mode").value = mode;
  }
}

$("google-key")?.addEventListener("change", () => {
  const key = $("google-key")?.value.trim() || "";
  if (key && $("tracker-mode") && $("tracker-mode").value === "auto") {
    $("tracker-mode").value = "gemini";
  }
  saveScoutKeys();
});

function saveScoutKeys() {
  localStorage.setItem("ramscout.tbaKey", $("tba-key").value.trim());
  if ($("openai-key")) localStorage.setItem("ramscout.openaiKey", $("openai-key").value.trim());
  if ($("google-key")) localStorage.setItem("ramscout.googleKey", $("google-key").value.trim());
  if ($("tracker-mode")) localStorage.setItem("ramscout.trackerMode", $("tracker-mode").value);
}

function trackerPayload() {
  return {
    tracker_mode: $("tracker-mode")?.value || "auto",
    auto_multicam: $("auto-multicam")?.checked !== false,
    openai_key: $("openai-key")?.value.trim() || "",
    google_key: $("google-key")?.value.trim() || "",
  };
}

$("start-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  saveScoutKeys();
  const fileInput = $("video-file");
  const file = fileInput?.files?.[0] || null;
  await createJob({
    url: $("url").value.trim(),
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
    form.append("auto_multicam", payload.auto_multicam === false ? "false" : "true");
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
    alert(err.detail || "Could not start job.");
    return;
  }
  const job = await res.json();
  state.job = job;
  $("workspace").hidden = false;
  $("workspace").scrollIntoView({ behavior: "smooth", block: "start" });
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
  if (job.status === "ready" || job.status === "error") {
    clearInterval(state.poll);
    state.poll = null;
  }
}

function renderJob(job) {
  window.__ramscoutJobId = job.id;
  window.dispatchEvent(new CustomEvent("ramscout:job", { detail: job }));
  const ytHelp = $("yt-help");
  if (ytHelp) {
    const err = `${job.error || ""} ${job.message || ""} ${(job.warnings || []).join(" ")}`.toLowerCase();
    const blocked = err.includes("bot") || err.includes("sign in") || err.includes("youtube") && err.includes("block");
    ytHelp.hidden = !blocked;
  }

  $("progress-status").textContent = labelStatus(job.status);
  $("progress-message").textContent = job.error || job.message || "";
  $("progress-fill").style.width = `${Math.max(4, job.progress || 0)}%`;
  $("progress-panel").hidden = job.status === "ready";

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
  renderTimeline(job);
  renderWarnings(job);
  drawField();
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

function renderRobots(job) {
  const grid = $("robot-grid");
  grid.innerHTML = "";
  for (const card of job.cards || []) {
    const el = document.createElement("article");
    el.className = "robot-card";
    el.innerHTML = `
      <header>
        <div class="robot-ident">
          <div class="bot-thumb-wrap">
            <img class="bot-thumb" alt="" src="${card.alliance === "red" ? "/static/robots/red.png" : "/static/robots/blue.png"}" />
            <span class="bot-num">${escapeHtml(card.team)}</span>
          </div>
          <div>
            <h3>${escapeHtml(card.team)}</h3>
            <div class="nick">${escapeHtml(card.nickname || "")}</div>
          </div>
        </div>
        <span class="chip ${card.alliance}">${card.alliance}</span>
      </header>
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
        <span>Team override</span>
        <input class="team-input" data-track="${escapeHtml(card.team)}" value="${escapeHtml(card.team)}" />
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
    tracking: "Tracking robots",
    scouting: "Auto-scouting",
    ready: "Ready",
    error: "Error",
  })[status] || status;
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
  const byTeam = {};
  for (const sample of job.samples || []) {
    const team = sample.team || `T${sample.track_id}`;
    (byTeam[team] ||= []).push(sample);
  }
  for (const [team, samples] of Object.entries(byTeam)) {
    samples.sort((a, b) => a.t - b.t);
    const alliance = samples[0]?.alliance || "blue";
    const color = alliance === "red" ? "#f28b82" : "#8ab4f8";
    drawPath(ctx, X, Y, samples, state.t, color, autoEnd, endgameStart);
    const now = lastAt(samples, state.t);
    if (!now) continue;
    drawBot(ctx, X(now.x), Y(now.y), alliance, team);
  }
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

function drawBot(ctx, x, y, alliance, team) {
  const img = state.bots[alliance] || state.bots.blue;
  const size = 58;
  if (img) {
    ctx.drawImage(img, x - size / 2, y - size / 2, size, size);
  } else {
    ctx.fillStyle = alliance === "red" ? "#c62828" : "#1565c0";
    ctx.beginPath();
    if (typeof ctx.roundRect === "function") ctx.roundRect(x - 18, y - 18, 36, 36, 6);
    else ctx.rect(x - 18, y - 18, 36, 36);
    ctx.fill();
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(x - 12, y - 16, 24, 8);
    ctx.fillRect(x - 12, y + 8, 24, 8);
  }
  const label = String(team);
  ctx.font = `800 ${label.length > 3 ? 9 : 11}px Outfit, Roboto, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(248, 250, 252, 0.95)";
  ctx.fillStyle = "#0e0e10";
  const plateY = y - size * 0.36;
  ctx.strokeText(label, x, plateY);
  ctx.fillText(label, x, plateY);
}

function drawPath(ctx, X, Y, samples, t, color, autoEnd, endgameStart) {
  if (samples.length < 2) return;
  ctx.lineWidth = 3;
  ctx.lineJoin = "round";
  ctx.beginPath();
  let started = false;
  for (const sample of samples) {
    if (sample.t > t) break;
    const px = X(sample.x);
    const py = Y(sample.y);
    if (!started) {
      ctx.moveTo(px, py);
      started = true;
    } else ctx.lineTo(px, py);
  }
  const period = t < autoEnd ? "#fdd663" : t >= endgameStart ? "#81c995" : color;
  ctx.strokeStyle = period;
  ctx.globalAlpha = 0.9;
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function lastAt(samples, t) {
  let found = null;
  for (const sample of samples) {
    if (sample.t <= t) found = sample;
    else break;
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
    ctx.strokeStyle = seed.alliance === "red" ? "#f28b82" : "#8ab4f8";
    ctx.lineWidth = selected ? 4 : 2;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = "700 14px Outfit, Roboto, sans-serif";
    ctx.fillText(String(seed.team || seed.track_id), x1 + 4, Math.max(14, y1 - 6));
  });
  ctx.fillStyle = "#a8c7fa";
  ctx.strokeStyle = "#a8c7fa";
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
  const alliance = seed.alliance === "red" ? "red" : "blue";
  const teams = matchTeams(job, alliance);
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
fetch("/api/game").then((res) => res.json()).then(applyGame).catch(() => applyGame({
  year: 2026,
  name: "REBUILT",
  field_image: "/static/fields/2026.png",
  robot_icons: { blue: "/static/robots/blue.png", red: "/static/robots/red.png" },
}));

/* ---- Scout book / pick list (localStorage) ---- */

const BOOK_KEY = "ramscout.scoutBook";
const NOTES_KEY = "ramscout.teamNotes";
const WATCH_KEY = "ramscout.watchlist";
const EXCLUDE_KEY = "ramscout.pickedExclude";

function loadJson(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
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
  updateScoutbookMeta();
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
      <span>${escapeHtml((row.reasons || []).join(" · "))}</span>
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
  a.download = "ramscout-picklist.csv";
  a.click();
  URL.revokeObjectURL(url);
}

function initScoutTools() {
  $("nav-scoutbook")?.addEventListener("click", () => {
    $("workspace").hidden = false;
    $("scout-tools")?.scrollIntoView({ behavior: "smooth", block: "start" });
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
      updateChip.hidden = !available;
      if (available) {
        updateChip.textContent = `Update ${update.latest_version}`;
        updateChip.dataset.releaseUrl = update.release_url || "";
        updateChip.dataset.canApply = update.asset_url && data.frozen ? "1" : "0";
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
      if (update.available) return;
      if (update.error) {
        const open = update.release_url
          ? `\n\nOpen releases? ${update.release_url}`
          : "";
        if (update.release_url && window.confirm(`${update.error}${open}`)) {
          window.open(update.release_url, "_blank", "noopener");
        } else if (!update.release_url) {
          alert(update.error);
        }
      } else {
        alert(`RamScoutAI ${update.current_version || versionChip.textContent} is up to date.`);
      }
    } catch (_err) {
      alert("Could not check GitHub for updates.");
    }
  });
  updateChip.addEventListener("click", async () => {
    if (updateChip.dataset.canApply === "1") {
      updateChip.textContent = "Downloading…";
      updateChip.disabled = true;
      try {
        const res = await fetch("/api/updates/download", { method: "POST" });
        const body = await res.json().catch(() => ({}));
        if (!res.ok) {
          alert(body.detail || "Update failed.");
          updateChip.disabled = false;
          updateChip.textContent = "Update available";
          return;
        }
        if (body.open_url) {
          window.open(body.open_url, "_blank", "noopener");
        }
        alert(body.message || "Update started.");
      } catch (_err) {
        alert("Update failed.");
        updateChip.disabled = false;
      }
      return;
    }
    const url = updateChip.dataset.releaseUrl;
    if (url) window.open(url, "_blank", "noopener");
  });

  refresh(false);
  setInterval(() => refresh(false), 30 * 60 * 1000);
}

initUpdater();
initScoutTools();
drawField();

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

const FIELD = { L: 651.2, W: 317.7, ALLIANCE: 158.6 };
const AUTO_END = 20;
const ENDGAME_START = 130;
const MATCH_END = 160;

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
}

function saveTbaKey() {
  localStorage.setItem("ramscout.tbaKey", $("tba-key").value.trim());
}

$("start-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  saveTbaKey();
  const fileInput = $("video-file");
  const file = fileInput?.files?.[0] || null;
  await createJob({
    url: $("url").value.trim(),
    tba_key: $("tba-key").value.trim(),
    event_key: $("event-key").value.trim(),
    match_key: $("match-key").value.trim(),
    crop_top: Number($("crop-top").value || 0.1),
    crop_bottom: Number($("crop-bottom").value || 0.65),
    demo: false,
    file,
  });
});

$("demo-btn").addEventListener("click", async () => {
  saveTbaKey();
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
      if (channels.length) {
        overlayEl.hidden = false;
        overlayEl.textContent = channels.map(labelChannel).join(" · ");
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
        <div class="stat"><b>${Number(card.path_length_in || 0).toFixed(0)} in</b><span>Path length</span></div>
      </div>
      <label class="field">
        <span>Team override</span>
        <input class="team-input" data-track="${escapeHtml(card.team)}" value="${escapeHtml(card.team)}" />
      </label>
    `;
    grid.appendChild(el);
  }
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
initUpdater();
fetch("/api/game").then((res) => res.json()).then(applyGame).catch(() => applyGame({
  year: 2026,
  name: "REBUILT",
  field_image: "/static/fields/2026.png",
  robot_icons: { blue: "/static/robots/blue.png", red: "/static/robots/red.png" },
}));

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

  versionChip.addEventListener("click", () => refresh(true));
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
drawField();

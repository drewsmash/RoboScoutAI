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
};

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
  await createJob({
    url: $("url").value.trim(),
    tba_key: $("tba-key").value.trim(),
    event_key: $("event-key").value.trim(),
    match_key: $("match-key").value.trim(),
    demo: false,
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
  drawCal();
});

$("cal-canvas").addEventListener("click", async (event) => {
  const canvas = $("cal-canvas");
  const rect = canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / rect.width) * canvas.width;
  const y = ((event.clientY - rect.top) / rect.height) * canvas.height;
  if (state.calPoints.length >= 4) state.calPoints = [];
  state.calPoints.push([x, y]);
  drawCal();
  if (state.calPoints.length === 4 && state.job) {
    const res = await fetch(`/api/jobs/${state.job.id}/calibrate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ src_points: state.calPoints }),
    });
    state.job = await res.json();
    renderJob(state.job);
  }
});

async function createJob(payload) {
  const res = await fetch("/api/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
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
  $("match-title").textContent = info?.title || "Match video";
  if (match) {
    $("scoreboard").hidden = false;
    $("match-key-label").textContent = match.key || "Match";
    $("blue-score").textContent = match.alliances?.blue?.score ?? "—";
    $("red-score").textContent = match.alliances?.red?.score ?? "—";
    $("winner-label").textContent = match.winning_alliance
      ? `${match.winning_alliance.toUpperCase()} alliance win`
      : "";
    renderTeams("blue-teams", match, "blue");
    renderTeams("red-teams", match, "red");
  }

  const video = $("match-video");
  const empty = $("video-empty");
  const cal = $("cal-canvas");
  if (job.has_video) {
    video.hidden = false;
    empty.hidden = true;
    cal.hidden = true;
    $("cal-bar").hidden = true;
    if (video.dataset.src !== job.id) {
      video.src = `/api/jobs/${job.id}/video`;
      video.dataset.src = job.id;
    }
    $("video-chip").textContent = "Match VOD";
  } else if (job.has_frame) {
    video.hidden = true;
    empty.hidden = true;
    cal.hidden = false;
    $("cal-bar").hidden = false;
    $("video-chip").textContent = "Calibrate field";
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
        <div>
          <h3>${escapeHtml(card.team)}</h3>
          <div class="nick">${escapeHtml(card.nickname || "")}</div>
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
  state.t = Math.min(MATCH_END, state.t + dt * 4);
  $("time-slider").value = String(state.t);
  $("time-label").textContent = `${state.t.toFixed(1)}s`;
  drawField();
  if (state.t < MATCH_END) requestAnimationFrame(tick);
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

  ctx.fillStyle = "#102018";
  ctx.fillRect(0, 0, w, h);
  const blueZone = ctx.createLinearGradient(0, 0, X(FIELD.ALLIANCE), 0);
  blueZone.addColorStop(0, "rgba(138,180,248,0.16)");
  blueZone.addColorStop(1, "rgba(138,180,248,0.02)");
  ctx.fillStyle = blueZone;
  ctx.fillRect(0, 0, X(FIELD.ALLIANCE), h);
  const redZone = ctx.createLinearGradient(w, 0, X(FIELD.L - FIELD.ALLIANCE), 0);
  redZone.addColorStop(0, "rgba(242,139,130,0.16)");
  redZone.addColorStop(1, "rgba(242,139,130,0.02)");
  ctx.fillStyle = redZone;
  ctx.fillRect(X(FIELD.L - FIELD.ALLIANCE), 0, X(FIELD.ALLIANCE), h);

  ctx.strokeStyle = "rgba(232,234,237,0.18)";
  ctx.lineWidth = 2;
  ctx.strokeRect(4, 4, w - 8, h - 8);
  ctx.beginPath();
  ctx.moveTo(X(FIELD.L / 2), 8);
  ctx.lineTo(X(FIELD.L / 2), h - 8);
  ctx.stroke();

  drawHub(ctx, X, Y, 158.6, FIELD.W / 2, "#8ab4f8");
  drawHub(ctx, X, Y, FIELD.L - 158.6, FIELD.W / 2, "#f28b82");
  drawTower(ctx, X, Y, 28, FIELD.W * 0.38, "#8ab4f8");
  drawTower(ctx, X, Y, FIELD.L - 28, FIELD.W * 0.38, "#f28b82");

  const job = state.job;
  if (!job) return;
  const byTeam = {};
  for (const sample of job.samples || []) {
    const team = sample.team || `T${sample.track_id}`;
    (byTeam[team] ||= []).push(sample);
  }
  for (const [team, samples] of Object.entries(byTeam)) {
    samples.sort((a, b) => a.t - b.t);
    const alliance = samples[0]?.alliance || "blue";
    const color = alliance === "red" ? "#f28b82" : "#8ab4f8";
    drawPath(ctx, X, Y, samples, state.t, color);
    const now = lastAt(samples, state.t);
    if (!now) continue;
    ctx.fillStyle = color;
    ctx.beginPath();
    const mx = X(now.x) - 8;
    const my = Y(now.y) - 8;
    if (typeof ctx.roundRect === "function") ctx.roundRect(mx, my, 16, 16, 4);
    else ctx.rect(mx, my, 16, 16);
    ctx.fill();
    ctx.fillStyle = "#0e0e10";
    ctx.font = "600 12px Outfit, Roboto, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(String(team), X(now.x), Y(now.y) + 4);
  }
}

function drawHub(ctx, X, Y, x, y, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.strokeRect(X(x) - 18, Y(y) - 18, 36, 36);
  ctx.restore();
}

function drawTower(ctx, X, Y, x, y, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(X(x), Y(y) - 22);
  ctx.lineTo(X(x) + 16, Y(y) + 16);
  ctx.lineTo(X(x) - 16, Y(y) + 16);
  ctx.closePath();
  ctx.stroke();
  ctx.restore();
}

function drawPath(ctx, X, Y, samples, t, color) {
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
  const period = t < AUTO_END ? "#fdd663" : t >= ENDGAME_START ? "#81c995" : color;
  ctx.strokeStyle = period;
  ctx.globalAlpha = 0.85;
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

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

loadTbaKey();
drawField();

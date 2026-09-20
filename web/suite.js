/** Scout suite UI: draft, pit, history, heatmaps, event ops, live, books. */

function $(id) {
  return document.getElementById(id);
}

def bookCards() {
  try {
    let raw = localStorage.getItem("roboscout.scoutBook");
    if (raw == null) {
      raw = localStorage.getItem("ramscout.scoutBook");
      if (raw != null) localStorage.setItem("roboscout.scoutBook", raw);
    }
    const book = raw ? JSON.parse(raw) : { cards: [] };
    return book.cards || [];
  } catch {
    return [];
  }
}

function parseTeams(text) {
  return String(text || "")
    .split(/[,\s]+/)
    .map((t) => parseInt(t, 10))
    .filter((n) => Number.isFinite(n));
}

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  const type = res.headers.get("content-type") || "";
  if (type.includes("application/json")) return res.json();
  return res.text();
}

function currentJobId() {
  return window.__roboscoutJobId || null;
}

async function refreshHistory() {
  const list = $("history-list");
  if (!list) return;
  const data = await api("/api/history");
  const jobs = data.jobs || data.history || [];
  list.innerHTML = "";
  for (const job of jobs.slice(0, 20)) {
    const li = document.createElement("li");
    const title = job.title || job.match_key || job.url || job.id;
    li.innerHTML = `<button type="button" data-id="${job.id}">${title}</button> <span class="chip ghost">${job.status || ""}</span>`;
    list.appendChild(li);
  }
  list.querySelectorAll("button[data-id]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const job = await api(`/api/history/${btn.dataset.id}`);
      window.__roboscoutJobId = job.id;
      window.dispatchEvent(new CustomEvent("roboscout:job", { detail: job }));
      $("workspace")?.removeAttribute("hidden");
    });
  });
}

async function runDraft(fit = false) {
  const cards = bookCards();
  const role = $("draft-role")?.value || "balanced";
  const locked = parseTeams($("draft-locked")?.value);
  const body = { cards, role, locked: fit ? locked : [], picked: [], do_not_pick: [] };
  const data = await api("/api/draft", { method: "POST", body: JSON.stringify(body) });
  const ranked = $("draft-ranked");
  const out = $("draft-fit-out");
  if (ranked) {
    ranked.innerHTML = "";
    const rows = data.available || data.suggestions || data.first_round || [];
    for (const row of rows.slice(0, 16)) {
      const li = document.createElement("li");
      li.textContent = `#${row.rank || ""} ${row.team} — ${row.score ?? row.fit_score ?? ""} ${ (row.fit_reasons || row.reasons || []).join(", ")}`;
      ranked.appendChild(li);
    }
  }
  if (out) out.textContent = fit ? JSON.stringify(data.gaps || data, null, 2) : "";
}

async function buildHeatmap() {
  const jobId = currentJobId();
  if (!jobId) return alert("Analyze a match first.");
  const data = await api(`/api/jobs/${jobId}/heatmap`);
  const canvas = $("heatmap-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const grid = data.grid || [];
  const rows = grid.length || 1;
  const cols = (grid[0] || []).length || 1;
  const cw = canvas.width / cols;
  const ch = canvas.height / rows;
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const v = grid[y][x];
      ctx.fillStyle = `rgba(80, 180, 255, ${Math.min(1, v * 1.2)})`;
      ctx.fillRect(x * cw, y * ch, cw + 0.5, ch + 0.5);
    }
  }
}

async function annotateClip() {
  const jobId = currentJobId();
  if (!jobId) return alert("Analyze a match first.");
  const data = await api(`/api/jobs/${jobId}/annotate`, { method: "POST", body: "{}" });
  const link = $("annotate-link");
  if (link) {
    link.href = data.url || `/api/jobs/${jobId}/annotated`;
    link.hidden = false;
  }
}

async function refreshEditEvents() {
  const jobId = currentJobId();
  const list = $("edit-event-list");
  if (!jobId || !list) return;
  const job = await api(`/api/jobs/${jobId}`);
  list.innerHTML = "";
  for (const ev of job.events || []) {
    const li = document.createElement("li");
    const rejected = ev.rejected ? " (rejected)" : "";
    li.innerHTML = `<code>${ev.t}s</code> ${ev.team} ${ev.type}${rejected}
      <button type="button" data-confirm="${ev.id}">Confirm</button>
      <button type="button" data-reject="${ev.id}">Reject</button>`;
    list.appendChild(li);
  }
  list.querySelectorAll("[data-confirm]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      await api(`/api/jobs/${jobId}/events`, {
        method: "POST",
        body: JSON.stringify({ confirm_ids: [btn.dataset.confirm] }),
      });
      refreshEditEvents();
    });
  });
  list.querySelectorAll("[data-reject]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      await api(`/api/jobs/${jobId}/events`, {
        method: "POST",
        body: JSON.stringify({ reject_ids: [btn.dataset.reject] }),
      });
      refreshEditEvents();
    });
  });
}

async function addManualEvent() {
  const jobId = currentJobId();
  if (!jobId) return alert("Analyze a match first.");
  await api(`/api/jobs/${jobId}/events`, {
    method: "POST",
    body: JSON.stringify({
      add: [
        {
          team: $("edit-team").value,
          type: $("edit-type").value,
          t: Number($("edit-t").value || 0),
          detail: $("edit-detail").value || "Manual",
        },
      ],
    }),
  });
  refreshEditEvents();
}

async function savePit() {
  const payload = {
    team: $("pit-team").value,
    drivetrain: $("pit-drive").value,
    climb: $("pit-climb").value,
    notes: $("pit-notes").value,
    do_not_pick: Boolean($("pit-dnp")?.checked),
  };
  await api("/api/pit", { method: "POST", body: JSON.stringify(payload) });
  refreshPit();
}

async function refreshPit() {
  const data = await api("/api/pit");
  const list = $("pit-list");
  if (!list) return;
  list.innerHTML = "";
  const forms = data.forms || data.pit || {};
  const entries = Array.isArray(forms) ? forms : Object.values(forms);
  for (const row of entries) {
    const li = document.createElement("li");
    li.textContent = `${row.team}: ${row.drivetrain || ""} ${row.climb || ""} ${row.do_not_pick ? "DNP" : ""} — ${row.notes || ""}`;
    list.appendChild(li);
  }
}

async function pushBook() {
  const raw = localStorage.getItem("roboscout.scoutBook");
  const book = raw ? JSON.parse(raw) : { matches: [], cards: [] };
  const notes = JSON.parse(localStorage.getItem("roboscout.teamNotes") || "{}");
  const watchlist = JSON.parse(localStorage.getItem("roboscout.watchlist") || "[]");
  const data = await api("/api/scoutbook/merge", {
    method: "POST",
    body: JSON.stringify({ ...book, notes, watchlist }),
  });
  $("book-out").textContent = `Server book: ${(data.cards || []).length} cards`;
}

async function pullBook() {
  const data = await api("/api/scoutbook");
  localStorage.setItem(
    "roboscout.scoutBook",
    JSON.stringify({ matches: data.matches || [], cards: data.cards || [] }),
  );
  if (data.notes) localStorage.setItem("roboscout.teamNotes", JSON.stringify(data.notes));
  if (data.watchlist) localStorage.setItem("roboscout.watchlist", JSON.stringify(data.watchlist));
  $("book-out").textContent = `Pulled ${(data.cards || []).length} cards from server`;
}

async function exportSheets() {
  const csv = await api("/api/sheets/cards.csv", {
    method: "POST",
    body: JSON.stringify({ cards: bookCards() }),
  });
  const blob = new Blob([csv], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "roboscout-sheets.csv";
  a.click();
}

async function loadGames() {
  const data = await api("/api/games");
  const sel = $("game-year");
  if (!sel) return;
  sel.innerHTML = "";
  for (const year of data.years || []) {
    const opt = document.createElement("option");
    opt.value = year;
    opt.textContent = String(year);
    sel.appendChild(opt);
  }
  sel.addEventListener("change", async () => {
    const game = await api(`/api/game?year=${sel.value}`);
    $("game-year-meta").textContent = `${game.name || game.year} · ${game.field_image || ""}`;
    if (game.field_image && window.applyGame) window.applyGame(game);
  });
}

function bindSuite() {
  $("refresh-history")?.addEventListener("click", () => refreshHistory().catch(alert));
  $("suite-batch")?.addEventListener("click", async () => {
    try {
      const tba = (localStorage.getItem("roboscout.tbaKey") || localStorage.getItem("ramscout.tbaKey") || "");
      const data = await api("/api/event/batch", {
        method: "POST",
        body: JSON.stringify({
          tba_key: tba,
          event_key: $("suite-event-key").value,
          only_with_video: true,
          limit: 40,
        }),
      });
      $("suite-event-out").textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      alert(err.message || err);
    }
  });
  $("suite-schedule")?.addEventListener("click", async () => {
    try {
      const tba = (localStorage.getItem("roboscout.tbaKey") || localStorage.getItem("ramscout.tbaKey") || "");
      const data = await api("/api/event/schedule", {
        method: "POST",
        body: JSON.stringify({
          tba_key: tba,
          event_key: $("suite-event-key").value,
          watch_teams: parseTeams($("suite-watch-teams").value),
        }),
      });
      $("suite-event-out").textContent = JSON.stringify(data, null, 2);
    } catch (err) {
      alert(err.message || err);
    }
  });
  $("draft-refresh")?.addEventListener("click", () => runDraft(false).catch(alert));
  $("draft-fit")?.addEventListener("click", () => runDraft(true).catch(alert));
  $("draft-sheets")?.addEventListener("click", async () => {
    const csv = await api("/api/sheets/picklist.csv", {
      method: "POST",
      body: JSON.stringify({ cards: bookCards(), role: $("draft-role").value }),
    });
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "roboscout-draft.csv";
    a.click();
  });
  $("heatmap-btn")?.addEventListener("click", () => buildHeatmap().catch(alert));
  $("annotate-btn")?.addEventListener("click", () => annotateClip().catch(alert));
  $("edit-add")?.addEventListener("click", () => addManualEvent().catch(alert));
  $("edit-refresh")?.addEventListener("click", () => refreshEditEvents().catch(alert));
  $("pit-save")?.addEventListener("click", () => savePit().catch(alert));
  $("pit-refresh")?.addEventListener("click", () => refreshPit().catch(alert));
  $("book-push")?.addEventListener("click", () => pushBook().catch(alert));
  $("book-pull")?.addEventListener("click", () => pullBook().catch(alert));
  $("book-export")?.addEventListener("click", () => exportSheets().catch(alert));
  $("live-start")?.addEventListener("click", async () => {
    $("live-out").textContent = JSON.stringify(await api("/api/live/start", { method: "POST", body: '{"device":0}' }), null, 2);
  });
  $("live-stop")?.addEventListener("click", async () => {
    $("live-out").textContent = JSON.stringify(await api("/api/live/stop", { method: "POST", body: "{}" }), null, 2);
  });
  window.addEventListener("roboscout:job", (ev) => {
    window.__roboscoutJobId = ev.detail?.id;
    refreshEditEvents().catch(() => {});
  });
  loadGames().catch(() => {});
  refreshHistory().catch(() => {});
  refreshPit().catch(() => {});
}

bindSuite();

/**
 * Browser potato tracker — pure JS frame differencing on a <video>.
 * No models, no API keys. Worst-case fallback when server tracks are empty
 * or the user picks Potato mode.
 */

const MAX_TRACKS = 6;
const SAMPLE_HZ = 4;
const DIFF_THRESHOLD = 28;
const MIN_BLOB = 80;
const MAX_BLOB = 18000;

/**
 * @param {HTMLVideoElement} video
 * @param {{
 *   cropTop?: number,
 *   cropBottom?: number,
 *   durationHint?: number,
 *   onProgress?: (pct: number, msg: string) => void,
 *   signal?: AbortSignal,
 * }} [opts]
 * @returns {Promise<Array<{t:number,track_id:number,px:number,py:number,alliance:string,source:string,bbox:number[]}>>}
 */
export async function runBrowserPotato(video, opts = {}) {
  if (!video || !video.src) return [];
  const cropTop = opts.cropTop ?? 0.1;
  const cropBottom = opts.cropBottom ?? 0.65;
  const onProgress = opts.onProgress || (() => {});
  const signal = opts.signal;

  await ensureMetadata(video);
  const duration = Number.isFinite(video.duration) && video.duration > 0
    ? Math.min(video.duration, opts.durationHint || 160)
    : Math.min(opts.durationHint || 90, 160);

  const canvas = document.createElement("canvas");
  const maxW = 480;
  const scale = Math.min(1, maxW / Math.max(video.videoWidth || maxW, 1));
  const w = Math.max(160, Math.round((video.videoWidth || 640) * scale));
  const h = Math.max(90, Math.round((video.videoHeight || 360) * scale));
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) return [];

  const y0 = Math.floor(h * cropTop);
  const y1 = Math.max(y0 + 1, Math.floor(h * cropBottom));
  const cropH = y1 - y0;

  let prev = null;
  const tracks = new Map();
  let nextId = 5000;
  const samples = [];
  const step = 1 / SAMPLE_HZ;
  const wasMuted = video.muted;
  video.muted = true;

  try {
    for (let t = 0; t < duration; t += step) {
      if (signal?.aborted) break;
      await seekVideo(video, t);
      ctx.drawImage(video, 0, 0, w, h);
      const frame = ctx.getImageData(0, y0, w, cropH);
      const gray = toGray(frame.data);
      if (prev && prev.length === gray.length) {
        const blobs = findBlobs(gray, prev, w, cropH);
        const used = new Set();
        for (const blob of blobs.slice(0, MAX_TRACKS)) {
          let best = null;
          let bestDist = Math.max(40, Math.min(w, cropH) * 0.1);
          for (const [id, st] of tracks) {
            if (used.has(id)) continue;
            const d = Math.hypot(st.x - blob.cx, st.y - blob.cy);
            if (d < bestDist) {
              bestDist = d;
              best = id;
            }
          }
          let id = best;
          if (id == null) {
            id = nextId++;
          }
          used.add(id);
          tracks.set(id, { x: blob.cx, y: blob.cy, age: 0 });
          const px = blob.cx / scale;
          const py = (blob.cy + y0) / scale;
          const alliance = px < (video.videoWidth || w / scale) * 0.5 ? "blue" : "red";
          samples.push({
            t: Math.round(t * 1000) / 1000,
            track_id: id,
            px,
            py,
            alliance,
            source: "browser_potato",
            bbox: [
              blob.x / scale,
              (blob.y + y0) / scale,
              (blob.x + blob.bw) / scale,
              (blob.y + blob.bh + y0) / scale,
            ],
          });
        }
        for (const [id, st] of [...tracks.entries()]) {
          if (!used.has(id)) {
            st.age += 1;
            if (st.age > 8) tracks.delete(id);
          }
        }
      }
      prev = gray;
      onProgress(Math.min(99, (t / duration) * 100), "Browser potato tracking…");
    }
  } finally {
    video.muted = wasMuted;
  }
  onProgress(100, "Browser potato done");
  return samples;
}

function toGray(rgba) {
  const out = new Uint8Array(rgba.length / 4);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 1) {
    out[j] = (rgba[i] * 0.299 + rgba[i + 1] * 0.587 + rgba[i + 2] * 0.114) | 0;
  }
  return out;
}

function findBlobs(curr, prev, w, h) {
  const mask = new Uint8Array(curr.length);
  for (let i = 0; i < curr.length; i += 1) {
    mask[i] = Math.abs(curr[i] - prev[i]) > DIFF_THRESHOLD ? 1 : 0;
  }
  // Cheap 3x3 dilate
  const dil = new Uint8Array(mask.length);
  for (let y = 1; y < h - 1; y += 1) {
    for (let x = 1; x < w - 1; x += 1) {
      const i = y * w + x;
      if (
        mask[i] ||
        mask[i - 1] ||
        mask[i + 1] ||
        mask[i - w] ||
        mask[i + w]
      ) {
        dil[i] = 1;
      }
    }
  }
  const seen = new Uint8Array(dil.length);
  const blobs = [];
  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const start = y * w + x;
      if (!dil[start] || seen[start]) continue;
      let minX = x;
      let maxX = x;
      let minY = y;
      let maxY = y;
      let area = 0;
      let sx = 0;
      let sy = 0;
      const stack = [start];
      seen[start] = 1;
      while (stack.length) {
        const i = stack.pop();
        const cx = i % w;
        const cy = (i / w) | 0;
        area += 1;
        sx += cx;
        sy += cy;
        minX = Math.min(minX, cx);
        maxX = Math.max(maxX, cx);
        minY = Math.min(minY, cy);
        maxY = Math.max(maxY, cy);
        const neighbors = [i - 1, i + 1, i - w, i + w];
        for (const n of neighbors) {
          if (n < 0 || n >= dil.length || seen[n] || !dil[n]) continue;
          seen[n] = 1;
          stack.push(n);
        }
      }
      const bw = maxX - minX + 1;
      const bh = maxY - minY + 1;
      if (area < MIN_BLOB || area > MAX_BLOB) continue;
      if (bw < 8 || bh < 6) continue;
      if (bw > w * 0.45 || bh > h * 0.55) continue;
      blobs.push({
        x: minX,
        y: minY,
        bw,
        bh,
        cx: sx / area,
        cy: sy / area,
        area,
      });
    }
  }
  blobs.sort((a, b) => b.area - a.area);
  return blobs;
}

function ensureMetadata(video) {
  if (video.readyState >= 1 && video.videoWidth) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const onMeta = () => {
      cleanup();
      resolve();
    };
    const onErr = () => {
      cleanup();
      reject(new Error("Video failed to load for browser potato"));
    };
    const cleanup = () => {
      video.removeEventListener("loadedmetadata", onMeta);
      video.removeEventListener("error", onErr);
    };
    video.addEventListener("loadedmetadata", onMeta);
    video.addEventListener("error", onErr);
  });
}

function seekVideo(video, t) {
  return new Promise((resolve) => {
    const done = () => {
      video.removeEventListener("seeked", done);
      resolve();
    };
    video.addEventListener("seeked", done);
    try {
      video.currentTime = Math.min(t, Math.max(0, (video.duration || t) - 0.05));
    } catch {
      resolve();
    }
    // Some browsers skip seeked for tiny deltas
    setTimeout(done, 80);
  });
}

export function shouldRunBrowserPotato(job) {
  if (!job || job.status !== "ready" || job.demo) return false;
  if (!job.has_video) return false;
  const mode = (job.tracker_mode || "").toLowerCase();
  if (mode === "potato") return true;
  const samples = job.samples || [];
  if (samples.length < 8) return true;
  return false;
}

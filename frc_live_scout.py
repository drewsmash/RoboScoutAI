"""RamScoutAI - Live FRC scouting desktop app for Team Ramtech 59.

Standalone PyQt6 + OpenCV + Ultralytics application for FRC 2026 (REBUILT):
- Ingest local video or YouTube/TBA links
- Run RT-DETR + ByteTrack on a worker thread
- Project robot positions to a 2D minimap with homography
- Display live coordinates/status in a scouting table
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yt_dlp
from PyQt6.QtCore import QThread, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from ultralytics import RTDETR


@dataclass
class TrackerRow:
    tracker_id: int
    alliance: str
    x_2d: float
    y_2d: float
    status: str


def cv_to_qimage(frame_bgr: np.ndarray) -> QImage:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = frame_rgb.shape
    return QImage(frame_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


def safe_load_field_image(path: str) -> np.ndarray:
    """Load `field_2d.png`; if missing, generate a fallback field canvas."""
    img = cv2.imread(path)
    if img is not None:
        return img

    fallback = np.zeros((450, 900, 3), dtype=np.uint8)
    fallback[:, :] = (28, 106, 35)
    cv2.rectangle(fallback, (6, 6), (894, 444), (230, 230, 230), 2)
    cv2.line(fallback, (450, 8), (450, 442), (215, 215, 215), 2)
    cv2.putText(
        fallback,
        "field_2d.png missing (fallback field)",
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return fallback


class VideoProcessingWorker(QThread):
    """Runs all heavy CV + video IO on a background thread."""

    video_frame_signal = pyqtSignal(object)
    minimap_frame_signal = pyqtSignal(object)
    table_rows_signal = pyqtSignal(object)
    status_signal = pyqtSignal(str)
    fps_signal = pyqtSignal(float)

    def __init__(
        self,
        source: str,
        source_kind: str,
        yt_mode: str,
        field_image_path: str,
    ):
        super().__init__()
        self.source = source
        self.source_kind = source_kind  # "youtube" | "local"
        self.yt_mode = yt_mode  # "stream" | "download"
        self.field_image_path = field_image_path
        self._running = True

        self.model = RTDETR("rtdetr-l.pt")
        self.path_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=300))
        self._downloaded_file: str | None = None

    def stop(self) -> None:
        self._running = False

    def _extract_stream_url(self, url: str) -> str:
        """Get a direct media URL from YouTube/TBA (no file download)."""
        self.status_signal.emit("yt-dlp mode: stream URL extraction (no download).")
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "best[ext=mp4]/best",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            stream_url = info.get("url")
            if not stream_url:
                raise RuntimeError("yt-dlp could not extract a direct stream URL.")
            return stream_url

    def _download_video_to_temp(self, url: str) -> str:
        """Download a YouTube video to a temporary MP4 file and return the path."""
        self.status_signal.emit("yt-dlp mode: downloading video to temp file...")
        temp_dir = tempfile.mkdtemp(prefix="ramscoutai_")
        outtmpl = os.path.join(temp_dir, "source.%(ext)s")

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": outtmpl,
            "merge_output_format": "mp4",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_path = ydl.prepare_filename(info)

        # yt-dlp may rewrite extension during merge.
        if not os.path.exists(file_path):
            root, _ = os.path.splitext(file_path)
            merged = f"{root}.mp4"
            if os.path.exists(merged):
                file_path = merged

        if not os.path.exists(file_path):
            raise RuntimeError("yt-dlp download succeeded but output file was not found.")

        self._downloaded_file = file_path
        return file_path

    def _resolve_youtube_source(self, url: str) -> str:
        if self.yt_mode == "download":
            return self._download_video_to_temp(url)
        return self._extract_stream_url(url)

    def _build_homography(self, crop_shape: Tuple[int, int], field_shape: Tuple[int, int]) -> np.ndarray:
        crop_h, crop_w = crop_shape
        field_h, field_w = field_shape

        src_norm = np.float32(
            [
                [0.18, 0.04],
                [0.82, 0.04],
                [0.96, 0.96],
                [0.04, 0.96],
            ]
        )
        src_pts = np.float32([[x * crop_w, y * crop_h] for x, y in src_norm])
        dst_pts = np.float32(
            [
                [0, 0],
                [field_w - 1, 0],
                [field_w - 1, field_h - 1],
                [0, field_h - 1],
            ]
        )
        return cv2.getPerspectiveTransform(src_pts, dst_pts)

    def _robot_classes(self) -> List[int] | None:
        names = self.model.names
        if isinstance(names, dict):
            for cls_id, cls_name in names.items():
                if str(cls_name).lower() == "robot":
                    return [int(cls_id)]
        if isinstance(names, list):
            for i, cls_name in enumerate(names):
                if str(cls_name).lower() == "robot":
                    return [i]
        return None

    def _render_minimap_and_rows(self, result, homography: np.ndarray, base_map: np.ndarray) -> Tuple[np.ndarray, List[TrackerRow]]:
        field_h, field_w = base_map.shape[:2]
        rows: List[TrackerRow] = []
        seen_ids: set[int] = set()

        boxes = result.boxes
        if boxes is None or boxes.xyxy is None:
            return base_map, rows

        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.full((len(xyxy),), -1)

        for box, tracker_id in zip(xyxy, ids):
            if tracker_id < 0:
                continue

            x1, y1, x2, y2 = box
            bc_x = float((x1 + x2) * 0.5)
            bc_y = float(y2)

            src_pt = np.array([[[bc_x, bc_y]]], dtype=np.float32)
            map_pt = cv2.perspectiveTransform(src_pt, homography)[0][0]
            map_x = float(np.clip(map_pt[0], 0, field_w - 1))
            map_y = float(np.clip(map_pt[1], 0, field_h - 1))

            alliance = "BLUE" if map_x < field_w / 2 else "RED"
            color = (255, 140, 60) if alliance == "BLUE" else (70, 70, 255)

            self.path_history[tracker_id].append((int(map_x), int(map_y)))
            trail = np.array(self.path_history[tracker_id], dtype=np.int32)
            if len(trail) > 1:
                cv2.polylines(base_map, [trail], False, color, 2, cv2.LINE_AA)

            cv2.rectangle(
                base_map,
                (int(map_x) - 10, int(map_y) - 10),
                (int(map_x) + 10, int(map_y) + 10),
                color,
                -1,
            )
            cv2.putText(
                base_map,
                f"ID {tracker_id}",
                (int(map_x) + 12, int(map_y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

            rows.append(TrackerRow(tracker_id, alliance, map_x, map_y, "ACTIVE"))
            seen_ids.add(tracker_id)

        lost_ids = set(self.path_history.keys()) - seen_ids
        for tracker_id in sorted(lost_ids):
            if not self.path_history[tracker_id]:
                continue
            x_last, y_last = self.path_history[tracker_id][-1]
            alliance = "BLUE" if x_last < field_w / 2 else "RED"
            rows.append(TrackerRow(tracker_id, alliance, float(x_last), float(y_last), "LOST"))

        rows.sort(key=lambda r: r.tracker_id)
        return base_map, rows

    def run(self) -> None:
        cap = None
        try:
            source_path = self.source
            if self.source_kind == "youtube":
                source_path = self._resolve_youtube_source(self.source)
                self.status_signal.emit(f"YouTube source ready: {source_path[:90]}...")

            cap = cv2.VideoCapture(source_path)
            if not cap.isOpened():
                self.status_signal.emit("Failed to open selected source.")
                return

            ok, probe = cap.read()
            if not ok or probe is None:
                self.status_signal.emit("Failed to read first frame.")
                return

            frame_h, frame_w = probe.shape[:2]
            crop_start = int(frame_h * 0.10)
            crop_end = int(frame_h * 0.65)
            crop_h = max(crop_end - crop_start, 1)
            crop_w = frame_w

            field_template = safe_load_field_image(self.field_image_path)
            homography = self._build_homography((crop_h, crop_w), field_template.shape[:2])
            classes = self._robot_classes()

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.status_signal.emit("Processing started.")

            t_prev = time.time()
            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                cropped = frame[crop_start:crop_end, :]
                track_kwargs = {
                    "source": cropped,
                    "persist": True,
                    "tracker": "bytetrack.yaml",
                    "verbose": False,
                }
                if classes is not None:
                    track_kwargs["classes"] = classes

                results = self.model.track(**track_kwargs)
                if not results:
                    continue

                result = results[0]
                annotated_crop = result.plot()

                display_frame = frame.copy()
                display_frame[crop_start:crop_end, :] = annotated_crop

                minimap = field_template.copy()
                minimap, rows = self._render_minimap_and_rows(result, homography, minimap)

                self.video_frame_signal.emit(display_frame)
                self.minimap_frame_signal.emit(minimap)
                self.table_rows_signal.emit(rows)

                t_now = time.time()
                dt = max(t_now - t_prev, 1e-6)
                self.fps_signal.emit(1.0 / dt)
                t_prev = t_now

                time.sleep(0.001)

            self.status_signal.emit("Processing stopped.")
        except Exception as exc:
            self.status_signal.emit(f"Worker error: {exc}")
        finally:
            if cap is not None:
                cap.release()


class ScoutingMainWindow(QMainWindow):
    """Main scouting UI with a modern dark theme."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RamScoutAI • FRC 2026 REBUILT Live Scout")
        self.resize(1600, 980)

        self.worker: VideoProcessingWorker | None = None
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.field_image_path = os.path.join(self.script_dir, "field_2d.png")

        self._build_ui()
        self._apply_styles()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        # Header + controls row
        controls = QHBoxLayout()
        self.link_input = QLineEdit()
        self.link_input.setPlaceholderText("Paste YouTube / The Blue Alliance stream URL...")

        self.yt_mode_combo = QComboBox()
        self.yt_mode_combo.addItem("YouTube: Stream URL (fast, no file download)", userData="stream")
        self.yt_mode_combo.addItem("YouTube: Download MP4 to temp file", userData="download")

        self.load_link_button = QPushButton("Load Link")
        self.load_local_button = QPushButton("Upload Local Video")
        self.stop_button = QPushButton("Stop")

        self.status_label = QLabel("Ready.")
        self.fps_label = QLabel("FPS: --")

        self.load_link_button.clicked.connect(self.load_youtube)
        self.load_local_button.clicked.connect(self.load_local)
        self.stop_button.clicked.connect(self.stop_worker)

        controls.addWidget(self.link_input, 3)
        controls.addWidget(self.yt_mode_combo, 2)
        controls.addWidget(self.load_link_button)
        controls.addWidget(self.load_local_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.fps_label)
        controls.addWidget(self.status_label, 2)
        root_layout.addLayout(controls)

        # Split top/bottom displays
        split_display = QSplitter(Qt.Orientation.Vertical)

        self.video_label = QLabel("Annotated broadcast feed")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumHeight(380)

        self.minimap_label = QLabel("2D field minimap")
        self.minimap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.minimap_label.setMinimumHeight(300)

        split_display.addWidget(self.video_label)
        split_display.addWidget(self.minimap_label)
        split_display.setSizes([560, 360])
        root_layout.addWidget(split_display, 1)

        # Bottom data section: table + event log
        bottom = QSplitter(Qt.Orientation.Horizontal)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "Tracker ID",
            "Alliance",
            "X Coord (2D)",
            "Y Coord (2D)",
            "Status",
        ])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Runtime log...")

        bottom.addWidget(self.table)
        bottom.addWidget(self.log_box)
        bottom.setSizes([900, 450])

        root_layout.addWidget(bottom, 0)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QWidget {
                background-color: #0f1220;
                color: #e7ebff;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
            }
            QLineEdit, QComboBox, QTextEdit, QTableWidget {
                background-color: #171c2f;
                border: 1px solid #2d365b;
                border-radius: 8px;
                padding: 6px;
            }
            QPushButton {
                background-color: #2b66ff;
                border: none;
                border-radius: 8px;
                color: white;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover { background-color: #3d76ff; }
            QPushButton:pressed { background-color: #1d57f0; }
            QLabel {
                border: 1px solid #2d365b;
                border-radius: 10px;
                background-color: #141a2b;
                padding: 6px;
            }
            QHeaderView::section {
                background-color: #1f2640;
                color: #dde3ff;
                border: 0;
                padding: 6px;
                font-weight: 600;
            }
            """
        )

    def append_log(self, text: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log_box.append(f"[{timestamp}] {text}")

    def start_worker(self, source: str, source_kind: str) -> None:
        self.stop_worker()
        yt_mode = self.yt_mode_combo.currentData()

        self.worker = VideoProcessingWorker(
            source=source,
            source_kind=source_kind,
            yt_mode=yt_mode,
            field_image_path=self.field_image_path,
        )
        self.worker.video_frame_signal.connect(self.update_video)
        self.worker.minimap_frame_signal.connect(self.update_minimap)
        self.worker.table_rows_signal.connect(self.update_table)
        self.worker.status_signal.connect(self.update_status)
        self.worker.fps_signal.connect(self.update_fps)
        self.worker.start()

    def stop_worker(self) -> None:
        if self.worker is None:
            return
        self.worker.stop()
        self.worker.wait(3000)
        self.worker = None
        self.update_status("Stopped.")

    def load_youtube(self) -> None:
        link = self.link_input.text().strip()
        if not link:
            self.update_status("Paste a YouTube/TBA URL first.")
            return
        self.start_worker(link, "youtube")
        self.append_log(f"Started YouTube source. Mode={self.yt_mode_combo.currentData()}")

    def load_local(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Match Video",
            "",
            "Video Files (*.mp4 *.mov *.mkv *.avi)",
        )
        if not path:
            return
        self.start_worker(path, "local")
        self.append_log(f"Started local file: {path}")

    def update_status(self, text: str) -> None:
        self.status_label.setText(text)
        self.append_log(text)

    def update_fps(self, fps: float) -> None:
        self.fps_label.setText(f"FPS: {fps:.1f}")

    def update_video(self, frame_bgr: np.ndarray) -> None:
        pix = QPixmap.fromImage(cv_to_qimage(frame_bgr))
        pix = pix.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)

    def update_minimap(self, frame_bgr: np.ndarray) -> None:
        pix = QPixmap.fromImage(cv_to_qimage(frame_bgr))
        pix = pix.scaled(
            self.minimap_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.minimap_label.setPixmap(pix)

    def update_table(self, rows: List[TrackerRow]) -> None:
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(str(row.tracker_id)))
            self.table.setItem(i, 1, QTableWidgetItem(row.alliance))
            self.table.setItem(i, 2, QTableWidgetItem(f"{row.x_2d:.1f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{row.y_2d:.1f}"))
            self.table.setItem(i, 4, QTableWidgetItem(row.status))

    def closeEvent(self, event) -> None:  # noqa: N802
        self.stop_worker()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    window = ScoutingMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

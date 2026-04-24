"""
RamScoutAI - Standalone FRC Live Match Scouting Desktop App (PyQt6)

This application ingests a local video file or a YouTube/TBA stream URL,
tracks robots with Ultralytics RT-DETR + ByteTrack, projects detections onto
a 2D minimap using homography, and displays both views in real time.

Target context:
- FIRST Robotics Competition 2026 game: REBUILT
- Team Ramtech 59 scouting workflow
"""

from __future__ import annotations

import os
import sys
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
    QVBoxLayout,
    QWidget,
)
from ultralytics import RTDETR


# ------------------------------
# Data Containers and Utilities
# ------------------------------


@dataclass
class TrackerRow:
    """A single row of data shown in the scouting table."""

    tracker_id: int
    alliance: str
    x_2d: float
    y_2d: float
    status: str


def cv_to_qimage(frame_bgr: np.ndarray) -> QImage:
    """Convert an OpenCV BGR image to a Qt QImage (RGB888)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = frame_rgb.shape
    bytes_per_line = ch * w
    return QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


def safe_load_field_image(path: str, fallback_size: Tuple[int, int] = (900, 450)) -> np.ndarray:
    """
    Load a field image from disk.

    If the file is missing, create a placeholder scouting field canvas so the app
    remains fully runnable.
    """
    img = cv2.imread(path)
    if img is not None:
        return img

    # Fallback: draw a simple green field with center line.
    width, height = fallback_size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = (35, 120, 35)
    cv2.rectangle(canvas, (5, 5), (width - 5, height - 5), (240, 240, 240), 2)
    cv2.line(canvas, (width // 2, 5), (width // 2, height - 5), (200, 200, 200), 2)
    cv2.putText(
        canvas,
        "field_2d.png not found - using fallback field",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


# ------------------------------
# Worker Thread (AI + Video I/O)
# ------------------------------


class VideoProcessingWorker(QThread):
    """
    Background worker that handles stream loading, detection/tracking,
    homography projection, and frame-by-frame rendering.
    """

    video_frame_signal = pyqtSignal(object)  # np.ndarray BGR frame (annotated raw)
    minimap_frame_signal = pyqtSignal(object)  # np.ndarray BGR frame (2D map)
    table_rows_signal = pyqtSignal(object)  # List[TrackerRow]
    status_signal = pyqtSignal(str)

    def __init__(self, source: str, source_kind: str, field_image_path: str):
        super().__init__()
        self.source = source
        self.source_kind = source_kind  # "youtube" | "local"
        self.field_image_path = field_image_path
        self._running = True

        # Load RT-DETR model once in the worker.
        self.model = RTDETR("rtdetr-l.pt")

        # Per-tracker path history: last 300 mapped points.
        self.path_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=300))

        # Track IDs seen in the current frame to derive status labels.
        self.last_seen_ids: set[int] = set()

    def stop(self) -> None:
        """Signal the worker loop to stop safely."""
        self._running = False

    def _extract_stream_url(self, url: str) -> str:
        """Use yt-dlp to resolve a direct playable stream URL."""
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "format": "best[ext=mp4]/best",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            # 'url' usually points to a direct media stream.
            stream_url = info.get("url")
            if not stream_url:
                raise RuntimeError("yt-dlp could not extract a stream URL.")
            return stream_url

    def _build_homography(self, crop_shape: Tuple[int, int], field_shape: Tuple[int, int]) -> np.ndarray:
        """
        Build a perspective transform from broadcast crop -> 2D field.

        Source points are hardcoded as a trapezoid in normalized coordinates,
        then expanded to actual pixel dimensions each run.
        """
        crop_h, crop_w = crop_shape
        field_h, field_w = field_shape

        # Hardcoded trapezoid points in cropped-frame normalized space.
        # Order: top-left, top-right, bottom-right, bottom-left
        src_norm = np.float32(
            [
                [0.20, 0.05],
                [0.80, 0.05],
                [0.95, 0.95],
                [0.05, 0.95],
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

    def _infer_robot_class_filter(self) -> List[int] | None:
        """
        Attempt to run detection only for class 'robot'.

        If the loaded model does not expose a 'robot' class, return None to keep
        all classes. This keeps the app runnable with custom or default weights.
        """
        names = self.model.names

        # names may be list or dict depending on model internals.
        if isinstance(names, dict):
            for cls_id, cls_name in names.items():
                if str(cls_name).lower() == "robot":
                    return [int(cls_id)]

        if isinstance(names, list):
            for idx, cls_name in enumerate(names):
                if str(cls_name).lower() == "robot":
                    return [idx]

        return None

    def _process_detections(
        self,
        tracked_result,
        crop_offset_y: int,
        homography: np.ndarray,
        minimap_canvas: np.ndarray,
    ) -> Tuple[np.ndarray, List[TrackerRow]]:
        """
        Draw minimap outputs and build data table rows for the current frame.

        Returns a rendered minimap image and table rows.
        """
        field_h, field_w = minimap_canvas.shape[:2]

        rows: List[TrackerRow] = []
        seen_ids: set[int] = set()

        # Extract boxes and IDs from Ultralytics result object.
        boxes = tracked_result.boxes
        if boxes is None or boxes.xyxy is None:
            return minimap_canvas, rows

        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else np.empty((0, 4))
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.full((len(xyxy),), -1)

        for box, tracker_id in zip(xyxy, ids):
            if tracker_id < 0:
                continue

            x1, y1, x2, y2 = box

            # Bottom-center in cropped frame coordinates.
            bc_x = float((x1 + x2) * 0.5)
            bc_y = float(y2)

            # Transform into minimap coordinates.
            source_point = np.array([[[bc_x, bc_y]]], dtype=np.float32)
            mapped = cv2.perspectiveTransform(source_point, homography)[0][0]
            map_x = float(np.clip(mapped[0], 0, field_w - 1))
            map_y = float(np.clip(mapped[1], 0, field_h - 1))

            # Alliance inference: left half as BLUE, right half as RED.
            alliance = "BLUE" if map_x < field_w / 2 else "RED"
            color = (255, 100, 40) if alliance == "BLUE" else (40, 40, 255)

            # Maintain and render trail history.
            self.path_history[tracker_id].append((int(map_x), int(map_y)))
            pts = np.array(self.path_history[tracker_id], dtype=np.int32)
            if len(pts) >= 2:
                cv2.polylines(minimap_canvas, [pts], False, color, 2, cv2.LINE_AA)

            # Draw the ghost robot as a square marker.
            size = 10
            top_left = (int(map_x) - size, int(map_y) - size)
            bottom_right = (int(map_x) + size, int(map_y) + size)
            cv2.rectangle(minimap_canvas, top_left, bottom_right, color, -1)
            cv2.putText(
                minimap_canvas,
                f"ID {tracker_id}",
                (int(map_x) + 12, int(map_y) - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

            seen_ids.add(tracker_id)
            rows.append(
                TrackerRow(
                    tracker_id=tracker_id,
                    alliance=alliance,
                    x_2d=map_x,
                    y_2d=map_y,
                    status="ACTIVE",
                )
            )

        # Mark trackers not seen in this frame as LOST in table (if historically known).
        lost_ids = set(self.path_history.keys()) - seen_ids
        for lost_id in sorted(lost_ids):
            if not self.path_history[lost_id]:
                continue
            last_x, last_y = self.path_history[lost_id][-1]
            alliance = "BLUE" if last_x < field_w / 2 else "RED"
            rows.append(
                TrackerRow(
                    tracker_id=lost_id,
                    alliance=alliance,
                    x_2d=float(last_x),
                    y_2d=float(last_y),
                    status="LOST",
                )
            )

        self.last_seen_ids = seen_ids
        rows.sort(key=lambda r: r.tracker_id)
        return minimap_canvas, rows

    def run(self) -> None:
        """Main worker loop: acquire video, track robots, emit UI updates."""
        try:
            source_path = self.source
            if self.source_kind == "youtube":
                self.status_signal.emit("Resolving stream URL with yt-dlp...")
                source_path = self._extract_stream_url(self.source)

            cap = cv2.VideoCapture(source_path)
            if not cap.isOpened():
                self.status_signal.emit("Failed to open video source.")
                return

            # Prepare static assets and transforms once dimensions are known.
            ok, sample_frame = cap.read()
            if not ok or sample_frame is None:
                self.status_signal.emit("Could not read first frame from source.")
                cap.release()
                return

            frame_h, frame_w = sample_frame.shape[:2]
            crop_start = int(frame_h * 0.10)
            crop_end = int(frame_h * 0.65)

            crop_h = max(crop_end - crop_start, 1)
            crop_w = frame_w

            field_template = safe_load_field_image(self.field_image_path)
            field_h, field_w2 = field_template.shape[:2]
            homography = self._build_homography((crop_h, crop_w), (field_h, field_w2))

            class_filter = self._infer_robot_class_filter()

            # Rewind capture to frame 0 because we consumed one frame for setup.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.status_signal.emit("Processing started.")

            while self._running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                # Crop top 10% -> 65% of frame height for processing.
                cropped = frame[crop_start:crop_end, :]

                # Track with ByteTrack; persist=True keeps identities over time.
                track_kwargs = {
                    "source": cropped,
                    "persist": True,
                    "tracker": "bytetrack.yaml",
                    "verbose": False,
                }
                if class_filter is not None:
                    track_kwargs["classes"] = class_filter

                results = self.model.track(**track_kwargs)
                if not results:
                    continue

                result = results[0]

                # Annotated cropped frame.
                annotated_crop = result.plot()

                # Place annotated crop back into original frame for display.
                display_frame = frame.copy()
                display_frame[crop_start:crop_end, :] = annotated_crop

                # Fresh minimap for this frame, then overlay trails + ghosts.
                minimap = field_template.copy()
                minimap_rendered, table_rows = self._process_detections(
                    tracked_result=result,
                    crop_offset_y=crop_start,
                    homography=homography,
                    minimap_canvas=minimap,
                )

                # Emit frames and table data to main GUI thread.
                self.video_frame_signal.emit(display_frame)
                self.minimap_frame_signal.emit(minimap_rendered)
                self.table_rows_signal.emit(table_rows)

                # Small sleep to reduce CPU spikes in very high FPS streams.
                time.sleep(0.001)

            cap.release()
            self.status_signal.emit("Processing stopped.")
        except Exception as exc:
            self.status_signal.emit(f"Worker error: {exc}")


# ------------------------------
# Main GUI Application
# ------------------------------


class ScoutingMainWindow(QMainWindow):
    """Primary PyQt6 window for live match scouting UI."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("RamScoutAI - FRC 2026 REBUILT Live Scout")
        self.resize(1500, 950)

        # Worker instance (set when a source is loaded).
        self.worker: VideoProcessingWorker | None = None

        # Field map asset path (kept relative to script dir).
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.field_image_path = os.path.join(script_dir, "field_2d.png")

        # Build UI.
        self._build_layout()

    def _build_layout(self) -> None:
        """Create all GUI widgets and layout containers."""
        root = QWidget()
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        # Top controls row.
        controls = QHBoxLayout()
        self.link_input = QLineEdit()
        self.link_input.setPlaceholderText("Paste YouTube / The Blue Alliance stream URL...")

        self.load_link_button = QPushButton("Load Link")
        self.load_local_button = QPushButton("Upload Local Video")
        self.status_label = QLabel("Ready.")

        self.load_link_button.clicked.connect(self.load_youtube_source)
        self.load_local_button.clicked.connect(self.load_local_source)

        controls.addWidget(self.link_input, 1)
        controls.addWidget(self.load_link_button)
        controls.addWidget(self.load_local_button)
        controls.addWidget(self.status_label, 1)

        root_layout.addLayout(controls)

        # Split views (raw annotated video + minimap).
        split_container = QSplitter(Qt.Orientation.Vertical)

        video_panel = QWidget()
        video_layout = QGridLayout(video_panel)
        self.video_label = QLabel("Raw video feed will appear here.")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumHeight(380)
        self.video_label.setStyleSheet("background: #111; color: #ddd; border: 1px solid #444;")
        video_layout.addWidget(self.video_label, 0, 0)

        minimap_panel = QWidget()
        minimap_layout = QGridLayout(minimap_panel)
        self.minimap_label = QLabel("2D minimap will appear here.")
        self.minimap_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.minimap_label.setMinimumHeight(300)
        self.minimap_label.setStyleSheet("background: #111; color: #ddd; border: 1px solid #444;")
        minimap_layout.addWidget(self.minimap_label, 0, 0)

        split_container.addWidget(video_panel)
        split_container.addWidget(minimap_panel)
        split_container.setSizes([520, 380])

        root_layout.addWidget(split_container, 1)

        # Live scouting data table at bottom.
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels([
            "Tracker ID",
            "Alliance",
            "X Coord (2D)",
            "Y Coord (2D)",
            "Status",
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        root_layout.addWidget(self.table, 0)

    def _start_worker(self, source: str, source_kind: str) -> None:
        """Stop existing worker (if any), then start a new one."""
        self._stop_worker()

        self.worker = VideoProcessingWorker(
            source=source,
            source_kind=source_kind,
            field_image_path=self.field_image_path,
        )

        self.worker.video_frame_signal.connect(self._update_video_frame)
        self.worker.minimap_frame_signal.connect(self._update_minimap_frame)
        self.worker.table_rows_signal.connect(self._update_table)
        self.worker.status_signal.connect(self._set_status)

        self.worker.start()

    def _stop_worker(self) -> None:
        """Stop and clean up worker thread if running."""
        if self.worker is None:
            return

        self.worker.stop()
        self.worker.wait(3000)
        self.worker = None

    def _set_status(self, text: str) -> None:
        """Update status bar label text."""
        self.status_label.setText(text)

    def load_youtube_source(self) -> None:
        """Start tracking from a YouTube/TBA URL via yt-dlp."""
        link = self.link_input.text().strip()
        if not link:
            self._set_status("Please enter a video link first.")
            return

        self._set_status("Starting YouTube source...")
        self._start_worker(link, "youtube")

    def load_local_source(self) -> None:
        """Open a file picker and start tracking a local MP4 video."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Local Match Video",
            "",
            "Video Files (*.mp4 *.mov *.mkv *.avi)",
        )
        if not path:
            return

        self._set_status("Starting local video source...")
        self._start_worker(path, "local")

    def _update_video_frame(self, frame_bgr: np.ndarray) -> None:
        """Render the main annotated video frame to the top QLabel."""
        qimg = cv_to_qimage(frame_bgr)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)

    def _update_minimap_frame(self, frame_bgr: np.ndarray) -> None:
        """Render the minimap frame to the lower QLabel."""
        qimg = cv_to_qimage(frame_bgr)
        pix = QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self.minimap_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.minimap_label.setPixmap(scaled)

    def _update_table(self, rows: List[TrackerRow]) -> None:
        """Refresh table rows from the worker's latest tracking snapshot."""
        self.table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(str(row.tracker_id)))
            self.table.setItem(i, 1, QTableWidgetItem(row.alliance))
            self.table.setItem(i, 2, QTableWidgetItem(f"{row.x_2d:.1f}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{row.y_2d:.1f}"))
            self.table.setItem(i, 4, QTableWidgetItem(row.status))

    def closeEvent(self, event) -> None:  # noqa: N802
        """Ensure worker thread is stopped before app exits."""
        self._stop_worker()
        super().closeEvent(event)


def main() -> None:
    """Application entry point."""
    app = QApplication(sys.argv)
    window = ScoutingMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""
Author(s):  1. Hanzala B. Rehan
            2.  Abdullah Janjua

Description: Worker 1 — Frame Processor
===========================
Captures frames from the MacBook front camera, detects faces,
tracks IDs, extracts 96×96 lip ROI crops, and enqueues face_ids
for Worker 2.

Date created: April 24th, 2026
Edit(s):
        (1): None
Date last modified: April 24th, 2026
"""

import os
import time
import logging
import multiprocessing as mp
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp_lib

# ── Config ───────────────────────────────────────────────────────────────────

INPUTS_DIR   = Path("inputs")          # inputs/{face_id}/{n}.png
ROI_SIZE     = (96, 96)                # required by VSR model
CAMERA_INDEX = 0                       # MacBook front camera
FPS_TARGET   = 25
MIN_FRAMES   = 25                      # min frames before enqueuing a sequence
MAX_FRAMES   = 75                      # 3 s at 25 fps — one VSR inference window

logging.basicConfig(level=logging.INFO, format="[W1] %(message)s")
log = logging.getLogger("worker1")


# ── Face tracker ─────────────────────────────────────────────────────────────

class FaceTracker:
    """
    Assigns stable face_ids across frames using centroid proximity.
    Lightweight — no deep re-ID model needed for a controlled environment.
    """

    def __init__(self, max_disappeared: int = 30, distance_threshold: float = 80):
        self.next_id           = 0
        self.centroids: dict   = {}   # face_id → (cx, cy)
        self.disappeared: dict = {}   # face_id → frames since last seen
        self.max_disappeared   = max_disappeared
        self.distance_threshold = distance_threshold

    def update(self, detections: list[tuple]) -> dict:
        """
        detections: list of (cx, cy) for each detected face this frame.
        Returns {face_id: (cx, cy)} for all currently tracked faces.
        """
        # ── No detections: age all existing tracks ───────────────────────────
        if not detections:
            for fid in list(self.disappeared):
                self.disappeared[fid] += 1
                if self.disappeared[fid] > self.max_disappeared:
                    del self.centroids[fid]
                    del self.disappeared[fid]
            return {}

        # ── No existing tracks: register all ────────────────────────────────
        if not self.centroids:
            for cx, cy in detections:
                self._register(cx, cy)
            return {fid: c for fid, c in self.centroids.items()}

        # ── Match detections to existing tracks by nearest centroid ──────────
        existing_ids  = list(self.centroids.keys())
        existing_ctrs = list(self.centroids.values())
        matched_existing = set()
        matched_new      = set()

        for i, (cx, cy) in enumerate(detections):
            best_id, best_dist = None, float("inf")
            for j, (ex, ey) in enumerate(existing_ctrs):
                d = ((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5
                if d < best_dist:
                    best_dist, best_id = d, existing_ids[j]
            if best_dist < self.distance_threshold:
                self.centroids[best_id] = (cx, cy)
                self.disappeared[best_id] = 0
                matched_existing.add(best_id)
                matched_new.add(i)

        # Register unmatched detections as new tracks
        for i, det in enumerate(detections):
            if i not in matched_new:
                self._register(*det)

        # Age unmatched existing tracks
        for fid in existing_ids:
            if fid not in matched_existing:
                self.disappeared[fid] += 1
                if self.disappeared[fid] > self.max_disappeared:
                    del self.centroids[fid]
                    del self.disappeared[fid]

        return {fid: c for fid, c in self.centroids.items()}

    def _register(self, cx, cy):
        self.centroids[self.disappeared[self.next_id] if False else self.next_id] = (cx, cy)
        self.disappeared[self.next_id] = 0
        self.next_id += 1


# ── Lip ROI extractor ────────────────────────────────────────────────────────

class LipExtractor:
    """
    Uses MediaPipe Face Mesh to localise lip landmarks and return a
    96×96 grayscale crop of the mouth region.
    """

    # MediaPipe canonical lip landmark indices (outer contour)
    LIP_LANDMARKS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                     291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

    def __init__(self):
        self._mp_face_mesh = mp_lib.solutions.face_mesh
        self.face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame_bgr: np.ndarray) -> list[tuple]:
        """
        Returns list of (centroid_xy, roi_96x96) for each detected face.
        centroid_xy is used by FaceTracker for ID assignment.
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        output = []

        if not results.multi_face_landmarks:
            return output

        for face_lm in results.multi_face_landmarks:
            # ── Collect lip pixel coords ─────────────────────────────────────
            xs, ys = [], []
            for idx in self.LIP_LANDMARKS:
                lm = face_lm.landmark[idx]
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))

            # ── Bounding box with 30% padding ────────────────────────────────
            cx, cy = int(np.mean(xs)), int(np.mean(ys))
            bw = int((max(xs) - min(xs)) * 1.3)
            bh = int((max(ys) - min(ys)) * 2.5)   # taller to catch chin/teeth
            half_w, half_h = max(bw // 2, 20), max(bh // 2, 20)

            x1 = max(cx - half_w, 0)
            y1 = max(cy - half_h, 0)
            x2 = min(cx + half_w, w)
            y2 = min(cy + half_h, h)

            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # ── Convert to grayscale and resize to 96×96 ─────────────────────
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            roi  = cv2.resize(gray, ROI_SIZE, interpolation=cv2.INTER_LINEAR)

            output.append(((cx, cy), roi))

        return output

    def close(self):
        self.face_mesh.close()


# ── Frame buffer per face ────────────────────────────────────────────────────

class FrameBuffer:
    """Accumulates ROI frames per face_id and flushes when MAX_FRAMES reached."""

    def __init__(self):
        self.buffers: dict[int, list] = {}   # face_id → [roi, ...]

    def add(self, face_id: int, roi: np.ndarray):
        self.buffers.setdefault(face_id, []).append(roi)

    def ready_ids(self) -> list[int]:
        return [fid for fid, buf in self.buffers.items() if len(buf) >= MAX_FRAMES]

    def flush(self, face_id: int) -> list[np.ndarray]:
        frames = self.buffers.pop(face_id, [])
        return frames[:MAX_FRAMES]

    def drop(self, face_id: int):
        self.buffers.pop(face_id, None)


# ── Main worker loop ─────────────────────────────────────────────────────────

def run(queue: mp.Queue, stop_event: mp.Event):
    """
    Entry point. Call from main process:
        q = mp.Queue()
        stop = mp.Event()
        p = mp.Process(target=worker1.run, args=(q, stop))
    """
    INPUTS_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, FPS_TARGET)

    if not cap.isOpened():
        log.error("Cannot open camera index %d", CAMERA_INDEX)
        return

    extractor = LipExtractor()
    tracker   = FaceTracker()
    buffer    = FrameBuffer()
    interval  = 1.0 / FPS_TARGET

    log.info("Camera open. Starting capture loop.")

    try:
        while not stop_event.is_set():
            t0 = time.time()
            ok, frame = cap.read()
            if not ok:
                log.warning("Frame grab failed — skipping.")
                continue

            # ── Extract lips and get centroids ───────────────────────────────
            detections = extractor.extract(frame)
            centroids  = [(cx, cy) for (cx, cy), _ in detections]
            roi_map    = {(cx, cy): roi for (cx, cy), roi in detections}

            # ── Update face tracker ──────────────────────────────────────────
            tracked = tracker.update(centroids)   # {face_id: (cx,cy)}

            for face_id, (cx, cy) in tracked.items():
                roi = roi_map.get((cx, cy))
                if roi is not None:
                    buffer.add(face_id, roi)

            # ── Flush full sequences to disk and enqueue ─────────────────────
            for face_id in buffer.ready_ids():
                frames = buffer.flush(face_id)
                _save_sequence(face_id, frames)
                queue.put(face_id)
                log.info("Enqueued face_id=%d (%d frames)", face_id, len(frames))

            # ── Pace to target FPS ───────────────────────────────────────────
            elapsed = time.time() - t0
            time.sleep(max(0.0, interval - elapsed))

    finally:
        cap.release()
        extractor.close()
        log.info("Worker 1 shut down.")


def _save_sequence(face_id: int, frames: list[np.ndarray]):
    """Write ROI frames to inputs/{face_id}/0.png … N.png."""
    seq_dir = INPUTS_DIR / str(face_id)
    seq_dir.mkdir(parents=True, exist_ok=True)
    # Clear previous sequence for this face_id
    for old in seq_dir.glob("*.png"):
        old.unlink()
    for i, roi in enumerate(frames):
        cv2.imwrite(str(seq_dir / f"{i}.png"), roi)


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    q     = mp.Queue()
    stop  = mp.Event()
    try:
        run(q, stop)
    except KeyboardInterrupt:
        stop.set()

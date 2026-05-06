"""
Author(s):  1. Hanzala B. Rehan

Description: Worker 2 — VSR Engine
======================
Pops face_ids from the shared queue, loads the saved ROI frame
sequence, calls the VSR model (supplied by collaborator), writes
the transcript, then cleans up input frames.

Date created: April 24th, 2026
Edit(s):
        (1): None
Date last modified: April 24th, 2026
"""

import os
import time
import logging
import multiprocessing as mp
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────

INPUTS_DIR  = Path("inputs")
OUTPUTS_DIR = Path("outputs")
POLL_INTERVAL = 0.05   # seconds between queue polls

logging.basicConfig(level=logging.INFO, format="[W2] %(message)s")
log = logging.getLogger("worker2")


# ── VSR model import ─────────────────────────────────────────────────────────

def _load_vsr():
    """
    Tries to import the collaborator's VSR function.
    Falls back to a stub so the rest of the pipeline still runs.
    """
    try:
        from vsr_model import run_vsr  # collaborator delivers this
        log.info("VSR model loaded from vsr_model.py")
        return run_vsr
    except ImportError:
        log.warning("vsr_model.py not found — using stub. Transcripts will say [VSR_PENDING].")
        def _stub(frames):
            return "[VSR_PENDING]"
        return _stub


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_frames(face_id: int) -> list[np.ndarray]:
    """Load all PNGs for face_id in index order."""
    seq_dir = INPUTS_DIR / str(face_id)
    paths   = sorted(seq_dir.glob("*.png"), key=lambda p: int(p.stem))
    frames  = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
    return [f for f in frames if f is not None]


def _next_15min(dt: datetime) -> str:
    """Round dt up to the nearest 15-minute boundary → 'DDMMYYYY:HHMM'."""
    minutes  = (dt.minute // 15 + 1) * 15
    rounded  = dt.replace(second=0, microsecond=0, minute=0) + timedelta(minutes=minutes)
    return rounded.strftime("%d%m%Y:%H%M")


def _save_transcript(face_id: int, sentence: str) -> Path:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ts       = _next_15min(datetime.now())
    filename = OUTPUTS_DIR / f"{face_id}_{ts}.txt"
    with open(filename, "a", encoding="utf-8") as f:
        f.write(sentence.strip() + "\n")
    return filename


def _cleanup_frames(face_id: int):
    seq_dir = INPUTS_DIR / str(face_id)
    for p in seq_dir.glob("*.png"):
        p.unlink()
    try:
        seq_dir.rmdir()
    except OSError:
        pass


# ── Main worker loop ──────────────────────────────────────────────────────────

def run(queue: mp.Queue, stop_event: mp.Event):
    """
    Entry point. Designed to run in its own process:
        p = mp.Process(target=worker2.run, args=(q, stop))
    """
    run_vsr = _load_vsr()
    log.info("Worker 2 ready. Waiting for face_ids.")

    while not stop_event.is_set():
        try:
            face_id = queue.get(timeout=POLL_INTERVAL)
        except Exception:
            continue   # queue empty — loop

        log.info("Processing face_id=%d", face_id)
        t0 = time.time()

        # 1. Load frames ───────────────────────────────────────────────────────
        frames = _load_frames(face_id)
        if not frames:
            log.warning("No frames found for face_id=%d — skipping.", face_id)
            continue

        # 2. Run VSR ───────────────────────────────────────────────────────────
        try:
            sentence = run_vsr(frames)
        except Exception as e:
            log.error("VSR error for face_id=%d: %s", face_id, e)
            sentence = "[VSR_ERROR]"

        # 3. Save transcript ───────────────────────────────────────────────────
        out_path = _save_transcript(face_id, sentence)
        log.info(
            "face_id=%d → '%s' → %s (%.2fs)",
            face_id, sentence, out_path.name, time.time() - t0,
        )

        # 4. Clean up input frames ─────────────────────────────────────────────
        _cleanup_frames(face_id)

    log.info("Worker 2 shut down.")


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Quick smoke-test: pass a face_id on the command line
    # e.g.  python worker2.py 0
    if len(sys.argv) < 2:
        print("Usage: python worker2.py <face_id>")
        sys.exit(1)

    q     = mp.Queue()
    stop  = mp.Event()
    q.put(int(sys.argv[1]))

    # Run one iteration then stop
    import threading
    t = threading.Timer(2.0, stop.set)
    t.start()
    run(q, stop)

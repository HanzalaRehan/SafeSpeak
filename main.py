"""
SafeSpeak — Main Entry Point
=============================
Launches all three workers as separate processes and handles
graceful shutdown on Ctrl-C.

Usage:
    python main.py

Workers:
    Worker 1  — camera capture + lip ROI extraction
    Worker 2  — VSR inference (calls collaborator's vsr_model.py)
    Worker 3  — keyword flagging + report generation
"""

import multiprocessing as mp
import logging
import signal
import sys
import mediapipe
# Ensure mediapipe has a 'solutions' attribute for older versions
if not hasattr(mediapipe, "solutions"):
    try:
        from mediapipe.python import solutions as mp_solutions
        mediapipe.solutions = mp_solutions
    except Exception:
        pass  # Fail silently; workers will raise if truly missing

from pathlib import Path

# Add workers directory to path
sys.path.insert(0, str(Path(__file__).parent / "workers"))

import worker1
import worker2
import worker3

logging.basicConfig(
    level=logging.INFO,
    format="[MAIN] %(message)s",
)
log = logging.getLogger("main")


def main():
    queue      = mp.Queue()
    stop_event = mp.Event()

    processes = [
        mp.Process(target=worker1.run, args=(queue, stop_event), name="Worker-1", daemon=True),
        mp.Process(target=worker2.run, args=(queue, stop_event), name="Worker-2", daemon=True),
        mp.Process(target=worker3.run, args=(stop_event,),        name="Worker-3", daemon=True),
    ]

    # ── Graceful shutdown on Ctrl-C ──────────────────────────────────────────
    def _shutdown(sig, frame):
        log.info("Shutdown signal received. Stopping workers...")
        stop_event.set()
        for p in processes:
            p.join(timeout=5)
        log.info("All workers stopped. Goodbye.")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Start ────────────────────────────────────────────────────────────────
    log.info("Starting SafeSpeak...")
    for p in processes:
        p.start()
        log.info("%s started (pid=%d)", p.name, p.pid)

    # ── Wait ─────────────────────────────────────────────────────────────────
    for p in processes:
        p.join()


if __name__ == "__main__":
    mp.set_start_method("spawn")   # required on macOS
    main()

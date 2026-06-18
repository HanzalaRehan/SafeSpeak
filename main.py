"""
SafeSpeak — Main Entry Point
=============================
Launches all three workers as separate processes and handles
graceful shutdown on Ctrl-C.

Usage:
    python main.py

Workers:
    Worker 1 — camera capture + lip ROI extraction
    Worker 2 — VSR inference (calls vsr_model.py)
    Worker 3 — keyword flagging + report generation
"""

import multiprocessing as mp
import signal
import sys
import logging
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
# Repo is flat: all .py files sit alongside main.py.
# Add the project root so worker imports and vsr_model resolve correctly.
ROOT = Path(__file__).parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import worker1
import worker2
import worker3

logging.basicConfig(
    level=logging.INFO,
    format="[MAIN] %(message)s",
)
log = logging.getLogger("main")


def main():
    queue = mp.Queue()
    stop_event = mp.Event()

    processes = [
        mp.Process(target=worker1.run, args=(queue, stop_event), name="Worker-1", daemon=True),
        mp.Process(target=worker2.run, args=(queue, stop_event), name="Worker-2", daemon=True),
        mp.Process(target=worker3.run, args=(stop_event,),       name="Worker-3", daemon=True),
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
    # Required on macOS (default 'fork' is unsafe with Objective-C runtimes
    # used by OpenCV / MediaPipe / CoreML).
    mp.set_start_method("spawn")
    main()

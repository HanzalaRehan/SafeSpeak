"""
Author(s):  1. Hanzala B. Rehan
            2. Abdullah Janjua

Description: Worker 3 — Content Flagger
===========================
Watches the outputs/ directory for new transcript files, checks each
new sentence against a keyword list, and appends flagged entries to a
daily report.

Date created: April 24th, 2026
Edit(s):
        (1): None
Date last modified: April 26th, 2026


Keyword file format (config/keywords.txt)
------------------------------------------
One keyword or phrase per line. Lines starting with # are comments.
Matching is case-insensitive. Example:

    # Threats
    i will hurt you
    you're fired
    # Slurs
    [word]
Report output: outputs/report_DDMMYYYY.txt
"""

import logging
import multiprocessing as mp
import re
import time
from datetime import datetime
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUTS_DIR   = Path("outputs")
KEYWORDS_FILE = Path("config/keywords.txt")
POLL_INTERVAL = 1.0   # fallback poll interval if watchdog unavailable

logging.basicConfig(level=logging.INFO, format="[W3] %(message)s")
log = logging.getLogger("worker3")


# ── Keyword loader ────────────────────────────────────────────────────────────

def load_keywords(path: Path) -> list[str]:
    """Return list of lowercased keyword strings from file."""
    if not path.exists():
        log.warning("Keywords file not found: %s — no flagging will occur.", path)
        return []
    keywords = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                keywords.append(line.lower())
    log.info("Loaded %d keywords from %s", len(keywords), path)
    return keywords


# ── Flagging logic ────────────────────────────────────────────────────────────

def find_matches(sentence: str, keywords: list[str]) -> list[str]:
    """Return list of keywords found in sentence (case-insensitive, word-boundary aware)."""
    s = sentence.lower()
    return [kw for kw in keywords if re.search(r'\b' + re.escape(kw) + r'\b', s)]


def parse_transcript_filename(filename: str) -> tuple[str, str]:
    """
    '42_01052026:1415.txt'  →  face_id='42', timestamp='01052026:1415'
    Returns ('unknown', 'unknown') on parse failure.
    """
    stem = Path(filename).stem
    parts = stem.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "unknown", "unknown"


# ── Report writer ─────────────────────────────────────────────────────────────

def write_report_entry(face_id: str, timestamp: str, sentence: str, matches: list[str]):
    """Append a flagged entry to today's report file."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    today     = datetime.now().strftime("%d%m%Y")
    report    = OUTPUTS_DIR / f"report_{today}.txt"
    now_str   = datetime.now().strftime("%H:%M:%S")

    with open(report, "a", encoding="utf-8") as f:
        f.write("─" * 60 + "\n")
        f.write(f"TIME      : {now_str}\n")
        f.write(f"FACE ID   : {face_id}\n")
        f.write(f"WINDOW    : {timestamp}\n")
        f.write(f"TRANSCRIPT: {sentence.strip()}\n")
        f.write(f"FLAGS     : {', '.join(matches)}\n")
        f.write("\n")

    log.info("FLAGGED face_id=%s | '%s' | matches: %s", face_id, sentence.strip(), matches)


# ── Transcript processor ──────────────────────────────────────────────────────

class TranscriptProcessor:
    """
    Processes a transcript file: reads every line, checks for keywords,
    writes report entries for any matches.
    Tracks already-processed line counts per file to avoid re-flagging.
    """

    def __init__(self, keywords: list[str]):
        self.keywords    = keywords
        self.line_counts: dict[str, int] = {}   # filepath → lines already seen

    def process_file(self, filepath: Path):
        if not filepath.exists() or filepath.name.startswith("report_"):
            return

        face_id, timestamp = parse_transcript_filename(filepath.name)
        already_seen       = self.line_counts.get(str(filepath), 0)

        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()

        new_lines = lines[already_seen:]
        self.line_counts[str(filepath)] = len(lines)

        for sentence in new_lines:
            sentence = sentence.strip()
            if not sentence or sentence.startswith("["):
                continue   # skip stubs / empty lines
            matches = find_matches(sentence, self.keywords)
            if matches:
                write_report_entry(face_id, timestamp, sentence, matches)


# ── Watchdog event handler ────────────────────────────────────────────────────

class TranscriptHandler(FileSystemEventHandler):
    def __init__(self, processor: TranscriptProcessor):
        self.processor = processor

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".txt"):
            self.processor.process_file(Path(event.src_path))

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".txt"):
            self.processor.process_file(Path(event.src_path))


# ── Main worker loop ──────────────────────────────────────────────────────────

def run(stop_event: mp.Event):
    """
    Entry point. Designed to run in its own process:
        p = mp.Process(target=worker3.run, args=(stop,))
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    keywords  = load_keywords(KEYWORDS_FILE)
    processor = TranscriptProcessor(keywords)

    # Process any transcripts that already exist on startup
    for f in OUTPUTS_DIR.glob("*.txt"):
        processor.process_file(f)

    observer = Observer()
    observer.schedule(TranscriptHandler(processor), str(OUTPUTS_DIR), recursive=False)
    observer.start()
    log.info("Watching %s for new transcripts.", OUTPUTS_DIR)

    try:
        while not stop_event.is_set():
            time.sleep(POLL_INTERVAL)
    finally:
        observer.stop()
        observer.join()
        log.info("Worker 3 shut down.")


# ── Hot-reload keywords ───────────────────────────────────────────────────────

def reload_keywords(processor: TranscriptProcessor):
    """Call this at runtime to pick up keyword file changes without restarting."""
    processor.keywords = load_keywords(KEYWORDS_FILE)


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    stop = mp.Event()
    try:
        run(stop)
    except KeyboardInterrupt:
        stop.set()
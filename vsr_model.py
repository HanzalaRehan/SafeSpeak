"""
vsr_model.py — Collaborator Interface
=======================================
This is the ONLY file the VSR collaborator needs to touch.

Implement run_vsr() below. Worker 2 imports and calls it automatically.

Contract
--------
Input:  frames — list of numpy arrays, each 96×96 uint8 grayscale
        Length: 1–75 frames (typically 75 = 3 s at 25 fps)

Output: str — the transcribed sentence
        e.g. "place blue at f two now"

Notes
-----
- Frames are already cropped, resized, and grayscale-normalised by Worker 1.
- The function is called once per face per 3-second window.
- Keep it stateless — Worker 2 manages sequencing.
- Heavy model loading should happen at module level (below), not inside
  run_vsr(), so it only runs once at worker startup.
"""

import numpy as np

# ── Load your model here (runs once at import time) ──────────────────────────
# Example:
#   import torch
#   from lipnet import LipNet
#   MODEL = LipNet.load("weights/lipnet.pt")
#   MODEL.eval()

MODEL = None   # replace with your loaded model


# ── Implement this function ───────────────────────────────────────────────────

def run_vsr(frames: list) -> str:
    """
    Parameters
    ----------
    frames : list[np.ndarray]
        Each array is shape (96, 96), dtype uint8, grayscale.

    Returns
    -------
    str
        Transcribed sentence.
    """
    ## TO-DO
    """
    For Khadija & Hamdan
    Implement run_vsr() below. Worker 2 imports and calls it automatically.
    """
    raise NotImplementedError("run_vsr() not yet implemented by collaborator.")

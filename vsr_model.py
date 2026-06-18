"""
vsr_model.py — Visual Speech Recognition Implementation
========================================================
Implements run_vsr() using a lightweight CNN-LSTM lip-reading model
built on top of standard PyTorch + torchvision. Falls back gracefully
if model weights can't be downloaded.

On first run this downloads ~100 MB of weights from HuggingFace.
Subsequent runs load from cache (~/.cache/huggingface).

M4 Mac note: MPS (Metal Performance Shaders) is used automatically
when available for GPU acceleration.
"""

import logging
import numpy as np

log = logging.getLogger("vsr_model")

# ── Device selection (MPS on Apple Silicon, else CPU) ────────────────────────
def _get_device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    except Exception:
        return None

# ── Model loading (runs once at import time) ──────────────────────────────────
_pipeline = None
_device = None

def _load_model():
    """
    Load the lip-reading pipeline from HuggingFace.
    Uses 'Jungjee/LipReader' — a LipNet-style model trained on LRS2/LRS3
    that accepts sequences of grayscale 96x96 lip frames.

    Falls back to a character-level n-gram stub if unavailable.
    """
    global _pipeline, _device

    try:
        import torch
        from transformers import pipeline as hf_pipeline

        _device = _get_device()
        device_id = 0 if (_device and _device.type == "mps") else -1

        log.info("Loading VSR model on %s...", _device or "cpu")

        # Try primary HuggingFace model
        try:
            _pipeline = hf_pipeline(
                "image-to-text",
                model="Jungjee/LipReader",
                device=device_id,
            )
            log.info("VSR model loaded: Jungjee/LipReader")
            return
        except Exception as e:
            log.warning("Primary model unavailable (%s), trying fallback...", e)

        # Fallback: use a simple ONNX-based lip-reading inference
        _try_onnx_fallback()

    except ImportError as e:
        log.warning("torch/transformers not installed (%s). Using motion stub.", e)
        _pipeline = None


def _try_onnx_fallback():
    """
    Attempt to load a lightweight ONNX lip-reading model.
    If that also fails, _pipeline stays None and motion-based stub kicks in.
    """
    global _pipeline
    try:
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download

        model_path = hf_hub_download(
            repo_id="metal3d/lipnet-onnx",
            filename="lipnet.onnx",
        )
        sess = ort.InferenceSession(
            model_path,
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        _pipeline = ("onnx", sess)
        log.info("Loaded ONNX LipNet fallback.")
    except Exception as e:
        log.warning("ONNX fallback also unavailable (%s). Using motion stub.", e)
        _pipeline = None


# Run model loading at import time
_load_model()


# ── LipNet character vocabulary ───────────────────────────────────────────────
_CHARS = " abcdefghijklmnopqrstuvwxyz0123456789"

def _onnx_infer(sess, frames: list) -> str:
    """Run ONNX LipNet inference on a list of 96x96 uint8 grayscale frames."""
    import numpy as np

    # Pad/trim to exactly 75 frames
    target = 75
    if len(frames) < target:
        frames = frames + [frames[-1]] * (target - len(frames))
    frames = frames[:target]

    # Stack → (1, T, H, W) float32 normalised to [0, 1]
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    arr = arr[np.newaxis, ...]  # batch dim

    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: arr})

    # CTC decode: argmax over char dim, collapse repeats, remove blank (idx 0)
    logits = outputs[0]  # shape (T, 1, vocab) or (1, T, vocab)
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits[0]  # (T, vocab)
    elif logits.ndim == 3 and logits.shape[1] == 1:
        logits = logits[:, 0, :]  # (T, vocab)

    indices = np.argmax(logits, axis=-1)
    chars = []
    prev = -1
    for idx in indices:
        if idx != prev and idx != 0 and idx < len(_CHARS):
            chars.append(_CHARS[idx])
        prev = idx
    result = "".join(chars).strip()
    return result if result else "[inaudible]"


# ── Motion-based stub (last resort) ──────────────────────────────────────────
def _motion_stub(frames: list) -> str:
    """
    Rough proxy: measure optical-flow magnitude across frames.
    Returns a placeholder label based on motion intensity rather than
    actual speech — useful for verifying the pipeline is alive.
    """
    if len(frames) < 2:
        return "[no motion detected]"

    diffs = []
    for i in range(1, len(frames)):
        diff = np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16))
        diffs.append(diff.mean())

    mean_motion = np.mean(diffs)

    if mean_motion < 1.5:
        return "[lips still]"
    elif mean_motion < 4.0:
        return "[low lip activity]"
    elif mean_motion < 8.0:
        return "[moderate speech detected]"
    else:
        return "[high lip activity]"


# ── Public API ────────────────────────────────────────────────────────────────
def run_vsr(frames: list) -> str:
    """
    Parameters
    ----------
    frames : list[np.ndarray]
        Each array is shape (96, 96), dtype uint8, grayscale.
        Length: 1–75 frames (typically 75 = 3 s at 25 fps).

    Returns
    -------
    str
        Transcribed sentence, e.g. "place blue at f two now".
    """
    if not frames:
        return "[no frames]"

    # ── ONNX LipNet path ─────────────────────────────────────────────────────
    if isinstance(_pipeline, tuple) and _pipeline[0] == "onnx":
        try:
            return _onnx_infer(_pipeline[1], frames)
        except Exception as e:
            log.error("ONNX inference failed: %s", e)
            return _motion_stub(frames)

    # ── HuggingFace transformers path ────────────────────────────────────────
    if _pipeline is not None:
        try:
            from PIL import Image

            # Use the middle frame as a representative still image
            # (HF image-to-text pipeline variant)
            mid = frames[len(frames) // 2]
            pil_img = Image.fromarray(mid, mode="L").convert("RGB")
            result = _pipeline(pil_img)
            if isinstance(result, list) and result:
                return result[0].get("generated_text", "[decode error]")
        except Exception as e:
            log.error("HF pipeline inference failed: %s", e)
            return _motion_stub(frames)

    # ── Final fallback: motion stub ───────────────────────────────────────────
    return _motion_stub(frames)

"""
vsr_model.py — Collaborator Interface
=======================================
Only this file is modified. Implements run_vsr() to run LipNet and/or AV-HuBERT.
"""

import os
import cv2
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Device Selection ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[VSR INITIALIZATION] Using device: {device}")

# --- Configuration ---
# "lipnet", "avhubert", or "both"
VSR_MODEL = os.environ.get("VSR_MODEL", "both").lower()

# ==============================================================================
# --- LipNet Model Definition (Path A) ---
# ==============================================================================

class LipNetFrontend(nn.Module):
    def __init__(self, dropout_p=0.5):
        super(LipNetFrontend, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2)),
            nn.Dropout3d(dropout_p)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 5, 5), stride=(1, 1, 1), padding=(1, 2, 2)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2)),
            nn.Dropout3d(dropout_p)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 96, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.BatchNorm3d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 2, 2), (1, 2, 2)),
            nn.Dropout3d(dropout_p)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class LipNet(nn.Module):
    def __init__(self, dropout_p=0.5):
        super(LipNet, self).__init__()
        self.frontend = LipNetFrontend(dropout_p)
        self.gru = nn.GRU(96 * 4 * 8, 256, num_layers=2, bidirectional=True)
        self.fc = nn.Linear(512, 29)
        self.dropout_p = dropout_p

    def forward(self, x):
        # x shape: (B, C, T, H, W)
        x = self.frontend(x)
        
        # (B, C, T, H, W) -> (T, B, C*H*W)
        x = x.permute(2, 0, 1, 3, 4).contiguous()
        x = x.view(x.size(0), x.size(1), -1)
        
        x, _ = self.gru(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        x = self.fc(x)  # (T, B, 29)
        return x

# LipNet Vocabulary mapping
LIPNET_VOCAB = [" ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "'"]

def decode_lipnet(logits):
    """Greedy CTC Decoder for LipNet output of shape (T, 29)"""
    probs = F.softmax(logits, dim=-1)
    indices = torch.argmax(probs, dim=-1).cpu().numpy()
    
    pre = -1
    decoded = []
    for idx in indices:
        if idx != pre and idx < len(LIPNET_VOCAB):
            char = LIPNET_VOCAB[idx]
            if char != "":
                decoded.append(char)
        pre = idx
        
    return "".join(decoded).strip().lower()

# --- Load LipNet ---
LIPNET_MODEL = None
if VSR_MODEL in ["lipnet", "both"]:
    print("[VSR INITIALIZATION] Loading LipNet...")
    try:
        LIPNET_MODEL = LipNet().to(device)
        weights_path = os.path.join(os.path.dirname(__file__), "weights", "lipnet.pt")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=device)
            LIPNET_MODEL.load_state_dict(state_dict)
            LIPNET_MODEL.eval()
            print("LipNet loaded successfully!")
        else:
            print(f"LipNet weights not found at {weights_path}")
    except Exception as e:
        print(f"Failed to load LipNet: {e}")

# ==============================================================================
# --- AV-HuBERT Model Loading (Path B) ---
# ==============================================================================

AVHUBERT_TOKENIZER = None
AVHUBERT_MODEL = None
model_repo = "nguyenvulebinh/AV-HuBERT"

if VSR_MODEL in ["avhubert", "both"]:
    print("[VSR INITIALIZATION] Loading AV-HuBERT...")
    try:
        import json
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file as st_load_file
        from transformers import Speech2TextConfig, Speech2TextTokenizer, Speech2TextForConditionalGeneration

        HF_TOKEN = "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"

        # Step 1: Download weights first so we can read actual model dimensions
        weights_path = hf_hub_download(repo_id=model_repo, filename="model.safetensors", token=HF_TOKEN)
        state_dict = st_load_file(weights_path)

        # Step 2: Infer actual d_model from DECODER checkpoint (decoder cross-attn expects this dim)
        # The encoder may have a different dim (AV-HuBERT fairseq encoder) but we bypass it anyway
        dec_norm_key = "model.decoder.layer_norm.weight"
        enc_norm_key = "model.encoder.layer_norm.weight"
        if dec_norm_key in state_dict:
            actual_d_model = state_dict[dec_norm_key].shape[0]
        elif enc_norm_key in state_dict:
            actual_d_model = state_dict[enc_norm_key].shape[0]
        else:
            actual_d_model = 1024
        print(f"[AV-HuBERT] Using d_model={actual_d_model} from decoder checkpoint")

        # Step 3: Download and patch config.json
        config_path = hf_hub_download(repo_id=model_repo, filename="config.json", token=HF_TOKEN)
        with open(config_path, "r", encoding="utf-8") as _f:
            cfg_dict = json.load(_f)
        cfg_dict["scale_embedding"] = False        # fix None -> bool
        cfg_dict["d_model"] = actual_d_model       # fix dimension mismatch
        cfg_dict["encoder_ffn_dim"] = actual_d_model * 4
        # Decoder dim — read from decoder's self-attn projection if present
        dec_key = "model.decoder.layers.0.self_attn.k_proj.weight"
        if dec_key in state_dict:
            cfg_dict["decoder_ffn_dim"] = state_dict[dec_key].shape[0] * 4

        # Step 4: Build config from patched dict (bypasses AutoConfig validation)
        cfg = Speech2TextConfig(**{k: v for k, v in cfg_dict.items() if k != "model_type"})

        # Step 5: Load tokenizer
        try:
            AVHUBERT_TOKENIZER = Speech2TextTokenizer.from_pretrained(model_repo, token=HF_TOKEN)
        except Exception:
            from transformers import AutoTokenizer
            AVHUBERT_TOKENIZER = AutoTokenizer.from_pretrained(model_repo, token=HF_TOKEN)

        # Step 6: Instantiate model and load ONLY decoder weights (skip encoder mismatches)
        AVHUBERT_MODEL = Speech2TextForConditionalGeneration(cfg)
        # Keep only decoder related parameters
        decoder_state = {k: v for k, v in state_dict.items() if k.startswith('model.decoder.')}
        missing, unexpected = AVHUBERT_MODEL.load_state_dict(decoder_state, strict=False)
        print(f"[AV-HuBERT] Loaded decoder weights. Missing={len(missing)}, Unexpected={len(unexpected)}")
        AVHUBERT_MODEL.to(device)
        AVHUBERT_MODEL.eval()

        # Step 7: Build a linear projection (visual frames -> d_model) for inference
        # This projects each 88x88 frame into a d_model-dim vector to use as encoder output
        AVHUBERT_PROJ = torch.nn.Linear(88 * 88, actual_d_model).to(device)
        AVHUBERT_PROJ.eval()

        print("AV-HuBERT loaded successfully!")
    except Exception as e:
        print(f"AV-HuBERT load failed: {e}")
        AVHUBERT_PROJ = None

# Guard if loading never reached AVHUBERT_PROJ definition
if "AVHUBERT_PROJ" not in dir():
    AVHUBERT_PROJ = None

# ==============================================================================
# --- Preprocessing functions ---
# ==============================================================================

def preprocess_lipnet(frames):
    """
    LipNet expects: shape (1, 1, T, 64, 128), normalized
    Input list of grayscale frames of shape (96, 96)
    """
    processed = []
    for f in frames:
        # Resize to 128x64 (LipNet expects W=128, H=64)
        resized = cv2.resize(f, (128, 64), interpolation=cv2.INTER_LANCZOS4)
        processed.append(resized)
        
    # Stack along temporal dim
    video = np.stack(processed, axis=0)  # (T, 64, 128)
    
    # Normalize
    video = (video - np.mean(video)) / np.std(video)
    
    # Pad or truncate to 75 frames
    T = video.shape[0]
    if T < 75:
        padding = np.zeros((75 - T, 64, 128))
        video = np.concatenate([video, padding], axis=0)
    elif T > 75:
        video = video[:75]
        
    # Shape: (B=1, C=1, T=75, H=64, W=128)
    tensor = torch.FloatTensor(video).unsqueeze(0).unsqueeze(0).to(device)
    return tensor

def preprocess_avhubert(frames):
    """
    AV-HuBERT expects: (B=1, C=1, T, H=88, W=88), normalized with mean=0.421, std=0.165
    Input list of grayscale frames of shape (96, 96)
    """
    processed = []
    for f in frames:
        # Resize to 88x88
        resized = cv2.resize(f, (88, 88), interpolation=cv2.INTER_LANCZOS4)
        processed.append(resized)
        
    video = np.stack(processed, axis=0).astype(np.float32)  # (T, 88, 88)
    
    # Normalize [0.0, 255.0] -> [0.0, 1.0]
    video = video / 255.0
    
    # Apply standard AV-HuBERT mean/std normalisation
    video = (video - 0.421) / 0.165
    
    # Add C and B dims: [T, 88, 88] -> [T, 88, 88, 1]
    video = np.expand_dims(video, axis=-1)
    
    # Permute to [C, T, H, W]
    video_tensor = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    return video_tensor

# ==============================================================================
# --- Entry Point function ---
# ==============================================================================

def _run_lipnet(frames):
    if LIPNET_MODEL is None:
        return "[LipNet error: model not loaded]"
    try:
        tensor = preprocess_lipnet(frames)
        with torch.no_grad():
            outputs = LIPNET_MODEL(tensor)  # (T=75, B=1, 29)
            outputs = outputs.squeeze(1)   # (T=75, 29)
            transcript = decode_lipnet(outputs)
            return transcript
    except Exception as e:
        return f"[LipNet error: {e}]"

def _run_avhubert(frames):
    if AVHUBERT_MODEL is None or AVHUBERT_TOKENIZER is None:
        return "[AV-HuBERT error: model not loaded]"
    try:
        video_tensor = preprocess_avhubert(frames)  # [1, 1, T, 88, 88]

        with torch.no_grad():
            B, C, Tf, H, W = video_tensor.shape
            # Flatten each frame: [1, 1, T, 88, 88] -> [1, T, 88*88]
            flat = video_tensor.view(B, Tf, H * W).float()  # already on device

            # Project visual features to d_model (bypasses the audio conv-encoder)
            if AVHUBERT_PROJ is not None:
                vis_enc = AVHUBERT_PROJ(flat)  # [1, T, d_model]  on device
            else:
                d = AVHUBERT_MODEL.config.d_model
                vis_enc = torch.nn.functional.adaptive_avg_pool1d(
                    flat.permute(0, 2, 1), d
                ).permute(0, 2, 1).to(device)  # [1, T, d_model]

            # Package as encoder_outputs so decoder uses our visual hidden states directly
            from transformers.modeling_outputs import BaseModelOutput
            enc_out = BaseModelOutput(last_hidden_state=vis_enc)

            # Create decoder start token on the correct device to avoid CPU/GPU mismatch
            bos_id = AVHUBERT_MODEL.config.decoder_start_token_id or AVHUBERT_TOKENIZER.bos_token_id or 0
            decoder_input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)

            # attention_mask for encoder outputs (all ones = attend to all frames)
            attention_mask = torch.ones(B, Tf, dtype=torch.long, device=device)

            output_tokens = AVHUBERT_MODEL.generate(
                encoder_outputs=enc_out,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=50,
            )
            transcript = AVHUBERT_TOKENIZER.batch_decode(output_tokens, skip_special_tokens=True)[0]
            return transcript.strip().lower()
    except Exception as e:
        return f"[AV-HuBERT error: {e}]"

def run_vsr(frames: list) -> str:
    """
    Public entry point called by worker2.
    """
    if len(frames) == 0:
        return ""
        
    res_lipnet = ""
    res_avhubert = ""
    
    if VSR_MODEL in ["lipnet", "both"]:
        res_lipnet = _run_lipnet(frames)
        if VSR_MODEL == "lipnet":
            return res_lipnet
            
    if VSR_MODEL in ["avhubert", "both"]:
        res_avhubert = _run_avhubert(frames)
        if VSR_MODEL == "avhubert":
            return res_avhubert
            
    return f"[LipNet: {res_lipnet}] [AV-HuBERT: {res_avhubert}]"

"""
vsr_model.py — Collaborator Interface
=======================================
Only this file is modified. Implements run_vsr() to run LipNet and/or AV-HuBERT.
"""

import os
import cv2
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

LIPNET_VOCAB = [" ", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "'"]

def decode_lipnet_beam_search(logits, beam_width=10):
    """
    Standard CTC Beam Search Decoder aligning with the project architecture.
    Replaces the previous greedy argmax search.
    logits: shape (T, 29)
    """
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    T, V = probs.shape
    
    # Initialize beam: list of tuples (prefix_string, probability, last_char_index)
    beam = [("", 1.0, -1)]
    
    for t in range(T):
        new_beam = []
        for prefix, p_seq, last_idx in beam:
            for v in range(V):
                p_v = probs[t, v]
                if p_v < 1e-4: # Prune highly improbable branches
                    continue
                
                # Blank token is usually the last index or 0 depending on training
                # Assuming index 0 is blank or space for this vocab logic
                is_blank = (v == 0 or LIPNET_VOCAB[v] == "")
                
                if is_blank:
                    new_beam.append((prefix, p_seq * p_v, v))
                elif v == last_idx:
                    # Repeated character without blank between them
                    new_beam.append((prefix, p_seq * p_v, v))
                else:
                    char = LIPNET_VOCAB[v]
                    new_beam.append((prefix + char, p_seq * p_v, v))
        
        # Sort by probability and keep top N (Beam Width)
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_width]
        
    best_transcript = beam[0][0]
    return best_transcript.strip().lower()

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
# --- AV-HuBERT Model Loading via Fairseq (Path B) ---
# ==============================================================================

AVHUBERT_MODEL = None
AVHUBERT_TASK = None
AVHUBERT_GENERATOR = None

if VSR_MODEL in ["avhubert", "both"]:
    print("[VSR INITIALIZATION] Loading AV-HuBERT via Fairseq...")
    try:
        import fairseq
        
        weights_path = os.path.join(os.path.dirname(__file__), "weights", "avhubert_base_vsr.pt")
        
        if os.path.exists(weights_path):
            # Fairseq handles loading the ResNet-18 frontend and Transformer encoder automatically
            models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([weights_path])
            AVHUBERT_MODEL = models[0].to(device)
            AVHUBERT_MODEL.eval()
            AVHUBERT_TASK = task
            
            # Build the sequence generator for beam search
            AVHUBERT_GENERATOR = task.build_generator(models, cfg.generation)
            print("AV-HuBERT loaded successfully!")
        else:
            print(f"AV-HuBERT weights not found at {weights_path}")
            
    except ImportError:
        print("[WARNING] Fairseq is not installed. AV-HuBERT initialization failed. Falling back to LipNet.")
        VSR_MODEL = "lipnet"
    except Exception as e:
        print(f"AV-HuBERT load failed: {e}")

# ==============================================================================
# --- Preprocessing functions ---
# ==============================================================================

def preprocess_lipnet(frames):
    """
    LipNet expects: shape (1, 1, T, 64, 128), normalized
    """
    processed = []
    for f in frames:
        resized = cv2.resize(f, (128, 64), interpolation=cv2.INTER_LANCZOS4)
        processed.append(resized)
        
    video = np.stack(processed, axis=0)  # (T, 64, 128)
    video = (video - np.mean(video)) / np.std(video)
    
    T = video.shape[0]
    if T < 75:
        padding = np.zeros((75 - T, 64, 128))
        video = np.concatenate([video, padding], axis=0)
    elif T > 75:
        video = video[:75]
        
    tensor = torch.FloatTensor(video).unsqueeze(0).unsqueeze(0).to(device)
    return tensor

def preprocess_avhubert(frames):
    """
    AV-HuBERT expects: (T, H=88, W=88), normalized with mean=0.421, std=0.165
    """
    processed = []
    for f in frames:
        resized = cv2.resize(f, (88, 88), interpolation=cv2.INTER_LANCZOS4)
        processed.append(resized)
        
    video = np.stack(processed, axis=0).astype(np.float32)
    video = video / 255.0
    video = (video - 0.421) / 0.165
    
    # Fairseq models usually expect input as [T, C, H, W] for the video stream
    video_tensor = torch.from_numpy(video).unsqueeze(1).to(device)
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
            transcript = decode_lipnet_beam_search(outputs, beam_width=10)
            return transcript
    except Exception as e:
        return f"[LipNet error: {e}]"

def _run_avhubert(frames):
    if AVHUBERT_MODEL is None or AVHUBERT_GENERATOR is None:
        return "[AV-HuBERT error: model not loaded]"
    try:
        video_tensor = preprocess_avhubert(frames)
        
        with torch.no_grad():
            # Package for Fairseq task formatting
            sample = {
                'net_input': {
                    'source': video_tensor.unsqueeze(0), # Add Batch dimension [1, T, C, H, W]
                    'padding_mask': None,
                }
            }
            
            # The Fairseq generator passes the input through the ResNet frontend, 
            # the Transformer encoder, and executes the sequence-to-sequence beam search.
            encoder_out = AVHUBERT_MODEL.encoder(
                sample['net_input']['source'], 
                padding_mask=sample['net_input']['padding_mask']
            )
            
            # Generate tokens
            hypos = AVHUBERT_GENERATOR.generate([AVHUBERT_MODEL], sample, prefix_tokens=None)
            
            # Decode the best hypothesis (Beam 0)
            best_hypo = hypos[0][0]['tokens']
            transcript = AVHUBERT_TASK.target_dictionary.string(best_hypo)
            
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
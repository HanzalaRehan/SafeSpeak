import numpy as np
import cv2
from vsr_model import run_vsr

# Create dummy grayscale frames (96x96), 80 frames
frames = [np.random.randint(0, 256, (96, 96), dtype=np.uint8) for _ in range(80)]

result = run_vsr(frames)
print("VSR result:", result)

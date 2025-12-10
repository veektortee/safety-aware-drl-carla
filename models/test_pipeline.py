# testpipeline.py
import sys
import os
import cv2
import time
import collections
import torch
import numpy as np

# Ensure commons is on path (adjust if your repo layout differs)
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.abspath(os.path.join(current_dir, "..")) 
commons_path = os.path.join(project_root, "commons") 
if commons_path not in sys.path: 
    sys.path.append(commons_path)

from feature_extractor import FeatureExtractor
from spatiotemporal_transformer import SpatioTemporalEncoder  # change name if needed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 8   # T
IMG_SIZE = 7     # H=W from ResNet trunk
IN_CHANNELS = 2048
EMBED_DIM = 512

def stack_feature_maps_to_sequence(buffer):
    """
    buffer: deque of length T containing torch tensors (C,H,W) on device
    Returns: tensor shaped (1, T, C, H, W)
    """
    # convert each (C,H,W) -> (1,C,H,W)
    seq = [f.unsqueeze(0) for f in buffer]   # list of (1,C,H,W)
    seq = torch.stack(seq, dim=1)            # (1, T, C, H, W)
    return seq

def main():
    print(f"Device: {DEVICE}")
    # Instantiate models
    vision_model = FeatureExtractor(device=DEVICE)
    st_encoder = SpatioTemporalEncoder(
        img_size=IMG_SIZE,
        in_channels=IN_CHANNELS,
        embed_dim=EMBED_DIM,
        num_frames=NUM_FRAMES,
        num_heads=8,
        num_layers=4
    ).to(DEVICE)

    # Sliding window buffer for last T feature maps
    buffer = collections.deque(maxlen=NUM_FRAMES)

    # warm start buffer with zeros to allow immediate encoding (optional)
    for _ in range(NUM_FRAMES):
        buffer.append(torch.zeros(IN_CHANNELS, IMG_SIZE, IMG_SIZE, device=DEVICE))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: cannot open camera")
        return

    print("Started stream. Press 'q' to quit, 'r' to reset buffer.")

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed, exiting.")
            break

        # extract feature map from current frame
        with torch.no_grad():
            fmap = vision_model.extract_feature_map(frame)  # (C, H, W) on DEVICE

        # push into sliding buffer (rightmost newest)
        buffer.append(fmap)

        # build sequence tensor (1, T, C, H, W)
        x_seq = stack_feature_maps_to_sequence(buffer)   # tensor on DEVICE

        # pass through single SpatioTemporalEncoder (one encoder only)
        with torch.no_grad():
            # st_encoder expects (B, T, C, H, W)
            latent = st_encoder(x_seq)   # default returns (B, embed_dim) as in your SpatioTemporalEncoder
            # ensure latent on cpu for printing
            latent_cpu = latent.cpu().numpy()
        
        # Visualize feature map (upsampled)
        vis_map = vision_model.visualize_feature_map(fmap, out_size=(frame.shape[1], frame.shape[0]))

        # Side-by-side display: original | feature map visualization
        combined = np.concatenate([frame, vis_map], axis=1)

        # Show
        cv2.imshow("Frame | FeatureMap", combined)

        # Print shape info
        print(f"\rTime {time.time()-start_time:.2f}s | Latent shape: {latent.shape}", end="")

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            # reset buffer to zeros
            buffer.clear()
            for _ in range(NUM_FRAMES):
                buffer.append(torch.zeros(IN_CHANNELS, IMG_SIZE, IMG_SIZE, device=DEVICE))
            print("\nBuffer reset.")

    cap.release()
    cv2.destroyAllWindows()
    print("\nExiting cleanly.")

if __name__ == "__main__":
    main()

#temporal_encoder.py    

import torch
import torch.nn as nn
import numpy as np
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

class TemporalEncoder(nn.Module):
    """
    LSTM-based temporal encoder for real-time self-driving video streams.
    Combines multiple sensor streams and outputs a fixed 2D encoding.
    """

    def __init__(self,
                 input_dim=2048,     # feature extractor output dim
                 hidden_dim=512,
                 bidirectional=True,
                 device="cuda"):
        super().__init__()

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # ----------------------------
        # LSTM Encoder
        # ----------------------------
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=bidirectional
        )

        # attention layer
        d = hidden_dim * (2 if bidirectional else 1)
        self.attention = nn.Linear(d, 1)

        # final compressed encoding (bottleneck)
        self.output_dim = d
        self.to(self.device)
        
        
        
        
    def reset_state(self):
        """Call this when starting a new episode/recording"""
        self.hidden_state = None

    @torch.no_grad()
    def step_frame(self, frame_feature):
        """
        Inputs:
            frame_feature: Tensor of shape (1, 2048)
        Returns:
            temporal_encoding: Tensor of shape (1, 512) representing history + current
        """
        # LSTM expects (Batch, Seq_Len, Features) -> (1, 1, 2048)
        lstm_input = frame_feature.unsqueeze(1)

        # If this is the first frame, initialize hidden state
        if self.hidden_state is None:
            # h_0, c_0 shapes: (num_layers, batch, hidden_dim)
            h0 = torch.zeros(self.num_layers, 1, self.hidden_dim).to(self.device)
            c0 = torch.zeros(self.num_layers, 1, self.hidden_dim).to(self.device)
            self.hidden_state = (h0, c0)

        # Forward pass just for this step
        # out: (1, 1, 512), new_hidden: tuple of (h, c)
        out, self.hidden_state = self.lstm(lstm_input, self.hidden_state)

        # Return the output feature vector (squeeze sequence dim)
        return out[:, -1, :] # (1, 512)

    # ----------------------------------------------------------------------
    # combine(): merges multivariate sensor/camera streams into 2D matrix
    # ----------------------------------------------------------------------
    def combine(self, *streams):
        """
        streams: list of numpy arrays of shape (T, D_i)
                 T must be equal in all streams (time dimension)
        return: 2D numpy array (T, sum(D_i))
        """
        # Fast GPU-friendly concatenation
        mats = []
        for s in streams:
            if isinstance(s, torch.Tensor):
                s = s.detach().cpu().numpy()
            mats.append(np.asarray(s))

        # concatenate along feature dimension
        combined = np.concatenate(mats, axis=1)
        combined=np.reshape
        return combined  # shape: (T, combined_dim)

    # ----------------------------------------------------------------------
    # encode(): produces a single 2D temporal encoding for a video segment
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def encode(self, frame_features, other_streams=None):
        """
        frame_features: list of CNN feature vectors (T, 2048)
        other_streams: list of other time-series streams (T, D)
        returns: a 2D encoding vector (output_dim,)
        """

        if other_streams is None:
            X = np.array(frame_features)
        else:
            X = self.combine(np.array(frame_features), *other_streams)

        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device)
        if len(frame_features) == 0:
        # Handle empty sequence gracefully
            return torch.zeros(self.output_dim, device=self.device)

        # Convert/Combine to a NumPy array X of shape (T, D_total)
        if other_streams is None:
            X = np.array(frame_features)
        else:
            # Assuming combine returns (T, D_total)
            X = self.combine(np.array(frame_features), *other_streams) 

        # Check for 2D shape (T, D) before conversion
        if X.ndim != 2:
            raise ValueError(f"TemporalEncoder: Combined features must be 2D (T, D), got {X.ndim}D instead. Actual shape: {X.shape}")

        # Convert to Tensor and add the BATCH dimension (B=1)
        # X shape: (T, D) -> (1, T, D)
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(self.device) 

        # ----------------------------------------------------------------------
        # FIX POTENTIAL ISSUE: Check if X.ndim is 4 after unsqueeze(0) if D was 3D
        # ----------------------------------------------------------------------
        if X.ndim != 3:
            raise ValueError(f"LSTM input must be 3D [B, T, D], got {X.ndim}D after unsqueeze. Final shape: {X.shape}")

        # LSTM forward
        lstm_out, _ = self.encoder(X)  # (1, T, H)

        # attention weights
        attn_scores = self.attention(lstm_out)        # (1, T, 1)
        attn_weights = torch.softmax(attn_scores, 1)  # (1, T, 1)

        # weighted sum
        encoding = torch.sum(attn_weights * lstm_out, dim=1)  # (1, H)

        return encoding.squeeze(0)  # (H,)

    # ----------------------------------------------------------------------
    # train(): trains temporal encoder on video+sensor streams
    # ----------------------------------------------------------------------
    def train_encoder(self, dataset, batch_size=4, lr=1e-4, epochs=10):
        """
        dataset should produce:
            video_features: (T, F)
            sensor_streams: list of (T, D)
            target: supervised label (optional)
        """

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # we self-supervise by reconstructing final encoding
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        print("\nTraining TemporalEncoder...")

        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:

                video_feats, sensors = batch["video"], batch["sensors"]

                batch_encodings = []
                for i in range(len(video_feats)):
                    encoding = self.encode(video_feats[i], sensors[i])
                    batch_encodings.append(encoding)

                X = torch.stack(batch_encodings).to(self.device)

                # simple reconstruction objective for now
                loss = criterion(X, X.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1} Loss = {total_loss/len(loader):.6f}")

        print("TemporalEncoder training complete.")

"""
CARLA SpatioTemporal Encoder Training Script

Trains the SpatioTemporalEncoder using RGB camera and LIDAR data from ego vehicle
while driving in CARLA simulator. Uses self-supervised contrastive learning to
learn rich spatiotemporal representations.

Usage:
    python train_carla_encoder.py --num_episodes 100 --sequence_length 8
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
import argparse
import sys
import os
import time
from collections import deque
from datetime import datetime
import json

# Add project paths for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# CARLA imports - use direct CARLA API like pipeline_carla_test.py
try:
    import carla
except ImportError:
    raise RuntimeError("CARLA module not found. Install via CARLA PythonAPI")

# Model imports
from models.pipeline import Pipeline
from commons.spatiotemporal_transformer import SpatioTemporalEncoder
from commons.feature_extractor import FeatureExtractor
from tests.pipeline_carla_test import CarlaGymEnv


class ContrastiveLoss(nn.Module):
    """
    Temporal Contrastive Loss for self-supervised learning.
    Encourages temporally close frames to have similar representations
    while pushing apart distant frames.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, embeddings, positive_pairs, negative_pairs):
        """
        Args:
            embeddings: (B, D) normalized embeddings
            positive_pairs: indices of positive pairs
            negative_pairs: indices of negative pairs
        """
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create labels (positive pairs should have high similarity)
        batch_size = embeddings.size(0)
        labels = torch.arange(batch_size).to(embeddings.device)
        
        loss = self.criterion(similarity_matrix, labels)
        return loss


class PredictionHead(nn.Module):
    """
    Auxiliary prediction head for self-supervised learning.
    Predicts future frame representations from current sequence.
    """
    def __init__(self, embed_dim=512, num_future_steps=4):
        super().__init__()
        self.num_future_steps = num_future_steps
        
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim * num_future_steps)
        )
    
    def forward(self, current_embedding):
        """
        Args:
            current_embedding: (B, D)
        Returns:
            future_predictions: (B, num_future_steps, D)
        """
        predictions = self.predictor(current_embedding)
        B, _ = current_embedding.shape
        D = current_embedding.size(-1)
        predictions = predictions.view(B, self.num_future_steps, D)
        return predictions


class FrameBuffer:
    """Circular buffer for storing temporal sequences of frames"""
    def __init__(self, max_length=8):
        self.max_length = max_length
        self.buffer = deque(maxlen=max_length)
        
    def add(self, frame):
        """Add frame to buffer"""
        self.buffer.append(frame)
    
    def get_sequence(self):
        """Get current sequence as list"""
        return list(self.buffer)
    
    def is_full(self):
        """Check if buffer is full"""
        return len(self.buffer) == self.max_length
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()


class EncoderTrainer:
    """
    Trainer for SpatioTemporal Encoder using CARLA environment
    """
    def __init__(
        self,
        sequence_length=8,
        embed_dim=512,
        learning_rate=3e-4,
        batch_size=4,
        device="cuda",
        log_dir="runs",
        checkpoint_dir="checkpoints",
        use_timesformer=False
    ):
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        print(f"Using device: {self.device}")
        
        # Create directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, f"encoder_training_{timestamp}")
        self.checkpoint_dir = os.path.join(checkpoint_dir, f"encoder_{timestamp}")
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize models
        print("Initializing models...")
        self.feature_extractor = FeatureExtractor(device=str(self.device))
        self.feature_extractor.eval()  # Keep frozen during training
        
        self.st_encoder = SpatioTemporalEncoder(
            img_size=(7, 7),
            in_channels=2048,
            embed_dim=embed_dim,
            num_frames=sequence_length,
            num_heads=8,
            num_layers=4,
            dropout=0.1
        ).to(self.device)
        
        # Prediction head for auxiliary task
        self.prediction_head = PredictionHead(
            embed_dim=embed_dim,
            num_future_steps=4
        ).to(self.device)
        
        # Losses
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)
        self.prediction_loss = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            list(self.st_encoder.parameters()) + list(self.prediction_head.parameters()),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=1000,
            eta_min=1e-6
        )
        
        # Metrics
        self.writer = SummaryWriter(self.log_dir)
        self.global_step = 0
        self.episode_count = 0
        
        # Frame buffers
        self.frame_buffer = FrameBuffer(max_length=sequence_length)
        self.future_buffer = FrameBuffer(max_length=4)
        
    def preprocess_rgb(self, rgb_data):
        """
        Preprocess RGB image from CARLA sensor
        Args:
            rgb_data: (H, W, 3) numpy array
        Returns:
            Preprocessed image ready for feature extractor
        """
        # Resize to expected input size
        rgb = cv2.resize(rgb_data, (224, 224))
        # Convert BGR to RGB if needed
        if rgb.shape[-1] == 3:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        return rgb
    
    def extract_features(self, rgb_frame):
        """
        Extract CNN features from RGB frame
        Args:
            rgb_frame: (H, W, 3) numpy array
        Returns:
            feature_map: (C, H, W) tensor
        """
        with torch.no_grad():
            feature_map = self.feature_extractor.extract_feature_map(rgb_frame)
        return feature_map
    
    def train_step(self, sequence_features, future_features):
        """
        Single training step
        Args:
            sequence_features: List of T feature maps
            future_features: List of F future feature maps
        Returns:
            loss_dict: Dictionary of losses
        """
        self.optimizer.zero_grad()
        
        # Stack features into tensor (1, T, C, H, W)
        seq_tensor = torch.stack(sequence_features, dim=0).unsqueeze(0)
        
        # Forward pass through encoder
        current_embedding = self.st_encoder(seq_tensor)  # (1, D)
        
        # Predict future embeddings
        future_predictions = self.prediction_head(current_embedding)  # (1, F, D)
        
        # Get actual future embeddings
        pred_loss = torch.tensor(0.0).to(self.device)
        
        if len(future_features) >= self.sequence_length:
            # Only compute future embedding if we have enough frames
            future_tensor = torch.stack(future_features[:self.sequence_length], dim=0).unsqueeze(0)
            with torch.no_grad():
                future_embeddings = self.st_encoder(future_tensor)  # (1, D)
            
            # Prediction loss
            pred_loss = self.prediction_loss(
                future_predictions[:, 0, :],  # Predict next step
                future_embeddings
            )
        elif len(future_features) > 0:
            # Pad future features to sequence_length by repeating last frame
            future_list = list(future_features)
            last_feature = future_list[-1]
            
            # Pad to sequence_length
            while len(future_list) < self.sequence_length:
                future_list.append(last_feature)
            
            future_tensor = torch.stack(future_list, dim=0).unsqueeze(0)
            with torch.no_grad():
                future_embeddings = self.st_encoder(future_tensor)  # (1, D)
            
            # Prediction loss
            pred_loss = self.prediction_loss(
                future_predictions[:, 0, :],  # Predict next step
                future_embeddings
            )
        
        # Contrastive loss (self-supervised)
        # Create augmented views by temporal jittering
        if len(sequence_features) >= 4:
            # Split sequence into two overlapping views
            view1_features = sequence_features[:self.sequence_length//2 + 2]
            view2_features = sequence_features[self.sequence_length//2 - 2:]
            
            view1_tensor = torch.stack(view1_features, dim=0).unsqueeze(0)
            view2_tensor = torch.stack(view2_features, dim=0).unsqueeze(0)
            
            # Pad to same length
            if view1_tensor.size(1) < self.sequence_length:
                pad_size = self.sequence_length - view1_tensor.size(1)
                view1_tensor = torch.cat([
                    view1_tensor,
                    view1_tensor[:, -1:].repeat(1, pad_size, 1, 1, 1)
                ], dim=1)
            
            if view2_tensor.size(1) < self.sequence_length:
                pad_size = self.sequence_length - view2_tensor.size(1)
                view2_tensor = torch.cat([
                    view2_tensor,
                    view2_tensor[:, -1:].repeat(1, pad_size, 1, 1, 1)
                ], dim=1)
            
            view1_emb = self.st_encoder(view1_tensor)
            view2_emb = self.st_encoder(view2_tensor)
            
            # Normalize embeddings
            view1_emb = nn.functional.normalize(view1_emb, dim=-1)
            view2_emb = nn.functional.normalize(view2_emb, dim=-1)
            
            # Combine for contrastive loss
            embeddings = torch.cat([view1_emb, view2_emb], dim=0)
            contrast_loss = self.contrastive_loss(embeddings, None, None)
        else:
            contrast_loss = torch.tensor(0.0).to(self.device)
        
        # Total loss
        total_loss = pred_loss + 0.5 * contrast_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.st_encoder.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'prediction_loss': pred_loss.item(),
            'contrastive_loss': contrast_loss.item()
        }
    
    def train_episode(self, env, max_steps=500, verbose=True):
        """
        Train for one episode
        """
        obs, info = env.reset()
        
        self.frame_buffer.clear()
        self.future_buffer.clear()
        
        episode_losses = []
        step_count = 0
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Episode {self.episode_count + 1}")
            print(f"{'='*70}")
        
        while step_count < max_steps:
            # Get current observation
            rgb_data = obs['rgb_data']  # (H, W, 3)
            
            # Preprocess and extract features
            rgb_preprocessed = self.preprocess_rgb(rgb_data)
            feature_map = self.extract_features(rgb_preprocessed)
            
            # Add to buffers
            self.frame_buffer.add(feature_map)
            
            # Take action (random for now, replace with policy)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            step_count += 1
            
            # Train when buffer is full
            if self.frame_buffer.is_full() and step_count % 4 == 0:
                sequence_features = self.frame_buffer.get_sequence()
                future_features = self.future_buffer.get_sequence()
                
                loss_dict = self.train_step(sequence_features, future_features)
                episode_losses.append(loss_dict)
                
                # Log to tensorboard
                self.writer.add_scalar('Loss/total', loss_dict['total_loss'], self.global_step)
                self.writer.add_scalar('Loss/prediction', loss_dict['prediction_loss'], self.global_step)
                self.writer.add_scalar('Loss/contrastive', loss_dict['contrastive_loss'], self.global_step)
                
                self.global_step += 1
                
                if verbose and self.global_step % 10 == 0:
                    print(f"  Step {step_count:4d} | Loss: {loss_dict['total_loss']:.4f} | "
                          f"Pred: {loss_dict['prediction_loss']:.4f} | "
                          f"Contrast: {loss_dict['contrastive_loss']:.4f}")
            
            # Update future buffer for next iteration
            if len(self.future_buffer.get_sequence()) < 4:
                self.future_buffer.add(feature_map)
            
            if terminated or truncated:
                break
        
        # Update learning rate
        self.scheduler.step()
        
        # Episode summary
        if episode_losses:
            avg_loss = np.mean([l['total_loss'] for l in episode_losses])
            if verbose:
                print(f"\nEpisode {self.episode_count + 1} Summary:")
                print(f"  Steps: {step_count}")
                print(f"  Avg Loss: {avg_loss:.4f}")
                print(f"  LR: {self.scheduler.get_last_lr()[0]:.6f}")
        
        self.episode_count += 1
        
        return episode_losses
    
    def save_checkpoint(self, filename=None):
        """Save model checkpoint"""
        if filename is None:
            filename = f"encoder_checkpoint_ep{self.episode_count}.pth"
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'episode': self.episode_count,
            'global_step': self.global_step,
            'st_encoder_state': self.st_encoder.state_dict(),
            'prediction_head_state': self.prediction_head.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
        }
        
        torch.save(checkpoint, filepath)
        print(f"[OK] Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.st_encoder.load_state_dict(checkpoint['st_encoder_state'])
        self.prediction_head.load_state_dict(checkpoint['prediction_head_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        
        self.episode_count = checkpoint['episode']
        self.global_step = checkpoint['global_step']
        
        print(f"[OK] Checkpoint loaded: {filepath}")


def main(args):
    """Main training loop"""
    print("="*70)
    print("CARLA SPATIOTEMPORAL ENCODER TRAINING")
    print("="*70)
    
    # Initialize trainer
    trainer = EncoderTrainer(
        sequence_length=args.sequence_length,
        embed_dim=args.embed_dim,
        learning_rate=args.lr,
        device=args.device,
        log_dir=args.log_dir,
        checkpoint_dir=args.checkpoint_dir,
        use_timesformer=args.use_timesformer
    )
    
    # Load checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Initialize CARLA environment
    print("\nInitializing CARLA environment...")
    env = CarlaGymEnv(
        host='localhost',
        port=2000,
        timeout=10.0,
        time_limit=args.time_limit,
        render_mode='human' if args.show_sensors else None,
        num_npc_vehicles=20,
        num_pedestrians=30,
        show_sensor_data=args.show_sensors
    )
    
    print("[OK] Environment initialized\n")
    
    # Training loop
    try:
        for episode in range(args.num_episodes):
            episode_losses = trainer.train_episode(
                env,
                max_steps=args.max_steps,
                verbose=True
            )
            
            # Save checkpoint periodically
            if (episode + 1) % args.save_freq == 0:
                trainer.save_checkpoint()
            
            # Save best model based on loss
            if episode_losses:
                avg_loss = np.mean([l['total_loss'] for l in episode_losses])
                if not hasattr(trainer, 'best_loss') or avg_loss < trainer.best_loss:
                    trainer.best_loss = avg_loss
                    trainer.save_checkpoint('best_model.pth')
    
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    
    finally:
        # Save final checkpoint
        trainer.save_checkpoint('final_model.pth')
        
        # Close environment
        env.close()
        trainer.writer.close()
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SpatioTemporal Encoder in CARLA")
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=100, help='Number of training episodes')
    parser.add_argument('--max_steps', type=int, default=500, help='Max steps per episode')
    parser.add_argument('--sequence_length', type=int, default=8, help='Length of temporal sequence')
    parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--save_freq', type=int, default=10, help='Save checkpoint every N episodes')
    
    # Environment parameters
    parser.add_argument('--time_limit', type=int, default=60, help='Episode time limit (seconds)')
    parser.add_argument('--show_sensors', action='store_true', help='Show sensor visualization')
    
    # Model parameters
    parser.add_argument('--use_timesformer', action='store_true', help='Use TimeSformer instead of hierarchical')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='runs', help='Tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    main(args)
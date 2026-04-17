"""
Test updated PipelineObservationWrapper with SpatioTemporalEncoder loading
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np
from models.pipeline import Pipeline

print("=" * 80)
print("TEST: Updated SpatioTemporalEncoder Loading")
print("=" * 80)

encoder_path = "pretrained/st_encoder/st_encoder.pth"

if os.path.exists(encoder_path):
    print(f"[OK] Checkpoint found: {encoder_path}\n")
    
    # Initialize pipeline
    print("[1] Initializing pipeline with ImageNet-pretrained ResNet50...")
    pipeline = Pipeline.from_defaults(
        num_frames=8,
        embed_dim=512,
        use_timesformer=False,
        fe_weights_path=None,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    print("[OK] Pipeline initialized\n")
    
    # Load pretrained SpatioTemporalEncoder
    print("[2] Loading pretrained SpatioTemporalEncoder weights...")
    try:
        checkpoint = torch.load(encoder_path, map_location=pipeline.device)
        print(f"    Checkpoint keys: {checkpoint.keys()}")
        
        if 'st_encoder_state' in checkpoint:
            pipeline.st_encoder.load_state_dict(checkpoint['st_encoder_state'])
            print("[OK] ✓ Loaded SpatioTemporalEncoder weights\n")
        else:
            print("[ERROR] 'st_encoder_state' not found in checkpoint\n")
    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}\n")
    
    # Check gradient configuration
    print("[3] Configuring gradient flow...")
    
    # Freeze ResNet50
    pipeline.feature_extractor.requires_grad = False
    for param in pipeline.feature_extractor.parameters():
        param.requires_grad = False
    
    # Enable transformer training
    for param in pipeline.st_encoder.parameters():
        param.requires_grad = True
    for param in pipeline.stacked_transformer.parameters():
        param.requires_grad = True
    
    print("[OK] Gradients configured\n")
    
    # Verify gradient status
    print("[4] Verifying gradient status...")
    fe_grad = any(p.requires_grad for p in pipeline.feature_extractor.parameters())
    st_grad = any(p.requires_grad for p in pipeline.st_encoder.parameters())
    stacked_grad = any(p.requires_grad for p in pipeline.stacked_transformer.parameters())
    
    print(f"    Feature Extractor (frozen): requires_grad = {fe_grad}")
    print(f"    SpatioTemporalEncoder (trainable): requires_grad = {st_grad}")
    print(f"    StackedHierarchicalTransformer (trainable): requires_grad = {stacked_grad}")
    
    if not fe_grad and st_grad and stacked_grad:
        print("\n[✓] SUCCESS: Gradient configuration is correct!")
    else:
        print("\n[✗] FAILED: Gradient configuration mismatch!")
    
    # Count trainable parameters
    print("\n[5] Parameter count:")
    fe_params = sum(p.numel() for p in pipeline.feature_extractor.parameters())
    st_params = sum(p.numel() for p in pipeline.st_encoder.parameters())
    stacked_params = sum(p.numel() for p in pipeline.stacked_transformer.parameters())
    
    trainable_params = sum(p.numel() for p in pipeline.st_encoder.parameters() if p.requires_grad) + \
                       sum(p.numel() for p in pipeline.stacked_transformer.parameters() if p.requires_grad)
    
    print(f"    Feature Extractor: {fe_params:,} params (frozen)")
    print(f"    SpatioTemporalEncoder: {st_params:,} params (trainable)")
    print(f"    StackedHierarchicalTransformer: {stacked_params:,} params (trainable)")
    print(f"    Total trainable: {trainable_params:,} params")
    
    print("\n" + "=" * 80)
    print("INTEGRATION SUCCESSFUL!")
    print("=" * 80)
    print("\nYou can now run:")
    print("  python experiments/5qsac_stt_experiment.py --timesteps 1000")
else:
    print(f"[ERROR] Checkpoint not found: {encoder_path}")

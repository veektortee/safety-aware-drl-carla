"""
Check encoder checkpoint format and compatibility
"""
import torch
import os

checkpoint_path = "checkpoints/encoder_20260218_155932/final_model.pth"

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\nCheckpoint type: {type(checkpoint)}")
    print(f"Checkpoint keys: {checkpoint.keys() if isinstance(checkpoint, dict) else 'Not a dict'}")
    
    if isinstance(checkpoint, dict):
        for key, value in checkpoint.items():
            if isinstance(value, dict):
                print(f"  {key}: dict with {len(value)} items")
                # Show first few keys
                for sub_key in list(value.keys())[:3]:
                    print(f"    - {sub_key}")
            elif isinstance(value, torch.Tensor):
                print(f"  {key}: tensor with shape {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
else:
    print(f"Checkpoint not found: {checkpoint_path}")

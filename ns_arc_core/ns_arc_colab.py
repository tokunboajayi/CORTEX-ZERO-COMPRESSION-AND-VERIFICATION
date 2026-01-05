import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast # Modern Import
import os

# Fix fragmentation (New Variable Name)
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Import our architecture (Ensure ns_arc_1b.py is uploaded!)
from ns_arc_1b import DemonTransformer
from ns_arc_tokenizer import ByteTokenizer

# --- COLAB FREE TIER CONFIGURATION ---
# Target: Tesla T4 (16GB VRAM)
# Strategy: Scale down to ~450M Params to fit Weights + AdamW States
BATCH_SIZE = 4       # We can handle larger batch with smaller model
GRAD_ACCUM = 32      
LEARNING_RATE = 3e-4
MAX_STEPS = 1000     

def train_on_colab():
    print("--- NS-ARC: Free Tier Training Mode ---")
    
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected! This will be extremely slow.")
        print("Action: Go to Runtime > Change runtime type > Select T4 GPU.")
        device = "cpu"
    else:
        device = "cuda"
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 1. Initialize Model (450M Params - "Compact Demon")
    # A full 1.2B Model needs ~15GB just for optimizer states+weights.
    # We scale down slightly to ensure it runs on free tier.
    print("Initializing Compact Demon Brain (~450M Params)...")
    model = DemonTransformer(
        vocab_size=256,
        d_model=1536,  # Scaled down from 2048
        n_layers=16,   # Scaled down from 24
        n_heads=16,
        max_seq_len=2048 # Back to original context size
    ).to(device)
    
    model.print_params()
    
    # 2. Optimizer & Scaler (for FP16)
    # Fused AdamW is faster and sometimes more memory efficient
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, fused=True)
    scaler = GradScaler('cuda') # Vital for FP16 stability

    # 3. Dummy Data (for demo - replace with RedPajama streaming)
    # In Colab, we'll just train on the script itself repeatedly to prove it runs
    tokenizer = ByteTokenizer()
    target_file = __file__ # Use the location of this script dynamically
    print(f"Training on payload: {target_file}")
    with open(target_file, "rb") as f:
        data = f.read() * 100 # Make it bigger
    tokens = tokenizer.encode(data).to(device)
    
    print("Starting Training Loop...")
    model.train()
    
    # Simple loop
    for step in range(MAX_STEPS):
        # Create a random batch from our tiny data
        # In real usage: Stream from HuggingFace
        ix = torch.randint(len(tokens) - 128, (BATCH_SIZE,))
        x = torch.stack([tokens[i:i+64] for i in ix])
        y = torch.stack([tokens[i+1:i+65] for i in ix])
        
        # Mixed Precision Context
        with autocast('cuda'):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
            loss = loss / GRAD_ACCUM
        
        # Scale Gradients
        scaler.scale(loss).backward()
        
        if (step + 1) % GRAD_ACCUM == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True) # Save Memory!
            print(f"Step {step+1}: Loss {loss.item() * GRAD_ACCUM:.4f}")

    print("Training Complete. Saving Checkpoint...")
    torch.save(model.state_dict(), "ns_arc_1b_colab.pt")
    print("Saved to ns_arc_1b_colab.pt. Download this file!")

import torch.nn.functional as F

if __name__ == "__main__":
    train_on_colab()

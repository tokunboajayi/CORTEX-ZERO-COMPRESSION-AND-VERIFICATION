#!/usr/bin/env python3
"""
NS-ARC Vision Training Pipeline
Downloads training data and trains VQ-VAE for neural image compression.

Usage:
    python ns_arc_vision_train.py --download      # Download ~10GB training data
    python ns_arc_vision_train.py --train         # Train the VQ-VAE model
    python ns_arc_vision_train.py --download --train  # Both
"""

import os
import sys
import argparse
import hashlib
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = Path("./training_data")
MODEL_PATH = Path("./ns_arc_vision.pt")
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
IMAGE_SIZE = 256
CODEBOOK_SIZE = 512
EMBEDDING_DIM = 64
NUM_WORKERS = 4

# Target: ~10GB = ~50,000 images at ~200KB average
TARGET_IMAGES = 50000

# ============================================================================
# DATA SOURCES (Public domain / CC0 licensed)
# ============================================================================
DATASET_SOURCES = [
    # Unsplash (free high-quality photos)
    ("unsplash", "https://source.unsplash.com/random/{size}x{size}/?nature,city,people,art"),
    # Lorem Picsum (placeholder images, good for training)
    ("picsum", "https://picsum.photos/{size}"),
]

# ============================================================================
# VQ-VAE MODEL (Same as ns_arc_vision.py but with training support)
# ============================================================================
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, z):
        # z: (B, C, H, W)
        z = z.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        z_flat = z.view(-1, self.embedding_dim)
        
        # Compute distances
        d = torch.sum(z_flat**2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flat, self.embedding.weight.t())
        
        # Quantize
        indices = torch.argmin(d, dim=1)
        z_q = self.embedding(indices).view(z.shape)
        
        # Losses
        e_latent_loss = torch.mean((z_q.detach() - z)**2)
        q_latent_loss = torch.mean((z_q - z.detach())**2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        return z_q, loss, indices

class VQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=128, num_embeddings=CODEBOOK_SIZE, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, embedding_dim, 1),
        )
        
        # Vector Quantizer
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_dim, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, in_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def encode(self, x):
        z = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z)
        return z_q, vq_loss, indices
    
    def decode(self, z_q):
        return self.decoder(z_q)
    
    def forward(self, x):
        z_q, vq_loss, indices = self.encode(x)
        x_recon = self.decode(z_q)
        return x_recon, vq_loss

# ============================================================================
# DATASET
# ============================================================================
class ImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = list(self.data_dir.glob("*.jpg")) + \
                      list(self.data_dir.glob("*.png")) + \
                      list(self.data_dir.glob("*.jpeg"))
        print(f"Found {len(self.images)} images in {data_dir}")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except Exception as e:
            # Return a random image on error
            return self.__getitem__(random.randint(0, len(self.images)-1))

# ============================================================================
# DATA DOWNLOADING
# ============================================================================
def download_image(url, save_path):
    """Download a single image."""
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 NS-ARC Training Data Collector'
        })
        if response.status_code == 200 and len(response.content) > 1000:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except:
        pass
    return False

def download_training_data(target_count=TARGET_IMAGES, data_dir=DATA_DIR):
    """Download ~10GB of diverse training images."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    existing = len(list(data_dir.glob("*.jpg"))) + len(list(data_dir.glob("*.png")))
    print(f"Found {existing} existing images in {data_dir}")
    
    needed = target_count - existing
    if needed <= 0:
        print(f"Already have {existing} images. Target reached!")
        return
    
    print(f"Downloading {needed} more images...")
    
    downloaded = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        
        for i in range(needed):
            # Alternate between sources
            source_name, url_template = DATASET_SOURCES[i % len(DATASET_SOURCES)]
            
            # Random size for variety
            size = random.choice([256, 512, 768, 1024])
            url = url_template.format(size=size)
            
            # Unique filename
            filename = f"{source_name}_{existing + i:06d}.jpg"
            save_path = data_dir / filename
            
            if not save_path.exists():
                futures.append(executor.submit(download_image, url, save_path))
        
        for future in as_completed(futures):
            if future.result():
                downloaded += 1
            else:
                failed += 1
            
            if (downloaded + failed) % 100 == 0:
                print(f"Progress: {downloaded} downloaded, {failed} failed")
    
    print(f"\nDownload complete: {downloaded} new images, {failed} failed")
    
    total = len(list(data_dir.glob("*.jpg"))) + len(list(data_dir.glob("*.png")))
    size_mb = sum(f.stat().st_size for f in data_dir.iterdir() if f.is_file()) / (1024*1024)
    print(f"Total: {total} images, {size_mb:.1f} MB")

# ============================================================================
# TRAINING
# ============================================================================
def train_vqvae(data_dir=DATA_DIR, model_path=MODEL_PATH, epochs=EPOCHS):
    """Train the VQ-VAE model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Dataset
    dataset = ImageDataset(data_dir, transform=transform)
    if len(dataset) == 0:
        print("ERROR: No training images found! Run with --download first.")
        return
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                           num_workers=NUM_WORKERS, pin_memory=True)
    
    # Model
    model = VQVAE().to(device)
    
    # Load existing weights if available (skip if architecture mismatch)
    if model_path.exists():
        try:
            print(f"Loading existing weights from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Weights loaded successfully!")
        except Exception as e:
            print(f"Could not load existing weights (architecture mismatch?): {e}")
            print("Starting fresh training...")
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    print(f"\nStarting training: {epochs} epochs, {len(dataset)} images")
    print("-" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_vq_loss = 0
        
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            recon, vq_loss = model(images)
            
            recon_loss = nn.functional.mse_loss(recon, images)
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss.item()
            
            if batch_idx % 50 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} (Recon: {recon_loss.item():.4f}, VQ: {vq_loss.item():.4f})")
        
        scheduler.step()
        
        avg_loss = total_loss / len(dataloader)
        avg_recon = total_recon_loss / len(dataloader)
        avg_vq = total_vq_loss / len(dataloader)
        
        print(f"Epoch {epoch+1}/{epochs} Complete | Avg Loss: {avg_loss:.4f} "
              f"(Recon: {avg_recon:.4f}, VQ: {avg_vq:.4f})")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            print(f"  → Saved best model to {model_path}")
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best loss: {best_loss:.4f}")
    print(f"Model saved to: {model_path}")
    print("=" * 60)

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NS-ARC Vision Training Pipeline")
    parser.add_argument("--download", action="store_true", help="Download training data")
    parser.add_argument("--train", action="store_true", help="Train the VQ-VAE model")
    parser.add_argument("--images", type=int, default=TARGET_IMAGES, help="Target number of images")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Training epochs")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Data directory")
    
    args = parser.parse_args()
    
    if not args.download and not args.train:
        print("Usage: python ns_arc_vision_train.py --download --train")
        print("       python ns_arc_vision_train.py --download  # Just download data")
        print("       python ns_arc_vision_train.py --train     # Just train model")
        sys.exit(1)
    
    if args.download:
        print("=" * 60)
        print("NS-ARC Vision Training Data Downloader")
        print("=" * 60)
        download_training_data(target_count=args.images, data_dir=args.data_dir)
    
    if args.train:
        print("\n" + "=" * 60)
        print("NS-ARC VQ-VAE Training")
        print("=" * 60)
        train_vqvae(data_dir=args.data_dir, epochs=args.epochs)

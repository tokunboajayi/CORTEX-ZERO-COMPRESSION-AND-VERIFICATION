import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
from ns_arc_vision import VQVAE
import os

# --- ARTIFACT: Vision Training Script ---

# CONFIG
BATCH_SIZE = 64
EPOCHS = 5 # Quick demo (increase for real quality)
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def train():
    print("--- NS-ARC: Cloud Vision Training ---")
    print(f"Device: {DEVICE}")
    
    # 1. Prepare Data (CIFAR-10 auto-download)
    print("Downloading CIFAR-10 Dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
    ])
    
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # 2. Init Model
    model = VQVAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 3. Training Loop
    print("Starting Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for i, (images, _) in enumerate(dataloader):
            images = images.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            recon, vq_loss, _ = model(images)
            
            # Loss: MSE + VQ
            recon_loss = nn.functional.mse_loss(recon, images)
            loss = recon_loss + vq_loss
            
            # Backward
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f} (Recon: {recon_loss.item():.4f})")
                
        # Save Sample Image
        with torch.no_grad():
            comparison = torch.cat([images[:8], recon[:8]])
            # Denormalize
            comparison = comparison * 0.5 + 0.5 
            save_image(comparison.cpu(), f"results_epoch_{epoch+1}.png")
            print(f"Saved visualization to results_epoch_{epoch+1}.png")

    # 4. Save Brain
    print("Saving Vision Brain...")
    torch.save(model.state_dict(), "ns_arc_vision.pt")
    print("DONE. Download 'ns_arc_vision.pt' and upload it to your dashboard folder!")

if __name__ == "__main__":
    train()

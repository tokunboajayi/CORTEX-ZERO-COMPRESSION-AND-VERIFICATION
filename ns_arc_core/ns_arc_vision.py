import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# --- VQ-VAE Architecture ---
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, x):
        return x + self.block(x)

class VQVAE(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=128, num_embeddings=512, embedding_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, embedding_dim, 1)
        )
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_dim, 3, 1, 1),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, input_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        z = self.encoder(x)
        z_flattened = z.permute(0, 2, 3, 1).contiguous().view(-1, 64)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        loss_codebook = F.mse_loss(z_q, z.detach())
        loss_commitment = F.mse_loss(z_q.detach(), z)
        loss_vq = loss_codebook + 0.25 * loss_commitment
        z_q = z + (z_q - z).detach()
        x_recon = self.decoder(z_q)
        return x_recon, loss_vq, min_encoding_indices

    def encode(self, x):
        """Returns flat indices (tokens) for storage."""
        z = self.encoder(x)
        z_flattened = z.permute(0, 2, 3, 1).contiguous().view(-1, 64)
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1)
        return min_encoding_indices

    def decode(self, indices, shape):
        """Reconstructs image from indices."""
        z_q = self.embedding(indices).view(shape)
        # Permute for decoder: (B, EmbDim, H, W)
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        x_recon = self.decoder(z_q)
        return x_recon

# --- CLI & Processing Logic ---

def load_brain():
    model = VQVAE()
    if os.path.exists("ns_arc_vision.pt"):
        try:
            state = torch.load("ns_arc_vision.pt", map_location="cpu")
            model.load_state_dict(state)
            print("Loaded Neural Vision Brain.")
        except Exception as e:
            print(f"Warning: Failed to load brain: {e}")
    else:
        print("Warning: No brain found (ns_arc_vision.pt). Using random initialization.")
    return model

def compress_image(image_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_brain().to(device)
    model.eval()

    # Load & Prep Image
    # Resize to multiple of 32 for VQVAE
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    
    # Store original dimensions (Basic header)
    dims = np.array([h, w], dtype=np.uint16)
    
    transform = transforms.Compose([
        transforms.Resize((h//4*4, w//4*4)), # Ensure even dims for simple VQVAE
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    x = transform(img).unsqueeze(0).to(device)

    # Encode
    with torch.no_grad():
        indices = model.encode(x)
    
    # Save as compressed Tokens (uint16)
    # File Structure: [Header: H, W] + [Tokens]
    indices_np = indices.cpu().numpy().astype(np.uint16)
    
    np.savez_compressed(output_path, dims=dims, tokens=indices_np) # Use zip storage for numpy
    
    # Calculate Ratio for Display
    original_size = os.path.getsize(image_path)
    compressed_size = os.path.getsize(output_path + ".npz") 
    
    # Rename to remove .npz extension added by savez
    final_path = output_path
    if os.path.exists(output_path + ".npz"):
        try:
            if os.path.exists(final_path): os.remove(final_path)
            os.rename(output_path + ".npz", final_path)
            compressed_size = os.path.getsize(final_path)
        except:
             pass

    print(f"COMPRESSION_RESULT:{compressed_size}")

def decompress_image(input_path, output_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_brain().to(device)
    model.eval()
    
    # Load .nsarc (.npz structure)
    try:
        # Check signature first or just try load
        # np.load will fail if it's not a zip/pickle
        data = np.load(input_path, allow_pickle=True)
        dims = data['dims']
        tokens = data['tokens']
        h, w = dims[0], dims[1]
    except Exception as e:
        print(f"Error: Could not load NS-ARC file. It might be a 'Stored' file (Original Format) instead of Neural Compressed.")
        print(f"Details: {e}")
        return

    tokens_tensor = torch.from_numpy(tokens).long().to(device)
    
    # Calculate Latent Shape: (Batch, H/4, W/4, EmbDim) implicitly handling EmbDim in Codebook
    # Indices shape from tokenizer was (Batch, H/4, W/4) flattened or 2D? 
    # encode() returned min_encoding_indices from flattened d.
    # d shape: (Batch*H*W, EmbDim) -> argmin -> (Batch*H*W)
    # We need to reshape back to (Batch, H/4, W/4)
    
    # Assuming standard downsample factor of 4 (2 layers of stride 2)
    latent_h = (h // 4 // 4) * 4 # Wait, logic in compress was h//4*4 resize?
    # compress logic: transforms.Resize((h//4*4, ...))
    # latent dim = resize_dim / 4
    
    # This shape logic is tricky without explicit storage.
    # For prototype, we infer from token count assuming square or Aspect Ratio?
    # Or strict math: Tokens = (H_resize/4) * (W_resize/4)
    # We stored dims.
    h_resize = int(h // 4 * 4)
    w_resize = int(w // 4 * 4)
    
    latent_shape = (1, h_resize // 4, w_resize // 4, 64) # 64 is embedding dim
    # Indices passed to decode are just indices (not 64 dim)
    # But decode needs to view them.
    # embedding(indices) -> (..., 64)
    
    with torch.no_grad():
        # Flattened tokens -> Indices Tensor
        # We need to reshape indices before embedding? 
        # No, embedding takes (Batch, H, W) indices directly?
        # My VQVAE implementation:
        # decode uses: self.embedding(indices).view(shape)
        # So we pass indices as is, and provide the target (B, H, W, Emb) shape
        
        recon = model.decode(tokens_tensor, latent_shape)
        
    # Post-process
    # Denormalize
    recon = recon * 0.5 + 0.5
    recon = torch.clamp(recon, 0, 1)
    
    # Save
    save_image = transforms.ToPILImage()(recon.squeeze(0).cpu())
    save_image = save_image.resize((w, h)) # Resize back to original EXACT dims
    save_image.save(output_path)
    print(f"DECOMPRESSION_SUCCESS:{output_path}")

def optimize_image(input_path, output_path):
    """
    Lossless Optimization Probe.
    Tries to save with max compression to see if we can squeeze bytes.
    """
    try:
        img = Image.open(input_path)
        # Save Attempt
        # We append a temporary extension to avoid overwrite issues until confirmed? 
        # But here specific output_path is provided.
        img.save(output_path, optimize=True, quality=95) # For JPG/PNG if applicable
        
        # Check size
        size = os.path.getsize(output_path)
        print(f"OPTIMIZATION_RESULT:{size}")
    except Exception as e:
        print(f"Optimization Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["encode", "decode", "optimize", "test"])
    parser.add_argument("--input", help="Input file path")
    parser.add_argument("--output", help="Output file path")
    args = parser.parse_args()

    if args.mode == "encode":
        if args.input and args.output:
            compress_image(args.input, args.output)
    elif args.mode == "decode":
        if args.input and args.output:
            decompress_image(args.input, args.output)
    elif args.mode == "optimize":
        if args.input and args.output:
            optimize_image(args.input, args.output)
    elif args.mode == "test":
        print("Neural Vision Engine Ready.")

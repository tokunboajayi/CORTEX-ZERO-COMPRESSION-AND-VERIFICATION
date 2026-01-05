import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import io

# ==========================================
# Module A: Spectrum Analyzer & Utils
# ==========================================

def get_mgrid(sidelen, dim=2):
    """Generates a flattened grid of (x,y) coordinates in [-1, 1]."""
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def array_to_tensor(array):
    """Converts a numpy array (H, W, C) to a tensor (N, C)."""
    tensor = torch.from_numpy(array).float() / 255.0
    return tensor.view(-1, 3)

def generate_synthetic_data(sidelen=256):
    """Generates a high-entropy 'Plasma' pattern to test the network."""
    x = np.linspace(-10, 10, sidelen)
    y = np.linspace(-10, 10, sidelen)
    X, Y = np.meshgrid(x, y)
    
    # Complex continuous function
    Z_r = np.sin(X) + np.cos(Y) + np.sin(X * 0.5) * np.cos(Y * 0.5)
    Z_g = np.sin(X + Y) + np.cos(X - Y)
    Z_b = np.sin(np.sqrt(X**2 + Y**2))
    
    # Normalize to 0-255
    def norm(z):
        return ((z - z.min()) / (z.max() - z.min()) * 255).astype(np.uint8)
        
    img = np.stack([norm(Z_r), norm(Z_g), norm(Z_b)], axis=2)
    return img

# ==========================================
# Module B: Resonance Engine (SIREN)
# ==========================================

class SineLayer(nn.Module):
    """
    See Sitzmann et al., 'Implicit Neural Representations with Periodic Activation Functions'
    """
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class SirenNet(nn.Module):
    def __init__(self, in_features=2, hidden_features=256, hidden_layers=3, out_features=3):
        super().__init__()
        self.net = []
        
        # Module A: Fourier Feature Mapping (Implicit in First Layer of SIREN with high omega_0)
        # We start with a high-frequency SineLayer
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=30))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=30))

        # Final layer is linear to output RGB
        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / 30, 
                                          np.sqrt(6 / hidden_features) / 30)
        self.net.append(final_linear)
        
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        return self.net(coords)

# ==========================================
# Core: The Compressor (Training Loop)
# ==========================================

def train_ert(target_img_array, steps=1000, hidden_features=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Stats: Device={device}, Target Shape={target_img_array.shape}")

    # Data Prep
    sidelen = target_img_array.shape[0]
    coords = get_mgrid(sidelen).to(device)
    pixels = array_to_tensor(target_img_array).to(device)
    
    # Model Init
    model = SirenNet(hidden_features=hidden_features).to(device)
    optimiser = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training
    model.train()
    print(">>> ERT Core Online... Ingesting Data Stream.")
    
    losses = []
    
    for step in range(steps):
        model_output = model(coords)
        loss = ((model_output - pixels)**2).mean()
        
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        
        losses.append(loss.item())
        
        if step % 100 == 0:
            psnr = -10 * np.log10(loss.item())
            print(f"Step {step}: Loss {loss.item():.6f} | PSNR {psnr:.2f}dB")

    return model, losses

# ==========================================
# Utils: Visualization & Analysis
# ==========================================

def reconstruct(model, sidelen, device='cpu'):
    model.eval()
    with torch.no_grad():
        coords = get_mgrid(sidelen).to(device)
        output = model(coords)
        output = output.cpu().view(sidelen, sidelen, 3).numpy()
    
    # Clamp and Denormalize
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)
    return output

if __name__ == "__main__":
    print("--- Entropy Resonance Transducer (Proto-1) ---")
    
    # 1. Generate Target
    sidelen = 256
    target_img = generate_synthetic_data(sidelen)
    
    # 2. Compress (Train)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, history = train_ert(target_img, steps=500, hidden_features=128)
    
    # 3. Crystallize (Quantize to Float16)
    model.float() # Ensure float32 before verify, or we can use half()
    # model.half() # NOTE: CPU inference for half is slow/unsupported in some torch versions, keeping float32 for compat
    
    # 4. Reconstruct
    rec_img = reconstruct(model, sidelen, device)
    
    # 5. Save & Compare
    Image.fromarray(target_img).save("ert_target.png")
    Image.fromarray(rec_img).save("ert_reconstructed.png")
    torch.save(model.state_dict(), "ert_formula.pth")
    
    # Calc Stats
    raw_size = sidelen * sidelen * 3
    formula_size = os.path.getsize("ert_formula.pth")
    ratio = raw_size / formula_size
    
    print(f"\n--- Analysis ---")
    print(f"Raw Bitmap Size: {raw_size/1024:.2f} KB")
    print(f"Neural Formula Size: {formula_size/1024:.2f} KB")
    print(f"Compression Ratio: {ratio:.2f}x")
    print(f"Reconstruction saved to 'ert_reconstructed.png'")
    
    # Plot Loss Iff interactive
    # plt.plot(history)
    # plt.show()

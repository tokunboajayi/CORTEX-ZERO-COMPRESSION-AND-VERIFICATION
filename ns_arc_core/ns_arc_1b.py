import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

class RoPE(nn.Module):
    """
    Rotary Positional Embeddings.
    Allows the model to generalize to sequence lengths seen during training.
    """
    def __init__(self, dim, max_seq_len=8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq) 
        # (max_seq, dim/2)
        
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())
        
    def forward(self, q, k, seq_len):
        # q, k: (B, T, Heads, Dim)
        cos = self.cos_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)
        sin = self.sin_cached[:seq_len, :].unsqueeze(0).unsqueeze(2)
        
        # Apply RoPE
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit. 
    State-of-the-art activation for LLMs (used in LLaMA, PaLM).
    """
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class DemonAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x, rope):
        B, T, C = x.shape
        
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim)
        
        # Apply RoPE
        q, k = rope(q, k, T)
        
        # S-D Attention (In prod: Use F.scaled_dot_product_attention for FlashAttn)
        att = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        
        y = att.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y)

class DemonBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = DemonAttention(d_model, n_heads)
        self.norm2 = RMSNorm(d_model)
        # SwiGLU MLP
        hidden = int(d_model * mlp_ratio * 2 / 3) # Standard LLaMA scaling
        self.mlp = SwiGLU(d_model, hidden)
        
    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        x = x + self.mlp(self.norm2(x))
        return x

class DemonTransformer(nn.Module):
    """
    NS-ARC 'Demon Brain' Architecture.
    Target: 1.2 Billion Parameters.
    """
    def __init__(self, vocab_size=256, d_model=2048, n_layers=24, n_heads=16, max_seq_len=8192):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.rope = RoPE(d_model // n_heads, max_seq_len)
        
        self.layers = nn.ModuleList([
            DemonBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        
        self.norm_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        h = self.embedding(x)
        
        for layer in self.layers:
            h = layer(h, self.rope)
            
        h = self.norm_f(h)
        return self.head(h)

    def print_params(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"Model Parameters: {total:,} ({total/1e9:.2f} B)")

if __name__ == "__main__":
    # verification
    print("Initializing Demon Brain (Small Config for Verification)...")
    # Using small config so my laptop doesn't explode, but architecture validates
    model = DemonTransformer(d_model=512, n_layers=4, n_heads=8)
    model.print_params()
    
    x = torch.randint(0, 256, (1, 64))
    y = model(x)
    print(f"Output Shape: {y.shape}")

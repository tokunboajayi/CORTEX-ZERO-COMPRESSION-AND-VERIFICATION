import torch
import torch.nn as nn

class ContextMixer(nn.Module):
    """
    NS-ARC Module III: Adaptive Context Mixer.
    Blends Neural Logits with Baseline Logits (LZ77/Struct) via Gating.
    """
    def __init__(self, d_model=256):
        super().__init__()
        # Input: Concat of [Neural_State, Baseline_State]
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # We assume we get features from the models to decide trust
        # OR we just mix the logits directly with a learned scalar?
        # Better: Mix based on recent context embedding.

    def forward(self, logits_neural, logits_baseline, context_embedding):
        """
        logits_neural:   (B, T, Vocab)
        logits_baseline: (B, T, Vocab)
        context_embedding: (B, T, d_model) - State of the system
        """
        # Calculate Trust Gate (0 = Trust Baseline, 1 = Trust Neural)
        # We project the context to a scalar weight
        gate = self.gate_net(torch.cat([context_embedding, context_embedding], dim=-1)) # (B, T, 1)
        
        # Weighted Mix
        mixed_logits = gate * logits_neural + (1 - gate) * logits_baseline
        
        return mixed_logits, gate

class DummyLZ77Predictor(nn.Module):
    """
    Simulates a baseline predictor (LZ77) that outputs high confidence
    when it finds a match, and uniform uniform distribution otherwise.
    """
    def __init__(self, vocab_size=256):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        # Fake logic: just predict the previous byte (repetition)
        # In reality, this wraps zstd's match finder
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab_size).to(input_ids.device)
        
        # Simple: Predict repeat of last token
        # logits.scatter_(...) 
        return logits

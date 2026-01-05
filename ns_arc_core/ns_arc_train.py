import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import os

from ns_arc_neural import SparseMoETransformer
from ns_arc_tokenizer import ByteTokenizer

class TextDataset(Dataset):
    """
    Simple dataset taht chucks a file into blocks of context_length.
    """
    def __init__(self, file_path, context_len=64):
        self.tokenizer = ByteTokenizer()
        with open(file_path, 'rb') as f:
            self.data = f.read()
            
        self.tokens = self.tokenizer.encode(self.data)
        self.context_len = context_len
        
    def __len__(self):
        # We need input + target (shifted by 1)
        # So len is data_len - context_len
        return max(0, len(self.tokens) - self.context_len - 1)
        
    def __getitem__(self, idx):
        # Input: x[0:T]
        # Target: x[1:T+1]
        chunk = self.tokens[idx : idx + self.context_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

class ArcTrainer:
    def __init__(self, model, learning_rate=3e-4):
        self.model = model
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        # Ignore index not needed for pure bytes, but good practice
        self.criterion = nn.CrossEntropyLoss() 
        
    def train_epoch(self, dataloader, epoch_idx):
        self.model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch_idx, (x, y) in enumerate(dataloader):
            # Forward
            # x shape: (B, T)
            # y shape: (B, T)
            
            self.optimizer.zero_grad()
            logits = self.model(x) # (B, T, Vocab)
            
            # Reshape for Loss: (B*T, Vocab) vs (B*T)
            B, T, V = logits.shape
            loss = self.criterion(logits.view(B*T, V), y.view(B*T))
            
            # Backward
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Loss {loss.item():.4f} (Bits/Byte: {loss.item()/0.69314:.2f})")
                
        avg_loss = total_loss / len(dataloader)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch_idx} Complete. Avg Loss: {avg_loss:.4f}. Time: {elapsed:.2f}s")
        return avg_loss

def run_training_experiment():
    print("--- NS-ARC Training Experiment ---")
    
    # 1. Setup Data (Use this script itself as training data!)
    # Ideally use a larger file, but 'self-training' is valid for debug
    target_file = __file__ 
    print(f"Training on: {target_file}")
    
    # Context length for prototype
    dataset = TextDataset(target_file, context_len=32)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 2. Setup Model
    # Small capacity for quick overfit check
    model = SparseMoETransformer(
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_experts=4,
        k=2,
        window_size=64
    )
    
    trainer = ArcTrainer(model)
    
    # 3. Train Loop
    epochs = 5
    for i in range(epochs):
        trainer.train_epoch(dataloader, i+1)
        
    # 4. Save
    torch.save(model.state_dict(), "ns_arc_model.pt")
    print("Model saved to ns_arc_model.pt")

if __name__ == "__main__":
    run_training_experiment()

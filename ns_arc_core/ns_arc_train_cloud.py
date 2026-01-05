import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from datasets import load_dataset
from ns_arc_1b import DemonTransformer
from ns_arc_tokenizer import ByteTokenizer

# Configuration
BATCH_SIZE = 16 # Adjust per GPU VRAM
GRAD_ACCUM = 4
LEARNING_RATE = 3e-4
MAX_TOKENS = 1_000_000_000 # 1B Token Budget for initial phase

class StreamingRedPajamaState(IterableDataset):
    def __init__(self, tokenizer, seq_len=8192):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        # Load RedPajama sample or full
        self.ds = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train", streaming=True)
        
    def __iter__(self):
        buffer = []
        for sample in self.ds:
            text = sample['text']
            tokens = self.tokenizer.encode(text).tolist()
            buffer.extend(tokens)
            
            while len(buffer) >= self.seq_len + 1:
                yield torch.tensor(buffer[:self.seq_len]), torch.tensor(buffer[1:self.seq_len+1])
                buffer = buffer[self.seq_len:]

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size
    else:
        return 0, 0, 1 # Single GPU/CPU Debug

def train():
    monitor_rank, local_rank, world_size = setup_distributed()
    
    if monitor_rank == 0:
        print("--- NS-ARC Cloud Trainer (RedPajama) ---")
        print(f"World Size: {world_size}")

    # 1. Model (1.2B Param Config)
    # Using smaller config for prototype; User should uncomment scaling params
    model = DemonTransformer(
        vocab_size=256,
        d_model=2048, # Scale to 2048 for 1B
        n_layers=24,
        n_heads=16,
        max_seq_len=8192
    ).cuda()
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
        
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.95))
    
    # 2. Data
    tokenizer = ByteTokenizer()
    dataset = StreamingRedPajamaState(tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    
    # 3. Loop
    step = 0
    model.train()
    
    for x, y in dataloader:
        x, y = x.cuda(), y.cuda() # (B, T)
        
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
        
        loss = loss / GRAD_ACCUM
        loss.backward()
        
        if (step + 1) % GRAD_ACCUM == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            if monitor_rank == 0 and step % 100 == 0:
                print(f"Step {step}: Loss {loss.item() * GRAD_ACCUM:.4f} (Bits: {loss.item() * GRAD_ACCUM / 0.693:.2f})")
                
                if step % 1000 == 0:
                     # Checkpoint
                     torch.save(model.module.state_dict(), f"ns_arc_1b_step{step}.pt")
                     
        step += 1
        if step >= MAX_TOKENS // (BATCH_SIZE * 8192):
            break

if __name__ == "__main__":
    train()

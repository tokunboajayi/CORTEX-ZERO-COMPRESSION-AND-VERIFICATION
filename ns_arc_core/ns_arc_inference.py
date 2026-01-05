import torch
import torch.nn.functional as F
from ns_arc_1b import DemonTransformer
from ns_arc_tokenizer import ByteTokenizer
import os

# CONFIGURATION (Must match ns_arc_colab.py exactly!)
CHECKPOINT_PATH = "ns_arc_1b_colab.pt"
D_MODEL = 1536
N_LAYERS = 16
N_HEADS = 16
MAX_SEQ_LEN = 2048
VOCAB_SIZE = 256

def generate():
    print("--- NS-ARC: Demon Brain Inference ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on: {device}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Could not find '{CHECKPOINT_PATH}'.")
        print("Please move the downloaded file into this folder!")
        return

    # 1. Load the Brain Structure
    print("Loading Model Architecture...")
    model = DemonTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_seq_len=MAX_SEQ_LEN
    ).to(device)

    # 2. Load the Learned Weights
    print(f"Loading Weights from {CHECKPOINT_PATH}...")
    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print("Brain Loaded Successfully! 🧠")
    except Exception as e:
        print(f"Weight Mismatch: {e}")
        return

    model.eval()
    tokenizer = ByteTokenizer()

    # 3. Interactive Loop
    while True:
        prompt_text = input("\nENTER PROMPT (or 'q' to quit): ")
        if prompt_text.lower() == 'q':
            break
            
        # Context
        context = tokenizer.encode(prompt_text).unsqueeze(0).to(device) # (1, T)
        
        # Generator
        print("Demon Thinking...", end="", flush=True)
        generated = context
        
        with torch.no_grad():
            for _ in range(50): # Generate 50 bytes
                logits = model(generated)
                # Take the logits for the last token
                next_token_logits = logits[:, -1, :]
                
                # Greedy Decode (Pick the most likely byte)
                # For more creativity, use torch.multinomial
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1).unsqueeze(0)
                
                generated = torch.cat((generated, next_token), dim=1)
                print(".", end="", flush=True)

        # Decode
        output_bytes = generated[0].tolist()
        try:
            output_text = bytes(output_bytes).decode('utf-8', errors='replace')
            print(f"\n\nRESULT:\n{output_text}")
        except:
            print(f"\n\nRESULT (Hex):\n{bytes(output_bytes).hex()}")

if __name__ == "__main__":
    generate()

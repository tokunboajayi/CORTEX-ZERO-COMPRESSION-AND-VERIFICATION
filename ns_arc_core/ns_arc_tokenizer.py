import torch

class ByteTokenizer:
    """
    NS-ARC Module VII: Byte-Level Tokenizer.
    
    Why Bytes?
    1. Robustness: No Out-Of-Vocabulary (OOV) tokens ever.
    2. Binary-Safe: Can compress any file stream.
    3. Simplicity: Vocab size is fixed at 256.
    """
    def __init__(self):
        self.vocab_size = 256
        
    def encode(self, text_or_bytes):
        """
        Converts input (str/bytes) to List[int] tensor.
        """
        if isinstance(text_or_bytes, str):
            data = text_or_bytes.encode('utf-8')
        else:
            data = text_or_bytes
            
        # Convert to tensor of longs
        # We assume input is reasonably sized (chunking happens elsewhere)
        return torch.tensor([b for b in data], dtype=torch.long)
        
    def decode(self, tokens):
        """
        Converts tensor/list of tokens back to bytes.
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
            
        return bytes(tokens)

if __name__ == "__main__":
    # Test
    tokenizer = ByteTokenizer()
    text = "Hello NS-ARC!"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    
    print(f"Original: {text}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded.decode('utf-8')}")
    assert text == decoded.decode('utf-8')
    print("Tokenizer Verified.")

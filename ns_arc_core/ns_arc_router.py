import torch
import json
from ert_splitter import JSONSplitter
from ert_cdc import ResonanceIndex
from ns_arc_neural import SparseMoETransformer
from ns_arc_visual import SemanticResidualCompressor
from ns_arc_genomics import GraSSCompressor

class SemanticRouter:
    """
    NS-ARC Module I: The Semantic Router.
    Dispatches input streams to the optimal compression 'Expert'.
    """
    def __init__(self):
        # Expert 1: Algorithm for Logs
        self.log_expert = JSONSplitter()
        
        # Expert 2: Algorithm for Backups/Binary
        self.cdc_expert = ResonanceIndex()
        
        # Expert 3: Neural for Text/Code
        # Initialize small prototype model
        self.neural_expert = SparseMoETransformer(
            vocab_size=256, d_model=64, n_layers=2, n_experts=4
        )
        self.neural_expert.eval()
        
        # Expert 4: Visual Cortex for Images
        self.visual_expert = SemanticResidualCompressor()
        
        # Expert 5: Genomics Engine
        self.genomics_expert = GraSSCompressor()

    def classify_stream(self, data_chunk):
        """
        Heuristic classification of the stream.
        In Prod: Use MobileNetV3 on first 64KB.
        """
        # 1. Check for Image Magic Bytes (PNG/JPG)
        head = data_chunk[:8]
        if head.startswith(b'\x89PNG\r\n\x1a\n') or head.startswith(b'\xFF\xD8\xFF'):
            return "IMAGE"
            
        # 2. Check for Genomics (FASTQ/FASTA or raw DNA)
        try:
            head_str = data_chunk[:1024].decode('utf-8')
            # FASTQ/FASTA markers + ACTGN density
            if head_str.startswith('@') or head_str.startswith('>'):
                # Strong signal, check content
                # Just assume yes for prototype
                return "GENOMICS"
                
            # Check ACTGN density logic if no header (Raw reads)
            # DNA is ACGTN + newline
            dna_chars = set("ACGTNacgtn\n\r")
            sample = head_str[:256] 
            matches = sum(1 for c in sample if c in dna_chars)
            if len(sample) > 0 and (matches / len(sample)) > 0.95:
                 return "GENOMICS"
                 
        except UnicodeDecodeError:
            pass
            
        # 3. Check for JSON/Logs indicators
        try:
            head_str = data_chunk[:1024].decode('utf-8')
            if head_str.strip().startswith('{') and ('"timestamp"' in head_str or '"ts"' in head_str):
                return "LOGS"
        except UnicodeDecodeError:
            pass # Binary
            
        # 4. Check for Text/Code identifiers (Robust Heuristic)
        try:
            head_str = data_chunk[:2048].decode('utf-8')
            
            # Simple keyword check is brittle. 
            # Better check: Ratio of printable characters.
            printable = sum(1 for c in head_str if c.isprintable() or c in ('\n', '\r', '\t'))
            ratio = printable / len(head_str)
            
            if ratio > 0.75: # 75% printable -> Likely Text/Code
                return "TEXT_CODE"
                
        except UnicodeDecodeError:
            pass # Contains non-utf8 bytes -> Likely Binary
            
        # 5. Default to Binary/Backup
        return "BINARY"

    def process_stream(self, data):
        """
        Routes and processes the data using the selected expert.
        Returns: (expert_name, compressed_size_estimate, ratio)
        """
        stream_type = self.classify_stream(data)
        
        if stream_type == "LOGS":
            # --- Route to ERT Splitter ---
            print(">>> Router: Detected LOGS. Dispatching to Semantic Splitter.")
            # Process line by line
            lines = data.decode('utf-8').split('\n')
            for line in lines:
                if line: self.log_expert.ingest(line)
            
            comp_size = self.log_expert.compress_streams()
            raw_size = len(data)
            return "LOGS (Splitter)", comp_size, raw_size / (comp_size + 1)

        elif stream_type == "IMAGE":
            # --- Route to Visual Cortex ---
            print(">>> Router: Detected IMAGE. Dispatching to Visual Cortex (Residuals).")
            # For prototype, we might need to strip magic bytes or just pass raw
            # The visual expert expects raw pixel data in the Mock, but let's assume it handles it.
            # We'll pass raw data. The Mock in ns_arc_visual handles raw bytes as pixels.
            comp_size = self.visual_expert.compress_image_stream(data)
            raw_size = len(data)
            return "IMAGE (Visual)", comp_size, raw_size / (comp_size + 1)
            
        elif stream_type == "GENOMICS":
             # --- Route to GraSS ---
            print(">>> Router: Detected GENOMICS. Dispatching to GraSS Engine.")
            comp_size = self.genomics_expert.compress_stream(data)
            raw_size = len(data)
            return "GENOMICS (GraSS)", comp_size, raw_size / (comp_size + 1)

        elif stream_type == "BINARY":
            # --- Route to CDC Resonance ---
            print(">>> Router: Detected BINARY. Dispatching to Resonance Index (CDC).")
            comp_size = self.cdc_expert.add_stream(data)
            raw_size = len(data)
            return "BINARY (CDC)", comp_size, raw_size / (comp_size + 1)
            
        elif stream_type == "TEXT_CODE":
            # --- Route to Neural MoE ---
            print(">>> Router: Detected TEXT/CODE. Dispatching to Neural MoE.")
            # Simulating compression ratio calculation for Neural
            # In reality, we'd run the model and sum -log2(probs)
            
            # Convert to tensor
            # proto limit: just take first 1024 bytes
            input_tensor = torch.from_numpy(
                torch.ByteTensor(list(data[:1024])).numpy()
            ).long().unsqueeze(0) # (1, T)
            
            with torch.no_grad():
                logits = self.neural_expert(input_tensor) # (1, T, 256)
                
            # Calculate Cross Entropy (Bits per byte estimate)
            # Fake target: offset by 1
            # For prototype, just returning a placeholder "Neural Ratio"
            # Real ratio would be ~4.0x for code
            
            return "TEXT (Neural MoE)", len(data) // 4, 4.0 

        return "UNKNOWN", len(data), 1.0

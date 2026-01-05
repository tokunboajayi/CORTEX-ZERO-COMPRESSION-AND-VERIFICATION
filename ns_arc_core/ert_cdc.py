import hashlib

class FastCDC:
    """
    Simulated FastCDC (Content Defined Chunking).
    Uses a rolling hash to find cut points based on a mask.
    """
    def __init__(self, min_size=2048, avg_size=4096, max_size=8192):
        self.min_size = min_size
        self.avg_size = avg_size
        self.max_size = max_size
        
        # Mask calculation for 'avg_size' (must be power of 2 roughly)
        # If we want a cut every N bytes, we generally look for N lower bits == 0
        # A simple approximation for the prototype
        self.mask_bits = avg_size.bit_length() - 1
        self.mask = (1 << self.mask_bits) - 1

    def _rolling_hash_cut(self, buffer):
        """
        Generator that yields (chunk_bytes, chunk_hash).
        Implementing a very basic windowed check for prototype speed.
        True Rabin/Gear hashing would be optimization.
        """
        start = 0
        pos = 0
        buf_len = len(buffer)
        
        while start < buf_len:
            curr_len = 0
            # Minimum chunk skip
            pos = start + self.min_size
            if pos >= buf_len:
                # Remainder
                chunk = buffer[start:]
                yield chunk, hashlib.sha256(chunk).hexdigest()
                break
                
            # Rolling search
            # We search from min_size to max_size
            cut_found = False
            for i in range(pos, min(start + self.max_size, buf_len)):
                # Fake rolling hash: just check the byte value or small window
                # For a python prototype, checking every byte is SLOW.
                # We'll use a stride or simple check.
                
                # Heuristic: Check if the last 2 bytes hash to 0 mod mask
                # This is NOT a real rolling hash but statistically similar for cuts
                # (Simulating 'Gear' hash weak check)
                val = (buffer[i] << 8) | buffer[i-1]
                if (val & self.mask) == 0:
                     # Cut here
                     cut_pos = i + 1
                     chunk = buffer[start:cut_pos]
                     yield chunk, hashlib.sha256(chunk).hexdigest()
                     start = cut_pos
                     cut_found = True
                     break
            
            if not cut_found:
                # Force cut at max_size
                cut_pos = min(start + self.max_size, buf_len)
                chunk = buffer[start:cut_pos]
                yield chunk, hashlib.sha256(chunk).hexdigest()
                start = cut_pos

class ResonanceIndex:
    """
    Manages the 'Resonance' (Dedup) Index.
    Stores chunk hashes and detects duplicates.
    """
    def __init__(self):
        self.chunks = {} # checksum -> stored_size
        self.total_input_bytes = 0
        self.unique_bytes = 0
        
    def add_stream(self, data):
        """
        Ingest a binary stream, chunk it, and update index.
        Returns (compressed_size, dedup_ratio)
        """
        self.total_input_bytes += len(data)
        
        cdc = FastCDC()
        stream_stored_size = 0
        
        chunk_refs = []
        
        for chunk, checksum in cdc._rolling_hash_cut(data):
            if checksum not in self.chunks:
                # New chunk -> Store it (simulate storage overhead)
                self.chunks[checksum] = len(chunk)
                self.unique_bytes += len(chunk)
                stream_stored_size += len(chunk)
            else:
                # Duplicate -> Reference it (Reference size ~ 32 bytes)
                stream_stored_size += 32 
                
            chunk_refs.append(checksum)
            
        return stream_stored_size
        
    def get_stats(self):
        return {
            "total_input": self.total_input_bytes,
            "unique_stored": self.unique_bytes,
            "global_ratio": self.total_input_bytes / (self.unique_bytes + 1) # Avoid div0
        }

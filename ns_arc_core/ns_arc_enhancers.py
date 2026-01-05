#!/usr/bin/env python3
"""
NS-ARC Compression Enhancers
Additional algorithms to boost compression performance across all data types.

These modules provide:
1. Adaptive Context Modeling - learns patterns from data
2. Delta Encoding Engine - efficient for time-series data
3. Dictionary Learning - learns domain-specific patterns
4. Run-Length + BWT hybrid - for repetitive data
5. Predictive Pre-processing - reduces entropy before compression
"""

import zlib
import hashlib
import struct
from collections import defaultdict, Counter
import math


# ============================================================================
# 1. ADAPTIVE CONTEXT MODELER
# ============================================================================

class AdaptiveContextModeler:
    """
    Learns byte-level context patterns for improved prediction.
    Reduces entropy by predicting the next byte based on context.
    """
    
    def __init__(self, order=4):
        self.order = order
        self.contexts = defaultdict(lambda: defaultdict(int))
        self.total_counts = defaultdict(int)
    
    def train(self, data: bytes):
        """Train on data to learn context patterns."""
        for i in range(self.order, len(data)):
            context = data[i-self.order:i]
            symbol = data[i]
            self.contexts[context][symbol] += 1
            self.total_counts[context] += 1
    
    def predict(self, context: bytes) -> dict:
        """Return probability distribution for next byte."""
        if context not in self.contexts:
            return {i: 1/256 for i in range(256)}  # Uniform
        
        total = self.total_counts[context]
        return {sym: count/total for sym, count in self.contexts[context].items()}
    
    def compute_entropy_reduction(self, data: bytes) -> float:
        """Compute theoretical entropy reduction from context modeling."""
        if len(data) < self.order + 1:
            return 0.0
        
        bits_uniform = len(data) * 8  # 8 bits per byte without model
        bits_model = 0.0
        
        for i in range(self.order, len(data)):
            context = data[i-self.order:i]
            symbol = data[i]
            probs = self.predict(context)
            prob = probs.get(symbol, 1/256)
            bits_model += -math.log2(prob) if prob > 0 else 8
        
        return (bits_uniform - bits_model) / bits_uniform * 100


# ============================================================================
# 2. DELTA ENCODING ENGINE
# ============================================================================

class DeltaEncoder:
    """
    Efficient delta encoding for time-series and sequential data.
    Stores differences instead of absolute values.
    """
    
    @staticmethod
    def encode_bytes(data: bytes) -> bytes:
        """Encode bytes as deltas."""
        if len(data) == 0:
            return b''
        
        result = bytearray([data[0]])  # First byte unchanged
        for i in range(1, len(data)):
            delta = (data[i] - data[i-1]) % 256
            result.append(delta)
        return bytes(result)
    
    @staticmethod
    def decode_bytes(data: bytes) -> bytes:
        """Decode delta-encoded bytes."""
        if len(data) == 0:
            return b''
        
        result = bytearray([data[0]])
        for i in range(1, len(data)):
            value = (result[i-1] + data[i]) % 256
            result.append(value)
        return bytes(result)
    
    @staticmethod
    def encode_integers(values: list, width=4) -> bytes:
        """Encode a list of integers as deltas (for timestamps, counters)."""
        if len(values) == 0:
            return b''
        
        # Store first value and then deltas
        result = struct.pack('>I', values[0])
        for i in range(1, len(values)):
            delta = values[i] - values[i-1]
            # Variable-length encoding for deltas
            if -127 <= delta <= 127:
                result += struct.pack('b', delta)
            else:
                result += struct.pack('b', -128) + struct.pack('>i', delta)
        return result
    
    @staticmethod
    def decode_integers(data: bytes) -> list:
        """Decode delta-encoded integers."""
        if len(data) < 4:
            return []
        
        result = [struct.unpack('>I', data[:4])[0]]
        pos = 4
        
        while pos < len(data):
            delta = struct.unpack('b', data[pos:pos+1])[0]
            pos += 1
            if delta == -128:
                delta = struct.unpack('>i', data[pos:pos+4])[0]
                pos += 4
            result.append(result[-1] + delta)
        
        return result


# ============================================================================
# 3. DICTIONARY LEARNING
# ============================================================================

class DictionaryLearner:
    """
    Learns common patterns (phrases) from data for dictionary-based compression.
    Uses LZ-style phrase extraction with persistence.
    """
    
    def __init__(self, min_phrase_len=4, max_phrases=4096):
        self.min_phrase_len = min_phrase_len
        self.max_phrases = max_phrases
        self.dictionary = {}
        self.phrase_to_id = {}
        self.id_to_phrase = {}
    
    def learn(self, data: bytes):
        """Extract common phrases from data."""
        phrase_counts = Counter()
        
        # Extract all possible phrases
        for length in range(self.min_phrase_len, min(64, len(data))):
            for i in range(len(data) - length + 1):
                phrase = data[i:i+length]
                # Only count if occurs multiple times
                if data.count(phrase) > 1:
                    phrase_counts[phrase] += data.count(phrase)
        
        # Select top phrases by savings (occurrences * length)
        savings = [(phrase, count * len(phrase)) for phrase, count in phrase_counts.items()]
        savings.sort(key=lambda x: -x[1])
        
        # Build dictionary
        for idx, (phrase, _) in enumerate(savings[:self.max_phrases]):
            self.phrase_to_id[phrase] = idx
            self.id_to_phrase[idx] = phrase
    
    def encode(self, data: bytes) -> bytes:
        """Encode data using learned dictionary."""
        if not self.phrase_to_id:
            return data
        
        result = bytearray()
        i = 0
        
        while i < len(data):
            best_match = None
            best_len = 0
            
            # Find longest matching phrase
            for phrase, idx in self.phrase_to_id.items():
                plen = len(phrase)
                if plen > best_len and data[i:i+plen] == phrase:
                    best_match = idx
                    best_len = plen
            
            if best_match is not None and best_len >= self.min_phrase_len:
                # Emit dictionary reference (2 bytes: marker + id)
                result.append(0xFF)  # Escape marker
                result.extend(struct.pack('>H', best_match))
                i += best_len
            else:
                # Emit literal byte
                if data[i] == 0xFF:
                    result.extend([0xFF, 0xFF, 0xFF])  # Escape literal 0xFF
                else:
                    result.append(data[i])
                i += 1
        
        return bytes(result)
    
    def get_dictionary_size(self) -> int:
        """Return approximate serialized dictionary size."""
        return sum(len(p) + 2 for p in self.phrase_to_id.keys())


# ============================================================================
# 4. BWT + MTF + RLE HYBRID
# ============================================================================

class BWTCompressor:
    """
    Burrows-Wheeler Transform with Move-To-Front and Run-Length Encoding.
    Excellent for highly repetitive text data.
    """
    
    @staticmethod
    def bwt_transform(data: bytes, block_size=1024) -> tuple:
        """Apply BWT to data in blocks."""
        result = bytearray()
        indices = []
        
        for start in range(0, len(data), block_size):
            block = data[start:start+block_size]
            transformed, idx = BWTCompressor._bwt_block(block)
            result.extend(transformed)
            indices.append(idx)
        
        return bytes(result), indices
    
    @staticmethod
    def _bwt_block(block: bytes) -> tuple:
        """Apply BWT to a single block."""
        n = len(block)
        if n == 0:
            return b'', 0
        
        # Build rotation table (indices only for memory efficiency)
        rotations = sorted(range(n), key=lambda i: block[i:] + block[:i])
        
        # Find original index
        original_idx = rotations.index(0)
        
        # Build last column
        last_col = bytes([block[(i - 1) % n] for i in rotations])
        
        return last_col, original_idx
    
    @staticmethod
    def mtf_encode(data: bytes) -> bytes:
        """Move-To-Front encoding - reduces entropy for clustered data."""
        alphabet = list(range(256))
        result = bytearray()
        
        for byte in data:
            idx = alphabet.index(byte)
            result.append(idx)
            # Move to front
            alphabet.pop(idx)
            alphabet.insert(0, byte)
        
        return bytes(result)
    
    @staticmethod
    def rle_encode(data: bytes) -> bytes:
        """Run-Length Encoding for zeros (common after MTF)."""
        result = bytearray()
        i = 0
        
        while i < len(data):
            if data[i] == 0:
                # Count zeros
                count = 0
                while i < len(data) and data[i] == 0 and count < 255:
                    count += 1
                    i += 1
                result.extend([0x00, count])
            else:
                result.append(data[i])
                i += 1
        
        return bytes(result)
    
    @classmethod
    def compress(cls, data: bytes) -> bytes:
        """Full compression pipeline: BWT -> MTF -> RLE -> zlib."""
        if len(data) == 0:
            return b''
        
        # Apply transforms
        bwt_data, indices = cls.bwt_transform(data)
        mtf_data = cls.mtf_encode(bwt_data)
        rle_data = cls.rle_encode(mtf_data)
        
        # Final compression with zlib
        compressed = zlib.compress(rle_data, 9)
        
        # Store indices for reconstruction
        header = struct.pack('>I', len(indices))
        for idx in indices:
            header += struct.pack('>H', idx)
        
        return header + compressed


# ============================================================================
# 5. PREDICTIVE PRE-PROCESSOR
# ============================================================================

class PredictivePreprocessor:
    """
    Pre-processes data to reduce entropy before compression.
    Uses multiple prediction models and selects the best.
    """
    
    @staticmethod
    def preprocess_text(data: bytes) -> bytes:
        """Preprocess text-like data for better compression."""
        result = bytearray()
        
        # Convert to lowercase differences (if mostly text)
        text_chars = sum(1 for b in data if 32 <= b <= 126)
        if text_chars / max(1, len(data)) > 0.8:
            # Text mode: store case info separately
            case_bits = bytearray()
            text = bytearray()
            
            for byte in data:
                if 65 <= byte <= 90:  # Uppercase
                    case_bits.append(1)
                    text.append(byte + 32)  # To lowercase
                elif 97 <= byte <= 122:  # Lowercase
                    case_bits.append(0)
                    text.append(byte)
                else:
                    case_bits.append(2)  # Other
                    text.append(byte)
            
            # Pack case bits (4 per byte)
            packed_cases = bytearray()
            for i in range(0, len(case_bits), 4):
                val = 0
                for j in range(4):
                    if i + j < len(case_bits):
                        val |= (case_bits[i + j] & 0x03) << (j * 2)
                packed_cases.append(val)
            
            # Header: 'T' for text mode + lengths
            result = bytearray([ord('T')])
            result.extend(struct.pack('>I', len(text)))
            result.extend(struct.pack('>I', len(packed_cases)))
            result.extend(packed_cases)
            result.extend(text)
            
            return bytes(result)
        
        # Binary mode: just return with 'B' header
        return b'B' + data
    
    @staticmethod
    def preprocess_numeric(data: bytes) -> bytes:
        """Preprocess numeric data (integers, floats) for better compression."""
        # Try to detect and delta-encode numeric sequences
        if len(data) % 4 == 0:
            # Might be 32-bit integers
            try:
                values = [struct.unpack('>I', data[i:i+4])[0] for i in range(0, len(data), 4)]
                
                # Check if delta encoding helps
                deltas = [values[i] - values[i-1] for i in range(1, len(values))]
                
                # If deltas are small, use delta encoding
                if deltas and max(abs(d) for d in deltas) < 65536:
                    encoded = DeltaEncoder.encode_integers(values)
                    if len(encoded) < len(data) * 0.9:
                        return b'D' + encoded
            except:
                pass
        
        return b'R' + data  # Raw mode


# ============================================================================
# 6. COMPRESSION SELECTOR
# ============================================================================

class CompressionSelector:
    """
    Intelligently selects the best compression algorithm for given data.
    Runs quick heuristics to choose optimal method without full compression.
    """
    
    @staticmethod
    def analyze(data: bytes) -> dict:
        """Analyze data characteristics."""
        if len(data) == 0:
            return {'type': 'empty', 'entropy': 0}
        
        # Byte frequency analysis
        freqs = Counter(data)
        entropy = -sum((c/len(data)) * math.log2(c/len(data)) for c in freqs.values() if c > 0)
        
        # Detect data type
        text_chars = sum(1 for b in data if 32 <= b <= 126)
        text_ratio = text_chars / len(data)
        
        zero_ratio = data.count(0) / len(data)
        
        unique_bytes = len(freqs)
        
        # Check for repetition
        sample = data[:min(1000, len(data))]
        repetition_score = 1 - len(set(sample)) / len(sample)
        
        return {
            'entropy': entropy,
            'text_ratio': text_ratio,
            'zero_ratio': zero_ratio,
            'unique_bytes': unique_bytes,
            'repetition_score': repetition_score,
            'size': len(data)
        }
    
    @staticmethod
    def select_method(data: bytes) -> str:
        """Select best compression method based on analysis."""
        stats = CompressionSelector.analyze(data)
        
        if stats.get('type') == 'empty':
            return 'none'
        
        # High text content -> BWT works well
        if stats['text_ratio'] > 0.85 and stats['repetition_score'] > 0.3:
            return 'bwt_hybrid'
        
        # High repetition with binary data -> dictionary learning
        if stats['repetition_score'] > 0.5 and stats['text_ratio'] < 0.5:
            return 'dictionary'
        
        # Low entropy (already somewhat compressed or random)
        if stats['entropy'] > 7.5:
            return 'store'  # Don't waste time, barely compressible
        
        # Sequential numeric data
        if stats['zero_ratio'] < 0.1 and stats['unique_bytes'] > 200:
            return 'delta_zlib'
        
        # Default: standard zlib
        return 'zlib'


# ============================================================================
# 7. UNIFIED ENHANCER
# ============================================================================

class NSARCEnhancer:
    """
    Unified interface for all compression enhancements.
    Automatically selects and applies best compression strategy.
    """
    
    def __init__(self):
        self.context_modeler = AdaptiveContextModeler(order=4)
        self.dict_learner = DictionaryLearner()
        self.selector = CompressionSelector()
    
    def train(self, training_data: list):
        """Train enhancement models on sample data."""
        for data in training_data:
            self.context_modeler.train(data)
            self.dict_learner.learn(data)
    
    def compress(self, data: bytes) -> tuple:
        """
        Compress data using best available method.
        Returns (compressed_data, method_used, compression_ratio).
        """
        if len(data) == 0:
            return b'', 'none', 1.0
        
        method = self.selector.select_method(data)
        
        if method == 'bwt_hybrid':
            compressed = BWTCompressor.compress(data)
        elif method == 'dictionary':
            compressed = self.dict_learner.encode(data)
            compressed = zlib.compress(compressed, 9)
        elif method == 'delta_zlib':
            delta = DeltaEncoder.encode_bytes(data)
            compressed = zlib.compress(delta, 9)
        elif method == 'store':
            compressed = data
        else:
            compressed = zlib.compress(data, 9)
        
        # Add method header
        header = method.encode()[:8].ljust(8, b'\0')
        result = header + compressed
        
        ratio = len(data) / max(1, len(result))
        return result, method, ratio
    
    def get_stats(self) -> dict:
        """Return statistics about learned models."""
        return {
            'context_order': self.context_modeler.order,
            'contexts_learned': len(self.context_modeler.contexts),
            'dictionary_phrases': len(self.dict_learner.phrase_to_id),
            'dictionary_size_bytes': self.dict_learner.get_dictionary_size()
        }


# ============================================================================
# MAIN - Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NS-ARC Compression Enhancers - Demo")
    print("=" * 60)
    
    # Test data
    text_data = b"The quick brown fox jumps over the lazy dog. " * 100
    binary_data = bytes(range(256)) * 50
    log_data = b'{"ts":1234567890,"level":"INFO","msg":"Request handled"}\n' * 200
    
    enhancer = NSARCEnhancer()
    
    # Train on samples
    print("\n[1] Training on sample data...")
    enhancer.train([text_data[:1000], log_data[:1000]])
    print(f"    Stats: {enhancer.get_stats()}")
    
    # Test compression
    print("\n[2] Testing compression methods...")
    
    for name, data in [("Text", text_data), ("Binary", binary_data), ("Logs", log_data)]:
        compressed, method, ratio = enhancer.compress(data)
        zlib_size = len(zlib.compress(data, 9))
        
        print(f"\n    {name}:")
        print(f"      Original: {len(data):,} bytes")
        print(f"      Enhanced: {len(compressed):,} bytes (Method: {method})")
        print(f"      Zlib-9:   {zlib_size:,} bytes")
        print(f"      Ratio:    {ratio:.2f}x")
        print(f"      vs Zlib:  {(zlib_size - len(compressed))/zlib_size*100:+.1f}%")
    
    # Test individual components
    print("\n[3] Component Tests...")
    
    # Delta encoding
    nums = list(range(1000, 2000, 10))
    encoded = DeltaEncoder.encode_integers(nums)
    decoded = DeltaEncoder.decode_integers(encoded)
    print(f"    Delta Encoding: {len(nums)*4} -> {len(encoded)} bytes ({len(encoded)/len(nums)/4*100:.0f}%)")
    
    # Context modeling
    entropy_reduction = enhancer.context_modeler.compute_entropy_reduction(text_data[:1000])
    print(f"    Context Model:  {entropy_reduction:.1f}% entropy reduction")
    
    print("\n" + "=" * 60)
    print("Enhancement algorithms ready for NS-ARC integration!")
    print("=" * 60)

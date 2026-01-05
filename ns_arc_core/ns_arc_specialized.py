#!/usr/bin/env python3
"""
NS-ARC Specialized Compressors
Domain-specific compression algorithms for maximum efficiency.

Includes:
1. JSON/Log Compressor - optimized for structured logs
2. Time-Series Compressor - for sensor/metric data
3. Image Residual Compressor - for surveillance/video frames
4. DNA/Sequence Compressor - for genomic data
5. Code/Source Compressor - for programming languages
"""

import zlib
import struct
import json
import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Optional


# ============================================================================
# 1. JSON/LOG COMPRESSOR
# ============================================================================

class JSONLogCompressor:
    """
    Specialized compressor for JSON logs with high redundancy.
    Separates structure from data for better compression.
    """
    
    def __init__(self):
        self.templates = {}
        self.template_id = 0
    
    def extract_template(self, json_str: str) -> Tuple[str, List]:
        """Extract template and variable values from JSON."""
        try:
            obj = json.loads(json_str)
        except:
            return json_str, []
        
        values = []
        template = self._extract_recursive(obj, values)
        return json.dumps(template, separators=(',', ':')), values
    
    def _extract_recursive(self, obj, values) -> any:
        """Recursively extract variable values."""
        if isinstance(obj, dict):
            return {k: self._extract_recursive(v, values) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._extract_recursive(item, values) for item in obj]
        elif isinstance(obj, (int, float)):
            values.append(('N', obj))
            return '__N__'
        elif isinstance(obj, str):
            # Check if it's a constant-like string (short, alphanumeric)
            if len(obj) < 20 and obj.isalnum():
                return obj  # Keep as part of template
            else:
                values.append(('S', obj))
                return '__S__'
        else:
            return obj
    
    def compress_logs(self, lines: List[str]) -> bytes:
        """Compress multiple JSON log lines."""
        result = bytearray()
        template_map = {}
        value_streams = defaultdict(list)
        
        for line in lines:
            template, values = self.extract_template(line.strip())
            
            if template not in template_map:
                template_map[template] = len(template_map)
            
            tid = template_map[template]
            value_streams['tid'].append(tid)
            
            for i, (vtype, val) in enumerate(values):
                value_streams[f'{vtype}_{i}'].append(val)
        
        # Serialize templates
        templates_data = json.dumps(list(template_map.keys())).encode('utf-8')
        result.extend(struct.pack('>I', len(templates_data)))
        result.extend(templates_data)
        
        # Serialize value streams
        result.extend(struct.pack('>I', len(value_streams)))
        
        for key, vals in value_streams.items():
            key_bytes = key.encode('utf-8')
            result.extend(struct.pack('>B', len(key_bytes)))
            result.extend(key_bytes)
            
            # Serialize values based on type
            val_data = json.dumps(vals).encode('utf-8')
            result.extend(struct.pack('>I', len(val_data)))
            result.extend(val_data)
        
        # Final compression
        return zlib.compress(bytes(result), 9)


# ============================================================================
# 2. TIME-SERIES COMPRESSOR
# ============================================================================

class TimeSeriesCompressor:
    """
    Optimized for time-series data (metrics, sensors, IoT).
    Uses XOR encoding for floats and delta encoding for timestamps.
    """
    
    @staticmethod
    def compress_floats(values: List[float]) -> bytes:
        """Compress float values using XOR of IEEE 754 representation."""
        if not values:
            return b''
        
        result = bytearray()
        
        # First value stored as-is
        prev_bits = struct.unpack('>Q', struct.pack('>d', values[0]))[0]
        result.extend(struct.pack('>d', values[0]))
        
        for val in values[1:]:
            curr_bits = struct.unpack('>Q', struct.pack('>d', val))[0]
            xor = prev_bits ^ curr_bits
            
            if xor == 0:
                result.append(0x00)  # No change
            else:
                # Count leading zeros
                leading = 0
                temp = xor
                while temp > 0 and (temp >> 63) == 0:
                    leading += 1
                    temp <<= 1
                
                # Simple encoding: leading zeros + significant bytes
                significant_bytes = (64 - leading + 7) // 8
                result.append(0x80 | ((leading & 0x3F) << 1) | (significant_bytes - 1))
                
                for i in range(significant_bytes):
                    shift = (significant_bytes - 1 - i) * 8
                    result.append((xor >> shift) & 0xFF)
            
            prev_bits = curr_bits
        
        return bytes(result)
    
    @staticmethod
    def compress_timestamps(timestamps: List[int]) -> bytes:
        """Compress timestamps using delta-of-delta encoding."""
        if not timestamps:
            return b''
        
        result = bytearray()
        
        # First timestamp
        result.extend(struct.pack('>Q', timestamps[0]))
        
        if len(timestamps) == 1:
            return bytes(result)
        
        # First delta
        delta = timestamps[1] - timestamps[0]
        result.extend(struct.pack('>i', delta))
        
        prev_delta = delta
        
        for i in range(2, len(timestamps)):
            delta = timestamps[i] - timestamps[i-1]
            dod = delta - prev_delta  # Delta of delta
            
            # Variable-length encoding based on magnitude
            if dod == 0:
                result.append(0x00)
            elif -63 <= dod <= 64:
                result.append(0x40 | ((dod + 63) & 0x7F))
            elif -8191 <= dod <= 8192:
                result.append(0x80 | (((dod + 8191) >> 8) & 0x3F))
                result.append((dod + 8191) & 0xFF)
            else:
                result.append(0xC0)
                result.extend(struct.pack('>i', dod))
            
            prev_delta = delta
        
        return bytes(result)
    
    @classmethod
    def compress(cls, timestamps: List[int], values: List[float]) -> bytes:
        """Compress a complete time series."""
        ts_data = cls.compress_timestamps(timestamps)
        val_data = cls.compress_floats(values)
        
        header = struct.pack('>II', len(ts_data), len(val_data))
        return header + ts_data + val_data


# ============================================================================
# 3. IMAGE RESIDUAL COMPRESSOR
# ============================================================================

class ImageResidualCompressor:
    """
    Compresses residual differences between video frames.
    Optimized for surveillance and security camera footage.
    """
    
    @staticmethod
    def compute_residual(prev_frame: bytes, curr_frame: bytes) -> bytes:
        """Compute XOR residual between frames."""
        if len(prev_frame) != len(curr_frame):
            return curr_frame  # Can't compute residual
        
        return bytes(a ^ b for a, b in zip(prev_frame, curr_frame))
    
    @staticmethod
    def encode_sparse_residual(residual: bytes, threshold: int = 5) -> bytes:
        """Encode sparse residual using run-length for zeros."""
        result = bytearray()
        zero_count = 0
        
        for byte in residual:
            if byte < threshold:
                zero_count += 1
                if zero_count == 255:
                    result.extend([0x00, 255])
                    zero_count = 0
            else:
                if zero_count > 0:
                    result.extend([0x00, zero_count])
                    zero_count = 0
                
                if byte == 0x00:
                    result.extend([0x00, 0])  # Literal zero
                else:
                    result.append(byte)
        
        if zero_count > 0:
            result.extend([0x00, zero_count])
        
        return bytes(result)
    
    @staticmethod
    def compress_block_motion(blocks_changed: List[Tuple[int, int, bytes]]) -> bytes:
        """Compress motion-detected blocks efficiently."""
        result = bytearray()
        result.extend(struct.pack('>I', len(blocks_changed)))
        
        for x, y, block_data in blocks_changed:
            # Position (2 bytes each)
            result.extend(struct.pack('>HH', x, y))
            # Block data (compressed)
            compressed = zlib.compress(block_data, 1)  # Fast compression
            result.extend(struct.pack('>H', len(compressed)))
            result.extend(compressed)
        
        return bytes(result)


# ============================================================================
# 4. DNA/SEQUENCE COMPRESSOR
# ============================================================================

class DNACompressor:
    """
    Specialized compressor for genomic data.
    Uses 2-bit encoding for bases and reference-based compression.
    """
    
    BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    REVERSE_MAP = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    
    @classmethod
    def encode_bases(cls, sequence: str) -> bytes:
        """Encode DNA sequence using 2 bits per base (4:1 compression)."""
        result = bytearray()
        
        # Handle N's and other ambiguous bases
        n_positions = []
        clean_seq = []
        
        for i, base in enumerate(sequence):
            if base.upper() in 'ACGT':
                clean_seq.append(base)
            else:
                n_positions.append(i)
                clean_seq.append('A')  # Placeholder
        
        # Pack 4 bases per byte
        for i in range(0, len(clean_seq), 4):
            byte = 0
            for j in range(4):
                if i + j < len(clean_seq):
                    byte |= cls.BASE_MAP.get(clean_seq[i + j], 0) << (6 - j * 2)
            result.append(byte)
        
        # Store N positions
        n_data = struct.pack('>I', len(n_positions))
        for pos in n_positions:
            n_data += struct.pack('>I', pos)
        
        # Header: length + N positions + packed bases
        header = struct.pack('>I', len(sequence))
        return header + n_data + bytes(result)
    
    @classmethod
    def compress_reads(cls, reads: List[str]) -> bytes:
        """Compress multiple DNA reads with deduplication."""
        result = bytearray()
        
        # Find common subsequences
        seen = {}
        encoded_reads = []
        
        for read in reads:
            # Hash the read
            read_hash = hash(read)
            
            if read_hash in seen:
                # Reference to previous read
                encoded_reads.append(('R', seen[read_hash]))
            else:
                # New read
                seen[read_hash] = len(encoded_reads)
                encoded_reads.append(('N', cls.encode_bases(read)))
        
        # Serialize
        result.extend(struct.pack('>I', len(reads)))
        
        for etype, data in encoded_reads:
            if etype == 'R':
                result.append(0x00)  # Reference marker
                result.extend(struct.pack('>I', data))
            else:
                result.append(0x01)  # New read marker
                result.extend(struct.pack('>I', len(data)))
                result.extend(data)
        
        return zlib.compress(bytes(result), 9)


# ============================================================================
# 5. CODE/SOURCE COMPRESSOR
# ============================================================================

class CodeCompressor:
    """
    Specialized compressor for source code.
    Uses token-based compression with language-specific dictionaries.
    """
    
    # Common tokens across programming languages
    COMMON_TOKENS = [
        'def', 'class', 'import', 'from', 'return', 'if', 'else', 'elif',
        'for', 'while', 'try', 'except', 'with', 'as', 'in', 'not', 'and', 'or',
        'True', 'False', 'None', 'self', 'print', 'len', 'range', 'list', 'dict',
        'function', 'const', 'let', 'var', 'async', 'await', 'export', 'import',
        'public', 'private', 'protected', 'static', 'void', 'int', 'string',
        '(', ')', '[', ']', '{', '}', ':', ',', '.', '=', '==', '!=', '+=', '-=',
        '->', '=>', '::', '/**', '*/', '//', '/*', '#', '"""', "'''"
    ]
    
    def __init__(self):
        self.token_map = {tok: i for i, tok in enumerate(self.COMMON_TOKENS)}
        self.identifier_map = {}
        self.identifier_counter = len(self.COMMON_TOKENS)
    
    def tokenize(self, code: str) -> List[str]:
        """Tokenize source code."""
        # Simple tokenizer - splits on whitespace and common delimiters
        pattern = r'(\s+|[(){}\[\]:,;.=<>!&|+\-*/])'
        tokens = re.split(pattern, code)
        return [t for t in tokens if t]
    
    def compress(self, code: str) -> bytes:
        """Compress source code using token-based encoding."""
        tokens = self.tokenize(code)
        result = bytearray()
        
        local_identifiers = {}
        local_counter = 0
        
        for token in tokens:
            if token in self.token_map:
                # Common token (1 byte)
                result.append(self.token_map[token])
            elif token.isidentifier():
                # Identifier - use local mapping
                if token not in local_identifiers:
                    local_identifiers[token] = local_counter
                    local_counter += 1
                    # First occurrence: marker + length + name
                    result.append(0xFE)
                    token_bytes = token.encode('utf-8')
                    result.append(len(token_bytes))
                    result.extend(token_bytes)
                else:
                    # Repeated: reference
                    result.append(0xFD)
                    result.append(local_identifiers[token])
            elif token.isspace():
                # Whitespace compression
                spaces = len(token)
                if token[0] == ' ':
                    result.extend([0xFC, spaces])
                elif token[0] == '\n':
                    result.extend([0xFB, token.count('\n')])
                elif token[0] == '\t':
                    result.extend([0xFA, token.count('\t')])
            else:
                # Literal
                result.append(0xFF)
                token_bytes = token.encode('utf-8')
                result.extend(struct.pack('>H', len(token_bytes)))
                result.extend(token_bytes)
        
        return zlib.compress(bytes(result), 9)


# ============================================================================
# UNIFIED INTERFACE
# ============================================================================

class SpecializedCompressor:
    """
    Unified interface for all specialized compressors.
    Auto-detects data type and applies optimal compression.
    """
    
    def __init__(self):
        self.json_compressor = JSONLogCompressor()
        self.timeseries_compressor = TimeSeriesCompressor()
        self.image_compressor = ImageResidualCompressor()
        self.dna_compressor = DNACompressor()
        self.code_compressor = CodeCompressor()
    
    def detect_type(self, data: bytes) -> str:
        """Detect data type from content."""
        try:
            text = data.decode('utf-8', errors='ignore')
        except:
            return 'binary'
        
        # Check for JSON
        if text.strip().startswith('{') or text.strip().startswith('['):
            try:
                json.loads(text.strip().split('\n')[0])
                return 'json'
            except:
                pass
        
        # Check for DNA
        dna_chars = set('ACGTNacgtn\n\r ')
        if len(text) > 100 and all(c in dna_chars for c in text[:1000]):
            return 'dna'
        
        # Check for source code
        code_indicators = ['def ', 'class ', 'function ', 'import ', '#include', 'package ']
        for indicator in code_indicators:
            if indicator in text[:1000]:
                return 'code'
        
        return 'generic'
    
    def compress(self, data: bytes) -> Tuple[bytes, str, float]:
        """Compress data using detected specialized compressor."""
        dtype = self.detect_type(data)
        
        try:
            if dtype == 'json':
                text = data.decode('utf-8')
                lines = text.strip().split('\n')
                compressed = self.json_compressor.compress_logs(lines)
            elif dtype == 'dna':
                text = data.decode('utf-8').replace('\n', '').replace('\r', '')
                compressed = DNACompressor.encode_bases(text)
            elif dtype == 'code':
                text = data.decode('utf-8')
                compressed = self.code_compressor.compress(text)
            else:
                compressed = zlib.compress(data, 9)
        except Exception as e:
            # Fallback to zlib
            compressed = zlib.compress(data, 9)
            dtype = 'fallback'
        
        ratio = len(data) / max(1, len(compressed))
        return compressed, dtype, ratio


# ============================================================================
# MAIN - Demo
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NS-ARC Specialized Compressors - Demo")
    print("=" * 60)
    
    compressor = SpecializedCompressor()
    
    # Test JSON logs
    json_logs = '\n'.join([
        '{"ts":1234567890,"level":"INFO","user":"alice","action":"login"}',
        '{"ts":1234567891,"level":"INFO","user":"bob","action":"login"}',
        '{"ts":1234567892,"level":"DEBUG","user":"alice","action":"view"}',
    ] * 100)
    
    # Test DNA sequence
    dna_seq = 'ACGTACGTACGTNNNNACGTACGT' * 500
    
    # Test source code
    source_code = '''
def hello_world():
    print("Hello, World!")

def add_numbers(a, b):
    return a + b

class Calculator:
    def __init__(self):
        self.result = 0
    
    def add(self, x):
        self.result += x
        return self
''' * 50
    
    tests = [
        ("JSON Logs", json_logs.encode()),
        ("DNA Sequence", dna_seq.encode()),
        ("Source Code", source_code.encode()),
    ]
    
    print("\n[Testing Specialized Compressors]")
    
    for name, data in tests:
        compressed, dtype, ratio = compressor.compress(data)
        zlib_size = len(zlib.compress(data, 9))
        
        print(f"\n  {name} (detected as: {dtype}):")
        print(f"    Original:    {len(data):,} bytes")
        print(f"    Specialized: {len(compressed):,} bytes")
        print(f"    Zlib-9:      {zlib_size:,} bytes")
        print(f"    Ratio:       {ratio:.2f}x")
        improvement = (zlib_size - len(compressed)) / zlib_size * 100
        print(f"    vs Zlib:     {improvement:+.1f}%")
    
    # Test DNA encoding directly
    print("\n[DNA 2-bit Encoding Test]")
    dna_test = "ACGTACGTACGTACGT"
    encoded = DNACompressor.encode_bases(dna_test)
    print(f"  Original:  {len(dna_test)} chars")
    print(f"  Encoded:   {len(encoded)} bytes")
    print(f"  Ratio:     {len(dna_test)/len(encoded):.2f}x (theoretical max: 4x)")
    
    print("\n" + "=" * 60)
    print("Specialized compressors ready!")
    print("=" * 60)

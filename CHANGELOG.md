# Changelog

All notable changes to NS-ARC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-29

### Added

#### Core Features
- **Semantic Router** - Automatic file type detection (Image/Video/Code/Logs/Binary/Genomics)
- **ERT Orchestrator** - Entropy Resonance Transducer optimization engine
- **Compression Modes** - Store, Zstd, Format-Lossless, Semantic Split, Corpus Resonance
- **Decompressor** - Full decompression with integrity verification

#### AI Features
- **Neural Codec Selector** - Predicts best codec from file fingerprint
- **Semantic Tokenizer** - JSON/Log tokenization for better compression
- **VQ-VAE Image Codec** - Neural image compression (lossy, optional)
- **AI Feature Flags** - Configurable AI feature toggles

#### Infrastructure
- **SQLite Chunk Store** - Content-addressable storage for deduplication
- **Performance Instrumentation** - Timing, counters, throughput metrics
- **Speed Configuration** - Fastest/Balanced/MaxRatio modes
- **Verification Modes** - None/Chunk/Full SHA-256 verification

#### Dashboard
- **Streamlit Web UI** - Visual compression/decompression interface
- **File Upload** - Drag-and-drop file handling
- **Dedup Stats** - Real-time deduplication statistics
- **Download** - Compressed file download

#### Training
- **VQ-VAE Training Pipeline** - Image dataset download and model training
- **1B Parameter Model** - Large-scale neural compression architecture

### Fixed
- Decompressor download button now works for all file types
- Large image preview with automatic downscaling
- Compressor speed improved by using pre-built binary

### Security
- Bounds checking on all file operations
- Expansion limit enforcement (no zip-bomb vulnerabilities)
- SHA-256 integrity verification

## [Unreleased]

### Planned
- Learned Entropy Model (rANS integration)
- Semantic Chunk Boundary detection
- Neural Similarity Embeddings for near-dedup
- EnCodec audio compression

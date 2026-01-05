"""
NS-ARC Core: Neural-Symbolic Adaptive Resonance Compressor

This package contains the core compression algorithms and modules for NS-ARC.

Modules:
- ns_arc_specialized: Domain-specific compressors (JSON/Log, Time-Series, DNA, etc.)
- ns_arc_enhancers: Compression enhancement algorithms (BWT, Delta, Dictionary)
- ns_arc_vision: VQ-VAE neural codec for images
- ert_core: Enhanced Reasoning Transformer for compression
"""

# Specialized Compressors
from .ns_arc_specialized import (
    JSONLogCompressor,
    TimeSeriesCompressor,
    ImageResidualCompressor,
    DNACompressor,
)

# Compression Enhancers
from .ns_arc_enhancers import (
    AdaptiveContextModeler,
    DeltaEncoder,
    DictionaryLearner,
    BWTCompressor,
)

# Vision/Image Compression - import safely
try:
    from .ns_arc_vision import VQVAE
except ImportError:
    VQVAE = None

# ERT Components - these are functions and classes
from .ert_core import SineLayer, SirenNet, train_ert, reconstruct

# Note: SemanticRouter has complex dependencies, import explicitly if needed
# from .ns_arc_router import SemanticRouter

__all__ = [
    # Specialized
    "JSONLogCompressor",
    "TimeSeriesCompressor",
    "ImageResidualCompressor",
    "DNACompressor",
    # Enhancers
    "AdaptiveContextModeler",
    "DeltaEncoder",
    "DictionaryLearner",
    "BWTCompressor",
    # Vision
    "VQVAE",
    # ERT
    "SineLayer",
    "SirenNet",
    "train_ert",
    "reconstruct",
]

__version__ = "1.0.0"

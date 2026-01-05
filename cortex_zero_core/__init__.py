"""
Cortex Zero Core: HALT-NN Anti-Hallucination Pipeline

Evidence-grounded truth verification using Jøsang Subjective Logic.
"""

# Core models
from .models import (
    Opinion, Polarity, Source, Evidence, AtomicClaim,
    ClaimType, ClaimStatus, NLILabel, SourceTier, ActionDecision, IntentType,
    IntentResult, HaltClaim, HaltEvidence, NLIResult, EvidenceLink, AnswerAudit,
    TIER_RELIABILITY
)

# Jøsang logic operators
from .logic import discount_opinion, consensus_opinion, calculate_ess

# Knowledge graph
from .graph import CortexGraph

# HALT-NN pipeline
from .halt_pipeline import (
    run_halt_pipeline,
    HaltConfig,
    EmbeddingRetriever,
    Reranker,
    NLIModel,
    UncertaintyCalibrator,
    SimpleNLI,
    SimpleCalibrator,
    analyze_intent,
    decompose_claims,
    retrieve_evidence,
    build_evidence_graph,
    verify_gate,
    generate_controlled_answer,
    calibrate_confidence,
    decide_action
)

# Persistence
from .persistence import PersistenceManager

# Neural modules (optional - requires additional dependencies)
try:
    from .neural_nli import TransformerNLI, CrossEncoderNLI, CachedNLI
    _HAS_NEURAL_NLI = True
except ImportError:
    _HAS_NEURAL_NLI = False

try:
    from .neural_retriever import SentenceTransformerRetriever, HybridRetriever
    _HAS_NEURAL_RETRIEVER = True
except ImportError:
    _HAS_NEURAL_RETRIEVER = False

try:
    from .neural_calibrator import (
        IsotonicCalibrator, FeatureBasedCalibrator, EnsembleCalibrator,
        expected_calibration_error, brier_score, reliability_diagram_data
    )
    _HAS_NEURAL_CALIBRATOR = True
except ImportError:
    _HAS_NEURAL_CALIBRATOR = False

__all__ = [
    # Models
    "Opinion", "Polarity", "Source", "Evidence", "AtomicClaim",
    "ClaimType", "ClaimStatus", "NLILabel", "SourceTier", "ActionDecision", "IntentType",
    "IntentResult", "HaltClaim", "HaltEvidence", "NLIResult", "EvidenceLink", "AnswerAudit",
    "TIER_RELIABILITY",
    # Logic
    "discount_opinion", "consensus_opinion", "calculate_ess",
    # Graph
    "CortexGraph",
    # Pipeline
    "run_halt_pipeline", "HaltConfig",
    "EmbeddingRetriever", "Reranker", "NLIModel", "UncertaintyCalibrator",
    "SimpleNLI", "SimpleCalibrator",
    "analyze_intent", "decompose_claims", "retrieve_evidence",
    "build_evidence_graph", "verify_gate", "generate_controlled_answer",
    "calibrate_confidence", "decide_action",
    # Persistence
    "PersistenceManager",
]

# Add neural modules to exports if available
if _HAS_NEURAL_NLI:
    __all__.extend(["TransformerNLI", "CrossEncoderNLI", "CachedNLI"])

if _HAS_NEURAL_RETRIEVER:
    __all__.extend(["SentenceTransformerRetriever", "HybridRetriever"])

if _HAS_NEURAL_CALIBRATOR:
    __all__.extend([
        "IsotonicCalibrator", "FeatureBasedCalibrator", "EnsembleCalibrator",
        "expected_calibration_error", "brier_score", "reliability_diagram_data"
    ])

# Optimized inference (always available)
try:
    from .optimized_inference import OptimizedNLI, OptimizedRetriever, get_optimized_nli, get_optimized_retriever
    __all__.extend(["OptimizedNLI", "OptimizedRetriever", "get_optimized_nli", "get_optimized_retriever"])
except ImportError:
    pass

# Optimized pipeline
try:
    from .optimized_pipeline import OptimizedHaltPipeline, get_optimized_pipeline, run_halt_optimized
    __all__.extend(["OptimizedHaltPipeline", "get_optimized_pipeline", "run_halt_optimized"])
except ImportError:
    pass

# Chain-of-Thought Verification
try:
    from .chain_of_thought import (
        ChainExtractor, MultiHopVerifier, HALTCoTEngine, 
        ReasoningChain, ReasoningStep, verify_with_cot
    )
    __all__.extend([
        "ChainExtractor", "MultiHopVerifier", "HALTCoTEngine",
        "ReasoningChain", "ReasoningStep", "verify_with_cot"
    ])
except ImportError:
    pass

# Mega Algorithms (10 enhancement algorithms)
try:
    from .mega_algorithms import (
        KnowledgeGraph, SemanticDeduplicator, TemporalDecay,
        AdversarialDetector, ActiveLearner, HierarchicalAttention,
        SourceReputationLearner, QueryExpander, EnsembleVerifier,
        UncertaintyQuantifier, MegaHALTEngine, create_mega_engine
    )
    __all__.extend([
        "KnowledgeGraph", "SemanticDeduplicator", "TemporalDecay",
        "AdversarialDetector", "ActiveLearner", "HierarchicalAttention",
        "SourceReputationLearner", "QueryExpander", "EnsembleVerifier",
        "UncertaintyQuantifier", "MegaHALTEngine", "create_mega_engine"
    ])
except ImportError:
    pass

# HALT-ARC: Merged compression + verification
try:
    from .halt_arc import (
        HALTARCEngine, CompressedEvidenceStore, EmbeddingCompressor,
        ContentDefinedChunker, create_halt_arc_engine
    )
    __all__.extend([
        "HALTARCEngine", "CompressedEvidenceStore", "EmbeddingCompressor",
        "ContentDefinedChunker", "create_halt_arc_engine"
    ])
except ImportError:
    pass


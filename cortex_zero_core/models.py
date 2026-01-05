"""
HALT-NN: Evidence-Grounded Anti-Hallucination Models

Extends Jøsang Subjective Logic with claim verification and evidence mapping.
"""

from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator
import time
import hashlib

# =============================================================================
# ORIGINAL JØSANG CORE (Preserved)
# =============================================================================

class Polarity(str, Enum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"

class Opinion(BaseModel):
    """Jøsang Subjective Logic Opinion Tuple: (belief, disbelief, uncertainty)"""
    belief: float = Field(..., ge=0.0, le=1.0)
    disbelief: float = Field(..., ge=0.0, le=1.0)
    uncertainty: float = Field(..., ge=0.0, le=1.0)
    base_rate: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode='after')
    def check_sum(self) -> 'Opinion':
        total = self.belief + self.disbelief + self.uncertainty
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"b + d + u must sum to 1.0, got {total}")
        return self
    
    @classmethod
    def vacuous(cls) -> 'Opinion':
        """Return a vacuous (no evidence) opinion."""
        return cls(belief=0.0, disbelief=0.0, uncertainty=1.0)
    
    @classmethod
    def dogmatic_true(cls) -> 'Opinion':
        """Return a fully certain TRUE opinion."""
        return cls(belief=1.0, disbelief=0.0, uncertainty=0.0)
    
    @classmethod
    def dogmatic_false(cls) -> 'Opinion':
        """Return a fully certain FALSE opinion."""
        return cls(belief=0.0, disbelief=1.0, uncertainty=0.0)

class Source(BaseModel):
    """Evidence source with reliability rating."""
    id: str
    reliability: float = Field(..., ge=0.0, le=1.0)
    domain: str

class Evidence(BaseModel):
    """Original evidence model (preserved for compatibility)."""
    id: str
    content: str
    source_id: str
    timestamp: int
    polarity: Polarity

class AtomicClaim(BaseModel):
    """Original atomic claim (Subject-Predicate-Object triplet)."""
    id: str
    subject: str
    predicate: str
    object: str
    creation_time: int = Field(default_factory=lambda: int(time.time()))
    history: List[Tuple[int, float]] = Field(default_factory=list)
    current_ess: float = 0.0
    current_opinion: Opinion = Field(default_factory=Opinion.vacuous)


# =============================================================================
# HALT-NN EXTENSIONS
# =============================================================================

class ClaimType(str, Enum):
    """Classification of claim for verification requirements."""
    MUST_CITE = "MUST_CITE"       # Non-trivial factual - REQUIRES evidence
    DERIVATION = "DERIVATION"     # Math/logical - verify with computation
    SUBJECTIVE = "SUBJECTIVE"     # Opinion/preference - no citation needed
    META = "META"                 # Process description - no citation needed

class ClaimStatus(str, Enum):
    """Verification status of a claim."""
    SUPPORTED = "SUPPORTED"       # Evidence meets threshold
    UNSUPPORTED = "UNSUPPORTED"   # No/insufficient evidence - CANNOT state as fact
    DISPUTED = "DISPUTED"         # Conflicting evidence - must present both sides
    PENDING = "PENDING"           # Not yet verified

class NLILabel(str, Enum):
    """Natural Language Inference output labels."""
    ENTAILS = "ENTAILS"           # Evidence supports claim
    CONTRADICTS = "CONTRADICTS"   # Evidence refutes claim
    NEUTRAL = "NEUTRAL"           # Evidence is irrelevant

class SourceTier(str, Enum):
    """Evidence source quality tiers."""
    TIER_A = "A"  # Official, peer-reviewed, government (0.9-1.0)
    TIER_B = "B"  # Reputable journalism, books (0.7-0.9)
    TIER_C = "C"  # Blogs, forums, user content (0.3-0.7)

# Reliability mapping for source tiers
TIER_RELIABILITY = {
    SourceTier.TIER_A: 0.95,
    SourceTier.TIER_B: 0.80,
    SourceTier.TIER_C: 0.50,
}

class ActionDecision(str, Enum):
    """Pipeline action decisions."""
    ANSWER = "ANSWER"                       # Proceed with answer
    ANSWER_WITH_DISPUTE = "ANSWER_WITH_DISPUTE"  # Answer showing conflicts
    ASK_FOR_INFO = "ASK_FOR_INFO"           # Request missing information
    ABSTAIN = "ABSTAIN"                     # Refuse to answer

class IntentType(str, Enum):
    """Query intent classification."""
    FACT_LOOKUP = "FACT_LOOKUP"
    EXPLANATION = "EXPLANATION"
    PLAN = "PLAN"
    PREDICTION = "PREDICTION"
    OPINION = "OPINION"


# =============================================================================
# HALT-NN DATA STRUCTURES
# =============================================================================

class IntentResult(BaseModel):
    """Result of intent analysis (Phase 1)."""
    intent_type: IntentType
    time_bounds: Optional[str] = None       # e.g., "2024", "last week"
    jurisdiction: Optional[str] = None      # e.g., "US", "California"
    precision_required: bool = False        # High precision needed?
    recency_sensitive: bool = False         # Needs fresh data?
    high_stakes: bool = False               # Safety-critical domain?

class HaltClaim(BaseModel):
    """A claim requiring verification in the HALT-NN pipeline."""
    id: str
    text: str                               # The claim statement
    claim_type: ClaimType
    status: ClaimStatus = ClaimStatus.PENDING
    opinion: Opinion = Field(default_factory=Opinion.vacuous)
    confidence: float = 0.0                 # Calibrated confidence
    evidence_ids: List[str] = Field(default_factory=list)
    
    @classmethod
    def create(cls, text: str, claim_type: ClaimType) -> 'HaltClaim':
        """Factory method to create a claim with auto-generated ID."""
        claim_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        return cls(id=claim_id, text=text, claim_type=claim_type)

class HaltEvidence(BaseModel):
    """Evidence item with full provenance for HALT-NN."""
    id: str
    content: str                            # Full evidence text
    span: str                               # Exact quote used
    source_id: str                          # Source identifier
    source_tier: SourceTier = SourceTier.TIER_B
    reliability: float = 0.8
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    url: Optional[str] = None
    doc_ref: Optional[str] = None           # Document reference
    
    @classmethod
    def create(cls, content: str, source_id: str, tier: SourceTier) -> 'HaltEvidence':
        """Factory method with auto-generated ID and reliability."""
        ev_id = hashlib.sha256(content.encode()).hexdigest()[:16]
        return cls(
            id=ev_id, 
            content=content, 
            span=content[:200],
            source_id=source_id,
            source_tier=tier,
            reliability=TIER_RELIABILITY[tier]
        )

class NLIResult(BaseModel):
    """Result from NLI/entailment model."""
    label: NLILabel
    probability: float = Field(..., ge=0.0, le=1.0)
    
class EvidenceLink(BaseModel):
    """Link between a claim and evidence in the evidence graph."""
    claim_id: str
    evidence_id: str
    nli_label: NLILabel
    nli_probability: float = Field(..., ge=0.0, le=1.0)
    support_strength: float = 0.0           # Mapped to Opinion.belief
    conflict_flag: bool = False             # True if CONTRADICTS
    
    def to_opinion(self, source_reliability: float = 0.8) -> Opinion:
        """Convert this link to a Jøsang Opinion."""
        if self.nli_label == NLILabel.ENTAILS:
            b = self.nli_probability * source_reliability
            d = 0.0
        elif self.nli_label == NLILabel.CONTRADICTS:
            b = 0.0
            d = self.nli_probability * source_reliability
        else:
            b = 0.0
            d = 0.0
        u = 1.0 - (b + d)
        return Opinion(belief=b, disbelief=d, uncertainty=u)


class AnswerAudit(BaseModel):
    """Complete audit trail for an answer (Phase 7 output)."""
    query: str
    timestamp: int = Field(default_factory=lambda: int(time.time()))
    
    # Components
    claims: List[HaltClaim] = Field(default_factory=list)
    evidence: List[HaltEvidence] = Field(default_factory=list)
    links: List[EvidenceLink] = Field(default_factory=list)
    
    # Decision
    action: ActionDecision = ActionDecision.ANSWER
    
    # Metrics
    overall_confidence: float = 0.0
    coverage_ratio: float = 0.0             # supported / required claims
    conflict_count: int = 0
    
    # Output
    answer_text: str = ""
    abstentions: List[str] = Field(default_factory=list)  # Claims we couldn't verify
    next_actions: List[str] = Field(default_factory=list) # What would raise confidence
    
    def get_claim_evidence_map(self) -> Dict[str, List[str]]:
        """Return mapping of claim_id -> list of evidence_ids."""
        mapping: Dict[str, List[str]] = {}
        for link in self.links:
            if link.claim_id not in mapping:
                mapping[link.claim_id] = []
            mapping[link.claim_id].append(link.evidence_id)
        return mapping

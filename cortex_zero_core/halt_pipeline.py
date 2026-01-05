"""
HALT-NN: Evidence-Grounded Anti-Hallucination Pipeline

7-Phase Deterministic Truth Pipeline with Jøsang Subjective Logic integration.
Neural modules are helpers that CANNOT bypass the "no evidence → no claim" rule.
"""

from typing import List, Dict, Optional, Tuple
from abc import ABC, abstractmethod

from .models import (
    Opinion, HaltClaim, HaltEvidence, EvidenceLink, AnswerAudit,
    ClaimType, ClaimStatus, NLILabel, NLIResult, SourceTier,
    ActionDecision, IntentType, IntentResult, TIER_RELIABILITY
)
from .logic import discount_opinion, consensus_opinion, calculate_ess


# =============================================================================
# CONFIGURATION
# =============================================================================

class HaltConfig:
    """Pipeline configuration thresholds."""
    SUPPORT_THRESHOLD: float = 0.6      # Min belief to mark SUPPORTED
    CONFLICT_THRESHOLD: float = 0.3     # % of contradicting evidence for DISPUTED
    CONFIDENCE_THRESHOLD: float = 0.5   # Min confidence to proceed with ANSWER
    COVERAGE_THRESHOLD: float = 0.3     # Min coverage to avoid ABSTAIN
    TOP_K_EVIDENCE: int = 5             # Evidence items per claim


# =============================================================================
# NEURAL MODULE INTERFACES (Abstract - implement with real models)
# =============================================================================

class EmbeddingRetriever(ABC):
    """Interface for embedding-based evidence retrieval."""
    @abstractmethod
    def search(self, query: str, sources: List[str], k: int) -> List[HaltEvidence]:
        """Return top-k evidence candidates."""
        pass

class Reranker(ABC):
    """Interface for evidence reranking."""
    @abstractmethod
    def rerank(self, query: str, candidates: List[HaltEvidence], k: int) -> List[HaltEvidence]:
        """Rerank and return top-k most relevant."""
        pass

class NLIModel(ABC):
    """Interface for Natural Language Inference."""
    @abstractmethod
    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        """Determine if premise ENTAILS/CONTRADICTS/NEUTRAL to hypothesis."""
        pass

class UncertaintyCalibrator(ABC):
    """Interface for confidence calibration."""
    @abstractmethod
    def calibrate(self, claim: HaltClaim, links: List[EvidenceLink]) -> float:
        """Return calibrated probability that claim is true."""
        pass


# =============================================================================
# SIMPLE IMPLEMENTATIONS (For testing without neural models)
# =============================================================================

class SimpleNLI(NLIModel):
    """Keyword-based NLI for testing (replace with real model)."""
    
    # Common question words to filter out
    STOP_WORDS = {"what", "is", "the", "a", "an", "who", "where", "when", "how", 
                  "does", "did", "are", "was", "were", "will", "can", "could"}
    
    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        premise_lower = premise.lower()
        hypothesis_lower = hypothesis.lower()
        
        # Extract key terms from hypothesis (filter stopwords and short words)
        key_terms = [w.strip('?.,!') for w in hypothesis_lower.split() 
                     if len(w) > 2 and w.strip('?.,!') not in self.STOP_WORDS]
        
        if not key_terms:
            return NLIResult(label=NLILabel.NEUTRAL, probability=0.8)
        
        # Count matches - check if term appears anywhere in premise
        matches = sum(1 for term in key_terms if term in premise_lower)
        match_ratio = matches / len(key_terms)
        
        # Check for negation
        negations = ["not", "never", "no", "false", "incorrect", "wrong", "isn't", "wasn't"]
        has_negation = any(neg in premise_lower for neg in negations)
        
        # Lower threshold for entailment (0.3 instead of 0.5)
        if match_ratio >= 0.3:
            if has_negation:
                return NLIResult(label=NLILabel.CONTRADICTS, probability=0.6 + match_ratio * 0.2)
            else:
                return NLIResult(label=NLILabel.ENTAILS, probability=0.5 + match_ratio * 0.4)
        else:
            return NLIResult(label=NLILabel.NEUTRAL, probability=0.8)


class SimpleCalibrator(UncertaintyCalibrator):
    """Simple calibrator based on evidence count and agreement."""
    
    def calibrate(self, claim: HaltClaim, links: List[EvidenceLink]) -> float:
        if not links:
            return 0.0
        
        # Factor 1: Number of sources (more = higher confidence)
        source_factor = min(len(links) / 3, 1.0)
        
        # Factor 2: Agreement (supporting vs contradicting)
        supporting = sum(1 for l in links if l.nli_label == NLILabel.ENTAILS)
        contradicting = sum(1 for l in links if l.nli_label == NLILabel.CONTRADICTS)
        total = supporting + contradicting
        agreement_factor = supporting / max(total, 1)
        
        # Factor 3: Average NLI probability
        avg_prob = sum(l.nli_probability for l in links) / len(links)
        
        # Combine factors
        confidence = (source_factor * 0.3 + agreement_factor * 0.4 + avg_prob * 0.3)
        return min(max(confidence, 0.0), 1.0)


# =============================================================================
# PHASE 1-2: INTENT ANALYSIS & CLAIM DECOMPOSITION
# =============================================================================

def analyze_intent(query: str) -> IntentResult:
    """
    Phase 1: Analyze query intent and extract constraints.
    """
    query_lower = query.lower()
    
    # Detect intent type
    if any(w in query_lower for w in ["what is", "who is", "define", "meaning of"]):
        intent_type = IntentType.FACT_LOOKUP
    elif any(w in query_lower for w in ["how does", "why does", "explain"]):
        intent_type = IntentType.EXPLANATION
    elif any(w in query_lower for w in ["will", "predict", "forecast"]):
        intent_type = IntentType.PREDICTION
    elif any(w in query_lower for w in ["should", "best", "recommend"]):
        intent_type = IntentType.OPINION
    else:
        intent_type = IntentType.FACT_LOOKUP
    
    # Detect constraints
    recency_keywords = ["current", "latest", "now", "today", "2024", "2025"]
    recency_sensitive = any(k in query_lower for k in recency_keywords)
    
    high_stakes_domains = ["medical", "legal", "financial", "safety", "health"]
    high_stakes = any(d in query_lower for d in high_stakes_domains)
    
    return IntentResult(
        intent_type=intent_type,
        recency_sensitive=recency_sensitive,
        high_stakes=high_stakes,
        precision_required=high_stakes
    )


def decompose_claims(query: str, intent: IntentResult) -> List[HaltClaim]:
    """
    Phase 2: Break query into atomic claims that need verification.
    
    Each claim is classified by type:
    - MUST_CITE: Factual claims requiring evidence
    - DERIVATION: Mathematical/logical (verify with computation)
    - SUBJECTIVE: Opinions (no citation needed)
    - META: Process descriptions (no citation needed)
    """
    claims = []
    
    # Simple decomposition: treat the entire query as one claim for now
    # In production, use an LLM to extract multiple atomic claims
    
    if intent.intent_type == IntentType.FACT_LOOKUP:
        claim_type = ClaimType.MUST_CITE
    elif intent.intent_type == IntentType.EXPLANATION:
        claim_type = ClaimType.MUST_CITE
    elif intent.intent_type == IntentType.PREDICTION:
        claim_type = ClaimType.MUST_CITE  # Still needs evidence for basis
    else:
        claim_type = ClaimType.SUBJECTIVE
    
    main_claim = HaltClaim.create(text=query, claim_type=claim_type)
    claims.append(main_claim)
    
    return claims


# =============================================================================
# PHASE 3-4: EVIDENCE RETRIEVAL & GRAPH CONSTRUCTION
# =============================================================================

def retrieve_evidence(
    claims: List[HaltClaim],
    evidence_store: List[HaltEvidence],
    retriever: Optional[EmbeddingRetriever] = None,
    top_k: int = HaltConfig.TOP_K_EVIDENCE
) -> List[HaltEvidence]:
    """
    Phase 3: Retrieve relevant evidence for each claim.
    
    If retriever is None, uses simple keyword matching on evidence_store.
    """
    if retriever:
        # Use neural retriever
        all_evidence = []
        for claim in claims:
            results = retriever.search(claim.text, sources=[], k=top_k)
            all_evidence.extend(results)
        return _deduplicate_evidence(all_evidence)
    
    # Simple keyword matching fallback
    relevant = []
    for claim in claims:
        claim_words = set(claim.text.lower().split())
        
        scored = []
        for ev in evidence_store:
            ev_words = set(ev.content.lower().split())
            overlap = len(claim_words & ev_words)
            if overlap > 0:
                scored.append((overlap, ev))
        
        # Take top-k by overlap
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, ev in scored[:top_k]:
            relevant.append(ev)
    
    return _deduplicate_evidence(relevant)


def _deduplicate_evidence(evidence: List[HaltEvidence]) -> List[HaltEvidence]:
    """Remove duplicate evidence by ID."""
    seen = set()
    unique = []
    for ev in evidence:
        if ev.id not in seen:
            seen.add(ev.id)
            unique.append(ev)
    return unique


def build_evidence_graph(
    claims: List[HaltClaim],
    evidence: List[HaltEvidence],
    nli_model: Optional[NLIModel] = None
) -> List[EvidenceLink]:
    """
    Phase 4: Build bipartite Claim↔Evidence graph with NLI scoring.
    
    Each link contains:
    - NLI label (ENTAILS/CONTRADICTS/NEUTRAL)
    - NLI probability
    - Support strength (mapped to Jøsang belief)
    - Conflict flag
    """
    if nli_model is None:
        nli_model = SimpleNLI()
    
    links = []
    
    for claim in claims:
        for ev in evidence:
            # Run NLI: does evidence support/contradict claim?
            nli_result = nli_model.predict(premise=ev.span, hypothesis=claim.text)
            
            # Convert NLI → Jøsang Opinion (will use in fusion)
            opinion = nli_to_opinion(nli_result, ev.reliability)
            
            link = EvidenceLink(
                claim_id=claim.id,
                evidence_id=ev.id,
                nli_label=nli_result.label,
                nli_probability=nli_result.probability,
                support_strength=opinion.belief,
                conflict_flag=(nli_result.label == NLILabel.CONTRADICTS)
            )
            links.append(link)
            
            # Track evidence on claim
            if nli_result.label != NLILabel.NEUTRAL:
                claim.evidence_ids.append(ev.id)
    
    return links


# =============================================================================
# NLI → JØSANG OPINION CONVERSION
# =============================================================================

def nli_to_opinion(nli_result: NLIResult, source_reliability: float) -> Opinion:
    """
    Convert NLI output to Jøsang Opinion, discounted by source reliability.
    
    ENTAILS → high belief
    CONTRADICTS → high disbelief  
    NEUTRAL → high uncertainty
    
    Then apply Jøsang discount operator.
    """
    prob = nli_result.probability
    
    if nli_result.label == NLILabel.ENTAILS:
        raw = Opinion(belief=prob, disbelief=0.0, uncertainty=1.0 - prob)
    elif nli_result.label == NLILabel.CONTRADICTS:
        raw = Opinion(belief=0.0, disbelief=prob, uncertainty=1.0 - prob)
    else:  # NEUTRAL
        raw = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)
    
    # Apply Jøsang discount operator
    return discount_opinion(raw, source_reliability)


# =============================================================================
# PHASE 5: VERIFICATION GATE (HALLUCINATION KILL SWITCH)
# =============================================================================

def fuse_evidence_opinions(
    claim: HaltClaim,
    links: List[EvidenceLink],
    evidence_map: Dict[str, HaltEvidence]
) -> Opinion:
    """
    Fuse multiple evidence opinions using Jøsang consensus operator.
    
    Returns the combined opinion for a claim based on all linked evidence.
    """
    claim_links = [l for l in links if l.claim_id == claim.id]
    
    if not claim_links:
        return Opinion.vacuous()
    
    # Convert first link to opinion
    first_link = claim_links[0]
    ev = evidence_map.get(first_link.evidence_id)
    reliability = ev.reliability if ev else 0.8
    fused = first_link.to_opinion(reliability)
    
    # Iteratively fuse with remaining evidence
    for link in claim_links[1:]:
        ev = evidence_map.get(link.evidence_id)
        reliability = ev.reliability if ev else 0.8
        next_opinion = link.to_opinion(reliability)
        fused = consensus_opinion(fused, next_opinion)
    
    return fused


def verify_gate(
    claims: List[HaltClaim],
    links: List[EvidenceLink],
    evidence: List[HaltEvidence],
    threshold: float = HaltConfig.SUPPORT_THRESHOLD
) -> List[HaltClaim]:
    """
    Phase 5: Verification Gate - HALLUCINATION KILL SWITCH
    
    Rules:
    - UNSUPPORTED claims CANNOT be stated as fact
    - DISPUTED claims must show both sides
    - Only SUPPORTED claims can be asserted
    """
    evidence_map = {ev.id: ev for ev in evidence}
    
    for claim in claims:
        if claim.claim_type != ClaimType.MUST_CITE:
            # Non-factual claims don't need verification
            claim.status = ClaimStatus.SUPPORTED
            continue
        
        claim_links = [l for l in links if l.claim_id == claim.id]
        
        # No evidence = UNSUPPORTED
        if not claim_links:
            claim.status = ClaimStatus.UNSUPPORTED
            claim.opinion = Opinion.vacuous()
            continue
        
        # Fuse all evidence opinions
        fused_opinion = fuse_evidence_opinions(claim, links, evidence_map)
        claim.opinion = fused_opinion
        
        # Check for conflicts
        conflicts = [l for l in claim_links if l.conflict_flag]
        conflict_ratio = len(conflicts) / len(claim_links)
        
        if conflict_ratio > HaltConfig.CONFLICT_THRESHOLD:
            claim.status = ClaimStatus.DISPUTED
        elif fused_opinion.belief >= threshold:
            claim.status = ClaimStatus.SUPPORTED
        elif fused_opinion.disbelief >= threshold:
            # More evidence against than for
            claim.status = ClaimStatus.DISPUTED
        else:
            claim.status = ClaimStatus.UNSUPPORTED
    
    return claims


# =============================================================================
# PHASE 6: CONTROLLED GENERATION (ONLY SUPPORTED CLAIMS)
# =============================================================================

def generate_controlled_answer(
    claims: List[HaltClaim],
    links: List[EvidenceLink],
    evidence: List[HaltEvidence]
) -> Tuple[str, List[str]]:
    """
    Phase 6: Generate answer ONLY from supported claims with citations.
    
    Returns:
    - answer_text: The generated answer with evidence content
    - abstentions: List of claims we couldn't verify
    
    RULE: Cannot emit unsupported claims as facts.
    """
    evidence_map = {ev.id: ev for ev in evidence}
    answer_parts = []
    abstentions = []
    
    for claim in claims:
        if claim.status == ClaimStatus.SUPPORTED:
            # Find best evidence for citation
            best_ev = _get_best_evidence(claim.id, links, evidence_map)
            if best_ev:
                # Include the actual evidence content as the answer
                answer_parts.append(
                    f"{best_ev.content}\n\n[Source: {best_ev.source_id}]"
                )
            else:
                answer_parts.append(f"[SUPPORTED] {claim.text}")
                
        elif claim.status == ClaimStatus.DISPUTED:
            # Show both sides
            supporting, contradicting = _get_dispute_sides(claim.id, links, evidence_map)
            answer_parts.append(
                f"[DISPUTED] {claim.text}\n"
                f"   Supporting: {supporting}\n"
                f"   Contradicting: {contradicting}"
            )
            
        elif claim.status == ClaimStatus.UNSUPPORTED:
            # CANNOT state as fact - mark as abstention
            answer_parts.append(
                f"[INSUFFICIENT EVIDENCE] Cannot verify: {claim.text}"
            )
            abstentions.append(claim.text)
    
    return "\n\n".join(answer_parts), abstentions


def _get_best_evidence(
    claim_id: str,
    links: List[EvidenceLink],
    evidence_map: Dict[str, HaltEvidence]
) -> Optional[HaltEvidence]:
    """Get the highest-quality supporting evidence for a claim."""
    claim_links = [l for l in links if l.claim_id == claim_id and l.nli_label == NLILabel.ENTAILS]
    
    if not claim_links:
        return None
    
    # Sort by support strength
    claim_links.sort(key=lambda l: l.support_strength, reverse=True)
    best_link = claim_links[0]
    
    return evidence_map.get(best_link.evidence_id)


def _get_dispute_sides(
    claim_id: str,
    links: List[EvidenceLink],
    evidence_map: Dict[str, HaltEvidence]
) -> Tuple[str, str]:
    """Get summaries of supporting and contradicting evidence."""
    supporting = []
    contradicting = []
    
    for link in links:
        if link.claim_id != claim_id:
            continue
        ev = evidence_map.get(link.evidence_id)
        if not ev:
            continue
            
        if link.nli_label == NLILabel.ENTAILS:
            supporting.append(f"{ev.source_id}: \"{ev.span[:50]}...\"")
        elif link.nli_label == NLILabel.CONTRADICTS:
            contradicting.append(f"{ev.source_id}: \"{ev.span[:50]}...\"")
    
    return (
        "; ".join(supporting[:2]) if supporting else "None",
        "; ".join(contradicting[:2]) if contradicting else "None"
    )


# =============================================================================
# PHASE 7: AUDIT & CALIBRATION
# =============================================================================

def calibrate_confidence(
    claims: List[HaltClaim],
    links: List[EvidenceLink],
    calibrator: Optional[UncertaintyCalibrator] = None
) -> Tuple[float, float]:
    """
    Phase 7: Compute calibrated overall confidence score.
    
    Returns:
    - overall_confidence: Combined confidence score (0-1)
    - coverage_ratio: Proportion of required claims that are supported
    
    Formula:
    conf_total = weighted_mean(conf(c)) * coverage_penalty * conflict_penalty
    """
    if calibrator is None:
        calibrator = SimpleCalibrator()
    
    # Calibrate each claim
    for claim in claims:
        claim_links = [l for l in links if l.claim_id == claim.id]
        claim.confidence = calibrator.calibrate(claim, claim_links)
    
    # Calculate coverage
    required = [c for c in claims if c.claim_type == ClaimType.MUST_CITE]
    supported = [c for c in required if c.status == ClaimStatus.SUPPORTED]
    
    if not required:
        coverage_ratio = 1.0
    else:
        coverage_ratio = len(supported) / len(required)
    
    # Calculate conflict penalty
    disputed = [c for c in claims if c.status == ClaimStatus.DISPUTED]
    conflict_penalty = 1 / (1 + len(disputed))
    
    # Weighted mean of confidences
    if supported:
        weighted_conf = sum(c.confidence for c in supported) / len(supported)
    else:
        weighted_conf = 0.0
    
    # Overall confidence
    overall = weighted_conf * coverage_ratio * conflict_penalty
    
    return overall, coverage_ratio


def decide_action(
    coverage_ratio: float,
    conflict_count: int,
    overall_confidence: float
) -> ActionDecision:
    """
    Decide pipeline action based on verification results.
    
    ABSTAIN is a feature, not a failure.
    """
    if coverage_ratio < HaltConfig.COVERAGE_THRESHOLD:
        return ActionDecision.ABSTAIN
    elif conflict_count > 2:
        return ActionDecision.ANSWER_WITH_DISPUTE
    elif overall_confidence < HaltConfig.CONFIDENCE_THRESHOLD:
        return ActionDecision.ASK_FOR_INFO
    else:
        return ActionDecision.ANSWER


def suggest_next_actions(audit: AnswerAudit) -> List[str]:
    """Suggest what would raise confidence."""
    suggestions = []
    
    if audit.coverage_ratio < 1.0:
        unsupported = [c for c in audit.claims if c.status == ClaimStatus.UNSUPPORTED]
        for claim in unsupported[:2]:
            suggestions.append(f"Find evidence for: \"{claim.text[:50]}...\"")
    
    if audit.conflict_count > 0:
        suggestions.append("Resolve conflicting sources")
    
    if audit.overall_confidence < 0.7:
        suggestions.append("Retrieve more authoritative (Tier A) sources")
    
    return suggestions


# =============================================================================
# MAIN PIPELINE ORCHESTRATOR
# =============================================================================

def run_halt_pipeline(
    query: str,
    evidence_store: List[HaltEvidence],
    retriever: Optional[EmbeddingRetriever] = None,
    nli_model: Optional[NLIModel] = None,
    calibrator: Optional[UncertaintyCalibrator] = None
) -> AnswerAudit:
    """
    Execute full HALT-NN pipeline.
    
    Phases:
    1. Intent Analysis
    2. Claim Decomposition
    3. Evidence Retrieval
    4. Evidence Graph Construction
    5. Verification Gate (KILL SWITCH)
    6. Controlled Generation
    7. Audit & Calibration
    """
    # Phase 1: Intent
    intent = analyze_intent(query)
    
    # Phase 2: Decompose
    claims = decompose_claims(query, intent)
    
    # Phase 3: Retrieve
    evidence = retrieve_evidence(claims, evidence_store, retriever)
    
    # Phase 4: Build Graph
    links = build_evidence_graph(claims, evidence, nli_model)
    
    # Phase 5: Verify (KILL SWITCH)
    claims = verify_gate(claims, links, evidence)
    
    # Phase 6: Generate
    answer_text, abstentions = generate_controlled_answer(claims, links, evidence)
    
    # Phase 7: Calibrate
    confidence, coverage = calibrate_confidence(claims, links, calibrator)
    
    # Count conflicts
    conflict_count = sum(1 for c in claims if c.status == ClaimStatus.DISPUTED)
    
    # Decide action
    action = decide_action(coverage, conflict_count, confidence)
    
    # Build audit
    audit = AnswerAudit(
        query=query,
        claims=claims,
        evidence=evidence,
        links=links,
        action=action,
        overall_confidence=confidence,
        coverage_ratio=coverage,
        conflict_count=conflict_count,
        answer_text=answer_text,
        abstentions=abstentions
    )
    
    # Suggest improvements
    audit.next_actions = suggest_next_actions(audit)
    
    return audit

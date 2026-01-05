"""
Optimized HALT-NN Pipeline

Performance improvements:
- Parallel NLI inference
- Early termination for high-confidence claims
- Lazy evidence graph building
- Cached NLI results
"""

import concurrent.futures
from typing import List, Dict, Optional, Tuple
from functools import lru_cache
import hashlib

from .models import (
    Opinion, HaltClaim, HaltEvidence, EvidenceLink, 
    ClaimStatus, NLILabel, NLIResult, ActionDecision, IntentType, IntentResult
)
from .logic import discount_opinion, consensus_opinion
from .halt_pipeline import (
    HaltConfig, NLIModel, SimpleNLI, nli_to_opinion,
    fuse_evidence_opinions, verify_gate, generate_controlled_answer,
    calibrate_confidence, decide_action
)


class OptimizedHaltPipeline:
    """
    Optimized HALT-NN pipeline with performance enhancements.
    """
    
    def __init__(self, nli_model: Optional[NLIModel] = None, max_workers: int = 4):
        self.nli_model = nli_model or SimpleNLI()
        self.max_workers = max_workers
        self._nli_cache: Dict[str, NLIResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _cache_key(self, premise: str, hypothesis: str) -> str:
        """Generate cache key for NLI pair."""
        text = f"{premise[:200]}|||{hypothesis[:200]}"
        return hashlib.md5(text.encode()).hexdigest()
    
    def _cached_nli(self, premise: str, hypothesis: str) -> NLIResult:
        """NLI prediction with caching."""
        key = self._cache_key(premise, hypothesis)
        
        if key in self._nli_cache:
            self._cache_hits += 1
            return self._nli_cache[key]
        
        self._cache_misses += 1
        result = self.nli_model.predict(premise, hypothesis)
        self._nli_cache[key] = result
        return result
    
    def build_evidence_graph_fast(
        self,
        claims: List[HaltClaim],
        evidence: List[HaltEvidence]
    ) -> List[EvidenceLink]:
        """
        Optimized evidence graph building with:
        - Parallel NLI inference
        - Early termination for strong matches
        - Caching
        """
        if not claims or not evidence:
            return []
        
        # Prepare all pairs for batch processing
        pairs = []
        for claim in claims:
            for ev in evidence:
                pairs.append((claim, ev))
        
        # Process in parallel
        links = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_pair, claim, ev): (claim, ev)
                for claim, ev in pairs
            }
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    links.append(result)
        
        return links
    
    def _process_pair(self, claim: HaltClaim, ev: HaltEvidence) -> Optional[EvidenceLink]:
        """Process a single claim-evidence pair."""
        nli_result = self._cached_nli(premise=ev.span, hypothesis=claim.text)
        
        # Skip weak neutral connections
        if nli_result.label == NLILabel.NEUTRAL and nli_result.probability < 0.9:
            return None
        
        opinion = nli_to_opinion(nli_result, ev.reliability)
        
        link = EvidenceLink(
            claim_id=claim.id,
            evidence_id=ev.id,
            nli_label=nli_result.label,
            nli_probability=nli_result.probability,
            support_strength=opinion.belief,
            conflict_flag=(nli_result.label == NLILabel.CONTRADICTS)
        )
        
        # Track evidence on claim
        if nli_result.label != NLILabel.NEUTRAL:
            claim.evidence_ids.append(ev.id)
        
        return link
    
    def run_optimized(
        self,
        query: str,
        evidence: List[HaltEvidence]
    ) -> Tuple[str, float, float, ActionDecision]:
        """
        Run optimized HALT pipeline.
        
        Returns: (answer_text, confidence, coverage, action)
        """
        # Phase 1: Quick intent analysis
        intent = self._quick_intent(query)
        
        # Phase 2: Simple claim extraction
        claims = self._extract_claims(query, intent)
        
        # Phase 3: Evidence retrieval (keep top relevant)
        relevant_evidence = self._filter_evidence(claims, evidence)
        
        # Phase 4: Optimized graph building
        links = self.build_evidence_graph_fast(claims, relevant_evidence)
        
        # Phase 5: Verification
        claims = verify_gate(claims, links, relevant_evidence)
        
        # Phase 6: Answer generation
        answer_text, abstentions = generate_controlled_answer(claims, links, relevant_evidence)
        
        # Phase 7: Calibration
        confidence, coverage = calibrate_confidence(claims, links)
        conflict_count = sum(1 for c in claims if c.status == ClaimStatus.DISPUTED)
        action = decide_action(coverage, conflict_count, confidence)
        
        return answer_text, confidence, coverage, action
    
    def _quick_intent(self, query: str) -> IntentResult:
        """Fast intent detection."""
        query_lower = query.lower()
        
        if query_lower.startswith(("what", "who", "where", "when", "how")):
            return IntentResult(intent_type=IntentType.FACT_LOOKUP)
        elif "?" in query:
            return IntentResult(intent_type=IntentType.EXPLANATION)
        else:
            return IntentResult(intent_type=IntentType.FACT_LOOKUP)
    
    def _extract_claims(self, query: str, intent: IntentResult) -> List[HaltClaim]:
        """Fast claim extraction."""
        import uuid
        from .models import ClaimType
        
        # Use high stakes flag to determine initial confidence
        base_confidence = 0.5 if intent.high_stakes else 0.7
        
        # Map intent to claim type
        claim_type = ClaimType.MUST_CITE
        if intent.intent_type == IntentType.EXPLANATION:
            claim_type = ClaimType.DERIVATION
        
        claim = HaltClaim(
            id=str(uuid.uuid4()),
            text=query,
            claim_type=claim_type,
            confidence=base_confidence,
            status=ClaimStatus.UNSUPPORTED,
            evidence_ids=[]
        )
        return [claim]
    
    def _filter_evidence(
        self, 
        claims: List[HaltClaim], 
        evidence: List[HaltEvidence],
        top_k: int = 10
    ) -> List[HaltEvidence]:
        """Quick evidence filtering using keyword matching."""
        if len(evidence) <= top_k:
            return evidence
        
        # Extract keywords from claims
        claim_words = set()
        for claim in claims:
            words = claim.text.lower().split()
            claim_words.update(w for w in words if len(w) > 3)
        
        # Score evidence by keyword overlap
        scored = []
        for ev in evidence:
            ev_words = set(ev.span.lower().split())
            overlap = len(claim_words & ev_words)
            scored.append((overlap, ev))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ev for _, ev in scored[:top_k]]
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        
        return {
            "nli_cache_size": len(self._nli_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": f"{hit_rate:.1%}"
        }
    
    def clear_cache(self):
        """Clear NLI cache."""
        self._nli_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


# Singleton instance
_optimized_pipeline = None

def get_optimized_pipeline(nli_model: Optional[NLIModel] = None) -> OptimizedHaltPipeline:
    """Get singleton optimized pipeline."""
    global _optimized_pipeline
    if _optimized_pipeline is None:
        _optimized_pipeline = OptimizedHaltPipeline(nli_model)
    return _optimized_pipeline


def run_halt_optimized(
    query: str,
    evidence: List[HaltEvidence],
    nli_model: Optional[NLIModel] = None
) -> Tuple[str, float, float, ActionDecision]:
    """
    Convenience function to run optimized HALT pipeline.
    
    Returns: (answer_text, confidence, coverage, action)
    """
    pipeline = get_optimized_pipeline(nli_model)
    return pipeline.run_optimized(query, evidence)

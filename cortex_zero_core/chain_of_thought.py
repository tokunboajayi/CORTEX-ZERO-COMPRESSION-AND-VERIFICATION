"""
Chain-of-Thought Verification (CoT-V) Module

Extends HALT-NN with:
- Reasoning chain extraction
- Multi-hop evidence verification
- Weak link detection
- Chain confidence propagation
"""

import re
import uuid
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from .models import HaltClaim, HaltEvidence, ClaimStatus, NLILabel


class StepType(str, Enum):
    """Type of reasoning step."""
    PREMISE = "premise"           # Starting assumption
    INFERENCE = "inference"       # Derived from previous
    CONCLUSION = "conclusion"     # Final claim
    EVIDENCE_BASED = "evidence"   # Directly from evidence


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""
    id: str
    text: str
    step_type: StepType
    confidence: float = 0.0
    evidence_ids: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)  # IDs of previous steps
    status: str = "unverified"
    
    @classmethod
    def create(cls, text: str, step_type: StepType, depends_on: List[str] = None):
        return cls(
            id=str(uuid.uuid4())[:8],
            text=text,
            step_type=step_type,
            depends_on=depends_on or []
        )


@dataclass
class ReasoningChain:
    """A complete chain of reasoning steps."""
    query: str
    steps: List[ReasoningStep]
    chain_confidence: float = 0.0
    weak_link_index: Optional[int] = None
    explanation: str = ""
    
    def get_step(self, step_id: str) -> Optional[ReasoningStep]:
        for step in self.steps:
            if step.id == step_id:
                return step
        return None


class ChainExtractor:
    """
    Extract reasoning chains from queries.
    
    Identifies logical structure in complex claims:
    - "A because B" → [B is premise, A is conclusion]
    - "If A then B" → [A is premise, B is inference]
    - "A, therefore B" → [A is premise, B is conclusion]
    """
    
    # Causal connectors
    BECAUSE_PATTERNS = [
        r"(.+?)\s+because\s+(.+)",
        r"(.+?)\s+since\s+(.+)",
        r"(.+?)\s+as\s+(.+)",
        r"(.+?)\s+due to\s+(.+)",
    ]
    
    THEREFORE_PATTERNS = [
        r"(.+?),?\s+therefore\s+(.+)",
        r"(.+?),?\s+so\s+(.+)",
        r"(.+?),?\s+thus\s+(.+)",
        r"(.+?),?\s+hence\s+(.+)",
    ]
    
    CONDITIONAL_PATTERNS = [
        r"if\s+(.+?),?\s+then\s+(.+)",
        r"when\s+(.+?),?\s+(.+)",
        r"given\s+(.+?),?\s+(.+)",
    ]
    
    def extract(self, query: str) -> ReasoningChain:
        """Extract reasoning chain from query."""
        query = query.strip()
        steps = []
        
        # Try causal patterns first (because, since)
        for pattern in self.BECAUSE_PATTERNS:
            match = re.match(pattern, query, re.IGNORECASE)
            if match:
                conclusion, premise = match.groups()
                premise_step = ReasoningStep.create(premise.strip(), StepType.PREMISE)
                conclusion_step = ReasoningStep.create(
                    conclusion.strip(), 
                    StepType.CONCLUSION,
                    depends_on=[premise_step.id]
                )
                steps = [premise_step, conclusion_step]
                break
        
        # Try therefore patterns
        if not steps:
            for pattern in self.THEREFORE_PATTERNS:
                match = re.match(pattern, query, re.IGNORECASE)
                if match:
                    premise, conclusion = match.groups()
                    premise_step = ReasoningStep.create(premise.strip(), StepType.PREMISE)
                    conclusion_step = ReasoningStep.create(
                        conclusion.strip(),
                        StepType.CONCLUSION,
                        depends_on=[premise_step.id]
                    )
                    steps = [premise_step, conclusion_step]
                    break
        
        # Try conditional patterns
        if not steps:
            for pattern in self.CONDITIONAL_PATTERNS:
                match = re.match(pattern, query, re.IGNORECASE)
                if match:
                    condition, result = match.groups()
                    condition_step = ReasoningStep.create(condition.strip(), StepType.PREMISE)
                    result_step = ReasoningStep.create(
                        result.strip(),
                        StepType.INFERENCE,
                        depends_on=[condition_step.id]
                    )
                    steps = [condition_step, result_step]
                    break
        
        # Try compound statements (A and B)
        if not steps and " and " in query.lower():
            parts = re.split(r'\s+and\s+', query, flags=re.IGNORECASE)
            if len(parts) >= 2:
                for part in parts:
                    step = ReasoningStep.create(part.strip(), StepType.PREMISE)
                    steps.append(step)
        
        # Default: single claim
        if not steps:
            steps = [ReasoningStep.create(query, StepType.EVIDENCE_BASED)]
        
        return ReasoningChain(query=query, steps=steps)


class MultiHopVerifier:
    """
    Verify reasoning chains with multi-hop evidence.
    
    For each step:
    1. Find supporting evidence
    2. Verify via NLI
    3. Propagate confidence through chain
    """
    
    def __init__(self, nli_model=None):
        self.nli_model = nli_model
        if nli_model is None:
            from .halt_pipeline import SimpleNLI
            self.nli_model = SimpleNLI()
    
    def verify_chain(
        self, 
        chain: ReasoningChain, 
        evidence: List[HaltEvidence]
    ) -> ReasoningChain:
        """Verify each step in the reasoning chain."""
        
        # Verify each step
        for i, step in enumerate(chain.steps):
            step_conf, step_evidence = self._verify_step(step, evidence, chain)
            step.confidence = step_conf
            step.evidence_ids = step_evidence
            step.status = "verified" if step_conf > 0.5 else "unverified"
        
        # Compute chain confidence (product of all steps)
        confidences = [s.confidence for s in chain.steps if s.confidence > 0]
        if confidences:
            # Chain confidence = geometric mean (less harsh than product)
            from math import prod
            chain.chain_confidence = prod(confidences) ** (1/len(confidences))
        else:
            chain.chain_confidence = 0.0
        
        # Find weak link
        if chain.steps:
            min_conf = min(s.confidence for s in chain.steps)
            for i, step in enumerate(chain.steps):
                if step.confidence == min_conf:
                    chain.weak_link_index = i
                    break
        
        # Generate explanation
        chain.explanation = self._generate_explanation(chain)
        
        return chain
    
    def _verify_step(
        self, 
        step: ReasoningStep, 
        evidence: List[HaltEvidence],
        chain: ReasoningChain
    ) -> Tuple[float, List[str]]:
        """Verify a single reasoning step."""
        
        # For inference steps, check dependency first
        if step.depends_on:
            dependency_ok = all(
                chain.get_step(dep_id) and chain.get_step(dep_id).confidence > 0.3
                for dep_id in step.depends_on
            )
            if not dependency_ok:
                return 0.1, []  # Dependencies not satisfied
        
        # Find matching evidence
        matched_evidence = []
        max_confidence = 0.0
        
        for ev in evidence:
            nli_result = self.nli_model.predict(ev.span, step.text)
            
            if nli_result.label == NLILabel.ENTAILS:
                score = nli_result.probability * ev.reliability
                if score > max_confidence:
                    max_confidence = score
                matched_evidence.append(ev.id)
            elif nli_result.label == NLILabel.CONTRADICTS:
                max_confidence = max(0.0, max_confidence - 0.3)
        
        return max_confidence, matched_evidence
    
    def _generate_explanation(self, chain: ReasoningChain) -> str:
        """Generate human-readable explanation of verification."""
        lines = []
        
        for i, step in enumerate(chain.steps):
            status_icon = "✅" if step.status == "verified" else "❌"
            conf_str = f"{step.confidence*100:.0f}%"
            
            step_str = f"Step {i+1}: \"{step.text[:50]}...\" {status_icon} {conf_str}"
            
            if i == chain.weak_link_index:
                step_str += " ⚠️ WEAK LINK"
            
            lines.append(step_str)
        
        lines.append(f"\nChain Confidence: {chain.chain_confidence*100:.0f}%")
        
        if chain.weak_link_index is not None:
            weak = chain.steps[chain.weak_link_index]
            lines.append(f"Weak Link: Step {chain.weak_link_index + 1} - needs more evidence")
        
        return "\n".join(lines)


class HALTCoTEngine:
    """
    Combined HALT-NN + Chain-of-Thought engine.
    
    Provides enhanced verification with reasoning chain analysis.
    """
    
    def __init__(self, nli_model=None):
        self.chain_extractor = ChainExtractor()
        self.multi_hop_verifier = MultiHopVerifier(nli_model)
    
    def verify_with_reasoning(
        self,
        query: str,
        evidence: List[HaltEvidence]
    ) -> Dict:
        """
        Run enhanced verification with chain-of-thought analysis.
        
        Returns dict with:
        - chain: The reasoning chain
        - chain_confidence: Overall chain confidence
        - weak_link: The weakest step (if any)
        - explanation: Human-readable analysis
        - action: ANSWER/ABSTAIN based on chain confidence
        """
        
        # Extract reasoning chain
        chain = self.chain_extractor.extract(query)
        
        # Verify each step
        verified_chain = self.multi_hop_verifier.verify_chain(chain, evidence)
        
        # Determine action
        if verified_chain.chain_confidence >= 0.6:
            action = "ANSWER"
        elif verified_chain.chain_confidence >= 0.3:
            action = "HEDGE"
        else:
            action = "ABSTAIN"
        
        # Build answer
        if action == "ANSWER":
            verified_steps = [s for s in chain.steps if s.status == "verified"]
            answer_text = " → ".join(s.text for s in verified_steps)
        elif action == "HEDGE":
            answer_text = f"[UNCERTAIN] {query}"
        else:
            answer_text = f"[INSUFFICIENT EVIDENCE] Cannot verify reasoning chain"
        
        return {
            "query": query,
            "answer_text": answer_text,
            "action": action,
            "chain_confidence": verified_chain.chain_confidence,
            "steps": [
                {
                    "text": s.text,
                    "type": s.step_type.value,
                    "confidence": s.confidence,
                    "status": s.status,
                    "evidence_count": len(s.evidence_ids)
                }
                for s in verified_chain.steps
            ],
            "weak_link": chain.weak_link_index,
            "explanation": verified_chain.explanation,
            "is_multi_step": len(chain.steps) > 1
        }


# Convenience function
def verify_with_cot(query: str, evidence: List[HaltEvidence]) -> Dict:
    """Run Chain-of-Thought enhanced verification."""
    engine = HALTCoTEngine()
    return engine.verify_with_reasoning(query, evidence)

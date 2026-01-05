import networkx as nx
import hashlib
import time
import numpy as np
from typing import List, Dict, Optional

from .models import AtomicClaim, Evidence, Source, Opinion, Polarity
from .logic import discount_opinion, consensus_opinion, calculate_ess

class CortexGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        # Map claim_id to AtomicClaim object for easy retrieval
        self.claims: Dict[str, AtomicClaim] = {}

    def _generate_claim_id(self, subject: str, predicate: str, object: str) -> str:
        raw = f"{subject}:{predicate}:{object}".lower().encode('utf-8')
        return hashlib.sha256(raw).hexdigest()

    def add_claim(self, subject: str, predicate: str, object: str) -> AtomicClaim:
        """
        Add a new claim to the graph. If it exists, return the existing one.
        """
        claim_id = self._generate_claim_id(subject, predicate, object)
        
        if claim_id in self.claims:
            return self.claims[claim_id]
        
        # Create new claim
        claim = AtomicClaim(
            id=claim_id,
            subject=subject,
            predicate=predicate,
            object=object
        )
        
        self.claims[claim_id] = claim
        
        # Add to graph
        self.graph.add_node(subject, type='entity')
        self.graph.add_node(object, type='entity')
        self.graph.add_edge(subject, object, key=claim_id, predicate=predicate, claim=claim)
        
        return claim

    def ingest_evidence(self, claim_id: str, evidence: Evidence, source: Source) -> AtomicClaim:
        """
        Process new evidence for a claim, updating its opinion and ESS.
        """
        if claim_id not in self.claims:
            raise ValueError(f"Claim ID {claim_id} not found in graph.")
        
        claim = self.claims[claim_id]
        
        # 1. Convert Evidence Polarity to Raw Opinion
        if evidence.polarity == Polarity.SUPPORTS:
            raw_opinion = Opinion(belief=0.9, disbelief=0.0, uncertainty=0.1)
        elif evidence.polarity == Polarity.CONTRADICTS:
            raw_opinion = Opinion(belief=0.0, disbelief=0.9, uncertainty=0.1)
        else:
            # Fallback for unknown polarity, though Enum prevents this mostly
            raw_opinion = Opinion(belief=0.0, disbelief=0.0, uncertainty=1.0)

        # 2. Discount Opinion by Source Reliability
        discounted_op = discount_opinion(raw_opinion, source.reliability)
        
        # 3. Fuse with Current Opinion
        # If it's the first opinion (u=1.0, b=0, d=0), consensus will just take the new one effectively
        new_opinion = consensus_opinion(claim.current_opinion, discounted_op)
        claim.current_opinion = new_opinion
        
        # 4. Calculate Volatility
        # Volatility = Standard Deviation of past ESS scores
        # If not enough history, volatility is low or 0.
        if len(claim.history) < 2:
            volatility = 0.0
        else:
            # Get just the scores
            scores = [score for _, score in claim.history]
            volatility = float(np.std(scores))
            
        # 5. Calculate ESS
        age = time.time() - claim.creation_time
        # For simulation purposes, we might want to allow 'time' to be controlled, 
        # but the prompt implies real-time or simulation-time handling. 
        # The Evidence has a timestamp. We should probably use that to calculate age relative to creation?
        # But 'age' in calculating ESS usually refers to the claim's lifespan.
        # Let's use max(0, evidence.timestamp - claim.creation_time) if evidence is the "now" marker.
        # However, usually we just want the current state.
        # Let's use the evidence timestamp as the "current time" for this update.
        current_time = evidence.timestamp
        age_seconds = max(0, current_time - claim.creation_time)
        # Scale age? The formula `log(1 + age)` implies age is maybe in generic units or days?
        # If seconds, log(1+10000) is ~9. log(1+1) is ~0.
        # Let's assume generic time units from simulation.py.
        
        new_ess = calculate_ess(new_opinion, age_seconds, volatility)
        
        # Update Claim
        claim.current_ess = new_ess
        claim.history.append((current_time, new_ess))
        
        return claim

    def detect_contradictions(self, subject: str, predicate: str) -> List[AtomicClaim]:
        """
        Find claims that share the Subject and Predicate but have different Objects.
        """
        matches = []
        # Iterate over all claims (inefficient for valid graph DB, okay for MVP)
        # Using graph structure is better:
        # Outgoing edges from Subject
        if self.graph.has_node(subject):
            for _, neighbor, key, data in self.graph.out_edges(subject, keys=True, data=True):
                claim = data.get('claim')
                if claim and claim.predicate == predicate:
                    matches.append(claim)
        
        return matches

    def get_claim(self, claim_id: str) -> Optional[AtomicClaim]:
        return self.claims.get(claim_id)

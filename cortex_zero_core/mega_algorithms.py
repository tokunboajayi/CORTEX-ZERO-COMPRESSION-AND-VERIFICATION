"""
HALT-NN Mega Algorithms Module

10 Enhancement Algorithms for Truth Verification:
1. Knowledge Graph Reasoning (KGR)
2. Semantic Deduplication (SD)
3. Temporal Decay (TD)
4. Adversarial Detection (AD)
5. Active Learning (AL)
6. Hierarchical Attention (HA)
7. Source Reputation Learning (SRL)
8. Query Expansion (QE)
9. Ensemble Verification (EV)
10. Uncertainty Quantification (UQ)
"""

import re
import math
import hashlib
import time
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta
import random

from .models import HaltEvidence, HaltClaim, NLILabel, SourceTier


# =============================================================================
# 1. KNOWLEDGE GRAPH REASONING (KGR)
# =============================================================================

class KnowledgeGraph:
    """
    Build and traverse knowledge graphs for transitive inference.
    
    Example:
        "Paris is capital of France" + "France is in Europe"
        → Infer: "Paris is in Europe"
    """
    
    def __init__(self):
        self.entities: Dict[str, Set[str]] = defaultdict(set)  # entity -> attributes
        self.relations: Dict[Tuple[str, str], str] = {}  # (subject, object) -> relation
        self.entity_embeddings: Dict[str, List[float]] = {}
    
    def add_fact(self, subject: str, relation: str, obj: str):
        """Add a fact to the knowledge graph."""
        self.entities[subject.lower()].add(relation)
        self.entities[obj.lower()].add(relation)
        self.relations[(subject.lower(), obj.lower())] = relation
    
    def extract_facts_from_evidence(self, evidence: List[HaltEvidence]):
        """Extract entity-relation-entity triples from evidence."""
        patterns = [
            # "X is Y" pattern
            (r"(\w+)\s+is\s+(?:a|an|the)?\s*(\w+)", "is_a"),
            # "X capital of Y"
            (r"(\w+)\s+(?:is\s+)?(?:the\s+)?capital\s+of\s+(\w+)", "capital_of"),
            # "X in Y"
            (r"(\w+)\s+(?:is\s+)?in\s+(\w+)", "located_in"),
            # "X created by Y"
            (r"(\w+)\s+(?:was\s+)?created\s+by\s+(\w+)", "created_by"),
            # "X part of Y"
            (r"(\w+)\s+(?:is\s+)?part\s+of\s+(\w+)", "part_of"),
        ]
        
        for ev in evidence:
            text = ev.span.lower()
            for pattern, relation in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match) == 2:
                        self.add_fact(match[0], relation, match[1])
    
    def infer_transitive(self, entity: str, relation: str, max_hops: int = 3) -> List[str]:
        """Find all entities reachable via transitive relation."""
        entity = entity.lower()
        visited = {entity}
        frontier = [entity]
        reachable = []
        
        for _ in range(max_hops):
            next_frontier = []
            for current in frontier:
                for (subj, obj), rel in self.relations.items():
                    if subj == current and rel == relation:
                        if obj not in visited:
                            visited.add(obj)
                            next_frontier.append(obj)
                            reachable.append(obj)
            frontier = next_frontier
        
        return reachable
    
    def can_infer(self, subject: str, relation: str, obj: str) -> Tuple[bool, float]:
        """Check if relation can be inferred, return (can_infer, confidence)."""
        subject, obj = subject.lower(), obj.lower()
        
        # Direct relation
        if (subject, obj) in self.relations:
            return True, 1.0
        
        # Transitive inference
        reachable = self.infer_transitive(subject, relation)
        if obj in reachable:
            hops = reachable.index(obj) + 1
            confidence = 0.9 ** hops  # Decay per hop
            return True, confidence
        
        return False, 0.0


# =============================================================================
# 2. SEMANTIC DEDUPLICATION (SD)
# =============================================================================

class SemanticDeduplicator:
    """
    Merge semantically similar evidence to avoid inflated confidence.
    
    Example:
        "Python was created by Guido"
        "Guido van Rossum made Python"
        → Same fact, count as 1 source
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold
        self._embedder = None
    
    def _get_embedder(self):
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                return None
        return self._embedder
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        embedder = self._get_embedder()
        if embedder is None:
            # Fallback: word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            return len(words1 & words2) / max(len(words1), len(words2))
        
        import numpy as np
        emb1 = embedder.encode(text1)
        emb2 = embedder.encode(text2)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def deduplicate(self, evidence: List[HaltEvidence]) -> List[HaltEvidence]:
        """Remove semantically duplicate evidence."""
        if not evidence:
            return []
        
        clusters: List[List[HaltEvidence]] = []
        
        for ev in evidence:
            merged = False
            for cluster in clusters:
                # Check similarity with cluster representative
                sim = self.compute_similarity(cluster[0].span, ev.span)
                if sim >= self.threshold:
                    cluster.append(ev)
                    merged = True
                    break
            
            if not merged:
                clusters.append([ev])
        
        # Return representative from each cluster (highest reliability)
        unique = []
        for cluster in clusters:
            best = max(cluster, key=lambda e: e.reliability)
            unique.append(best)
        
        return unique
    
    def get_dedup_stats(self, original: int, deduped: int) -> Dict:
        return {
            "original_count": original,
            "unique_count": deduped,
            "duplicates_removed": original - deduped,
            "reduction_pct": (original - deduped) / original * 100 if original > 0 else 0
        }


# =============================================================================
# 3. TEMPORAL DECAY (TD)
# =============================================================================

class TemporalDecay:
    """
    Reduce confidence for old evidence.
    
    Example:
        Evidence (2015): "Python 2.7 is latest" → 60% decay
        Evidence (2024): "Python 3.12 is latest" → 100% fresh
    """
    
    def __init__(self, half_life_years: float = 3.0, min_factor: float = 0.3):
        self.half_life_years = half_life_years
        self.min_factor = min_factor
    
    def compute_decay(self, evidence_date: datetime, current_date: datetime = None) -> float:
        """Compute decay factor based on age."""
        if current_date is None:
            current_date = datetime.now()
        
        age_days = (current_date - evidence_date).days
        age_years = age_days / 365.25
        
        # Exponential decay: factor = 0.5^(age/half_life)
        decay_factor = 0.5 ** (age_years / self.half_life_years)
        
        return max(decay_factor, self.min_factor)
    
    def apply_decay(self, evidence: HaltEvidence, evidence_date: datetime = None) -> HaltEvidence:
        """Apply temporal decay to evidence reliability."""
        if evidence_date is None:
            # Try to extract date from source_id or use default
            evidence_date = datetime.now() - timedelta(days=365)  # Assume 1 year old default
        
        decay = self.compute_decay(evidence_date)
        evidence.reliability *= decay
        return evidence
    
    def apply_to_list(self, evidence: List[HaltEvidence], dates: Dict[str, datetime] = None) -> List[HaltEvidence]:
        """Apply decay to list of evidence."""
        dates = dates or {}
        for ev in evidence:
            date = dates.get(ev.id, datetime.now() - timedelta(days=180))
            self.apply_decay(ev, date)
        return evidence


# =============================================================================
# 4. ADVERSARIAL DETECTION (AD)
# =============================================================================

class AdversarialDetector:
    """
    Detect and block malicious/poisoned evidence.
    
    Checks for:
    - Contradictions with established facts
    - Anomalous patterns (too good to be true)
    - Repetitive content (spam)
    """
    
    def __init__(self):
        self.known_false: Set[str] = set()
        self.fingerprints: Dict[str, int] = {}  # content hash -> count
        self.source_violations: Dict[str, int] = defaultdict(int)
    
    def compute_fingerprint(self, text: str) -> str:
        """Create normalized fingerprint of text."""
        normalized = ' '.join(text.lower().split())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def is_suspicious(self, evidence: HaltEvidence, existing: List[HaltEvidence]) -> Tuple[bool, str]:
        """Check if evidence is suspicious."""
        
        # Check for near-duplicates (spam detection)
        fingerprint = self.compute_fingerprint(evidence.span)
        if fingerprint in self.fingerprints:
            if self.fingerprints[fingerprint] >= 3:
                return True, "Duplicate content detected (spam)"
        self.fingerprints[fingerprint] = self.fingerprints.get(fingerprint, 0) + 1
        
        # Check for known false claims
        for false_claim in self.known_false:
            if false_claim.lower() in evidence.span.lower():
                return True, f"Contains known false claim"
        
        # Check for contradiction with high-reliability existing evidence
        for existing_ev in existing:
            if existing_ev.reliability >= 0.9:
                # Simple contradiction check (contains "not" + similar content)
                if self._check_contradiction(evidence.span, existing_ev.span):
                    return True, "Contradicts trusted evidence"
        
        # Check source violation history
        if self.source_violations[evidence.source_id] >= 3:
            return True, "Source has history of violations"
        
        return False, "OK"
    
    def _check_contradiction(self, text1: str, text2: str) -> bool:
        """Simple contradiction detection."""
        t1_lower = text1.lower()
        t2_lower = text2.lower()
        
        # Check for negation patterns
        negations = ["not", "never", "no ", "false", "isn't", "aren't", "doesn't"]
        
        t1_has_neg = any(neg in t1_lower for neg in negations)
        t2_has_neg = any(neg in t2_lower for neg in negations)
        
        # Similar content but different polarity
        words1 = set(t1_lower.split()) - {"not", "never", "no", "is", "are", "the", "a"}
        words2 = set(t2_lower.split()) - {"not", "never", "no", "is", "are", "the", "a"}
        
        overlap = len(words1 & words2) / max(len(words1), len(words2)) if words1 and words2 else 0
        
        if overlap > 0.5 and t1_has_neg != t2_has_neg:
            return True
        
        return False
    
    def report_violation(self, source_id: str):
        """Record a violation for a source."""
        self.source_violations[source_id] += 1
    
    def add_known_false(self, claim: str):
        """Add a known false claim to blocklist."""
        self.known_false.add(claim.lower())


# =============================================================================
# 5. ACTIVE LEARNING (AL)
# =============================================================================

class ActiveLearner:
    """
    Identify what evidence is missing and suggest additions.
    
    Example:
        Query: "What is quantum computing?"
        Response: ABSTAIN (no evidence)
        Suggestion: "Add evidence about: quantum computing, qubits, superposition"
    """
    
    def __init__(self):
        self.query_history: List[Dict] = []
        self.gap_analysis: Dict[str, int] = defaultdict(int)  # topic -> miss count
    
    def analyze_gap(self, query: str, evidence: List[HaltEvidence], coverage: float) -> Dict:
        """Analyze what evidence is missing for a query."""
        
        # Extract query topics
        query_words = set(query.lower().split())
        stopwords = {"what", "is", "the", "a", "an", "how", "why", "when", "where", "who"}
        topics = query_words - stopwords
        
        # Check which topics have evidence
        covered_topics = set()
        for ev in evidence:
            ev_words = set(ev.span.lower().split())
            covered_topics.update(topics & ev_words)
        
        missing_topics = topics - covered_topics
        
        # Track gaps
        for topic in missing_topics:
            self.gap_analysis[topic] += 1
        
        # Record query
        self.query_history.append({
            "query": query,
            "coverage": coverage,
            "missing": list(missing_topics),
            "timestamp": time.time()
        })
        
        return {
            "coverage": coverage,
            "covered_topics": list(covered_topics),
            "missing_topics": list(missing_topics),
            "suggestions": self._generate_suggestions(missing_topics)
        }
    
    def _generate_suggestions(self, missing: Set[str]) -> List[str]:
        """Generate evidence acquisition suggestions."""
        suggestions = []
        for topic in missing:
            suggestions.append(f"Add evidence about: {topic}")
        return suggestions
    
    def get_priority_topics(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently missing topics."""
        sorted_gaps = sorted(self.gap_analysis.items(), key=lambda x: x[1], reverse=True)
        return sorted_gaps[:top_n]


# =============================================================================
# 6. HIERARCHICAL ATTENTION (HA)
# =============================================================================

class HierarchicalAttention:
    """
    Focus on claim-relevant spans in long evidence.
    
    Example:
        Long evidence: "In 1991... [500 words] ...Python was created by Guido"
        Query: "Who created Python?"
        → Focus on "Python was created by Guido" span
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
    
    def extract_relevant_spans(self, evidence_text: str, query: str, n_spans: int = 3) -> List[Tuple[str, float]]:
        """Extract most relevant spans from evidence."""
        
        # Tokenize
        words = evidence_text.split()
        query_words = set(query.lower().split())
        stopwords = {"what", "is", "the", "a", "an", "how", "why", "when", "where", "who", "?"}
        query_terms = query_words - stopwords
        
        if not words or not query_terms:
            return [(evidence_text[:200], 1.0)]
        
        # Score each window
        spans = []
        for i in range(0, len(words), self.window_size // 2):
            window = words[i:i + self.window_size]
            window_text = ' '.join(window)
            window_lower = window_text.lower()
            
            # Score = term frequency in window
            score = sum(1 for term in query_terms if term in window_lower)
            score = score / len(query_terms) if query_terms else 0
            
            if score > 0:
                spans.append((window_text, score))
        
        # Sort by score and return top n
        spans.sort(key=lambda x: x[1], reverse=True)
        return spans[:n_spans] if spans else [(evidence_text[:200], 0.5)]
    
    def focus_evidence(self, evidence: HaltEvidence, query: str) -> HaltEvidence:
        """Replace evidence content with focused span."""
        spans = self.extract_relevant_spans(evidence.span, query)
        if spans:
            best_span, score = spans[0]
            evidence.span = best_span
            evidence.reliability *= (0.7 + 0.3 * score)  # Boost if highly relevant
        return evidence


# =============================================================================
# 7. SOURCE REPUTATION LEARNING (SRL)
# =============================================================================

class SourceReputationLearner:
    """
    Dynamically adjust source trust based on verification history.
    
    Example:
        source "fake-news.com" starts at tier B
        → 5 contradictions detected
        → Auto-downgrade to tier C (reliability 0.3)
    """
    
    def __init__(self):
        self.reputation_scores: Dict[str, float] = {}
        self.verification_history: Dict[str, List[Tuple[bool, float]]] = defaultdict(list)
        self.base_reliability = {
            "A": 0.95,
            "B": 0.75,
            "C": 0.50
        }
    
    def get_reputation(self, source_id: str, base_tier: str = "B") -> float:
        """Get current reputation score for source."""
        if source_id not in self.reputation_scores:
            self.reputation_scores[source_id] = self.base_reliability.get(base_tier, 0.75)
        return self.reputation_scores[source_id]
    
    def record_verification(self, source_id: str, was_correct: bool, confidence: float):
        """Record a verification result for learning."""
        self.verification_history[source_id].append((was_correct, confidence))
        self._update_reputation(source_id)
    
    def _update_reputation(self, source_id: str):
        """Update reputation based on history."""
        history = self.verification_history[source_id]
        if not history:
            return
        
        # Compute weighted accuracy (recent = more weight)
        weights = [0.9 ** i for i in range(len(history))]
        weights.reverse()  # Most recent = highest weight
        
        weighted_correct = sum(w * (1.0 if correct else 0.0) for w, (correct, _) in zip(weights, history))
        total_weight = sum(weights)
        
        accuracy = weighted_correct / total_weight if total_weight > 0 else 0.5
        
        # Update reputation (bounded 0.1 to 0.99)
        current = self.reputation_scores.get(source_id, 0.75)
        new_score = 0.7 * current + 0.3 * accuracy  # Exponential moving average
        self.reputation_scores[source_id] = max(0.1, min(0.99, new_score))
    
    def get_top_sources(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get most reputable sources."""
        sorted_sources = sorted(self.reputation_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_sources[:n]
    
    def get_flagged_sources(self, threshold: float = 0.4) -> List[Tuple[str, float]]:
        """Get sources with poor reputation."""
        return [(s, r) for s, r in self.reputation_scores.items() if r < threshold]


# =============================================================================
# 8. QUERY EXPANSION (QE)
# =============================================================================

class QueryExpander:
    """
    Expand queries with synonyms and related concepts.
    
    Example:
        Query: "What is ML?"
        Expanded: ["What is ML?", "What is machine learning?", 
                   "What is artificial intelligence?"]
    """
    
    def __init__(self):
        # Common abbreviations and synonyms
        self.expansions = {
            "ml": ["machine learning", "ml algorithms", "supervised learning"],
            "ai": ["artificial intelligence", "machine learning", "deep learning"],
            "nlp": ["natural language processing", "text processing", "language models"],
            "nn": ["neural network", "deep learning", "neural nets"],
            "dl": ["deep learning", "neural networks", "deep neural networks"],
            "cv": ["computer vision", "image recognition", "visual ai"],
            "python": ["python programming", "python language", "python3"],
            "js": ["javascript", "ecmascript", "node.js"],
            "db": ["database", "sql", "data storage"],
            "api": ["application programming interface", "rest api", "web service"],
        }
        
        # Conceptual hierarchy
        self.hierarchy = {
            "machine learning": ["artificial intelligence"],
            "deep learning": ["machine learning", "neural networks"],
            "python": ["programming language"],
            "tensorflow": ["machine learning framework", "python library"],
        }
    
    def expand(self, query: str, max_expansions: int = 5) -> List[str]:
        """Expand query with synonyms and related terms."""
        query_lower = query.lower()
        expansions = [query]  # Original query first
        
        # Check for known abbreviations
        for abbrev, synonyms in self.expansions.items():
            if abbrev in query_lower:
                for syn in synonyms[:2]:  # Limit synonyms per term
                    expanded = query_lower.replace(abbrev, syn)
                    if expanded not in expansions:
                        expansions.append(expanded)
        
        # Add hierarchical concepts
        for term, parents in self.hierarchy.items():
            if term in query_lower:
                for parent in parents:
                    expanded = f"{query} ({parent})"
                    if len(expansions) < max_expansions:
                        expansions.append(expanded)
        
        return expansions[:max_expansions]
    
    def add_expansion(self, term: str, synonyms: List[str]):
        """Add custom expansion mapping."""
        self.expansions[term.lower()] = synonyms


# =============================================================================
# 9. ENSEMBLE VERIFICATION (EV)
# =============================================================================

class EnsembleVerifier:
    """
    Multiple verification strategies vote on truth.
    
    Example:
        NLI Model: ENTAILS (80%)
        Keyword Match: ENTAILS (70%)
        KG Inference: ENTAILS (90%)
        → Ensemble: ENTAILS (80%) with high agreement
    """
    
    def __init__(self):
        self.verifiers: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
    
    def add_verifier(self, name: str, verifier_fn, weight: float = 1.0):
        """Add a verification function."""
        self.verifiers[name] = verifier_fn
        self.weights[name] = weight
    
    def verify(self, premise: str, hypothesis: str) -> Dict:
        """Run all verifiers and combine results."""
        results = {}
        
        for name, verifier in self.verifiers.items():
            try:
                result = verifier(premise, hypothesis)
                results[name] = {
                    "label": result.get("label", "neutral"),
                    "confidence": result.get("confidence", 0.5)
                }
            except Exception as e:
                results[name] = {"label": "error", "confidence": 0.0}
        
        # Combine results
        ensemble_result = self._combine_results(results)
        
        return {
            "individual": results,
            "ensemble": ensemble_result
        }
    
    def _combine_results(self, results: Dict) -> Dict:
        """Combine individual results into ensemble decision."""
        if not results:
            return {"label": "neutral", "confidence": 0.5, "agreement": 0.0}
        
        # Vote counting
        label_scores = defaultdict(float)
        total_weight = 0
        
        for name, result in results.items():
            if result["label"] != "error":
                weight = self.weights.get(name, 1.0)
                label_scores[result["label"]] += result["confidence"] * weight
                total_weight += weight
        
        if total_weight == 0:
            return {"label": "neutral", "confidence": 0.5, "agreement": 0.0}
        
        # Find winning label
        best_label = max(label_scores.keys(), key=lambda k: label_scores[k])
        best_score = label_scores[best_label] / total_weight
        
        # Compute agreement (how many verifiers agree)
        agreeing = sum(1 for r in results.values() if r["label"] == best_label)
        agreement = agreeing / len(results)
        
        return {
            "label": best_label,
            "confidence": best_score,
            "agreement": agreement
        }


# =============================================================================
# 10. UNCERTAINTY QUANTIFICATION (UQ)
# =============================================================================

class UncertaintyQuantifier:
    """
    Bayesian confidence intervals instead of point estimates.
    
    Example:
        Confidence: 75% ± 10% (65-85% range)
        Epistemic uncertainty: high (need more evidence)
        Aleatoric uncertainty: low (evidence is clear)
    """
    
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.prior_alpha = prior_alpha  # Beta distribution prior
        self.prior_beta = prior_beta
    
    def compute_confidence_interval(
        self, 
        successes: int, 
        total: int, 
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Compute Bayesian credible interval.
        
        Returns: (mean, lower_bound, upper_bound)
        """
        from scipy import stats
        
        # Posterior parameters (Beta-Binomial conjugate)
        alpha = self.prior_alpha + successes
        beta = self.prior_beta + (total - successes)
        
        # Mean (point estimate)
        mean = alpha / (alpha + beta)
        
        # Credible interval
        lower_percentile = (1 - confidence_level) / 2
        upper_percentile = 1 - lower_percentile
        
        try:
            lower = stats.beta.ppf(lower_percentile, alpha, beta)
            upper = stats.beta.ppf(upper_percentile, alpha, beta)
        except:
            # Fallback if scipy not available
            std = math.sqrt(alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1)))
            lower = max(0, mean - 2 * std)
            upper = min(1, mean + 2 * std)
        
        return mean, lower, upper
    
    def quantify_uncertainty(
        self, 
        supporting_evidence: int, 
        total_evidence: int,
        model_confidence: float
    ) -> Dict:
        """
        Quantify both epistemic and aleatoric uncertainty.
        
        Epistemic: Uncertainty from lack of knowledge (reducible)
        Aleatoric: Uncertainty from inherent randomness (irreducible)
        """
        
        # Compute credible interval
        mean, lower, upper = self.compute_confidence_interval(
            supporting_evidence, 
            max(total_evidence, 1)
        )
        
        # Epistemic uncertainty (width of interval - decreases with more evidence)
        interval_width = upper - lower
        epistemic = interval_width
        
        # Aleatoric uncertainty (model's inherent uncertainty)
        aleatoric = 1 - model_confidence
        
        # Total uncertainty
        total_uncertainty = math.sqrt(epistemic ** 2 + aleatoric ** 2)
        
        return {
            "point_estimate": mean,
            "lower_bound": lower,
            "upper_bound": upper,
            "interval_width": interval_width,
            "epistemic_uncertainty": epistemic,
            "aleatoric_uncertainty": aleatoric,
            "total_uncertainty": total_uncertainty,
            "confidence_level": 0.95,
            "needs_more_evidence": epistemic > 0.3
        }


# =============================================================================
# UNIFIED MEGA ENGINE
# =============================================================================

class MegaHALTEngine:
    """
    Unified engine combining all 10 mega algorithms with HALT-NN.
    """
    
    def __init__(self):
        # Initialize all algorithms
        self.knowledge_graph = KnowledgeGraph()
        self.deduplicator = SemanticDeduplicator()
        self.temporal_decay = TemporalDecay()
        self.adversarial_detector = AdversarialDetector()
        self.active_learner = ActiveLearner()
        self.attention = HierarchicalAttention()
        self.reputation_learner = SourceReputationLearner()
        self.query_expander = QueryExpander()
        self.ensemble_verifier = EnsembleVerifier()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        
        # Stats
        self.stats = {
            "queries_processed": 0,
            "duplicates_removed": 0,
            "adversarial_blocked": 0,
            "queries_expanded": 0
        }
    
    def process_evidence(self, evidence: List[HaltEvidence]) -> List[HaltEvidence]:
        """Pre-process evidence with all algorithms."""
        
        # 1. Adversarial detection
        safe_evidence = []
        for ev in evidence:
            is_suspicious, reason = self.adversarial_detector.is_suspicious(ev, safe_evidence)
            if not is_suspicious:
                safe_evidence.append(ev)
            else:
                self.stats["adversarial_blocked"] += 1
        
        # 2. Semantic deduplication
        original_count = len(safe_evidence)
        unique_evidence = self.deduplicator.deduplicate(safe_evidence)
        self.stats["duplicates_removed"] += original_count - len(unique_evidence)
        
        # 3. Temporal decay
        decayed_evidence = self.temporal_decay.apply_to_list(unique_evidence)
        
        # 4. Apply source reputation
        for ev in decayed_evidence:
            reputation = self.reputation_learner.get_reputation(ev.source_id)
            ev.reliability *= reputation
        
        # 5. Build knowledge graph
        self.knowledge_graph.extract_facts_from_evidence(decayed_evidence)
        
        return decayed_evidence
    
    def process_query(self, query: str, evidence: List[HaltEvidence]) -> Dict:
        """Process a query with all mega algorithms."""
        self.stats["queries_processed"] += 1
        
        # 1. Query expansion
        expanded_queries = self.query_expander.expand(query)
        self.stats["queries_expanded"] += len(expanded_queries) - 1
        
        # 2. Pre-process evidence
        processed_evidence = self.process_evidence(evidence)
        
        # 3. Apply hierarchical attention
        focused_evidence = []
        for ev in processed_evidence:
            for exp_query in expanded_queries:
                focused = self.attention.focus_evidence(ev, exp_query)
                if focused not in focused_evidence:
                    focused_evidence.append(focused)
        
        # 4. Active learning analysis
        coverage = len(focused_evidence) / max(len(processed_evidence), 1)
        learning_suggestions = self.active_learner.analyze_gap(query, focused_evidence, coverage)
        
        # 5. Uncertainty quantification
        supporting = sum(1 for ev in focused_evidence if ev.reliability > 0.6)
        uncertainty = self.uncertainty_quantifier.quantify_uncertainty(
            supporting, 
            len(focused_evidence),
            0.75  # Base model confidence
        )
        
        return {
            "original_query": query,
            "expanded_queries": expanded_queries,
            "evidence_count": {
                "original": len(evidence),
                "processed": len(processed_evidence),
                "focused": len(focused_evidence)
            },
            "coverage": coverage,
            "confidence": {
                "point": uncertainty["point_estimate"],
                "lower": uncertainty["lower_bound"],
                "upper": uncertainty["upper_bound"],
                "uncertainty": uncertainty["total_uncertainty"]
            },
            "learning_suggestions": learning_suggestions["suggestions"],
            "needs_more_evidence": uncertainty["needs_more_evidence"],
            "stats": self.stats.copy()
        }
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            **self.stats,
            "knowledge_graph_entities": len(self.knowledge_graph.entities),
            "knowledge_graph_relations": len(self.knowledge_graph.relations),
            "source_reputations": len(self.reputation_learner.reputation_scores),
            "flagged_sources": len(self.reputation_learner.get_flagged_sources())
        }


# Convenience function
def create_mega_engine() -> MegaHALTEngine:
    """Create and return the unified mega engine."""
    return MegaHALTEngine()

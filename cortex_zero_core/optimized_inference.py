"""
HALT-NN Optimized Inference Engine

Provides:
- Lazy model loading with caching
- LRU result caching
- Batch inference
- FP16 quantization (optional)
"""

import os
import time
import hashlib
from functools import lru_cache
from typing import List, Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Global model cache
_model_cache = {}


class OptimizedNLI:
    """Optimized NLI inference with caching and batching."""
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern for model reuse."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.model = None
        self.model_path = "./models/nli_trained"
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._initialized = True
        
    def _load_model(self):
        """Lazy load the model on first use."""
        if self.model is not None:
            return
            
        logger.info("Loading optimized NLI model...")
        start = time.time()
        
        try:
            from sentence_transformers import CrossEncoder
            
            if os.path.exists(self.model_path):
                self.model = CrossEncoder(self.model_path)
            else:
                self.model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
                
            logger.info(f"Model loaded in {time.time() - start:.2f}s")
            
        except ImportError:
            logger.warning("CrossEncoder not available, using fallback")
            self.model = None
    
    def _cache_key(self, premise: str, hypothesis: str) -> str:
        """Generate cache key for premise-hypothesis pair."""
        text = f"{premise}|||{hypothesis}"
        return hashlib.md5(text.encode()).hexdigest()[:16]
    
    def predict(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """
        Predict NLI relationship with caching.
        
        Returns:
            dict with 'entailment', 'contradiction', 'neutral' scores
        """
        # Check cache
        key = self._cache_key(premise, hypothesis)
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        
        self._cache_misses += 1
        
        # Load model if needed
        self._load_model()
        
        if self.model is None:
            # Fallback: simple keyword matching
            return self._fallback_predict(premise, hypothesis)
        
        # Run inference
        import numpy as np
        scores = self.model.predict([(premise, hypothesis)])
        
        # Convert to probabilities
        from scipy.special import softmax
        probs = softmax(scores[0])
        
        result = {
            "contradiction": float(probs[0]),
            "entailment": float(probs[1]),
            "neutral": float(probs[2])
        }
        
        # Cache result (limit cache size)
        if len(self._cache) > 10000:
            # Remove oldest entries
            keys = list(self._cache.keys())[:1000]
            for k in keys:
                del self._cache[k]
        
        self._cache[key] = result
        return result
    
    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
        """
        Batch predict for multiple premise-hypothesis pairs.
        
        Much faster than individual predictions.
        """
        self._load_model()
        
        if self.model is None or not pairs:
            return [self._fallback_predict(p, h) for p, h in pairs]
        
        # Check cache for each pair
        results = []
        uncached_pairs = []
        uncached_indices = []
        
        for i, (premise, hypothesis) in enumerate(pairs):
            key = self._cache_key(premise, hypothesis)
            if key in self._cache:
                results.append(self._cache[key])
                self._cache_hits += 1
            else:
                results.append(None)
                uncached_pairs.append((premise, hypothesis))
                uncached_indices.append(i)
                self._cache_misses += 1
        
        # Batch inference for uncached
        if uncached_pairs:
            import numpy as np
            from scipy.special import softmax
            
            scores = self.model.predict(uncached_pairs)
            
            for idx, (j, score) in enumerate(zip(uncached_indices, scores)):
                probs = softmax(score)
                result = {
                    "contradiction": float(probs[0]),
                    "entailment": float(probs[1]),
                    "neutral": float(probs[2])
                }
                
                # Update results and cache
                results[j] = result
                key = self._cache_key(uncached_pairs[idx][0], uncached_pairs[idx][1])
                self._cache[key] = result
        
        return results
    
    def _fallback_predict(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Simple keyword-based fallback when model unavailable."""
        premise_lower = premise.lower()
        hypothesis_lower = hypothesis.lower()
        
        # Check for word overlap
        premise_words = set(premise_lower.split())
        hypothesis_words = set(hypothesis_lower.split())
        
        overlap = len(premise_words & hypothesis_words)
        total = len(hypothesis_words)
        
        if total == 0:
            return {"entailment": 0.33, "contradiction": 0.33, "neutral": 0.34}
        
        overlap_ratio = overlap / total
        
        # Check for negation
        negations = {"not", "no", "never", "neither", "cannot", "don't", "doesn't", "isn't", "aren't"}
        has_negation = bool(hypothesis_words & negations) != bool(premise_words & negations)
        
        if has_negation and overlap_ratio > 0.3:
            return {"entailment": 0.1, "contradiction": 0.7, "neutral": 0.2}
        elif overlap_ratio > 0.5:
            return {"entailment": 0.7, "contradiction": 0.1, "neutral": 0.2}
        else:
            return {"entailment": 0.2, "contradiction": 0.2, "neutral": 0.6}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        
        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": f"{hit_rate:.1%}",
            "model_loaded": self.model is not None
        }
    
    def clear_cache(self):
        """Clear the prediction cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


class OptimizedRetriever:
    """Optimized evidence retrieval with embedding cache."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.embedder = None
        self.evidence_embeddings = {}
        self._initialized = True
    
    def _load_embedder(self):
        """Lazy load sentence embedder."""
        if self.embedder is not None:
            return
            
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded embedding model")
        except ImportError:
            logger.warning("SentenceTransformer not available")
            self.embedder = None
    
    def embed_evidence(self, evidence_id: str, content: str):
        """Pre-compute and cache evidence embedding."""
        if evidence_id in self.evidence_embeddings:
            return
            
        self._load_embedder()
        
        if self.embedder is not None:
            embedding = self.embedder.encode(content)
            self.evidence_embeddings[evidence_id] = embedding
    
    def search(self, query: str, evidence_list: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Fast semantic search over cached embeddings.
        """
        self._load_embedder()
        
        if self.embedder is None or not evidence_list:
            # Fallback: keyword search
            return self._keyword_search(query, evidence_list, top_k)
        
        import numpy as np
        
        # Embed query
        query_embedding = self.embedder.encode(query)
        
        # Compute similarities
        scored = []
        for ev in evidence_list:
            ev_id = ev.get("id", ev.get("source_id", ""))
            content = ev.get("content", "")
            
            # Get or compute embedding
            if ev_id not in self.evidence_embeddings:
                self.embed_evidence(ev_id, content)
            
            if ev_id in self.evidence_embeddings:
                similarity = np.dot(query_embedding, self.evidence_embeddings[ev_id])
                scored.append((similarity, ev))
            else:
                scored.append((0, ev))
        
        # Sort by similarity
        scored.sort(key=lambda x: x[0], reverse=True)
        
        return [ev for _, ev in scored[:top_k]]
    
    def _keyword_search(self, query: str, evidence_list: List[Dict], top_k: int) -> List[Dict]:
        """Fallback keyword-based search."""
        query_words = set(query.lower().split())
        
        scored = []
        for ev in evidence_list:
            content = ev.get("content", "").lower()
            content_words = set(content.split())
            overlap = len(query_words & content_words)
            scored.append((overlap, ev))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [ev for _, ev in scored[:top_k]]


# Convenience functions
def get_optimized_nli() -> OptimizedNLI:
    """Get singleton optimized NLI instance."""
    return OptimizedNLI()


def get_optimized_retriever() -> OptimizedRetriever:
    """Get singleton optimized retriever instance."""
    return OptimizedRetriever()

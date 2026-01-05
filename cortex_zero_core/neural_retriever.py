"""
Neural Retriever: Embedding-based Evidence Retrieval

Uses sentence-transformers and FAISS for semantic search.
"""

from typing import List, Optional, Dict, Any
import logging
import numpy as np

from .models import HaltEvidence, SourceTier, TIER_RELIABILITY
from .halt_pipeline import EmbeddingRetriever

logger = logging.getLogger(__name__)


class SentenceTransformerRetriever(EmbeddingRetriever):
    """
    Evidence retriever using sentence-transformers embeddings and FAISS index.
    
    Features:
    - Semantic similarity search (not just keyword matching)
    - Fast approximate nearest neighbor search via FAISS
    - Lazy model loading
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        use_faiss: bool = True
    ):
        """
        Initialize the retriever.
        
        Args:
            model_name: Sentence-transformers model name
            device: Device for embeddings ('cuda', 'cpu', or None for auto)
            use_faiss: Whether to use FAISS for indexing (faster for large stores)
        """
        self.model_name = model_name
        self._model = None
        self._device = device
        self._use_faiss = use_faiss
        
        # Evidence store
        self._evidence_store: List[HaltEvidence] = []
        self._embeddings: Optional[np.ndarray] = None
        self._faiss_index = None
        self._is_indexed = False
        
    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is not None:
            return
            
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading SentenceTransformer model {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self._device)
            
        except ImportError:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
    
    def add_evidence(self, evidence: List[HaltEvidence]):
        """
        Add evidence to the store and update the index.
        
        Args:
            evidence: List of HaltEvidence items to add
        """
        self._load_model()
        
        # Add to store
        self._evidence_store.extend(evidence)
        
        # Compute embeddings for new evidence
        texts = [ev.content for ev in evidence]
        new_embeddings = self._model.encode(texts, convert_to_numpy=True)
        
        # Update embeddings array
        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])
        
        # Rebuild FAISS index
        self._rebuild_index()
        self._is_indexed = True
        
    def _rebuild_index(self):
        """Rebuild the FAISS index with current embeddings."""
        if not self._use_faiss or self._embeddings is None:
            return
            
        try:
            import faiss
            
            dim = self._embeddings.shape[1]
            
            # Use IVF index for larger stores, flat for small
            if len(self._evidence_store) < 1000:
                self._faiss_index = faiss.IndexFlatIP(dim)  # Inner product
            else:
                # IVF with 100 clusters
                nlist = min(100, len(self._evidence_store) // 10)
                quantizer = faiss.IndexFlatIP(dim)
                self._faiss_index = faiss.IndexIVFFlat(quantizer, dim, nlist)
                self._faiss_index.train(self._embeddings)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(self._embeddings)
            self._faiss_index.add(self._embeddings)
            
        except ImportError:
            logger.warning("FAISS not available. Falling back to numpy search.")
            self._use_faiss = False
    
    def search(self, query: str, sources: List[str], k: int) -> List[HaltEvidence]:
        """
        Search for relevant evidence.
        
        Args:
            query: The search query
            sources: Source IDs to filter by (empty = all sources)
            k: Number of results to return
            
        Returns:
            List of top-k most relevant evidence items
        """
        if not self._is_indexed or len(self._evidence_store) == 0:
            return []
            
        self._load_model()
        
        # Encode query
        query_embedding = self._model.encode([query], convert_to_numpy=True)
        
        if self._use_faiss and self._faiss_index is not None:
            return self._search_faiss(query_embedding, sources, k)
        else:
            return self._search_numpy(query_embedding, sources, k)
    
    def _search_faiss(
        self, 
        query_embedding: np.ndarray, 
        sources: List[str], 
        k: int
    ) -> List[HaltEvidence]:
        """Search using FAISS index."""
        import faiss
        
        # Normalize query
        faiss.normalize_L2(query_embedding)
        
        # Search more than k to allow filtering
        search_k = min(k * 3, len(self._evidence_store))
        scores, indices = self._faiss_index.search(query_embedding, search_k)
        
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:  # FAISS returns -1 for unfilled slots
                continue
                
            evidence = self._evidence_store[idx]
            
            # Filter by source if specified
            if sources and evidence.source_id not in sources:
                continue
                
            results.append(evidence)
            
            if len(results) >= k:
                break
                
        return results
    
    def _search_numpy(
        self, 
        query_embedding: np.ndarray, 
        sources: List[str], 
        k: int
    ) -> List[HaltEvidence]:
        """Fallback search using numpy cosine similarity."""
        # Normalize
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = self._embeddings / np.linalg.norm(
            self._embeddings, axis=1, keepdims=True
        )
        
        # Compute similarities
        similarities = np.dot(embeddings_norm, query_norm.T).flatten()
        
        # Get top indices
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            evidence = self._evidence_store[idx]
            
            # Filter by source if specified
            if sources and evidence.source_id not in sources:
                continue
                
            results.append(evidence)
            
            if len(results) >= k:
                break
                
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Return retriever statistics."""
        return {
            "evidence_count": len(self._evidence_store),
            "is_indexed": self._is_indexed,
            "using_faiss": self._use_faiss,
            "model_name": self.model_name
        }


class HybridRetriever(EmbeddingRetriever):
    """
    Combines keyword (BM25) and semantic search for better retrieval.
    
    Uses reciprocal rank fusion to combine results.
    """
    
    def __init__(
        self,
        semantic_model: str = "all-MiniLM-L6-v2",
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            semantic_model: Model for semantic search
            semantic_weight: Weight for semantic scores (0-1)
            keyword_weight: Weight for keyword scores (0-1)
        """
        self._semantic = SentenceTransformerRetriever(model_name=semantic_model)
        self._semantic_weight = semantic_weight
        self._keyword_weight = keyword_weight
        self._evidence_store: List[HaltEvidence] = []
        
    def add_evidence(self, evidence: List[HaltEvidence]):
        """Add evidence to both indexes."""
        self._evidence_store.extend(evidence)
        self._semantic.add_evidence(evidence)
    
    def _keyword_search(self, query: str, k: int) -> List[tuple]:
        """Simple TF-IDF-like keyword search."""
        query_words = set(query.lower().split())
        
        scored = []
        for i, ev in enumerate(self._evidence_store):
            ev_words = set(ev.content.lower().split())
            overlap = len(query_words & ev_words)
            if overlap > 0:
                score = overlap / len(query_words)
                scored.append((score, i, ev))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k * 2]  # Return more for fusion
    
    def search(self, query: str, sources: List[str], k: int) -> List[HaltEvidence]:
        """
        Search using both semantic and keyword methods.
        
        Uses reciprocal rank fusion to combine results.
        """
        # Get semantic results
        semantic_results = self._semantic.search(query, sources, k * 2)
        
        # Get keyword results
        keyword_results = self._keyword_search(query, k * 2)
        
        # Score by reciprocal rank fusion
        scores: Dict[str, float] = {}
        
        for rank, ev in enumerate(semantic_results):
            if sources and ev.source_id not in sources:
                continue
            scores[ev.id] = scores.get(ev.id, 0) + self._semantic_weight / (rank + 1)
        
        for rank, (_, _, ev) in enumerate(keyword_results):
            if sources and ev.source_id not in sources:
                continue
            scores[ev.id] = scores.get(ev.id, 0) + self._keyword_weight / (rank + 1)
        
        # Sort by combined score
        ev_map = {ev.id: ev for ev in self._evidence_store}
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        return [ev_map[eid] for eid in sorted_ids[:k] if eid in ev_map]

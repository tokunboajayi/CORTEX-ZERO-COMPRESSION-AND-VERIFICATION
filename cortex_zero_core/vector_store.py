"""
Vector Store: ChromaDB-based Evidence Storage

Provides persistent vector storage for semantic evidence retrieval.
"""

from typing import List, Optional, Dict, Any
import logging
import os

from .models import HaltEvidence, SourceTier, TIER_RELIABILITY
from .halt_pipeline import EmbeddingRetriever

logger = logging.getLogger(__name__)


class ChromaVectorStore(EmbeddingRetriever):
    """
    ChromaDB-based vector store for evidence.
    
    Features:
    - Persistent storage
    - Semantic search with embeddings
    - Metadata filtering (tier, source, timestamp)
    - Automatic collection management
    """
    
    def __init__(
        self,
        collection_name: str = "halt_evidence",
        persist_directory: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory for persistent storage
            embedding_model: Sentence transformer model for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory or os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "chroma_db"
        )
        self.embedding_model = embedding_model
        
        self._client = None
        self._collection = None
        self._embedding_function = None
        
    def _initialize(self):
        """Initialize ChromaDB client and collection."""
        if self._client is not None:
            return
            
        try:
            import chromadb
            from chromadb.utils import embedding_functions
            
            # Create persistent client
            self._client = chromadb.PersistentClient(
                path=self.persist_directory
            )
            
            # Use sentence transformers for embeddings
            self._embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.embedding_model
            )
            
            # Get or create collection
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_function,
                metadata={"description": "HALT-NN evidence store"}
            )
            
            logger.info(
                f"ChromaDB initialized: {self.persist_directory}, "
                f"collection: {self.collection_name}, "
                f"count: {self._collection.count()}"
            )
            
        except ImportError:
            raise ImportError(
                "chromadb is required. Install with: pip install chromadb"
            )
    
    def add_evidence(self, evidence: List[HaltEvidence]) -> int:
        """
        Add evidence to the vector store.
        
        Args:
            evidence: List of HaltEvidence items
            
        Returns:
            Number of items added
        """
        self._initialize()
        
        if not evidence:
            return 0
        
        ids = []
        documents = []
        metadatas = []
        
        for ev in evidence:
            ids.append(ev.id)
            documents.append(ev.content)
            metadatas.append({
                "source_id": ev.source_id,
                "source_tier": ev.source_tier.value,
                "reliability": ev.reliability,
                "timestamp": ev.timestamp,
                "span": ev.span[:500] if ev.span else "",
                "url": ev.url or "",
                "doc_ref": ev.doc_ref or ""
            })
        
        # Upsert to handle duplicates
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(evidence)} evidence items to vector store")
        return len(evidence)
    
    def search(
        self, 
        query: str, 
        sources: List[str] = None, 
        k: int = 5
    ) -> List[HaltEvidence]:
        """
        Search for relevant evidence.
        
        Args:
            query: Search query
            sources: Optional list of source_ids to filter by
            k: Number of results
            
        Returns:
            List of matching HaltEvidence items
        """
        self._initialize()
        
        # Build where filter
        where_filter = None
        if sources:
            where_filter = {"source_id": {"$in": sources}}
        
        # Query the collection
        results = self._collection.query(
            query_texts=[query],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to HaltEvidence
        evidence_list = []
        
        if results["ids"] and results["ids"][0]:
            for i, ev_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                content = results["documents"][0][i]
                
                ev = HaltEvidence(
                    id=ev_id,
                    content=content,
                    span=metadata.get("span", content[:200]),
                    source_id=metadata.get("source_id", "unknown"),
                    source_tier=SourceTier(metadata.get("source_tier", "B")),
                    reliability=metadata.get("reliability", 0.8),
                    timestamp=metadata.get("timestamp", 0),
                    url=metadata.get("url") or None,
                    doc_ref=metadata.get("doc_ref") or None
                )
                evidence_list.append(ev)
        
        return evidence_list
    
    def search_by_tier(
        self,
        query: str,
        min_tier: str = "C",
        k: int = 5
    ) -> List[HaltEvidence]:
        """
        Search with minimum tier requirement.
        
        Args:
            query: Search query
            min_tier: Minimum tier (A, B, or C)
            k: Number of results
        """
        self._initialize()
        
        # Map tier to reliability threshold
        tier_thresholds = {"A": 0.9, "B": 0.7, "C": 0.3}
        min_reliability = tier_thresholds.get(min_tier, 0.3)
        
        where_filter = {"reliability": {"$gte": min_reliability}}
        
        results = self._collection.query(
            query_texts=[query],
            n_results=k,
            where=where_filter,
            include=["documents", "metadatas"]
        )
        
        return self._results_to_evidence(results)
    
    def _results_to_evidence(self, results: Dict) -> List[HaltEvidence]:
        """Convert ChromaDB results to HaltEvidence list."""
        evidence_list = []
        
        if results["ids"] and results["ids"][0]:
            for i, ev_id in enumerate(results["ids"][0]):
                metadata = results["metadatas"][0][i]
                content = results["documents"][0][i]
                
                ev = HaltEvidence(
                    id=ev_id,
                    content=content,
                    span=metadata.get("span", content[:200]),
                    source_id=metadata.get("source_id", "unknown"),
                    source_tier=SourceTier(metadata.get("source_tier", "B")),
                    reliability=metadata.get("reliability", 0.8),
                    timestamp=metadata.get("timestamp", 0)
                )
                evidence_list.append(ev)
        
        return evidence_list
    
    def get_all(self, limit: int = 1000) -> List[HaltEvidence]:
        """Get all evidence from the store."""
        self._initialize()
        
        results = self._collection.get(
            limit=limit,
            include=["documents", "metadatas"]
        )
        
        evidence_list = []
        if results["ids"]:
            for i, ev_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                content = results["documents"][i]
                
                ev = HaltEvidence(
                    id=ev_id,
                    content=content,
                    span=metadata.get("span", content[:200]),
                    source_id=metadata.get("source_id", "unknown"),
                    source_tier=SourceTier(metadata.get("source_tier", "B")),
                    reliability=metadata.get("reliability", 0.8),
                    timestamp=metadata.get("timestamp", 0)
                )
                evidence_list.append(ev)
        
        return evidence_list
    
    def delete(self, evidence_ids: List[str]) -> int:
        """Delete evidence by IDs."""
        self._initialize()
        
        self._collection.delete(ids=evidence_ids)
        logger.info(f"Deleted {len(evidence_ids)} evidence items")
        return len(evidence_ids)
    
    def clear(self):
        """Clear all evidence from the store."""
        self._initialize()
        
        # Delete and recreate collection
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            embedding_function=self._embedding_function
        )
        logger.info("Cleared all evidence from vector store")
    
    def count(self) -> int:
        """Get total evidence count."""
        self._initialize()
        return self._collection.count()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        self._initialize()
        
        return {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "embedding_model": self.embedding_model,
            "total_count": self._collection.count()
        }


class InMemoryVectorStore(EmbeddingRetriever):
    """
    In-memory vector store for testing (no persistence).
    
    Uses sentence-transformers for embeddings and numpy for similarity.
    """
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize in-memory store."""
        self.embedding_model = embedding_model
        self._model = None
        self._evidence: List[HaltEvidence] = []
        self._embeddings = None
        
    def _load_model(self):
        """Load embedding model."""
        if self._model is not None:
            return
            
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.embedding_model)
        except ImportError:
            raise ImportError(
                "sentence-transformers required. "
                "Install with: pip install sentence-transformers"
            )
    
    def add_evidence(self, evidence: List[HaltEvidence]) -> int:
        """Add evidence to in-memory store."""
        self._load_model()
        
        self._evidence.extend(evidence)
        
        # Compute embeddings
        texts = [ev.content for ev in evidence]
        new_embeddings = self._model.encode(texts)
        
        import numpy as np
        if self._embeddings is None:
            self._embeddings = new_embeddings
        else:
            self._embeddings = np.vstack([self._embeddings, new_embeddings])
        
        return len(evidence)
    
    def search(
        self, 
        query: str, 
        sources: List[str] = None, 
        k: int = 5
    ) -> List[HaltEvidence]:
        """Search in-memory store."""
        if not self._evidence:
            return []
            
        self._load_model()
        
        import numpy as np
        
        # Encode query
        query_emb = self._model.encode([query])[0]
        
        # Compute similarities
        similarities = np.dot(self._embeddings, query_emb) / (
            np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_emb)
        )
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:k * 2]
        
        results = []
        for idx in top_indices:
            ev = self._evidence[idx]
            if sources and ev.source_id not in sources:
                continue
            results.append(ev)
            if len(results) >= k:
                break
        
        return results
    
    def count(self) -> int:
        """Get evidence count."""
        return len(self._evidence)

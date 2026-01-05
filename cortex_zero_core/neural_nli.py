"""
Neural NLI: Transformer-based Natural Language Inference

Provides production-ready NLI implementations using HuggingFace models.
"""

from typing import Optional, Dict, Any
import logging

from .models import NLIResult, NLILabel
from .halt_pipeline import NLIModel

logger = logging.getLogger(__name__)


class TransformerNLI(NLIModel):
    """
    NLI using HuggingFace transformer models (e.g., roberta-large-mnli).
    
    Uses standard 3-way classification: ENTAILMENT, CONTRADICTION, NEUTRAL.
    """
    
    # Label mapping from model output to our labels
    LABEL_MAP = {
        "ENTAILMENT": NLILabel.ENTAILS,
        "CONTRADICTION": NLILabel.CONTRADICTS,
        "NEUTRAL": NLILabel.NEUTRAL,
        # Some models use different casing
        "entailment": NLILabel.ENTAILS,
        "contradiction": NLILabel.CONTRADICTS,
        "neutral": NLILabel.NEUTRAL,
    }
    
    def __init__(
        self, 
        model_name: str = "roberta-large-mnli",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize transformer NLI model.
        
        Args:
            model_name: HuggingFace model name (must be trained for NLI)
            device: Device to run on ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache model weights
        """
        self.model_name = model_name
        self._pipeline = None
        self._device = device
        self._cache_dir = cache_dir
        
    def _load_model(self):
        """Lazy load the model on first use."""
        if self._pipeline is not None:
            return
            
        try:
            from transformers import pipeline
            import torch
            
            # Auto-detect device if not specified
            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading NLI model {self.model_name} on {self._device}")
            
            self._pipeline = pipeline(
                "text-classification",
                model=self.model_name,
                device=self._device if self._device != "cpu" else -1,
                model_kwargs={"cache_dir": self._cache_dir} if self._cache_dir else {}
            )
            
        except ImportError:
            raise ImportError(
                "transformers and torch are required for TransformerNLI. "
                "Install with: pip install transformers torch"
            )
    
    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        """
        Predict NLI label for premise-hypothesis pair.
        
        Args:
            premise: The evidence text
            hypothesis: The claim to verify
            
        Returns:
            NLIResult with label and probability
        """
        self._load_model()
        
        # Format input for NLI model
        # Most NLI models expect: "premise </s></s> hypothesis" or similar
        input_text = f"{premise} </s></s> {hypothesis}"
        
        try:
            result = self._pipeline(input_text, top_k=3)
            
            # Get the top prediction
            top_pred = result[0]
            label_str = top_pred["label"].upper()
            probability = top_pred["score"]
            
            # Map to our label
            nli_label = self.LABEL_MAP.get(label_str, NLILabel.NEUTRAL)
            
            return NLIResult(label=nli_label, probability=probability)
            
        except Exception as e:
            logger.warning(f"NLI prediction failed: {e}. Returning NEUTRAL.")
            return NLIResult(label=NLILabel.NEUTRAL, probability=0.5)


class CrossEncoderNLI(NLIModel):
    """
    NLI using Cross-Encoder models from sentence-transformers.
    
    More accurate than bi-encoders for NLI tasks, but slower.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: Optional[str] = None
    ):
        """
        Initialize cross-encoder NLI model.
        
        Args:
            model_name: Cross-encoder model from sentence-transformers
            device: Device to run on
        """
        self.model_name = model_name
        self._model = None
        self._device = device
        
    def _load_model(self):
        """Lazy load the model on first use."""
        if self._model is not None:
            return
            
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"Loading CrossEncoder NLI model {self.model_name}")
            self._model = CrossEncoder(self.model_name, device=self._device)
            
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderNLI. "
                "Install with: pip install sentence-transformers"
            )
    
    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        """
        Predict NLI label for premise-hypothesis pair.
        
        Cross-encoders typically output: [contradiction, entailment, neutral]
        """
        self._load_model()
        
        try:
            # Cross-encoder expects list of sentence pairs
            scores = self._model.predict([(premise, hypothesis)])[0]
            
            # Scores are typically [contradiction, entailment, neutral]
            # Get index of max score
            max_idx = scores.argmax()
            probability = float(scores[max_idx])
            
            # Map index to label
            if max_idx == 0:
                label = NLILabel.CONTRADICTS
            elif max_idx == 1:
                label = NLILabel.ENTAILS
            else:
                label = NLILabel.NEUTRAL
                
            return NLIResult(label=label, probability=probability)
            
        except Exception as e:
            logger.warning(f"CrossEncoder NLI prediction failed: {e}. Returning NEUTRAL.")
            return NLIResult(label=NLILabel.NEUTRAL, probability=0.5)


class CachedNLI(NLIModel):
    """
    Wrapper that caches NLI results for repeated queries.
    
    Useful for large-scale processing with repeated premise-hypothesis pairs.
    """
    
    def __init__(self, base_model: NLIModel, max_cache_size: int = 10000):
        """
        Initialize cached NLI wrapper.
        
        Args:
            base_model: The underlying NLI model to wrap
            max_cache_size: Maximum number of results to cache
        """
        self._base_model = base_model
        self._cache: Dict[str, NLIResult] = {}
        self._max_cache_size = max_cache_size
        self._hits = 0
        self._misses = 0
    
    def _cache_key(self, premise: str, hypothesis: str) -> str:
        """Generate cache key from inputs."""
        return f"{premise[:100]}|||{hypothesis[:100]}"
    
    def predict(self, premise: str, hypothesis: str) -> NLIResult:
        """Predict with caching."""
        key = self._cache_key(premise, hypothesis)
        
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        
        self._misses += 1
        result = self._base_model.predict(premise, hypothesis)
        
        # Add to cache if not full
        if len(self._cache) < self._max_cache_size:
            self._cache[key] = result
            
        return result
    
    @property
    def cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "cache_size": len(self._cache)
        }

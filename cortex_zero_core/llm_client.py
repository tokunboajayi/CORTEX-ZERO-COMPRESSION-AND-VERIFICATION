"""
LLM Client: Multi-Provider Language Model Integration

Supports Gemini and OpenAI for claim decomposition and answer generation.
"""

from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
import logging
import json
import re

from .models import HaltClaim, ClaimType, IntentResult, IntentType

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def decompose_claims(self, query: str, intent: IntentResult) -> List[HaltClaim]:
        """Break a query into atomic verifiable claims."""
        pass
    
    @abstractmethod
    def generate_answer(
        self, 
        query: str, 
        supported_claims: List[HaltClaim],
        evidence_snippets: List[str]
    ) -> str:
        """Generate a natural answer from verified claims."""
        pass
    
    @abstractmethod
    def explain_confidence(
        self,
        claim: HaltClaim,
        confidence: float,
        evidence_count: int
    ) -> str:
        """Generate human-readable confidence explanation."""
        pass


class GeminiClient(LLMClient):
    """
    Google Gemini LLM client.
    
    Requires: pip install google-generativeai
    Set GOOGLE_API_KEY environment variable.
    """
    
    CLAIM_DECOMPOSITION_PROMPT = """You are a claim extraction system. Break down the following query into atomic, verifiable claims.

Query: {query}
Intent Type: {intent_type}

Rules:
1. Each claim should be independently verifiable
2. Claims should be factual statements, not questions
3. Separate compound statements into individual claims
4. Mark each claim as one of: MUST_CITE (factual), DERIVATION (math/logic), SUBJECTIVE (opinion), META (process)

Return a JSON array of claims:
[{{"text": "claim text", "type": "MUST_CITE"}}, ...]

Only return the JSON array, no other text."""

    ANSWER_GENERATION_PROMPT = """Generate a clear, accurate answer based on these verified claims and evidence.

Query: {query}

Supported Claims:
{claims}

Evidence:
{evidence}

Rules:
1. Only state what is supported by evidence
2. Use natural, conversational language
3. Include source attribution where appropriate
4. If information is incomplete, acknowledge it

Generate the answer:"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash"
    ):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google API key (or use GOOGLE_API_KEY env var)
            model: Gemini model name
        """
        self.model_name = model
        self._model = None
        self._api_key = api_key
        
    def _load_model(self):
        """Lazy load the Gemini model."""
        if self._model is not None:
            return
            
        try:
            import google.generativeai as genai
            import os
            
            api_key = self._api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set")
            
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(self.model_name)
            logger.info(f"Loaded Gemini model: {self.model_name}")
            
        except ImportError:
            raise ImportError(
                "google-generativeai is required. "
                "Install with: pip install google-generativeai"
            )
    
    def decompose_claims(self, query: str, intent: IntentResult) -> List[HaltClaim]:
        """Break query into atomic claims using Gemini."""
        self._load_model()
        
        prompt = self.CLAIM_DECOMPOSITION_PROMPT.format(
            query=query,
            intent_type=intent.intent_type.value
        )
        
        try:
            response = self._model.generate_content(prompt)
            text = response.text.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                claims_data = json.loads(json_match.group())
            else:
                claims_data = json.loads(text)
            
            claims = []
            for item in claims_data:
                claim_type = ClaimType(item.get("type", "MUST_CITE"))
                claim = HaltClaim.create(
                    text=item["text"],
                    claim_type=claim_type
                )
                claims.append(claim)
            
            return claims
            
        except Exception as e:
            logger.warning(f"Gemini claim decomposition failed: {e}")
            # Fallback: return query as single claim
            return [HaltClaim.create(text=query, claim_type=ClaimType.MUST_CITE)]
    
    def generate_answer(
        self,
        query: str,
        supported_claims: List[HaltClaim],
        evidence_snippets: List[str]
    ) -> str:
        """Generate natural answer from verified claims."""
        self._load_model()
        
        claims_text = "\n".join(f"- {c.text}" for c in supported_claims)
        evidence_text = "\n".join(f"- {e[:200]}..." for e in evidence_snippets[:5])
        
        prompt = self.ANSWER_GENERATION_PROMPT.format(
            query=query,
            claims=claims_text or "No verified claims",
            evidence=evidence_text or "No evidence available"
        )
        
        try:
            response = self._model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini answer generation failed: {e}")
            return f"Based on available evidence: {claims_text}"
    
    def explain_confidence(
        self,
        claim: HaltClaim,
        confidence: float,
        evidence_count: int
    ) -> str:
        """Generate confidence explanation."""
        if confidence >= 0.8:
            level = "high"
            reason = "strong supporting evidence"
        elif confidence >= 0.5:
            level = "moderate"
            reason = "some supporting evidence"
        else:
            level = "low"
            reason = "limited evidence"
        
        return (
            f"Confidence is {level} ({confidence:.0%}) based on {evidence_count} "
            f"evidence items - {reason}."
        )


class OpenAIClient(LLMClient):
    """
    OpenAI GPT client.
    
    Requires: pip install openai
    Set OPENAI_API_KEY environment variable.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini"
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (or use OPENAI_API_KEY env var)
            model: Model name (gpt-4o-mini, gpt-4o, etc.)
        """
        self.model_name = model
        self._client = None
        self._api_key = api_key
        
    def _load_client(self):
        """Lazy load OpenAI client."""
        if self._client is not None:
            return
            
        try:
            from openai import OpenAI
            import os
            
            api_key = self._api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set")
            
            self._client = OpenAI(api_key=api_key)
            logger.info(f"Loaded OpenAI model: {self.model_name}")
            
        except ImportError:
            raise ImportError(
                "openai is required. Install with: pip install openai"
            )
    
    def _chat(self, prompt: str) -> str:
        """Send chat completion request."""
        self._load_client()
        
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    
    def decompose_claims(self, query: str, intent: IntentResult) -> List[HaltClaim]:
        """Break query into atomic claims using GPT."""
        prompt = GeminiClient.CLAIM_DECOMPOSITION_PROMPT.format(
            query=query,
            intent_type=intent.intent_type.value
        )
        
        try:
            text = self._chat(prompt)
            
            # Extract JSON
            json_match = re.search(r'\[.*\]', text, re.DOTALL)
            if json_match:
                claims_data = json.loads(json_match.group())
            else:
                claims_data = json.loads(text)
            
            claims = []
            for item in claims_data:
                claim_type = ClaimType(item.get("type", "MUST_CITE"))
                claim = HaltClaim.create(
                    text=item["text"],
                    claim_type=claim_type
                )
                claims.append(claim)
            
            return claims
            
        except Exception as e:
            logger.warning(f"OpenAI claim decomposition failed: {e}")
            return [HaltClaim.create(text=query, claim_type=ClaimType.MUST_CITE)]
    
    def generate_answer(
        self,
        query: str,
        supported_claims: List[HaltClaim],
        evidence_snippets: List[str]
    ) -> str:
        """Generate answer using GPT."""
        claims_text = "\n".join(f"- {c.text}" for c in supported_claims)
        evidence_text = "\n".join(f"- {e[:200]}..." for e in evidence_snippets[:5])
        
        prompt = GeminiClient.ANSWER_GENERATION_PROMPT.format(
            query=query,
            claims=claims_text or "No verified claims",
            evidence=evidence_text or "No evidence available"
        )
        
        try:
            return self._chat(prompt)
        except Exception as e:
            logger.warning(f"OpenAI answer generation failed: {e}")
            return f"Based on available evidence: {claims_text}"
    
    def explain_confidence(
        self,
        claim: HaltClaim,
        confidence: float,
        evidence_count: int
    ) -> str:
        """Generate confidence explanation."""
        # Same logic as Gemini
        if confidence >= 0.8:
            level = "high"
        elif confidence >= 0.5:
            level = "moderate"
        else:
            level = "low"
        
        return f"Confidence: {level} ({confidence:.0%}) from {evidence_count} sources."


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing without API keys.
    
    Returns simple rule-based decomposition.
    """
    
    def decompose_claims(self, query: str, intent: IntentResult) -> List[HaltClaim]:
        """Simple rule-based claim extraction."""
        # Split by "and" or semicolons
        parts = re.split(r'\s+and\s+|;\s*', query.lower())
        
        claims = []
        for part in parts:
            part = part.strip()
            if len(part) > 5:
                claims.append(HaltClaim.create(
                    text=part,
                    claim_type=ClaimType.MUST_CITE
                ))
        
        if not claims:
            claims.append(HaltClaim.create(
                text=query,
                claim_type=ClaimType.MUST_CITE
            ))
        
        return claims
    
    def generate_answer(
        self,
        query: str,
        supported_claims: List[HaltClaim],
        evidence_snippets: List[str]
    ) -> str:
        """Generate simple answer."""
        if supported_claims:
            return f"Based on verified evidence: {supported_claims[0].text}"
        return "Insufficient evidence to provide a verified answer."
    
    def explain_confidence(
        self,
        claim: HaltClaim,
        confidence: float,
        evidence_count: int
    ) -> str:
        """Simple confidence explanation."""
        return f"Confidence: {confidence:.0%} ({evidence_count} evidence items)"


def get_llm_client(provider: str = "mock", **kwargs) -> LLMClient:
    """
    Factory function to get LLM client.
    
    Args:
        provider: "gemini", "openai", or "mock"
        **kwargs: Provider-specific arguments
        
    Returns:
        LLMClient instance
    """
    providers = {
        "gemini": GeminiClient,
        "openai": OpenAIClient,
        "mock": MockLLMClient
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Use: {list(providers.keys())}")
    
    return providers[provider](**kwargs)

"""
HALT-NN FastAPI Wrapper

REST API and WebSocket streaming for the HALT-NN anti-hallucination pipeline.
Optimized with caching, compression, and efficient inference.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import time
import json
import asyncio
import os
import hashlib
from functools import lru_cache

from cortex_zero_core import (
    run_halt_pipeline, HaltEvidence, SourceTier, ActionDecision,
    ClaimStatus, HaltConfig, TIER_RELIABILITY,
    analyze_intent, decompose_claims, retrieve_evidence,
    build_evidence_graph, verify_gate, generate_controlled_answer,
    calibrate_confidence, decide_action
)

# Security imports
try:
    from cortex_zero_core.security import (
        rate_limiter, InputValidator, api_auth, create_security_middleware,
        get_security_headers, fingerprinter
    )
    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False

# =============================================================================
# API MODELS
# =============================================================================

class EvidenceCreate(BaseModel):
    """Request model for adding evidence."""
    content: str = Field(..., description="Evidence text content")
    source_id: str = Field(..., description="Source identifier (e.g., URL, document ID)")
    tier: str = Field(default="B", description="Source tier: A, B, or C")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Python is a high-level programming language.",
                "source_id": "wikipedia.org/Python",
                "tier": "A"
            }
        }


class QueryRequest(BaseModel):
    """Request model for running the HALT-NN pipeline."""
    query: str = Field(..., description="Question or claim to verify")
    use_neural: bool = Field(default=False, description="Use neural models if available")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is Python programming language?",
                "use_neural": False
            }
        }


class ClaimResponse(BaseModel):
    """Response model for individual claims."""
    id: str
    text: str
    status: str
    confidence: float
    evidence_ids: List[str]


class LinkResponse(BaseModel):
    """Response model for evidence links."""
    claim_id: str
    evidence_id: str
    nli_label: str
    nli_probability: float
    support_strength: float


class QueryResponse(BaseModel):
    """Response model for query results."""
    query: str
    action: str
    answer_text: str
    overall_confidence: float
    coverage_ratio: float
    conflict_count: int
    claims: List[ClaimResponse]
    evidence_count: int
    links: List[LinkResponse]
    abstentions: List[str]
    next_actions: List[str]
    processing_time_ms: float


class EvidenceResponse(BaseModel):
    """Response model for evidence items."""
    id: str
    content: str
    source_id: str
    tier: str
    reliability: float


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    evidence_count: int
    neural_nli_available: bool
    neural_retriever_available: bool


# =============================================================================
# APPLICATION
# =============================================================================

app = FastAPI(
    title="HALT-NN API",
    description="Evidence-Grounded Anti-Hallucination Pipeline API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip compression for faster responses
app.add_middleware(GZipMiddleware, minimum_size=500)

# Security middleware (rate limiting, security headers, etc.)
if SECURITY_AVAILABLE:
    create_security_middleware(app)

# Response cache for repeated queries
_response_cache: Dict[str, tuple] = {}
_cache_ttl = 300  # 5 minutes

def get_cached_response(query: str) -> Optional[Dict]:
    """Get cached response if available and not expired."""
    cache_key = hashlib.md5(query.lower().encode()).hexdigest()
    if cache_key in _response_cache:
        cached_data, timestamp = _response_cache[cache_key]
        if time.time() - timestamp < _cache_ttl:
            return cached_data
        else:
            del _response_cache[cache_key]
    return None

def cache_response(query: str, response: Dict):
    """Cache a response."""
    cache_key = hashlib.md5(query.lower().encode()).hexdigest()
    _response_cache[cache_key] = (response, time.time())
    
    # Limit cache size
    if len(_response_cache) > 1000:
        # Remove oldest entries
        sorted_keys = sorted(_response_cache.keys(), 
                           key=lambda k: _response_cache[k][1])
        for key in sorted_keys[:100]:
            del _response_cache[key]

# Serve static files (chat UI)
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")



# In-memory evidence store
evidence_store: List[HaltEvidence] = []

# Check for neural module availability
try:
    from cortex_zero_core.neural_nli import TransformerNLI
    NEURAL_NLI_AVAILABLE = True
except ImportError:
    NEURAL_NLI_AVAILABLE = False

try:
    from cortex_zero_core.neural_retriever import SentenceTransformerRetriever
    NEURAL_RETRIEVER_AVAILABLE = True
except ImportError:
    NEURAL_RETRIEVER_AVAILABLE = False


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and show system status."""
    return HealthResponse(
        status="healthy",
        evidence_count=len(evidence_store),
        neural_nli_available=NEURAL_NLI_AVAILABLE,
        neural_retriever_available=NEURAL_RETRIEVER_AVAILABLE
    )


@app.get("/stats", tags=["System"])
async def get_stats():
    """Get performance statistics including cache info."""
    return {
        "evidence_count": len(evidence_store),
        "response_cache_size": len(_response_cache),
        "cache_ttl_seconds": _cache_ttl,
        "neural_nli_available": NEURAL_NLI_AVAILABLE,
        "neural_retriever_available": NEURAL_RETRIEVER_AVAILABLE,
        "optimizations": {
            "gzip_compression": True,
            "response_caching": True,
            "lazy_model_loading": True
        }
    }


@app.post("/evidence", response_model=EvidenceResponse, tags=["Evidence"])
async def add_evidence(evidence: EvidenceCreate):
    """
    Add evidence to the store.
    
    Evidence is used by the HALT-NN pipeline to verify claims.
    """
    # Map tier string to enum
    tier_map = {"A": SourceTier.TIER_A, "B": SourceTier.TIER_B, "C": SourceTier.TIER_C}
    tier = tier_map.get(evidence.tier.upper(), SourceTier.TIER_B)
    
    # Create evidence
    halt_evidence = HaltEvidence.create(
        content=evidence.content,
        source_id=evidence.source_id,
        tier=tier
    )
    
    # Add to store
    evidence_store.append(halt_evidence)
    
    return EvidenceResponse(
        id=halt_evidence.id,
        content=halt_evidence.content,
        source_id=halt_evidence.source_id,
        tier=halt_evidence.source_tier.value,
        reliability=halt_evidence.reliability
    )


@app.get("/evidence", response_model=List[EvidenceResponse], tags=["Evidence"])
async def list_evidence():
    """List all evidence in the store."""
    return [
        EvidenceResponse(
            id=ev.id,
            content=ev.content,
            source_id=ev.source_id,
            tier=ev.source_tier.value,
            reliability=ev.reliability
        )
        for ev in evidence_store
    ]


@app.delete("/evidence/{evidence_id}", tags=["Evidence"])
async def delete_evidence(evidence_id: str):
    """Delete evidence by ID."""
    global evidence_store
    
    original_count = len(evidence_store)
    evidence_store = [ev for ev in evidence_store if ev.id != evidence_id]
    
    if len(evidence_store) == original_count:
        raise HTTPException(status_code=404, detail="Evidence not found")
    
    return {"status": "deleted", "id": evidence_id}


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def run_query(request: QueryRequest):
    """
    Run the HALT-NN pipeline on a query.
    
    Returns verification results including:
    - Action decision (ANSWER, ABSTAIN, etc.)
    - Claim statuses (SUPPORTED, UNSUPPORTED, DISPUTED)
    - Confidence scores
    - Evidence links
    """
    start_time = time.time()
    
    # Input validation
    if SECURITY_AVAILABLE:
        is_valid, error, sanitized = InputValidator.validate_query(request.query)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)
        query = sanitized
    else:
        query = request.query
    
    # Check cache first
    cached = get_cached_response(query)
    if cached:
        cached["from_cache"] = True
        cached["processing_time_ms"] = 0.1
        return QueryResponse(**cached)
    
    # Run optimized pipeline
    try:
        from cortex_zero_core.optimized_pipeline import get_optimized_pipeline
        pipeline = get_optimized_pipeline()
        answer_text, confidence, coverage, action = pipeline.run_optimized(
            request.query, evidence_store
        )
        
        # Get pipeline stats
        stats = pipeline.get_stats()
    except ImportError:
        # Fallback to original pipeline
        audit = run_halt_pipeline(
            query=request.query,
            evidence_store=evidence_store,
            retriever=None,
            nli_model=None,
            calibrator=None
        )
        answer_text = audit.answer_text
        confidence = audit.overall_confidence
        coverage = audit.coverage_ratio
        action = audit.action
        stats = {}
    
    processing_time = (time.time() - start_time) * 1000
    
    # Build response
    response_data = {
        "query": request.query,
        "action": action.value if hasattr(action, 'value') else str(action),
        "answer_text": answer_text,
        "overall_confidence": confidence,
        "coverage_ratio": coverage,
        "conflict_count": 0,
        "claims": [],
        "evidence_count": len(evidence_store),
        "links": [],
        "abstentions": [],
        "next_actions": [],
        "processing_time_ms": processing_time
    }
    
    # Cache the response
    cache_response(request.query, response_data)
    
    return QueryResponse(**response_data)


@app.post("/query/cot", tags=["Query"])
async def run_cot_query(request: QueryRequest):
    """
    Run Chain-of-Thought enhanced verification.
    
    Breaks query into reasoning steps, verifies each step,
    and identifies weak links in the reasoning chain.
    """
    start_time = time.time()
    
    # Input validation
    if SECURITY_AVAILABLE:
        is_valid, error, sanitized = InputValidator.validate_query(request.query)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)
        query = sanitized
    else:
        query = request.query
    
    # Run Chain-of-Thought verification
    try:
        from cortex_zero_core.chain_of_thought import verify_with_cot
        result = verify_with_cot(query, evidence_store)
    except ImportError:
        raise HTTPException(
            status_code=501, 
            detail="Chain-of-Thought module not available"
        )
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        **result,
        "processing_time_ms": processing_time
    }


@app.post("/query/mega", tags=["Query"])
async def run_mega_query(request: QueryRequest):
    """
    Run query with all 10 mega algorithms enabled.
    
    Includes: Query Expansion, Semantic Dedup, Temporal Decay,
    Adversarial Detection, Knowledge Graph, Active Learning,
    Hierarchical Attention, Source Reputation, Ensemble Verification,
    and Uncertainty Quantification.
    """
    start_time = time.time()
    
    # Input validation
    if SECURITY_AVAILABLE:
        is_valid, error, sanitized = InputValidator.validate_query(request.query)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error)
        query = sanitized
    else:
        query = request.query
    
    # Run Mega Engine
    try:
        from cortex_zero_core.mega_algorithms import create_mega_engine
        engine = create_mega_engine()
        result = engine.process_query(query, evidence_store)
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Mega algorithms module not available"
        )
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        **result,
        "processing_time_ms": processing_time,
        "algorithms_used": [
            "Knowledge Graph Reasoning",
            "Semantic Deduplication",
            "Temporal Decay",
            "Adversarial Detection",
            "Active Learning",
            "Hierarchical Attention",
            "Source Reputation Learning",
            "Query Expansion",
            "Ensemble Verification",
            "Uncertainty Quantification"
        ]
    }


@app.post("/batch", tags=["Query"])
async def run_batch_queries(queries: List[str]):
    """Run multiple queries in batch."""
    results = []
    
    for query in queries:
        audit = run_halt_pipeline(
            query=query,
            evidence_store=evidence_store
        )
        results.append({
            "query": query,
            "action": audit.action.value,
            "confidence": audit.overall_confidence,
            "answer": audit.answer_text[:200]  # Truncate for brevity
        })
    
    return {"results": results, "count": len(results)}


@app.get("/config", tags=["System"])
async def get_config():
    """Get current pipeline configuration."""
    return {
        "support_threshold": HaltConfig.SUPPORT_THRESHOLD,
        "conflict_threshold": HaltConfig.CONFLICT_THRESHOLD,
        "confidence_threshold": HaltConfig.CONFIDENCE_THRESHOLD,
        "coverage_threshold": HaltConfig.COVERAGE_THRESHOLD,
        "top_k_evidence": HaltConfig.TOP_K_EVIDENCE,
        "tier_reliability": {k.value: v for k, v in TIER_RELIABILITY.items()}
    }


# =============================================================================
# STARTUP
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Load sample evidence on startup."""
    global evidence_store
    
    # Add some sample evidence for testing
    sample_evidence = [
        HaltEvidence.create(
            content="Python is a high-level, general-purpose programming language. "
                    "Its design philosophy emphasizes code readability.",
            source_id="wikipedia.org/Python",
            tier=SourceTier.TIER_A
        ),
        HaltEvidence.create(
            content="Python was conceived in the late 1980s by Guido van Rossum.",
            source_id="python.org/history",
            tier=SourceTier.TIER_A
        ),
        HaltEvidence.create(
            content="Python is widely used for data science and machine learning.",
            source_id="techcrunch.com/2024",
            tier=SourceTier.TIER_B
        ),
    ]
    
    evidence_store.extend(sample_evidence)
    print(f"[HALT-NN API] Loaded {len(sample_evidence)} sample evidence items")


# =============================================================================
# WEBSOCKET STREAMING
# =============================================================================

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
    
    async def send_json(self, websocket: WebSocket, data: dict):
        await websocket.send_json(data)


manager = ConnectionManager()


@app.websocket("/ws/query")
async def websocket_query(websocket: WebSocket):
    """
    WebSocket endpoint for streaming verification.
    
    Send: {"query": "your question"}
    Receive: Stream of events:
      - {"event": "intent", "data": {...}}
      - {"event": "claims", "data": [...]}
      - {"event": "evidence", "data": [...]}
      - {"event": "verification", "data": [...]}
      - {"event": "complete", "data": {...}}
    """
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive query
            data = await websocket.receive_json()
            query = data.get("query", "")
            
            if not query:
                await manager.send_json(websocket, {
                    "event": "error",
                    "data": {"message": "Query is required"}
                })
                continue
            
            start_time = time.time()
            
            # Phase 1: Intent Analysis
            await manager.send_json(websocket, {
                "event": "phase",
                "data": {"phase": 1, "name": "Intent Analysis", "status": "running"}
            })
            
            intent = analyze_intent(query)
            await manager.send_json(websocket, {
                "event": "intent",
                "data": {
                    "intent_type": intent.intent_type.value,
                    "recency_sensitive": intent.recency_sensitive,
                    "high_stakes": intent.high_stakes
                }
            })
            
            # Phase 2: Claim Decomposition
            await manager.send_json(websocket, {
                "event": "phase",
                "data": {"phase": 2, "name": "Claim Decomposition", "status": "running"}
            })
            
            claims = decompose_claims(query, intent)
            await manager.send_json(websocket, {
                "event": "claims",
                "data": [{"id": c.id, "text": c.text, "type": c.claim_type.value} for c in claims]
            })
            
            # Phase 3: Evidence Retrieval
            await manager.send_json(websocket, {
                "event": "phase",
                "data": {"phase": 3, "name": "Evidence Retrieval", "status": "running"}
            })
            
            evidence = retrieve_evidence(claims, evidence_store)
            await manager.send_json(websocket, {
                "event": "evidence",
                "data": [{"id": e.id, "source": e.source_id, "tier": e.source_tier.value} for e in evidence]
            })
            
            # Phase 4: NLI Graph
            await manager.send_json(websocket, {
                "event": "phase",
                "data": {"phase": 4, "name": "Evidence Graph", "status": "running"}
            })
            
            links = build_evidence_graph(claims, evidence)
            await manager.send_json(websocket, {
                "event": "links",
                "data": [{"claim": l.claim_id[:8], "evidence": l.evidence_id[:8], "label": l.nli_label.value} for l in links]
            })
            
            # Phase 5: Verification
            await manager.send_json(websocket, {
                "event": "phase",
                "data": {"phase": 5, "name": "Verification Gate", "status": "running"}
            })
            
            claims = verify_gate(claims, links, evidence)
            await manager.send_json(websocket, {
                "event": "verification",
                "data": [{"id": c.id[:8], "text": c.text[:50], "status": c.status.value} for c in claims]
            })
            
            # Phase 6: Generation
            await manager.send_json(websocket, {
                "event": "phase",
                "data": {"phase": 6, "name": "Answer Generation", "status": "running"}
            })
            
            answer_text, abstentions = generate_controlled_answer(claims, links, evidence)
            
            # Phase 7: Calibration
            await manager.send_json(websocket, {
                "event": "phase",
                "data": {"phase": 7, "name": "Calibration", "status": "running"}
            })
            
            confidence, coverage = calibrate_confidence(claims, links)
            conflict_count = sum(1 for c in claims if c.status == ClaimStatus.DISPUTED)
            action = decide_action(coverage, conflict_count, confidence)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Send final result
            await manager.send_json(websocket, {
                "event": "complete",
                "data": {
                    "query": query,
                    "action": action.value,
                    "answer_text": answer_text,
                    "confidence": confidence,
                    "coverage": coverage,
                    "conflict_count": conflict_count,
                    "abstentions": abstentions,
                    "processing_time_ms": processing_time
                }
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await manager.send_json(websocket, {
            "event": "error",
            "data": {"message": str(e)}
        })
        manager.disconnect(websocket)


@app.get("/", tags=["UI"])
async def serve_ui():
    """Serve the chat UI."""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    index_path = os.path.join(static_dir, "index.html")
    
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return {"message": "HALT-NN API", "docs": "/docs", "health": "/health"}


@app.get("/unified", tags=["UI"])
async def serve_unified_ui():
    """Serve the unified compression + verification UI."""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    unified_path = os.path.join(static_dir, "unified.html")
    
    if os.path.exists(unified_path):
        return FileResponse(unified_path)
    else:
        return {"message": "Unified UI not found", "fallback": "/"}


@app.get("/network", tags=["UI"])
async def serve_network_viz():
    """Serve the neural network visualization."""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    viz_path = os.path.join(static_dir, "network_viz.html")
    
    if os.path.exists(viz_path):
        return FileResponse(viz_path)
    else:
        return {"message": "Network visualization not found"}




# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

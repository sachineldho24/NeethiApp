"""
Neethi App - FastAPI Gateway
Main entry point for the API server

Refactored to use LibrarianAgent and LawyerAgent for clean separation of concerns.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from loguru import logger
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import uuid
import os
import json

# Load environment variables from .env file
load_dotenv()

# Configure logging
logger.add(
    "logs/api_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO"
)

# ================== Global Model Cache ==================
_qdrant_client = None
_embedder = None
_librarian = None
_lawyer = None
_llm_router = None


def get_qdrant_client():
    """Get or create Qdrant client singleton"""
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        url = os.getenv("QDRANT_URL")
        key = os.getenv("QDRANT_API_KEY")
        if url and key:
            _qdrant_client = QdrantClient(url=url, api_key=key)
            logger.info("Qdrant client initialized")
    return _qdrant_client


def get_embedder():
    """Get or create embedding model singleton"""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        from pathlib import Path
        
        # Use fine-tuned model if available, otherwise fall back to base
        finetuned_path = Path("models/inlegalbert-finetuned")
        if finetuned_path.exists():
            logger.info("Loading fine-tuned InLegalBERT model...")
            _embedder = SentenceTransformer(str(finetuned_path))
            logger.info("✅ Fine-tuned InLegalBERT model loaded")
        else:
            logger.info("Loading base InLegalBERT model...")
            _embedder = SentenceTransformer("law-ai/InLegalBERT")
            logger.info("InLegalBERT model loaded (base)")
    return _embedder


def get_librarian():
    """Get or create LibrarianAgent singleton"""
    global _librarian
    if _librarian is None:
        from agents.librarian import LibrarianAgent
        client = get_qdrant_client()
        embedder = get_embedder()
        if client and embedder:
            _librarian = LibrarianAgent(
                qdrant_client=client,
                embedding_model=embedder
            )
            logger.info("LibrarianAgent initialized")
    return _librarian


def get_lawyer():
    """Get or create LawyerAgent singleton with Ollama + LLM Router support"""
    global _lawyer
    if _lawyer is None:
        from agents.lawyer import LawyerAgent, OllamaClient
        # Configure Ollama
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        ollama_client = OllamaClient(base_url=ollama_url, model="llama3")
        
        # Get LLM Router for cloud fallback
        llm_router = get_llm_router()
        
        _lawyer = LawyerAgent(ollama_client=ollama_client, llm_router=llm_router)
        logger.info("LawyerAgent initialized (Ollama + LLM Router)")
    return _lawyer


def get_llm_router():
    """Get or create LLM Router singleton with Groq/Gemini support"""
    global _llm_router
    if _llm_router is None:
        from services.llm_router import create_llm_router
        _llm_router = create_llm_router()
        logger.info(f"LLM Router initialized: {_llm_router.get_status()}")
    return _llm_router


def generate_session_id() -> str:
    """Generate unique session ID using UUID"""
    return f"session_{uuid.uuid4().hex[:12]}"


# ================== FastAPI App ==================

# ================== Lifespan Context Manager ==================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Modern lifespan handler for startup/shutdown"""
    # Startup
    logger.info("Starting Neethi API server...")
    
    # Preload Qdrant client
    client = get_qdrant_client()
    if client:
        logger.info("✅ Qdrant client ready")
    else:
        logger.warning("⚠️ Qdrant client not initialized - check environment variables")
    
    # Preload embedding model and agents
    librarian = get_librarian()
    if librarian:
        logger.info("✅ LibrarianAgent ready")
    
    # Initialize LLM Router first (used by LawyerAgent)
    llm_router = get_llm_router()
    if llm_router:
        logger.info("✅ LLM Router ready")
    
    lawyer = get_lawyer()
    if lawyer:
        logger.info("✅ LawyerAgent ready")
    
    logger.info("API server ready - agents preloaded!")
    
    yield  # Server is running
    
    # Shutdown
    logger.info("Shutting down Neethi API server...")


app = FastAPI(
    title="Neethi Legal Assistant API",
    description="AI-powered legal guidance for Indian citizens",
    version="1.2.0",  # Bumped version for LLM Router + async fixes
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Configuration - use specific origins in production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if os.getenv("PRODUCTION") else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================== Request/Response Models ==================

class LegalQueryRequest(BaseModel):
    """Request for legal advice"""
    query: str = Field(..., min_length=10, description="Legal question")
    location: Optional[Dict[str, float]] = Field(None, description="User location {lat, lon}")
    include_news: bool = Field(False, description="Include related news")
    include_locations: bool = Field(False, description="Include nearby legal resources")
    include_judgments: bool = Field(True, description="Include case law from bail judgments")


class LegalQueryResponse(BaseModel):
    """Response with legal advice"""
    session_id: str
    query: str
    advice: str
    sources: List[Dict[str, Any]]
    case_law: Optional[List[Dict[str, Any]]] = None  # SC + Bail judgments
    nearby_resources: Optional[List[Dict[str, Any]]] = None
    news: Optional[List[Dict[str, Any]]] = None
    disclaimer: str = (
        "This is general legal information, not legal advice. "
        "Please consult a qualified lawyer for your specific situation."
    )
    processing_time_ms: float


class DocumentRequest(BaseModel):
    """Request for document generation"""
    doc_type: str = Field(..., description="Document type: FIR, RTI, NOTICE, COMPLAINT")
    details: Dict[str, Any] = Field(..., description="Document details")


class DocumentResponse(BaseModel):
    """Response with generated document"""
    doc_type: str
    filename: str
    download_url: str
    missing_fields: Optional[List[str]] = None


class FeedbackRequest(BaseModel):
    """Feedback submission request"""
    session_id: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    comment: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    services: Dict[str, str]


# ================== Endpoints ==================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Neethi Legal Assistant API",
        "version": "1.2.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint with connectivity verification"""
    librarian_status = "online" if get_librarian() else "offline"
    qdrant_status = "online" if get_qdrant_client() else "offline"
    
    # Check LLM Router status
    llm_status = "offline"
    llm_router = get_llm_router()
    if llm_router:
        router_status = llm_router.get_status()
        if router_status["groq"]["available"] or router_status["gemini"]["available"]:
            llm_status = "online"
    
    return HealthResponse(
        status="healthy",
        version="1.2.0",
        timestamp=datetime.now().isoformat(),
        services={
            "api": "online",
            "qdrant": qdrant_status,
            "librarian": librarian_status,
            "llm": llm_status
        }
    )


@app.post("/api/v1/query", response_model=LegalQueryResponse, tags=["Legal Advice"])
async def legal_query(request: LegalQueryRequest):
    """
    Get legal advice for a question.
    
    Uses LibrarianAgent for retrieval and LawyerAgent for response formatting.
    """
    import time
    start_time = time.time()
    
    logger.info(f"Query received: {request.query[:50]}...")
    
    # Get agents
    librarian = get_librarian()
    lawyer = get_lawyer()
    
    if not librarian:
        processing_time = (time.time() - start_time) * 1000
        return LegalQueryResponse(
            session_id=generate_session_id(),
            query=request.query,
            advice="⚠️ Service not available. Please check Qdrant credentials.",
            sources=[],
            processing_time_ms=processing_time
        )
    
    try:
        # Step 1: Librarian retrieves relevant chunks (statutes + optional judgments)
        chunks = librarian.search_all(request.query, include_judgments=request.include_judgments)
        
        # Step 2: Lawyer formats the response
        advice = lawyer.generate_advice(request.query, chunks)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Build response
        session_id = generate_session_id()
        
        return LegalQueryResponse(
            session_id=session_id,
            query=request.query,
            advice=advice.advice,
            sources=advice.sources,
            case_law=advice.case_law or [],
            nearby_resources=[] if request.include_locations else None,
            news=[] if request.include_news else None,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        processing_time = (time.time() - start_time) * 1000
        return LegalQueryResponse(
            session_id=generate_session_id(),
            query=request.query,
            advice=f"Error during search: {str(e)}",
            sources=[],
            processing_time_ms=processing_time
        )


@app.post("/api/v1/document/generate", response_model=DocumentResponse, tags=["Documents"])
async def generate_document(request: DocumentRequest):
    """Generate a legal document (FIR, RTI, Notice)."""
    logger.info(f"Document request: {request.doc_type}")
    
    valid_types = ["FIR", "RTI", "NOTICE", "COMPLAINT"]
    if request.doc_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid document type. Valid types: {valid_types}"
        )
    
    # TODO: Integrate with ScribeAgent
    return DocumentResponse(
        doc_type=request.doc_type,
        filename=f"{request.doc_type}_{datetime.now().strftime('%Y%m%d')}.txt",
        download_url="/api/v1/document/download/placeholder",
        missing_fields=None
    )


@app.get("/api/v1/news", tags=["News"])
async def get_legal_news(
    topic: Optional[str] = None,
    days: int = 7,
    limit: int = 10
):
    """Get recent legal news and updates."""
    logger.info(f"News request: topic={topic}, days={days}")
    
    # TODO: Integrate with JournalistAgent
    return {
        "topic": topic,
        "days": days,
        "articles": [],
        "message": "News integration pending"
    }


@app.get("/api/v1/locations", tags=["Location"])
async def find_legal_resources(
    lat: float,
    lon: float,
    resource_type: str = "police"
):
    """Find nearby legal resources (police stations, courts, legal aid)."""
    logger.info(f"Location request: ({lat}, {lon}), type={resource_type}")
    
    valid_types = ["police", "court", "legal_aid", "advocate"]
    if resource_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid resource type. Valid types: {valid_types}"
        )
    
    # TODO: Integrate with ClerkAgent
    return {
        "location": {"lat": lat, "lon": lon},
        "resource_type": resource_type,
        "resources": [],
        "message": "Location integration pending (requires Google Maps API key)"
    }


@app.post("/api/v1/feedback", tags=["Feedback"])
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback on a legal advice response.
    
    Stores feedback to JSONL file for future fine-tuning (HITL).
    """
    logger.info(f"Feedback received: session={request.session_id}, rating={request.rating}")
    
    # Create feedback entry
    feedback_entry = {
        "timestamp": datetime.now().isoformat(),
        "session_id": request.session_id,
        "rating": request.rating,
        "comment": request.comment
    }
    
    # Ensure data directory exists
    feedback_dir = Path("data")
    feedback_dir.mkdir(exist_ok=True)
    feedback_file = feedback_dir / "feedback.jsonl"
    
    # Append to JSONL file
    try:
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_entry, ensure_ascii=False) + "\n")
        logger.info(f"Feedback saved to {feedback_file}")
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
    
    return {
        "status": "received",
        "session_id": request.session_id,
        "rating": request.rating
    }


# ================== Run Server ==================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

"""
Shared State Management for Multi-Agent Crews

This module defines the shared state that agents use to communicate.
Like a whiteboard in a law firm's conference room.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid


@dataclass
class RetrievedDocument:
    """A document retrieved by the Librarian"""
    chunk_id: str
    text: str
    law_type: str
    section_num: Optional[str] = None
    case_name: Optional[str] = None
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LegalConsultationState:
    """
    Shared state for legal consultation workflow.
    
    This state object is passed between agents in a crew,
    allowing them to read from and write to a common context.
    """
    
    # Session identification
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    
    # Input
    query: str = ""
    user_location: Optional[Dict[str, float]] = None  # {"lat": x, "lon": y}
    
    # From Librarian Agent
    retrieved_chunks: List[RetrievedDocument] = field(default_factory=list)
    retrieval_completed: bool = False
    
    # From Lawyer Agent
    generated_advice: str = ""
    advice_sources: List[Dict[str, str]] = field(default_factory=list)
    advice_completed: bool = False
    
    # From Clerk Agent
    nearby_resources: List[Dict[str, Any]] = field(default_factory=list)
    location_completed: bool = False
    
    # From Journalist Agent
    relevant_news: List[Dict[str, Any]] = field(default_factory=list)
    news_completed: bool = False
    
    # From Scribe Agent
    drafted_document: Optional[bytes] = None
    document_path: Optional[str] = None
    document_completed: bool = False
    
    # Error tracking
    errors: List[Dict[str, str]] = field(default_factory=list)
    
    # Performance tracking
    agent_timings: Dict[str, float] = field(default_factory=dict)
    
    def add_error(self, agent: str, message: str):
        """Record an error from an agent"""
        self.errors.append({
            "agent": agent,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
    
    def record_timing(self, agent: str, duration_seconds: float):
        """Record how long an agent took"""
        self.agent_timings[agent] = duration_seconds
    
    def is_complete(self) -> bool:
        """Check if core workflow is complete"""
        return self.retrieval_completed and self.advice_completed
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary for API response"""
        return {
            "session_id": self.session_id,
            "query": self.query,
            "advice": self.generated_advice,
            "sources": self.advice_sources,
            "nearby_resources": self.nearby_resources,
            "news": self.relevant_news,
            "document_path": self.document_path,
            "errors": self.errors,
            "timings": self.agent_timings,
        }


@dataclass  
class DocumentDraftingState:
    """
    Shared state for document drafting workflow.
    """
    
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Input
    doc_type: str = ""  # "FIR", "RTI", "NOTICE"
    user_inputs: Dict[str, Any] = field(default_factory=dict)
    
    # Extraction
    extracted_fields: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
    
    # Output
    generated_document: Optional[bytes] = None
    document_path: Optional[str] = None
    format: str = "docx"
    
    # Status
    is_complete: bool = False
    errors: List[str] = field(default_factory=list)

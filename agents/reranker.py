"""
Reranker Agent - Result Refinement

Uses cross-encoder model to rerank retrieved chunks for improved relevance.
This addresses the issue identified in current_analysis.md where the reranker
defined in model_config.yaml was never used.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Lightweight (22M params)
- Good balance of speed and accuracy
- Works well for legal document reranking
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger
import os

# Try to import cross-encoder
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers CrossEncoder not available")


@dataclass
class RankedChunk:
    """A chunk with reranking score"""
    chunk_id: str
    text: str
    law_type: str
    section_num: Optional[str] = None
    case_name: Optional[str] = None
    original_score: float = 0.0
    rerank_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class RerankerAgent:
    """
    The Reranker Agent refines retrieval results using cross-encoder scoring.
    
    Unlike bi-encoders (InLegalBERT), cross-encoders process query-document pairs
    together, allowing for better relevance estimation at the cost of speed.
    
    Use after initial retrieval:
    1. LibrarianAgent retrieves top-20 candidates
    2. RerankerAgent reranks to top-5 most relevant
    """
    
    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    def __init__(
        self,
        model_name: str = None,
        top_k: int = 5,
        device: str = None
    ):
        """
        Initialize RerankerAgent.
        
        Args:
            model_name: Cross-encoder model name (default: ms-marco-MiniLM)
            top_k: Number of results to return after reranking
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.top_k = top_k
        self._model = None
        
        # Auto-detect device
        if device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device
    
    @property
    def model(self) -> Optional["CrossEncoder"]:
        """Lazy load cross-encoder model"""
        if not CROSS_ENCODER_AVAILABLE:
            return None
        
        if self._model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info(f"Cross-encoder loaded on {self.device}")
        
        return self._model
    
    def is_available(self) -> bool:
        """Check if reranker is available"""
        return CROSS_ENCODER_AVAILABLE
    
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[RankedChunk]:
        """
        Rerank chunks by relevance to query.
        
        Args:
            query: User's legal question
            chunks: List of retrieved chunks (from LibrarianAgent)
            top_k: Number of results to return (overrides default)
            
        Returns:
            List of RankedChunk sorted by rerank_score (descending)
        """
        if not chunks:
            return []
        
        top_k = top_k or self.top_k
        
        # If model not available, return original chunks
        if not self.model:
            logger.warning("Reranker not available, returning original order")
            return self._to_ranked_chunks(chunks[:top_k])
        
        try:
            # Create query-document pairs
            pairs = [(query, self._get_text(chunk)) for chunk in chunks]
            
            # Get cross-encoder scores
            scores = self.model.predict(pairs)
            
            # Create scored chunks
            ranked = []
            for chunk, score in zip(chunks, scores):
                ranked_chunk = RankedChunk(
                    chunk_id=chunk.get("chunk_id", ""),
                    text=self._get_text(chunk),
                    law_type=chunk.get("law_type", ""),
                    section_num=chunk.get("section_num"),
                    case_name=chunk.get("case_name"),
                    original_score=chunk.get("score", 0.0),
                    rerank_score=float(score),
                    metadata=chunk.get("metadata", {})
                )
                ranked.append(ranked_chunk)
            
            # Sort by rerank score (descending)
            ranked.sort(key=lambda x: x.rerank_score, reverse=True)
            
            logger.info(f"Reranked {len(chunks)} chunks, returning top {top_k}")
            return ranked[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return self._to_ranked_chunks(chunks[:top_k])
    
    def _get_text(self, chunk: Dict[str, Any]) -> str:
        """Extract text from chunk dict"""
        # Handle different chunk formats
        if isinstance(chunk, dict):
            return chunk.get("text", str(chunk))
        return str(chunk)
    
    def _to_ranked_chunks(self, chunks: List[Dict[str, Any]]) -> List[RankedChunk]:
        """Convert dicts to RankedChunk without reranking"""
        return [
            RankedChunk(
                chunk_id=c.get("chunk_id", ""),
                text=self._get_text(c),
                law_type=c.get("law_type", ""),
                section_num=c.get("section_num"),
                case_name=c.get("case_name"),
                original_score=c.get("score", 0.0),
                rerank_score=c.get("score", 0.0),  # Use original score
                metadata=c.get("metadata", {})
            )
            for c in chunks
        ]
    
    async def rerank_async(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[RankedChunk]:
        """
        Async wrapper for rerank using ThreadPoolExecutor.
        
        Use this in async endpoints to avoid blocking.
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        executor = ThreadPoolExecutor(max_workers=1)
        loop = asyncio.get_event_loop()
        
        return await loop.run_in_executor(
            executor,
            self.rerank,
            query,
            chunks,
            top_k
        )


# Singleton instance
_reranker: Optional[RerankerAgent] = None


def create_reranker_agent(
    model_name: str = None,
    top_k: int = 5,
    **kwargs
) -> RerankerAgent:
    """Factory function to create Reranker agent"""
    return RerankerAgent(
        model_name=model_name,
        top_k=top_k,
        **kwargs
    )


def get_reranker() -> RerankerAgent:
    """Get or create singleton Reranker agent"""
    global _reranker
    if _reranker is None:
        _reranker = create_reranker_agent()
    return _reranker

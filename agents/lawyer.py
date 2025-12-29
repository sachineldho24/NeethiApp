"""
Lawyer Agent - Legal Advice Synthesis Specialist

Role: Compassionate Indian legal advisor who explains laws in simple language
Goal: Generate accurate, actionable, empathetic legal advice
Tools: 
  - Primary: Ollama LLM (Llama3) for synthesized advice
  - Fallback: Template-based formatting if LLM unavailable
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from loguru import logger
import httpx
import re

# Import from librarian for type hints
from agents.librarian import RetrievedChunk


@dataclass
class LegalAdvice:
    """Structured legal advice response"""
    query: str
    advice: str
    sources: List[Dict[str, Any]]
    case_law: List[Dict[str, Any]] = None  # SC + Bail judgments
    llm_used: bool = False
    disclaimer: str = (
        "This is general legal information, not legal advice. "
        "Please consult a qualified lawyer for your specific situation."
    )


# ================== Prompt Templates ==================

LEGAL_SYNTHESIS_PROMPT = """You are an expert Indian legal advisor. Based on the following legal provisions and case law, provide a clear, synthesized legal opinion.

User Question: {query}

Relevant Legal Provisions:
{context}

STRICT INSTRUCTIONS:
1. Answer ONLY based on the provided Legal Provisions above
2. Do NOT use outside knowledge or make up information
3. Use citation markers like [Source 1], [Source 2] when referencing specific provisions
4. If the answer is not in the context, say: The provided legal sources do not cover this specific scenario
5. Use simple language a layperson can understand
6. Be concise but comprehensive
7. Output PLAIN TEXT only. DO NOT use asterisks, hashes, dashes, or any special formatting symbols
8. For lists use simple numbers followed by a closing parenthesis like 1) 2) 3)

Your Legal Opinion:"""


class OllamaClient:
    """Client for Ollama LLM API with timeout handling"""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        timeout: float = 60.0
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._available = None
    
    def is_available(self) -> bool:
        """Check if Ollama is running and model is available"""
        if self._available is not None:
            return self._available
        
        try:
            response = httpx.get(
                f"{self.base_url}/api/tags",
                timeout=5.0
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                self._available = self.model in model_names
                if self._available:
                    logger.info(f"✅ Ollama available with model: {self.model}")
                else:
                    logger.warning(f"Model {self.model} not found. Available: {model_names}")
            else:
                self._available = False
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            self._available = False
        
        return self._available
    
    def generate(self, prompt: str) -> Optional[str]:
        """Generate text using Ollama with timeout handling"""
        if not self.is_available():
            return None
        
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Lower temperature for factual responses
                        "num_predict": 1024  # Limit output length
                    }
                },
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                logger.error(f"Ollama error: {response.status_code}")
                return None
                
        except httpx.TimeoutException:
            logger.warning(f"Ollama timeout after {self.timeout}s")
            return None
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            return None


class LawyerAgent:
    """
    The Lawyer Agent synthesizes retrieved legal chunks into user-friendly advice.
    
    Tri-Mode LLM:
    1. Primary: Ollama (Llama3) for local inference
    2. Secondary: LLM Router (Groq/Gemini) for cloud LLMs
    3. Fallback: Template-based formatting if all LLMs unavailable
    """
    
    DISCLAIMER = (
        "This is general legal information, not legal advice. "
        "Please consult a qualified lawyer for your specific situation."
    )
    
    # Context window management
    MAX_CHARS_PER_CHUNK = 1000  # Truncate to prevent context overflow
    
    def __init__(self, ollama_client: OllamaClient = None, llm_router = None):
        """
        Initialize LawyerAgent.
        
        Args:
            ollama_client: Optional OllamaClient for local LLM synthesis
            llm_router: Optional LLMRouter for Groq/Gemini cloud LLMs
        """
        self.ollama = ollama_client or OllamaClient()
        self.llm_router = llm_router
        self._llm_available = None
    
    def _truncate_chunk(self, text: str, max_chars: int = None) -> str:
        """Truncate chunk text to prevent context overflow"""
        max_chars = max_chars or self.MAX_CHARS_PER_CHUNK
        if len(text) <= max_chars:
            return text
        # Truncate at sentence boundary if possible
        truncated = text[:max_chars]
        last_period = truncated.rfind(". ")
        if last_period > max_chars * 0.7:  # Only use if > 70% of text
            truncated = truncated[:last_period + 1]
        return truncated + "..."
    
    def _format_context(self, chunks: List[RetrievedChunk]) -> str:
        """Format chunks into context string with source markers"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            # Different truncation for different source types
            if chunk.law_type == "SC Judgment":
                # SC judgments need more context for verdicts
                text = self._truncate_chunk(chunk.text, max_chars=800)
                case_year = chunk.metadata.get("year", "") if chunk.metadata else ""
                source_header = f"[Source {i}] SC Judgment ({case_year})"
            elif chunk.law_type == "Bail Judgment":
                text = self._truncate_chunk(chunk.text, max_chars=600)
                source_header = f"[Source {i}] {chunk.law_type}"
            else:
                text = self._truncate_chunk(chunk.text)
                source_header = f"[Source {i}] {chunk.law_type} Section {chunk.section_num}"
            
            context_parts.append(f"{source_header}:\n{text}")
        
        return "\n\n".join(context_parts)
    
    def _generate_with_llm(self, query: str, chunks: List[RetrievedChunk]) -> Optional[str]:
        """Generate synthesized advice using Ollama LLM"""
        context = self._format_context(chunks)
        prompt = LEGAL_SYNTHESIS_PROMPT.format(query=query, context=context)
        
        logger.info(f"Generating LLM synthesis via Ollama (context: {len(context)} chars)")
        return self.ollama.generate(prompt)
    
    def _generate_with_router(self, query: str, chunks: List[RetrievedChunk]) -> Optional[str]:
        """Generate synthesized advice using LLM Router (Groq/Gemini)"""
        if not self.llm_router:
            return None
        
        context = self._format_context(chunks)
        prompt = LEGAL_SYNTHESIS_PROMPT.format(query=query, context=context)
        
        try:
            # Use LLM Router's generate method - returns LLMResponse dataclass
            response = self.llm_router.generate(prompt, use_legal_system_prompt=False)
            
            # Check if we got a valid response (not the fallback error message)
            if response and response.provider != "none":
                logger.info(f"✅ Generated advice via LLM Router ({response.provider})")
                return response.content
            else:
                logger.warning(f"LLM Router returned no valid response")
                return None
        except Exception as e:
            logger.error(f"LLM Router error: {e}")
            return None
    
    def _generate_heuristic(self, chunks: List[RetrievedChunk]) -> str:
        """Fallback: Generate advice using template-based formatting"""
        advice_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            law_type = chunk.law_type
            section_num = chunk.section_num
            score = chunk.score
            text = self._truncate_chunk(chunk.text, 500)  # Shorter for heuristic
            
            advice_parts.append(
                f"**{i}. {law_type} Section {section_num}** (Relevance: {score:.0%})\n{text}"
            )
        
        return (
            "Based on your query, here are the most relevant legal provisions:\n\n"
            + "\n\n---\n\n".join(advice_parts)
        )
    
    def _extract_sources(self, chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        """Extract source metadata for response (statutes only)"""
        return [
            {
                "source_id": i,
                "law_type": chunk.law_type,
                "section": chunk.section_num,
                "relevance_score": f"{chunk.score:.2f}"
            }
            for i, chunk in enumerate(chunks, 1)
            if chunk.law_type not in ["SC Judgment", "Bail Judgment"]
        ]
    
    def _summarize_case_for_layman(self, case_text: str, case_name: str) -> str:
        """Use LLM to create a simple summary for common people"""
        if len(case_text) < 200:
            # Too short to summarize
            return case_text[:400] + "..." if len(case_text) > 400 else case_text
        
        prompt = f"""Summarize this legal case in 2-3 simple sentences that a common person can understand. 
Focus on: What happened? What did the court decide? Why does it matter?
Do NOT use legal jargon. Write in plain simple English.

Case: {case_name}
Text: {case_text[:1500]}

Simple Summary:"""
        
        # Try Ollama first
        if self.ollama.is_available():
            try:
                summary = self.ollama.generate(prompt)
                if summary and len(summary) > 50:
                    return summary.strip()[:500]
            except Exception:
                pass
        
        # Try LLM Router (Groq/Gemini)
        if self.llm_router:
            try:
                response = self.llm_router.generate(prompt, use_legal_system_prompt=False)
                if response and response.provider != "none" and len(response.content) > 50:
                    logger.debug(f"SC summary via LLM Router ({response.provider})")
                    return response.content.strip()[:500]
            except Exception as e:
                logger.warning(f"LLM Router summarization failed: {e}")
        
        # Fallback: simple truncation
        return case_text[:400] + "..." if len(case_text) > 400 else case_text
    
    def _extract_case_law(self, chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
        """Extract case law (SC + Bail judgments) with LLM-summarized details"""
        case_law = []
        for chunk in chunks:
            if chunk.law_type in ["SC Judgment", "Bail Judgment"]:
                # Extract metadata
                year = ""
                doc_url = ""
                case_name = ""
                if chunk.metadata:
                    year = chunk.metadata.get("year", "")
                    doc_url = chunk.metadata.get("doc_url", "")  # Indian Kanoon URL
                
                # Use case_name from chunk if available
                if chunk.case_name and chunk.case_name != "Unknown":
                    case_name = chunk.case_name
                
                # Create simple summary using LLM (for SC judgments with enough text)
                if chunk.law_type == "SC Judgment" and len(chunk.text) > 300:
                    simple_summary = self._summarize_case_for_layman(chunk.text, case_name)
                else:
                    simple_summary = chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text
                
                case_entry = {
                    "case_type": chunk.law_type,
                    "case_id": chunk.section_num,
                    "case_name": case_name,
                    "year": year,
                    "summary": simple_summary,
                    "relevance_score": f"{chunk.score:.2f}"
                }
                
                # Add doc_url only if available (SC Judgments from 26K dataset)
                if doc_url:
                    case_entry["doc_url"] = doc_url
                
                case_law.append(case_entry)
        return case_law
    
    def generate_advice(
        self,
        query: str,
        chunks: List[RetrievedChunk]
    ) -> LegalAdvice:
        """
        Generate legal advice from retrieved chunks.
        
        Uses dual-mode:
        1. Primary: LLM synthesis (if available)
        2. Fallback: Template-based formatting
        
        Args:
            query: User's legal question
            chunks: Retrieved chunks from LibrarianAgent
            
        Returns:
            LegalAdvice object with synthesized response
        """
        logger.info(f"Lawyer generating advice for: {query[:50]}...")
        
        if not chunks:
            return LegalAdvice(
                query=query,
                advice="No relevant legal provisions found for your query. Please try rephrasing your question.",
                sources=[],
                llm_used=False
            )
        
        # Try LLM synthesis: Ollama first, then LLM Router (Groq/Gemini)
        llm_advice = None
        llm_provider = None
        
        # Priority 1: Ollama (local)
        if self.ollama.is_available():
            llm_advice = self._generate_with_llm(query, chunks)
            if llm_advice:
                llm_provider = "ollama"
        
        # Priority 2: LLM Router (Groq/Gemini cloud)
        if not llm_advice and self.llm_router:
            llm_advice = self._generate_with_router(query, chunks)
            if llm_advice:
                llm_provider = "cloud"
        
        # Determine final advice
        if llm_advice:
            advice_text = llm_advice
            llm_used = True
            logger.info(f"✅ Generated advice using LLM ({llm_provider})")
        else:
            advice_text = self._generate_heuristic(chunks)
            llm_used = False
            logger.info("⚠️ Using fallback heuristic formatting (all LLMs unavailable)")
        
        sources = self._extract_sources(chunks)
        case_law = self._extract_case_law(chunks)
        
        return LegalAdvice(
            query=query,
            advice=advice_text,
            sources=sources,
            case_law=case_law,
            llm_used=llm_used
        )
    
    def to_response_dict(
        self,
        advice: LegalAdvice,
        session_id: str,
        processing_time_ms: float
    ) -> Dict[str, Any]:
        """
        Convert LegalAdvice to API response dictionary.
        
        Args:
            advice: LegalAdvice object
            session_id: Session identifier
            processing_time_ms: Processing time in milliseconds
            
        Returns:
            Dictionary matching LegalQueryResponse schema
        """
        return {
            "session_id": session_id,
            "query": advice.query,
            "advice": advice.advice,
            "sources": advice.sources,
            "case_law": advice.case_law or [],
            "nearby_resources": None,
            "news": None,
            "disclaimer": advice.disclaimer,
            "llm_used": advice.llm_used,
            "processing_time_ms": processing_time_ms
        }


def create_lawyer_agent(ollama_url: str = None, model: str = "llama3") -> LawyerAgent:
    """Factory function to create a Lawyer agent with Ollama client"""
    ollama_client = OllamaClient(
        base_url=ollama_url or "http://localhost:11434",
        model=model
    )
    return LawyerAgent(ollama_client=ollama_client)

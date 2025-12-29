"""
Advice Crew - Legal Advice Workflow Orchestration

Coordinates: Librarian → Lawyer (→ Clerk → Journalist optional)
Purpose: Answer user's legal questions with cited sources
"""

from typing import Optional
from loguru import logger
import time

from crews.shared_state import LegalConsultationState


class AdviceCrew:
    """
    Orchestrates the legal advice workflow.
    
    Workflow:
    1. Librarian retrieves relevant legal documents
    2. Lawyer generates advice using retrieved context
    3. (Optional) Clerk finds nearby legal resources
    4. (Optional) Journalist adds relevant news
    """
    
    def __init__(
        self,
        librarian_agent,
        lawyer_agent,
        clerk_agent=None,
        journalist_agent=None
    ):
        self.librarian = librarian_agent
        self.lawyer = lawyer_agent
        self.clerk = clerk_agent
        self.journalist = journalist_agent
    
    async def execute(
        self,
        query: str,
        location: Optional[dict] = None,
        include_news: bool = False,
        include_locations: bool = False
    ) -> LegalConsultationState:
        """
        Execute the legal advice workflow.
        
        Args:
            query: User's legal question
            location: Optional user location for nearby services
            include_news: Whether to fetch related news
            include_locations: Whether to find nearby legal resources
            
        Returns:
            LegalConsultationState with all gathered information
        """
        state = LegalConsultationState(
            query=query,
            user_location=location
        )
        
        logger.info(f"AdviceCrew starting for query: {query[:50]}...")
        
        # Step 1: Retrieval (Required)
        try:
            start = time.time()
            chunks = await self.librarian.search(query)
            state.retrieved_chunks = chunks
            state.retrieval_completed = True
            state.record_timing("librarian", time.time() - start)
            logger.info(f"Librarian retrieved {len(chunks)} chunks")
        except Exception as e:
            state.add_error("librarian", str(e))
            logger.error(f"Librarian error: {e}")
            return state  # Can't proceed without retrieval
        
        # Step 2: Generation (Required)
        try:
            start = time.time()
            # Convert chunks to dict format for lawyer
            chunk_dicts = [
                {"text": c.text, "law_type": c.law_type, "section_num": c.section_num}
                for c in state.retrieved_chunks
            ]
            advice = await self.lawyer.generate_advice(query, chunk_dicts)
            state.generated_advice = advice.summary  # TODO: Full advice object
            state.advice_sources = advice.sources
            state.advice_completed = True
            state.record_timing("lawyer", time.time() - start)
            logger.info("Lawyer generated advice")
        except Exception as e:
            state.add_error("lawyer", str(e))
            logger.error(f"Lawyer error: {e}")
        
        # Step 3: Location (Optional)
        if include_locations and self.clerk and location:
            try:
                start = time.time()
                resources = await self.clerk.find_resources(
                    location.get("lat", 0),
                    location.get("lon", 0)
                )
                state.nearby_resources = [r.__dict__ for r in resources]
                state.location_completed = True
                state.record_timing("clerk", time.time() - start)
            except Exception as e:
                state.add_error("clerk", str(e))
                logger.error(f"Clerk error: {e}")
        
        # Step 4: News (Optional)
        if include_news and self.journalist:
            try:
                start = time.time()
                # Extract topic from query for filtering
                topic = self._extract_topic(query)
                news = await self.journalist.fetch_news(topic=topic, limit=3)
                state.relevant_news = [
                    {"title": n.title, "url": n.url, "source": n.source}
                    for n in news
                ]
                state.news_completed = True
                state.record_timing("journalist", time.time() - start)
            except Exception as e:
                state.add_error("journalist", str(e))
                logger.error(f"Journalist error: {e}")
        
        logger.info(f"AdviceCrew completed. Errors: {len(state.errors)}")
        return state
    
    def _extract_topic(self, query: str) -> Optional[str]:
        """Extract main topic from query for news filtering"""
        # Simple keyword extraction
        keywords = ["bail", "divorce", "property", "theft", "fraud", "murder"]
        query_lower = query.lower()
        for kw in keywords:
            if kw in query_lower:
                return kw
        return None

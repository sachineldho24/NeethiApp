"""
Online Pipeline - Query Processor

Handles incoming queries with sanitization, intent detection, and routing.
"""

import re
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from loguru import logger


class QueryIntent(Enum):
    """Detected intent of user query"""
    LEGAL_ADVICE = "legal_advice"
    DOCUMENT_DRAFT = "document_draft"
    LOCATION_SEARCH = "location_search"
    NEWS_SEARCH = "news_search"
    GENERAL_INFO = "general_info"
    UNCLEAR = "unclear"


@dataclass
class ProcessedQuery:
    """Processed and sanitized query"""
    original: str
    cleaned: str
    intent: QueryIntent
    entities: dict
    language: str = "en"


class QueryProcessor:
    """
    Processes and routes incoming user queries.
    
    Steps:
    1. Sanitize input (remove profanity, normalize text)
    2. Detect language
    3. Classify intent
    4. Extract entities (IPC sections, names, etc.)
    """
    
    # Intent detection keywords
    INTENT_KEYWORDS = {
        QueryIntent.DOCUMENT_DRAFT: [
            "draft", "write", "generate", "create", "fir", "rti", 
            "notice", "complaint", "application", "letter"
        ],
        QueryIntent.LOCATION_SEARCH: [
            "near", "nearby", "find", "where", "locate", "police station",
            "court", "lawyer", "advocate", "legal aid"
        ],
        QueryIntent.NEWS_SEARCH: [
            "news", "latest", "recent", "update", "judgment", "verdict",
            "today", "this week", "announced"
        ],
        QueryIntent.LEGAL_ADVICE: [
            "can i", "what if", "is it legal", "punishment", "bail",
            "section", "ipc", "bns", "law", "rights", "offense"
        ]
    }
    
    # Profanity/abuse filter (basic)
    BLOCKED_PATTERNS = [
        r'\b(kill|murder|attack)\s+(someone|person|people)\b',
        # Add more patterns as needed
    ]
    
    def __init__(self):
        self.entity_patterns = {
            "ipc_section": re.compile(r'(?:ipc|section)\s*(\d+[a-z]?)', re.I),
            "bns_section": re.compile(r'(?:bns|section)\s*(\d+)', re.I),
            "money_amount": re.compile(r'(?:rs\.?|₹|inr)\s*([\d,]+)', re.I),
            "date": re.compile(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}'),
        }
    
    def process(self, query: str) -> ProcessedQuery:
        """Process a user query"""
        logger.debug(f"Processing query: {query[:50]}...")
        
        # Step 1: Clean
        cleaned = self._sanitize(query)
        
        # Step 2: Check for blocked content
        if self._is_blocked(cleaned):
            logger.warning(f"Blocked query detected")
            return ProcessedQuery(
                original=query,
                cleaned="",
                intent=QueryIntent.UNCLEAR,
                entities={"blocked": True}
            )
        
        # Step 3: Detect intent
        intent = self._detect_intent(cleaned)
        
        # Step 4: Extract entities
        entities = self._extract_entities(cleaned)
        
        return ProcessedQuery(
            original=query,
            cleaned=cleaned,
            intent=intent,
            entities=entities
        )
    
    def _sanitize(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters except punctuation
        text = re.sub(r'[^\w\s.,?!₹-]', '', text)
        
        # Normalize common variations
        text = text.replace("IPC", "IPC").replace("BNS", "BNS")
        
        return text
    
    def _is_blocked(self, text: str) -> bool:
        """Check for harmful content"""
        text_lower = text.lower()
        
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    def _detect_intent(self, text: str) -> QueryIntent:
        """Classify query intent"""
        text_lower = text.lower()
        
        # Score each intent
        scores = {}
        for intent, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            scores[intent] = score
        
        # Get highest scoring intent
        if max(scores.values()) == 0:
            return QueryIntent.UNCLEAR
        
        best_intent = max(scores, key=scores.get)
        return best_intent
    
    def _extract_entities(self, text: str) -> dict:
        """Extract named entities from text"""
        entities = {}
        
        for entity_type, pattern in self.entity_patterns.items():
            matches = pattern.findall(text)
            if matches:
                entities[entity_type] = matches[0] if len(matches) == 1 else matches
        
        return entities


class SafetyFilter:
    """
    Content safety filter for responses.
    Ensures generated advice doesn't contain harmful content.
    """
    
    DISCLAIMERS = {
        "general": (
            "This is general legal information, not legal advice. "
            "Please consult a qualified lawyer for your specific situation."
        ),
        "bail": (
            "Bail eligibility depends on specific facts of your case. "
            "A lawyer can assess your situation and file appropriate applications."
        ),
        "criminal": (
            "Criminal matters require immediate professional legal assistance. "
            "Consider contacting a criminal lawyer or legal aid service."
        )
    }
    
    def add_disclaimer(self, response: str, query_type: str = "general") -> str:
        """Add appropriate disclaimer to response"""
        disclaimer = self.DISCLAIMERS.get(query_type, self.DISCLAIMERS["general"])
        return f"{response}\n\n---\n⚠️ **Disclaimer**: {disclaimer}"
    
    def check_response(self, response: str) -> Tuple[bool, Optional[str]]:
        """
        Check if response is safe to send.
        
        Returns: (is_safe, reason_if_unsafe)
        """
        # Check for empty response
        if not response or len(response.strip()) < 20:
            return False, "Response too short"
        
        # Check for obvious hallucination markers
        hallucination_markers = [
            "I don't have access to",
            "I cannot verify",
            "as an AI",
            "my training data"
        ]
        
        for marker in hallucination_markers:
            if marker.lower() in response.lower():
                logger.warning(f"Hallucination marker detected: {marker}")
                # Don't block, but flag
        
        return True, None

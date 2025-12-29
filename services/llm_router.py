"""
LLM Router - Multi-Provider LLM Orchestration

Implements automatic failover between LLM providers:
- Primary: Groq (Llama 3.1 70B) - Fast inference, generous free tier
- Fallback: Google Gemini Flash - Reliable backup

Features:
- Automatic rate limit detection and failover
- Response caching for repeated queries
- Configurable timeouts and retry logic
- Structured output support
"""

import os
import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from functools import lru_cache
from loguru import logger
import httpx

# Try to import providers
try:
    from groq import Groq, AsyncGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq package not installed")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai package not installed")


@dataclass
class LLMResponse:
    """Structured LLM response"""
    content: str
    provider: str
    model: str
    tokens_used: int = 0
    cached: bool = False


class GroqClient:
    """Groq API client for Llama 3.1 inference"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",
        timeout: float = 60.0,
        max_tokens: int = 1000,
        temperature: float = 0.3
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None
        self._async_client = None
        
        if not self.api_key:
            logger.warning("GROQ_API_KEY not set")
    
    @property
    def client(self) -> Optional["Groq"]:
        """Lazy initialization of sync client"""
        if not GROQ_AVAILABLE or not self.api_key:
            return None
        if self._client is None:
            self._client = Groq(api_key=self.api_key)
        return self._client
    
    @property
    def async_client(self) -> Optional["AsyncGroq"]:
        """Lazy initialization of async client"""
        if not GROQ_AVAILABLE or not self.api_key:
            return None
        if self._async_client is None:
            self._async_client = AsyncGroq(api_key=self.api_key)
        return self._async_client
    
    def is_available(self) -> bool:
        """Check if Groq is configured and available"""
        return GROQ_AVAILABLE and bool(self.api_key)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[LLMResponse]:
        """Synchronous generation"""
        if not self.client:
            return None
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider="groq",
                model=self.model,
                tokens_used=response.usage.total_tokens if response.usage else 0
            )
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            return None
    
    async def generate_async(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[LLMResponse]:
        """Asynchronous generation"""
        if not self.async_client:
            return None
        
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return LLMResponse(
                content=response.choices[0].message.content,
                provider="groq",
                model=self.model,
                tokens_used=response.usage.total_tokens if response.usage else 0
            )
        except Exception as e:
            logger.error(f"Groq async generation failed: {e}")
            return None


class GeminiClient:
    """Google Gemini API client"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        max_tokens: int = 1000,
        temperature: float = 0.3
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._model = None
        
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not set")
        elif GEMINI_AVAILABLE:
            genai.configure(api_key=self.api_key)
    
    @property
    def model(self):
        """Lazy initialization of Gemini model"""
        if not GEMINI_AVAILABLE or not self.api_key:
            return None
        if self._model is None:
            self._model = genai.GenerativeModel(self.model_name)
        return self._model
    
    def is_available(self) -> bool:
        """Check if Gemini is configured and available"""
        return GEMINI_AVAILABLE and bool(self.api_key)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[LLMResponse]:
        """Synchronous generation"""
        if not self.model:
            return None
        
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = self.model.generate_content(
                full_prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens
                }
            )
            
            return LLMResponse(
                content=response.text,
                provider="gemini",
                model=self.model_name,
                tokens_used=0  # Gemini doesn't return token count directly
            )
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return None
    
    async def generate_async(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[LLMResponse]:
        """Asynchronous generation"""
        if not self.model:
            return None
        
        try:
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            response = await self.model.generate_content_async(
                full_prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens
                }
            )
            
            return LLMResponse(
                content=response.text,
                provider="gemini",
                model=self.model_name,
                tokens_used=0
            )
        except Exception as e:
            logger.error(f"Gemini async generation failed: {e}")
            return None


class LLMRouter:
    """
    Multi-provider LLM router with automatic failover.
    
    Priority order:
    1. Groq (Llama 3.1 70B) - Primary, fast inference
    2. Gemini Flash - Fallback
    3. Ollama (local) - Offline fallback
    
    Features:
    - Automatic failover on rate limits or errors
    - Response caching (optional)
    - Configurable retry logic
    """
    
    LEGAL_SYSTEM_PROMPT = """You are an expert Indian legal advisor. Your role is to:
1. Provide accurate legal information based on Indian laws (IPC, BNS, CrPC, BNSS, etc.)
2. Explain legal concepts in simple language that a common citizen can understand
3. Always cite specific sections and acts when applicable
4. Include practical next steps for the user
5. Add appropriate disclaimers about seeking professional legal advice

Be empathetic, helpful, and thorough in your responses."""
    
    def __init__(
        self,
        groq_client: Optional[GroqClient] = None,
        gemini_client: Optional[GeminiClient] = None,
        enable_cache: bool = True,
        cache_size: int = 100
    ):
        self.groq = groq_client or GroqClient()
        self.gemini = gemini_client or GeminiClient()
        self.enable_cache = enable_cache
        self._cache: Dict[str, LLMResponse] = {}
        self._cache_size = cache_size
        
        # Track provider status
        self._groq_failures = 0
        self._gemini_failures = 0
        self._max_failures = 3
        
        logger.info(f"LLMRouter initialized - Groq: {self.groq.is_available()}, Gemini: {self.gemini.is_available()}")
    
    def _get_cache_key(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate cache key from prompt"""
        import hashlib
        content = f"{system_prompt or ''}|{prompt}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _check_cache(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[LLMResponse]:
        """Check cache for previous response"""
        if not self.enable_cache:
            return None
        key = self._get_cache_key(prompt, system_prompt)
        if key in self._cache:
            response = self._cache[key]
            response.cached = True
            logger.debug(f"Cache hit for prompt (key: {key[:8]}...)")
            return response
        return None
    
    def _update_cache(self, prompt: str, response: LLMResponse, system_prompt: Optional[str] = None):
        """Update cache with new response"""
        if not self.enable_cache:
            return
        
        # Simple LRU: remove oldest if at capacity
        if len(self._cache) >= self._cache_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        key = self._get_cache_key(prompt, system_prompt)
        self._cache[key] = response
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_legal_system_prompt: bool = True
    ) -> LLMResponse:
        """
        Generate response with automatic failover.
        
        Args:
            prompt: User prompt
            system_prompt: Optional custom system prompt
            use_legal_system_prompt: Use default legal system prompt if no custom provided
            
        Returns:
            LLMResponse with content and metadata
        """
        # Use legal system prompt by default
        if system_prompt is None and use_legal_system_prompt:
            system_prompt = self.LEGAL_SYSTEM_PROMPT
        
        # Check cache
        cached = self._check_cache(prompt, system_prompt)
        if cached:
            return cached
        
        # Try Groq first (unless too many failures)
        if self.groq.is_available() and self._groq_failures < self._max_failures:
            response = self.groq.generate(prompt, system_prompt)
            if response:
                self._groq_failures = 0  # Reset on success
                self._update_cache(prompt, response, system_prompt)
                logger.info(f"LLM response from Groq ({response.tokens_used} tokens)")
                return response
            else:
                self._groq_failures += 1
                logger.warning(f"Groq failed, failures: {self._groq_failures}")
        
        # Fallback to Gemini
        if self.gemini.is_available() and self._gemini_failures < self._max_failures:
            response = self.gemini.generate(prompt, system_prompt)
            if response:
                self._gemini_failures = 0
                self._update_cache(prompt, response, system_prompt)
                logger.info(f"LLM response from Gemini (fallback)")
                return response
            else:
                self._gemini_failures += 1
                logger.warning(f"Gemini failed, failures: {self._gemini_failures}")
        
        # All providers failed - return error response
        logger.error("All LLM providers failed")
        return LLMResponse(
            content="I apologize, but I'm currently unable to process your request. Please try again later.",
            provider="none",
            model="none",
            tokens_used=0
        )
    
    async def generate_async(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        use_legal_system_prompt: bool = True
    ) -> LLMResponse:
        """Async version of generate with failover"""
        if system_prompt is None and use_legal_system_prompt:
            system_prompt = self.LEGAL_SYSTEM_PROMPT
        
        # Check cache
        cached = self._check_cache(prompt, system_prompt)
        if cached:
            return cached
        
        # Try Groq first
        if self.groq.is_available() and self._groq_failures < self._max_failures:
            response = await self.groq.generate_async(prompt, system_prompt)
            if response:
                self._groq_failures = 0
                self._update_cache(prompt, response, system_prompt)
                return response
            else:
                self._groq_failures += 1
        
        # Fallback to Gemini
        if self.gemini.is_available() and self._gemini_failures < self._max_failures:
            response = await self.gemini.generate_async(prompt, system_prompt)
            if response:
                self._gemini_failures = 0
                self._update_cache(prompt, response, system_prompt)
                return response
            else:
                self._gemini_failures += 1
        
        return LLMResponse(
            content="I apologize, but I'm currently unable to process your request. Please try again later.",
            provider="none",
            model="none",
            tokens_used=0
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get router status for health checks"""
        return {
            "groq": {
                "available": self.groq.is_available(),
                "failures": self._groq_failures,
                "model": self.groq.model
            },
            "gemini": {
                "available": self.gemini.is_available(),
                "failures": self._gemini_failures,
                "model": self.gemini.model_name
            },
            "cache": {
                "enabled": self.enable_cache,
                "size": len(self._cache),
                "max_size": self._cache_size
            }
        }
    
    def reset_failures(self):
        """Reset failure counters (useful after transient issues resolve)"""
        self._groq_failures = 0
        self._gemini_failures = 0
        logger.info("LLM Router failure counters reset")


# Singleton instance
_llm_router: Optional[LLMRouter] = None


def create_llm_router(
    groq_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    **kwargs
) -> LLMRouter:
    """Factory function to create LLM Router"""
    groq_client = GroqClient(api_key=groq_api_key) if groq_api_key else GroqClient()
    gemini_client = GeminiClient(api_key=gemini_api_key) if gemini_api_key else GeminiClient()
    
    return LLMRouter(
        groq_client=groq_client,
        gemini_client=gemini_client,
        **kwargs
    )


def get_llm_router() -> LLMRouter:
    """Get or create singleton LLM Router instance"""
    global _llm_router
    if _llm_router is None:
        _llm_router = create_llm_router()
    return _llm_router

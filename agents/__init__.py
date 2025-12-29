"""
Neethi App - Agents Package
Contains all CrewAI agent definitions and specialized agents
"""

from agents.librarian import create_librarian_agent, LibrarianAgent
from agents.lawyer import create_lawyer_agent, LawyerAgent
from agents.clerk import create_clerk_agent, ClerkAgent
from agents.journalist import create_journalist_agent, JournalistAgent
from agents.scribe import create_scribe_agent, ScribeAgent
from agents.reranker import create_reranker_agent, RerankerAgent, get_reranker

__all__ = [
    # Core agents
    "create_librarian_agent",
    "create_lawyer_agent", 
    "create_clerk_agent",
    "create_journalist_agent",
    "create_scribe_agent",
    # Specialized agents
    "create_reranker_agent",
    # Agent classes
    "LibrarianAgent",
    "LawyerAgent",
    "ClerkAgent",
    "JournalistAgent",
    "ScribeAgent",
    "RerankerAgent",
    # Singletons
    "get_reranker",
]


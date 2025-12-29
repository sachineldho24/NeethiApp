"""
Neethi App - Crews Package
Multi-agent orchestration for different workflows
"""

from crews.advice_crew import AdviceCrew
from crews.document_crew import DocumentCrew
from crews.shared_state import LegalConsultationState

__all__ = [
    "AdviceCrew",
    "DocumentCrew",
    "LegalConsultationState",
]

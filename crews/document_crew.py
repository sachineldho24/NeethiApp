"""
Document Crew - Document Drafting Workflow Orchestration

Coordinates: NER Extraction → Validation → Scribe
Purpose: Generate legal documents (FIR, RTI, Notices) from user input
"""

from typing import Dict, Any, Optional
from loguru import logger

from crews.shared_state import DocumentDraftingState
from agents.scribe import DocumentRequest


class DocumentCrew:
    """
    Orchestrates the document drafting workflow.
    
    Workflow:
    1. Extract fields from user input (NER or structured)
    2. Validate required fields are present
    3. Ask for missing fields (if any)
    4. Generate document using Scribe agent
    """
    
    def __init__(self, scribe_agent, ner_model=None):
        self.scribe = scribe_agent
        self.ner = ner_model  # Optional NER for field extraction
    
    async def execute(
        self,
        doc_type: str,
        user_inputs: Dict[str, Any]
    ) -> DocumentDraftingState:
        """
        Execute the document drafting workflow.
        
        Args:
            doc_type: Type of document ("FIR", "RTI", "NOTICE", "COMPLAINT")
            user_inputs: Dictionary of user-provided field values
            
        Returns:
            DocumentDraftingState with generated document or missing fields
        """
        state = DocumentDraftingState(
            doc_type=doc_type,
            user_inputs=user_inputs
        )
        
        logger.info(f"DocumentCrew starting for {doc_type}")
        
        # Step 1: Extract/validate fields
        if self.ner:
            # Use NER to extract fields from unstructured text
            extracted = await self._extract_with_ner(user_inputs)
            state.extracted_fields = {**user_inputs, **extracted}
        else:
            state.extracted_fields = user_inputs
        
        # Step 2: Check for missing required fields
        schema = self.scribe.DOCUMENT_SCHEMAS.get(doc_type, {})
        required = schema.get("required", [])
        
        missing = []
        for field in required:
            if field not in state.extracted_fields or not state.extracted_fields[field]:
                missing.append(field)
        
        if missing:
            state.missing_fields = missing
            logger.info(f"Missing required fields: {missing}")
            return state  # Return early so caller can request missing info
        
        # Step 3: Generate document
        try:
            request = DocumentRequest(
                doc_type=doc_type,
                details=state.extracted_fields
            )
            result = await self.scribe.generate_document(request)
            
            if result:
                state.document_path = result.filename
                state.is_complete = True
                logger.info(f"Document generated: {result.filename}")
            else:
                state.errors.append("Failed to generate document")
        except Exception as e:
            state.errors.append(str(e))
            logger.error(f"Scribe error: {e}")
        
        return state
    
    async def _extract_with_ner(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured fields from unstructured text using NER"""
        # TODO: Implement NER extraction
        # This would use spaCy or a fine-tuned model to extract:
        # - Names (PERSON)
        # - Addresses (GPE, LOC)
        # - Dates (DATE)
        # - Organizations (ORG)
        return {}
    
    def get_field_prompts(self, doc_type: str) -> Dict[str, str]:
        """Get user-friendly prompts for each field"""
        prompts = {
            "FIR": {
                "complainant_name": "Your full name (as per ID proof)",
                "complainant_address": "Your complete residential address",
                "incident_date": "Date and time of the incident",
                "incident_place": "Location where the incident occurred",
                "incident_description": "Detailed description of what happened",
                "accused_details": "Details of the accused (name, description, etc.)",
            },
            "RTI": {
                "applicant_name": "Your full name",
                "applicant_address": "Your complete address for correspondence",
                "public_authority": "Name of the government department/authority",
                "information_sought": "Specific information you are seeking",
            },
            "NOTICE": {
                "sender_name": "Your name (sender)",
                "sender_address": "Your complete address",
                "recipient_name": "Name of the person you're sending notice to",
                "recipient_address": "Address of the recipient",
                "subject": "Subject/topic of the notice",
                "notice_content": "Main content/demands of the notice",
            }
        }
        return prompts.get(doc_type, {})

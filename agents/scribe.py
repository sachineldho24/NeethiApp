"""
Scribe Agent - Document Drafting

Role: Generate legal documents (FIR, RTI, Legal Notices)
Goal: Create properly formatted legal documents from user inputs
Tools: Document templates, NER extraction, DOCX generation
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from loguru import logger
import os


@dataclass
class DocumentRequest:
    """Request for document generation"""
    doc_type: str  # "FIR", "RTI", "NOTICE", "COMPLAINT"
    details: Dict[str, Any]
    language: str = "en"


@dataclass
class GeneratedDocument:
    """A generated legal document"""
    doc_type: str
    filename: str
    content: str
    format: str  # "docx", "pdf", "txt"
    created_at: datetime


class ScribeAgent:
    """
    The Scribe Agent handles document generation.
    
    Workflow:
    1. Extract required fields from user input (using NER)
    2. Validate all required fields are present
    3. Fill document template with extracted data
    4. Generate formatted document (DOCX/PDF)
    """
    
    # Required fields for each document type
    DOCUMENT_SCHEMAS = {
        "FIR": {
            "required": [
                "complainant_name", "complainant_address", "incident_date",
                "incident_place", "incident_description", "accused_details"
            ],
            "optional": ["witnesses", "evidence", "phone_number"]
        },
        "RTI": {
            "required": [
                "applicant_name", "applicant_address", "public_authority",
                "information_sought"
            ],
            "optional": ["bpl_status", "fee_paid"]
        },
        "NOTICE": {
            "required": [
                "sender_name", "sender_address", "recipient_name",
                "recipient_address", "subject", "notice_content"
            ],
            "optional": ["deadline", "consequences"]
        },
        "COMPLAINT": {
            "required": [
                "complainant_name", "complainant_address",
                "authority_name", "complaint_subject", "complaint_details"
            ],
            "optional": ["supporting_documents", "relief_sought"]
        }
    }
    
    def __init__(
        self,
        templates_dir: str = "templates/",
        output_dir: str = "output/"
    ):
        self.templates_dir = templates_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    async def generate_document(
        self,
        request: DocumentRequest
    ) -> Optional[GeneratedDocument]:
        """
        Generate a legal document.
        
        Args:
            request: Document request with type and details
            
        Returns:
            Generated document or None if validation fails
        """
        logger.info(f"Scribe generating {request.doc_type} document")
        
        # Validate document type
        if request.doc_type not in self.DOCUMENT_SCHEMAS:
            logger.error(f"Unknown document type: {request.doc_type}")
            return None
        
        # Validate required fields
        missing = self._validate_fields(request)
        if missing:
            logger.warning(f"Missing required fields: {missing}")
            return None
        
        # Generate document content
        content = self._fill_template(request)
        
        # Create output file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{request.doc_type}_{timestamp}.txt"
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info(f"Document generated: {filename}")
        
        return GeneratedDocument(
            doc_type=request.doc_type,
            filename=filename,
            content=content,
            format="txt",
            created_at=datetime.now()
        )
    
    def _validate_fields(self, request: DocumentRequest) -> list:
        """Check for missing required fields"""
        schema = self.DOCUMENT_SCHEMAS[request.doc_type]
        missing = []
        
        for field in schema["required"]:
            if field not in request.details or not request.details[field]:
                missing.append(field)
        
        return missing
    
    def _fill_template(self, request: DocumentRequest) -> str:
        """Fill document template with provided details"""
        # TODO: Implement proper template loading from files
        # For now, use inline templates
        
        if request.doc_type == "FIR":
            return self._fir_template(request.details)
        elif request.doc_type == "RTI":
            return self._rti_template(request.details)
        elif request.doc_type == "NOTICE":
            return self._notice_template(request.details)
        elif request.doc_type == "COMPLAINT":
            return self._complaint_template(request.details)
        
        return ""
    
    def _fir_template(self, details: Dict) -> str:
        """FIR document template"""
        return f"""
FIRST INFORMATION REPORT (FIR)

To,
The Station House Officer
[Police Station Name]

Subject: Complaint regarding {details.get('incident_description', 'N/A')[:50]}...

Respected Sir/Madam,

I, {details.get('complainant_name', '[Complainant Name]')}, residing at 
{details.get('complainant_address', '[Address]')}, hereby lodge this complaint 
as follows:

1. DATE & TIME OF INCIDENT: {details.get('incident_date', '[Date]')}

2. PLACE OF INCIDENT: {details.get('incident_place', '[Place]')}

3. DETAILS OF INCIDENT:
{details.get('incident_description', '[Description]')}

4. ACCUSED PERSON(S):
{details.get('accused_details', '[Accused Details]')}

5. WITNESSES (if any):
{details.get('witnesses', 'None')}

I request you to register an FIR and take necessary action against the accused.

Thanking you,

Name: {details.get('complainant_name', '')}
Date: {datetime.now().strftime('%d/%m/%Y')}
Signature: _______________
"""

    def _rti_template(self, details: Dict) -> str:
        """RTI application template"""
        return f"""
APPLICATION UNDER RIGHT TO INFORMATION ACT, 2005

To,
The Public Information Officer
{details.get('public_authority', '[Authority Name]')}

Subject: Request for Information under RTI Act, 2005

Respected Sir/Madam,

I, {details.get('applicant_name', '[Applicant Name]')}, would like to seek the 
following information under the RTI Act, 2005:

INFORMATION SOUGHT:
{details.get('information_sought', '[Information Details]')}

Personal Details:
Name: {details.get('applicant_name', '')}
Address: {details.get('applicant_address', '[Address]')}
BPL Status: {details.get('bpl_status', 'No')}

Fee Details:
Payment Mode: {details.get('fee_paid', 'Postal Order / DD')}

I hereby declare that I am a citizen of India.

Date: {datetime.now().strftime('%d/%m/%Y')}
Place: {details.get('applicant_address', '').split(',')[-1] if details.get('applicant_address') else ''}

Signature: _______________
"""

    def _notice_template(self, details: Dict) -> str:
        """Legal notice template"""
        return f"""
LEGAL NOTICE

From:
{details.get('sender_name', '[Sender Name]')}
{details.get('sender_address', '[Sender Address]')}

To:
{details.get('recipient_name', '[Recipient Name]')}
{details.get('recipient_address', '[Recipient Address]')}

Date: {datetime.now().strftime('%d/%m/%Y')}

Subject: {details.get('subject', '[Subject]')}

Dear Sir/Madam,

Under instructions from my client, I hereby serve upon you the following notice:

{details.get('notice_content', '[Notice Content]')}

You are hereby called upon to {details.get('relief_sought', 'comply with the above')} 
within {details.get('deadline', '15 days')} from the receipt of this notice, 
failing which my client shall be constrained to initiate appropriate legal 
proceedings against you.

Please treat this as a LEGAL NOTICE.

Yours faithfully,

{details.get('sender_name', '')}
"""

    def _complaint_template(self, details: Dict) -> str:
        """General complaint template"""
        return f"""
COMPLAINT

To,
{details.get('authority_name', '[Authority Name]')}

Subject: {details.get('complaint_subject', '[Subject]')}

Respected Sir/Madam,

I, {details.get('complainant_name', '[Name]')}, residing at 
{details.get('complainant_address', '[Address]')}, hereby submit this complaint 
for your kind consideration:

DETAILS OF COMPLAINT:
{details.get('complaint_details', '[Details]')}

RELIEF SOUGHT:
{details.get('relief_sought', 'Appropriate action as per law')}

I request you to look into this matter and take necessary action.

Thanking you,

Name: {details.get('complainant_name', '')}
Date: {datetime.now().strftime('%d/%m/%Y')}
Signature: _______________
"""


def create_scribe_agent(**kwargs) -> ScribeAgent:
    """Factory function to create a Scribe agent"""
    return ScribeAgent(**kwargs)

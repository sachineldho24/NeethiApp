"""
Statute Expert Agent - IPC/BNS Section Identification

This agent specializes in identifying applicable legal sections from incident
descriptions and providing structured information about statutes.

Key capabilities:
- Identify applicable IPC/BNS/CrPC sections from natural language
- Explain what each section covers
- Classify offenses (cognizable/non-cognizable, bailable/non-bailable)
- Map between old IPC sections and new BNS sections
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger
import os


@dataclass
class StatuteSection:
    """Represents a legal statute section"""
    section_num: str
    act: str  # IPC, BNS, CrPC, BNSS, BSA
    title: str
    description: str
    punishment: Optional[str] = None
    is_cognizable: bool = False
    is_bailable: bool = True
    equivalent_section: Optional[str] = None  # Cross-reference (IPC<->BNS)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "section": f"{self.act} {self.section_num}",
            "title": self.title,
            "description": self.description,
            "punishment": self.punishment,
            "cognizable": self.is_cognizable,
            "bailable": self.is_bailable,
            "equivalent": self.equivalent_section
        }


@dataclass
class SectionIdentificationResult:
    """Result of section identification"""
    primary_sections: List[StatuteSection] = field(default_factory=list)
    secondary_sections: List[StatuteSection] = field(default_factory=list)
    offense_type: str = ""  # "criminal", "civil", "administrative"
    recommended_action: str = ""
    confidence: float = 0.0


# Common IPC to BNS mappings
IPC_BNS_MAP = {
    # Theft and property
    "379": ("303", "Theft"),
    "380": ("304", "Theft in dwelling house"),
    "381": ("305", "Theft by clerk or servant"),
    "382": ("306", "Theft after preparation for causing death or hurt"),
    "383": ("307", "Extortion"),
    "384": ("308", "Punishment for extortion"),
    "390": ("309", "Robbery"),
    "392": ("310", "Punishment for robbery"),
    # Hurt and assault
    "319": ("114", "Hurt"),
    "320": ("115", "Grievous hurt"),
    "321": ("116", "Voluntarily causing hurt"),
    "322": ("117", "Voluntarily causing grievous hurt"),
    "323": ("118", "Punishment for voluntarily causing hurt"),
    "324": ("119", "Voluntarily causing hurt by dangerous weapons"),
    "325": ("120", "Punishment for voluntarily causing grievous hurt"),
    # Murder and homicide
    "299": ("100", "Culpable homicide"),
    "300": ("101", "Murder"),
    "302": ("103", "Punishment for murder"),
    "304": ("105", "Punishment for culpable homicide not amounting to murder"),
    "304A": ("106", "Causing death by negligence"),
    "304B": ("80", "Dowry death"),
    # Cheating and fraud
    "415": ("316", "Cheating"),
    "417": ("318", "Punishment for cheating"),
    "420": ("319", "Cheating and dishonestly inducing delivery of property"),
    # Defamation
    "499": ("354", "Defamation"),
    "500": ("355", "Punishment for defamation"),
    # Criminal intimidation
    "503": ("349", "Criminal intimidation"),
    "506": ("350", "Punishment for criminal intimidation"),
    # Trespass
    "441": ("327", "Criminal trespass"),
    "442": ("328", "House-trespass"),
    "447": ("329", "Punishment for criminal trespass"),
    "448": ("330", "House-trespass"),
    # Mischief
    "425": ("322", "Mischief"),
    "426": ("323", "Punishment for mischief"),
    "427": ("324", "Mischief causing damage to amount of fifty rupees"),
}


class StatuteExpertAgent:
    """
    Expert agent for statute identification and legal classification.
    
    Uses LLM to identify applicable sections from incident descriptions,
    then enriches with structured data from statute database.
    """
    
    def __init__(self, llm_router=None, statute_db=None):
        """
        Initialize StatuteExpertAgent.
        
        Args:
            llm_router: LLM router for section identification
            statute_db: Database/dict of statute information
        """
        self.llm_router = llm_router
        self.statute_db = statute_db or {}
        
    def _get_llm_router(self):
        """Lazy load LLM router"""
        if self.llm_router is None:
            try:
                from services.llm_router import get_llm_router
                self.llm_router = get_llm_router()
            except ImportError:
                logger.warning("LLM Router not available")
        return self.llm_router
    
    def get_ipc_equivalent(self, bns_section: str) -> Optional[str]:
        """Get IPC equivalent of a BNS section"""
        for ipc, (bns, _) in IPC_BNS_MAP.items():
            if bns == bns_section:
                return ipc
        return None
    
    def get_bns_equivalent(self, ipc_section: str) -> Optional[str]:
        """Get BNS equivalent of an IPC section"""
        if ipc_section in IPC_BNS_MAP:
            return IPC_BNS_MAP[ipc_section][0]
        return None
    
    def identify_sections(
        self,
        incident_description: str,
        prefer_act: str = "BNS"
    ) -> SectionIdentificationResult:
        """
        Identify applicable legal sections from an incident description.
        
        Args:
            incident_description: Natural language description of the incident
            prefer_act: Preferred act (BNS for new law, IPC for old references)
            
        Returns:
            SectionIdentificationResult with identified sections
        """
        llm = self._get_llm_router()
        
        if not llm:
            logger.warning("LLM not available for section identification")
            return SectionIdentificationResult(
                recommended_action="Please consult a lawyer for section identification"
            )
        
        prompt = f"""Analyze this incident and identify the most applicable Indian legal sections.

INCIDENT: {incident_description}

Instructions:
1. Identify PRIMARY sections (main offenses) and SECONDARY sections (related offenses)
2. Use BNS (Bharatiya Nyaya Sanhita) sections for new cases
3. Include the equivalent IPC section for reference
4. Specify if each offense is cognizable (police can arrest without warrant) and bailable

Return in this exact format:
PRIMARY SECTIONS:
- BNS <number>: <offense name> | Cognizable: <yes/no> | Bailable: <yes/no> | IPC Equivalent: <number>

SECONDARY SECTIONS:
- BNS <number>: <offense name> | Cognizable: <yes/no> | Bailable: <yes/no> | IPC Equivalent: <number>

OFFENSE TYPE: <criminal/civil/administrative>

RECOMMENDED ACTION: <what should the person do next>

CONFIDENCE: <percentage>"""

        try:
            response = llm.generate(prompt, use_legal_system_prompt=True)
            return self._parse_section_response(response.content)
        except Exception as e:
            logger.error(f"Section identification failed: {e}")
            return SectionIdentificationResult(
                recommended_action=f"Error during identification: {str(e)}"
            )
    
    def _parse_section_response(self, response_text: str) -> SectionIdentificationResult:
        """Parse LLM response into structured result"""
        result = SectionIdentificationResult()
        
        lines = response_text.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            if 'PRIMARY SECTIONS' in line.upper():
                current_section = 'primary'
            elif 'SECONDARY SECTIONS' in line.upper():
                current_section = 'secondary'
            elif line.startswith('OFFENSE TYPE:'):
                result.offense_type = line.replace('OFFENSE TYPE:', '').strip().lower()
            elif line.startswith('RECOMMENDED ACTION:'):
                result.recommended_action = line.replace('RECOMMENDED ACTION:', '').strip()
            elif line.startswith('CONFIDENCE:'):
                try:
                    conf_str = line.replace('CONFIDENCE:', '').replace('%', '').strip()
                    result.confidence = float(conf_str) / 100
                except:
                    result.confidence = 0.5
            elif line.startswith('- ') and current_section:
                section = self._parse_section_line(line[2:])
                if section:
                    if current_section == 'primary':
                        result.primary_sections.append(section)
                    else:
                        result.secondary_sections.append(section)
        
        return result
    
    def _parse_section_line(self, line: str) -> Optional[StatuteSection]:
        """Parse a single section line"""
        try:
            # Expected format: "BNS 303: Theft | Cognizable: yes | Bailable: yes | IPC: 379"
            parts = line.split('|')
            
            # Parse section and title
            section_part = parts[0].strip()
            if ':' in section_part:
                act_section, title = section_part.split(':', 1)
                act_section = act_section.strip()
                title = title.strip()
            else:
                act_section = section_part
                title = ""
            
            # Determine act and section number
            if act_section.upper().startswith('BNS'):
                act = "BNS"
                section_num = act_section.replace('BNS', '').strip()
            elif act_section.upper().startswith('IPC'):
                act = "IPC"
                section_num = act_section.replace('IPC', '').strip()
            elif act_section.upper().startswith('CRPC'):
                act = "CrPC"
                section_num = act_section.replace('CRPC', '').strip()
            else:
                act = "BNS"
                section_num = act_section
            
            # Parse cognizable/bailable
            is_cognizable = False
            is_bailable = True
            equivalent = None
            
            for part in parts[1:]:
                part = part.strip().lower()
                if 'cognizable' in part and 'yes' in part:
                    is_cognizable = True
                if 'bailable' in part and 'no' in part:
                    is_bailable = False
                if 'ipc' in part or 'equivalent' in part:
                    # Extract equivalent section
                    import re
                    nums = re.findall(r'\d+', part)
                    if nums:
                        equivalent = f"IPC {nums[0]}"
            
            return StatuteSection(
                section_num=section_num,
                act=act,
                title=title,
                description="",  # Would be filled from database
                is_cognizable=is_cognizable,
                is_bailable=is_bailable,
                equivalent_section=equivalent
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse section line: {line}, error: {e}")
            return None
    
    def lookup_section(
        self,
        section_num: str,
        act: str = "BNS"
    ) -> Optional[StatuteSection]:
        """
        Look up detailed information about a specific section.
        
        Args:
            section_num: Section number (e.g., "303")
            act: Act name (BNS, IPC, CrPC)
            
        Returns:
            StatuteSection with full details
        """
        # Check local database first
        key = f"{act}_{section_num}"
        if key in self.statute_db:
            return self.statute_db[key]
        
        # Try to get equivalent mapping
        equivalent = None
        if act == "IPC":
            equivalent = self.get_bns_equivalent(section_num)
            if equivalent:
                equivalent = f"BNS {equivalent}"
        elif act == "BNS":
            equivalent = self.get_ipc_equivalent(section_num)
            if equivalent:
                equivalent = f"IPC {equivalent}"
        
        # Return basic section from mapping
        if act == "IPC" and section_num in IPC_BNS_MAP:
            bns_num, title = IPC_BNS_MAP[section_num]
            return StatuteSection(
                section_num=section_num,
                act="IPC",
                title=title,
                description=f"See BNS {bns_num} for current law",
                equivalent_section=f"BNS {bns_num}"
            )
        
        # For BNS, reverse lookup
        for ipc, (bns, title) in IPC_BNS_MAP.items():
            if bns == section_num:
                return StatuteSection(
                    section_num=section_num,
                    act="BNS",
                    title=title,
                    description="",
                    equivalent_section=f"IPC {ipc}"
                )
        
        return None


def create_statute_expert_agent(llm_router=None, **kwargs) -> StatuteExpertAgent:
    """Factory function to create StatuteExpertAgent"""
    return StatuteExpertAgent(llm_router=llm_router, **kwargs)

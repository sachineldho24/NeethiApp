"""
Librarian Agent - Retrieval Specialist

Role: Expert legal researcher who locates relevant laws, cases, and precedents
Goal: Find the 5 most relevant legal documents for any user question
Tools: Qdrant vector search, keyword matching, primary section lookup
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger
from pathlib import Path
import json
import re

# For base model loading (SC judgments)
from sentence_transformers import SentenceTransformer


# ================== Load IPC-BNS Mapping ==================
def _load_ipc_bns_mapping() -> Dict[str, Dict[str, List[str]]]:
    """Load the IPC↔BNS equivalents mapping from JSON file."""
    mapping_file = Path(__file__).parent.parent / "data" / "mappings" / "ipc_bns_equivalents.json"
    if mapping_file.exists():
        with open(mapping_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Remove comments
            return {k: v for k, v in data.items() if not k.startswith("_")}
    return {}

IPC_BNS_MAPPING = _load_ipc_bns_mapping()


@dataclass
class RetrievedChunk:
    """A single retrieved legal document chunk"""
    chunk_id: str
    text: str
    law_type: str  # "IPC", "BNS", "BSA", "Judgment"
    section_num: Optional[str] = None
    case_name: Optional[str] = None
    score: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)


# ================== Legal Knowledge Base ==================

# Keywords for text matching (used for hybrid search)
LEGAL_KEYWORDS = {
    'theft': ['theft', 'steal', 'stolen', 'thief', 'dishonestly', 'movable property'],
    'murder': ['murder', 'homicide', 'culpable homicide'],
    'assault': ['assault', 'voluntarily causing hurt', 'grievous hurt'],
    'fraud': ['cheat', 'cheating', 'fraud', 'deceive', 'dishonestly inducing'],
    'rape': ['rape', 'sexual assault', 'molestation', 'outraging modesty'],
    'dowry': ['dowry', 'cruelty by husband', 'demand for dowry'],
    'kidnap': ['kidnap', 'kidnapping', 'abduct', 'abduction', 'wrongful confinement'],
    'defamation': ['defamation', 'imputation', 'reputation'],
    'robbery': ['robbery', 'roberry', 'robery', 'dacoity', 'extortion', 'loot'],
    'bail': ['bail', 'bailable', 'non-bailable'],
}

# PRIMARY SECTIONS: Direct section number lookup per crime category
# These are the EXACT sections that should appear first for each query type
PRIMARY_SECTIONS = {
    'theft': ['378', '379', '380', '381', '382'],
    'murder': ['302', '300', '299', '301', '304'],
    'assault': ['323', '324', '325', '326', '351', '352', '319', '320', '321', '322'],
    'fraud': ['420', '415', '416', '417', '418', '419'],
    'rape': ['375', '376', '354', '354A', '354B'],
    'dowry': ['498A', '304B', '406'],
    'kidnap': ['359', '360', '361', '362', '363', '364', '364A', '365', '366'],
    'defamation': ['499', '500', '501', '502'],
    'robbery': ['390', '391', '392', '393', '394', '395', '396', '397', '398'],
}


class LibrarianAgent:
    """
    The Librarian Agent handles all retrieval operations.
    
    Implements a 3-tier priority-based retrieval:
    1. Primary section matches (exact section numbers for detected category)
    2. Keyword matches in text (scored by match count)
    3. Semantic search fallback (vector similarity)
    
    Supports multi-collection search:
    - neethi-legal-kb: Statutes (IPC, BNS, BSA)
    - neethi-bail-judgments: Case law (bail decisions)
    """
    
    # Collection names
    STATUTES_COLLECTION = "neethi-legal-kb"
    BAIL_COLLECTION = "neethi-bail-judgments"
    SC_COLLECTION = "neethi-judgments"
    
    # Base model for SC judgments (must match what was used during ingestion)
    SC_BASE_MODEL = "law-ai/InLegalBERT"
    
    def __init__(
        self,
        qdrant_client,
        embedding_model,
        collection_name: str = "neethi-legal-kb",
        top_k: int = 5
    ):
        self.qdrant = qdrant_client
        self.embedder = embedding_model  # Fine-tuned model for statutes/bail
        self.collection = collection_name
        self.top_k = top_k
        self._sc_embedder = None  # Lazy-loaded base model for SC judgments
    
    @property
    def sc_embedder(self):
        """Lazy load base InLegalBERT for SC judgment retrieval.
        
        SC judgments were indexed with base model, so we need matching embeddings.
        """
        if self._sc_embedder is None:
            logger.info(f"Loading base model for SC judgments: {self.SC_BASE_MODEL}")
            self._sc_embedder = SentenceTransformer(self.SC_BASE_MODEL)
            logger.info("✅ Base InLegalBERT loaded for SC judgment retrieval")
        return self._sc_embedder
    
    def _detect_categories(self, query: str) -> Tuple[List[str], List[str], List[str]]:
        """
        Detect crime categories and extract relevant keywords/sections.
        Uses IPC_BNS_MAPPING for cross-law equivalent section expansion.
        
        Returns:
            Tuple of (matched_categories, matched_keywords, primary_section_nums)
        """
        query_lower = query.lower()
        matched_categories = []
        matched_keywords = []
        primary_section_nums = []
        
        for category, keywords in LEGAL_KEYWORDS.items():
            if any(kw in query_lower for kw in keywords):
                matched_categories.append(category)
                matched_keywords.extend(keywords)
                
                # Get sections from PRIMARY_SECTIONS (IPC-only)
                if category in PRIMARY_SECTIONS:
                    primary_section_nums.extend(PRIMARY_SECTIONS[category])
                
                # Also get BNS equivalents from IPC_BNS_MAPPING
                if category in IPC_BNS_MAPPING:
                    mapping = IPC_BNS_MAPPING[category]
                    # Add both IPC and BNS sections
                    primary_section_nums.extend(mapping.get("IPC", []))
                    primary_section_nums.extend(mapping.get("BNS", []))
        
        # Extract section numbers directly mentioned in query
        query_section_numbers = re.findall(r'\b(\d{2,3}[A-Za-z]?)\b', query)
        
        # Expand mentioned sections to equivalents (find both IPC and BNS)
        expanded_sections = list(query_section_numbers)
        for sec_num in query_section_numbers:
            for offense, mapping in IPC_BNS_MAPPING.items():
                if sec_num in mapping.get("IPC", []):
                    # Found in IPC, add BNS equivalents
                    expanded_sections.extend(mapping.get("BNS", []))
                    break
                elif sec_num in mapping.get("BNS", []):
                    # Found in BNS, add IPC equivalents
                    expanded_sections.extend(mapping.get("IPC", []))
                    break
        
        # Combine: query sections first, then category sections
        primary_section_nums = expanded_sections + primary_section_nums
        
        # Remove duplicates while preserving order
        seen = set()
        unique_sections = []
        for sec in primary_section_nums:
            if sec not in seen:
                seen.add(sec)
                unique_sections.append(sec)
        
        return matched_categories, matched_keywords, unique_sections
    
    def _fetch_all_sections(self, law_types: List[str] = None) -> List[Any]:
        """Fetch all sections from Qdrant for local filtering."""
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        
        if law_types is None:
            law_types = ["IPC", "BNS"]
        
        ipc_filter = Filter(
            must=[FieldCondition(key="law_type", match=MatchAny(any=law_types))]
        )
        
        all_sections = []
        offset = None
        
        while True:
            batch, offset = self.qdrant.scroll(
                collection_name=self.collection,
                scroll_filter=ipc_filter,
                limit=100,
                offset=offset
            )
            all_sections.extend(batch)
            if offset is None or len(all_sections) >= 1000:
                break
        
        return all_sections
    
    def _match_primary_sections(
        self,
        all_sections: List[Any],
        matched_categories: List[str],
        target_sections: List[str]
    ) -> List[Any]:
        """Match sections by primary section numbers (Priority 1).
        
        Uses offense-context filtering to prevent wrong sections.
        E.g., "murder 302" should NOT return BNS 302 (religious offense).
        """
        if not target_sections:
            return []
        
        logger.info(f"Target sections: {target_sections[:10]}")
        logger.info(f"Matched categories: {matched_categories}")
        
        # Build a lookup for priority ordering
        section_priority = {sec: idx for idx, sec in enumerate(target_sections)}
        
        # Build LAW-TYPE AWARE offense-context filter
        # Key: (law_type, section_num), e.g., ("IPC", "302") or ("BNS", "103")
        valid_offense_sections = {}  # {law_type: set of valid sections}
        for category in matched_categories:
            if category in IPC_BNS_MAPPING:
                for law_type in ["IPC", "BNS"]:
                    if law_type not in valid_offense_sections:
                        valid_offense_sections[law_type] = set()
                    valid_offense_sections[law_type].update(IPC_BNS_MAPPING[category].get(law_type, []))
            if category in PRIMARY_SECTIONS:
                # PRIMARY_SECTIONS are IPC-only
                if "IPC" not in valid_offense_sections:
                    valid_offense_sections["IPC"] = set()
                valid_offense_sections["IPC"].update(PRIMARY_SECTIONS[category])
        
        logger.info(f"Valid offense sections: {valid_offense_sections}")
        
        # Find all matching sections
        matches = []
        for point in all_sections:
            section_num = point.payload.get("section_num", "")
            law_type = point.payload.get("law_type", "")
            # Clean section number (e.g., "IPC323" -> "323")
            clean_section = re.sub(r'^[A-Za-z]+', '', section_num).strip()
            
            if clean_section in section_priority:
                # If we have offense context, verify this section belongs to that offense FOR THIS LAW TYPE
                if matched_categories and valid_offense_sections:
                    law_valid_sections = valid_offense_sections.get(law_type, set())
                    if clean_section in law_valid_sections:
                        matches.append((section_priority[clean_section], point))
                    # Skip sections not valid for this law type (e.g., BNS 302 not in BNS murder list)
                    else:
                        logger.debug(f"Skipping {law_type} {clean_section} - not in offense context")
                else:
                    # No category context, accept all matches
                    matches.append((section_priority[clean_section], point))
        
        # Sort by priority order (302 first, then 103, 300, etc.)
        matches.sort(key=lambda x: x[0])
        primary_matches = [m[1] for m in matches]
        
        logger.info(f"Primary section matches: {len(primary_matches)}")
        return primary_matches
    
    def _match_keywords(
        self,
        all_sections: List[Any],
        matched_keywords: List[str],
        exclude_ids: set
    ) -> List[Any]:
        """Match sections by keywords in text (Priority 2)."""
        if not matched_keywords:
            return []
        
        logger.info(f"Keywords detected: {matched_keywords[:5]}")
        keyword_matches = []
        
        for point in all_sections:
            if point.id in exclude_ids:
                continue
            text = point.payload.get("text", "").lower()
            
            # Score by keyword count
            match_score = sum(1 for kw in matched_keywords if kw.lower() in text)
            if match_score > 0:
                keyword_matches.append((point, match_score))
        
        # Sort by match score
        keyword_matches.sort(key=lambda x: x[1], reverse=True)
        results = [p for p, _ in keyword_matches]
        
        logger.info(f"Keyword matches: {len(results)}")
        return results
    
    def _semantic_search(self, query: str, exclude_ids: set) -> List[Any]:
        """Semantic search fallback (Priority 3)."""
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        
        logger.info("Adding semantic search results")
        expanded_query = f"Indian Penal Code: {query}"
        query_embedding = self.embedder.encode(expanded_query).tolist()
        
        ipc_filter = Filter(
            must=[FieldCondition(key="law_type", match=MatchAny(any=["IPC", "BNS"]))]
        )
        
        semantic_results = self.qdrant.query_points(
            collection_name=self.collection,
            query=query_embedding,
            query_filter=ipc_filter,
            limit=10
        ).points
        
        # Filter out already seen
        return [r for r in semantic_results if r.id not in exclude_ids]
    
    def search(self, query: str) -> List[RetrievedChunk]:
        """
        Search the vector database for relevant legal chunks.
        
        Implements 3-tier priority-based retrieval:
        1. Primary section matches (exact section numbers)
        2. Keyword matches in text
        3. Semantic search fallback
        
        Args:
            query: User's legal question
            
        Returns:
            List of top-k relevant chunks
        """
        logger.info(f"Librarian searching for: {query[:50]}...")
        
        # Step 1: Detect categories and get target sections
        matched_categories, matched_keywords, target_sections = self._detect_categories(query)
        
        # Step 2: Fetch all IPC/BNS sections
        all_sections = self._fetch_all_sections(["IPC", "BNS"])
        logger.info(f"Fetched {len(all_sections)} IPC/BNS sections")
        
        # Step 3: Priority 1 - Primary section matches
        primary_matches = self._match_primary_sections(
            all_sections, matched_categories, target_sections
        )
        
        # Step 4: Priority 2 - Keyword matches
        primary_ids = {p.id for p in primary_matches}
        keyword_matches = self._match_keywords(all_sections, matched_keywords, primary_ids)
        
        # Combine results
        search_results = primary_matches + keyword_matches
        
        # Step 5: Priority 3 - Semantic fallback if not enough
        if len(search_results) < 3:
            seen_ids = {r.id for r in search_results}
            semantic_results = self._semantic_search(query, seen_ids)
            search_results.extend(semantic_results)
        
        logger.info(f"Total results: {len(search_results)}")
        
        # Convert to RetrievedChunk objects
        chunks = []
        for result in search_results[:self.top_k]:
            payload = result.payload
            section_num_raw = payload.get("section_num", "N/A")
            law_type = payload.get("law_type", "Unknown")
            
            # Clean section number
            section_num = section_num_raw
            if section_num_raw.upper().startswith(law_type.upper()):
                section_num = section_num_raw[len(law_type):].strip()
            
            chunks.append(RetrievedChunk(
                chunk_id=str(result.id),
                text=payload.get("text", ""),
                law_type=law_type,
                section_num=section_num,
                score=getattr(result, 'score', None) or 0.8,
                metadata=payload
            ))
        
        return chunks
    
    def _collection_exists(self, collection_name: str) -> bool:
        """Check if a Qdrant collection exists"""
        try:
            collections = self.qdrant.get_collections().collections
            return any(c.name == collection_name for c in collections)
        except Exception:
            return False
    
    def search_judgments(self, query: str, target_sections: List[str] = None, limit: int = 3) -> List[RetrievedChunk]:
        """
        Search bail judgments collection for relevant case law.
        
        Args:
            query: User's legal question
            target_sections: IPC/BNS sections to filter by (for relevance)
            limit: Max results to return
            
        Returns:
            List of relevant judgment chunks
        """
        if not self._collection_exists(self.BAIL_COLLECTION):
            logger.warning(f"Collection {self.BAIL_COLLECTION} not found")
            return []
        
        logger.info(f"Searching bail judgments for: {query[:50]}...")
        
        # Generate embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Search bail judgments (fetch more to filter)
        fetch_limit = limit * 5 if target_sections else limit
        results = self.qdrant.query_points(
            collection_name=self.BAIL_COLLECTION,
            query=query_embedding,
            limit=fetch_limit
        ).points
        
        # Convert to chunks with optional filtering
        chunks = []
        for result in results:
            payload = result.payload
            
            # If we have target sections, filter for relevance
            if target_sections:
                # Check if this case mentions any of our target IPC sections
                case_sections = payload.get("ipc_sections", [])
                if isinstance(case_sections, str):
                    case_sections = [s.strip() for s in case_sections.split(",")]
                
                # Check for overlap
                has_relevant_section = any(
                    sec in case_sections or 
                    any(sec in cs for cs in case_sections)
                    for sec in target_sections[:5]  # Top 5 primary sections
                )
                
                if not has_relevant_section:
                    continue  # Skip irrelevant cases
            
            # Format case ID nicely
            case_id = payload.get("case_id", "Unknown")
            bail_outcome = payload.get("bail_outcome", "Unknown")
            
            chunks.append(RetrievedChunk(
                chunk_id=str(result.id),
                text=payload.get("text", payload.get("summary", "")),
                law_type="Bail Judgment",
                section_num=f"Case-{case_id}",
                case_name=f"Bail {bail_outcome}",
                score=getattr(result, 'score', None) or 0.7,
                metadata=payload
            ))
            
            if len(chunks) >= limit:
                break
        
        logger.info(f"Found {len(chunks)} relevant bail judgments")
        return chunks
    
    def search_all(self, query: str, include_judgments: bool = True) -> List[RetrievedChunk]:
        """
        Search across all collections (statutes + bail judgments + SC judgments).
        
        Args:
            query: User's legal question
            include_judgments: Whether to include case law
            
        Returns:
            Combined and ranked list of chunks
        """
        # Detect categories for filtering case law
        matched_categories, _, target_sections = self._detect_categories(query)
        
        # Search statutes (main search)
        statute_chunks = self.search(query)
        
        if not include_judgments:
            return statute_chunks
        
        # Search bail judgments with section filtering
        bail_chunks = self.search_judgments(query, target_sections=target_sections, limit=2)
        
        # Search SC judgments (fetch more for relevance)
        sc_chunks = self._search_sc_judgments(query, limit=5)
        
        # Merge results: statutes first, then case law
        # 3 statutes + 1-2 bail + 3-5 SC judgments
        merged = statute_chunks[:3] + bail_chunks[:2] + sc_chunks[:5]
        
        logger.info(f"Merged: {len(statute_chunks[:3])} statutes + {len(bail_chunks[:2])} bail + {len(sc_chunks[:5])} SC")
        return merged
    
    def _search_sc_judgments(self, query: str, limit: int = 5) -> List[RetrievedChunk]:
        """Search SC Judgments collection using BASE model (matches ingestion).
        
        Note: SC judgments were indexed with base InLegalBERT, not fine-tuned.
        Using fine-tuned embedder here would cause vector space mismatch.
        """
        if not self._collection_exists(self.SC_COLLECTION):
            return []
        
        # Use BASE model for SC judgments (matches ingestion model)
        query_embedding = self.sc_embedder.encode(query).tolist()
        
        results = self.qdrant.query_points(
            collection_name=self.SC_COLLECTION,
            query=query_embedding,
            limit=limit
        ).points
        
        chunks = []
        for result in results:
            payload = result.payload
            
            # Extract case details from new 26K dataset
            case_id = payload.get("case_id", payload.get("section_num", "Unknown"))
            filename = payload.get("filename", "")
            text = payload.get("text", "")
            doc_url = payload.get("doc_url", "")  # Indian Kanoon URL
            case_name = payload.get("case_name", "")  # Extracted case name
            year = payload.get("year", "")
            
            # Fallback: Try to extract year from case_id or filename if not in payload
            if not year:
                year_match = re.search(r'(19|20)\d{2}', case_id)
                year = year_match.group(0) if year_match else ""
            
            # Format case display name
            if case_name and case_name != "Unknown":
                case_display = case_name[:100]  # Truncate long names
            else:
                case_display = f"SC Case {year}" if year else "SC Case"
            
            # Include more text for context (up to 1000 chars for key HELD portions)
            held_match = re.search(r'HELD[:\s]*(.{200,800})', text, re.IGNORECASE)
            if held_match:
                judgment_text = f"HELD: {held_match.group(1)[:600]}..."
            else:
                judgment_text = text[:800] + "..." if len(text) > 800 else text
            
            chunks.append(RetrievedChunk(
                chunk_id=str(result.id),
                text=judgment_text,
                law_type="SC Judgment",
                section_num=case_id,
                case_name=case_display,
                score=getattr(result, 'score', None) or 0.7,
                metadata={
                    "case_id": case_id,
                    "year": year,
                    "filename": filename,
                    "doc_url": doc_url  # Include Indian Kanoon URL
                }
            ))
        
        logger.info(f"Found {len(chunks)} SC judgments")
        return chunks


def create_librarian_agent(
    qdrant_client,
    embedding_model,
    **kwargs
) -> LibrarianAgent:
    """Factory function to create a Librarian agent"""
    return LibrarianAgent(
        qdrant_client=qdrant_client,
        embedding_model=embedding_model,
        **kwargs
    )


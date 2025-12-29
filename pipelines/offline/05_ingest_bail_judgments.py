"""
Bail Judgments Ingestion Pipeline

Downloads IndianBailJudgments-1200 from HuggingFace and indexes into Qdrant.

Usage:
    python pipelines/offline/05_ingest_bail_judgments.py

Requirements:
    pip install datasets qdrant-client sentence-transformers
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from loguru import logger
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configure logging
logger.add("logs/bail_ingestion_{time}.log", rotation="10 MB", level="INFO")


@dataclass
class BailJudgmentChunk:
    """Represents a single bail judgment for indexing"""
    case_id: str
    bail_outcome: str  # "Granted" or "Rejected"
    ipc_sections: List[str]
    legal_issues: str
    summary: str
    full_text: str
    metadata: Dict[str, Any]


class BailJudgmentsIngester:
    """Handles downloading, processing, and indexing bail judgments"""
    
    COLLECTION_NAME = "neethi-bail-judgments"
    EMBEDDING_DIM = 768  # InLegalBERT
    
    def __init__(self):
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.client = None
        self.embedder = None
        self.stats = {
            "total_cases": 0,
            "indexed": 0,
            "errors": 0
        }
    
    def _init_clients(self):
        """Initialize Qdrant and embedding model"""
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
        
        logger.info("Initializing clients...")
        
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )
        logger.info("✅ Qdrant client connected")
        
        logger.info("Loading InLegalBERT model...")
        self.embedder = SentenceTransformer("law-ai/InLegalBERT")
        logger.info("✅ Embedding model loaded")
    
    def download_dataset(self) -> List[Dict]:
        """Download IndianBailJudgments-1200 from HuggingFace"""
        from datasets import load_dataset
        
        logger.info("Downloading IndianBailJudgments-1200 from HuggingFace...")
        
        try:
            dataset = load_dataset("SnehaDeshmukh/IndianBailJudgments-1200")
            
            # Convert to list of dicts
            cases = list(dataset["train"])
            self.stats["total_cases"] = len(cases)
            
            logger.info(f"✅ Downloaded {len(cases)} bail judgments")
            return cases
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    
    def process_case(self, case: Dict, idx: int) -> BailJudgmentChunk:
        """Process a single bail judgment into a chunk"""
        
        # Extract fields (adapt based on actual dataset schema)
        case_id = str(case.get("id", f"bail_{idx}"))
        bail_outcome = case.get("bail_outcome", case.get("Bail Outcome", "Unknown"))
        
        # Handle IPC sections (may be string or list)
        ipc_raw = case.get("ipc_sections", case.get("IPC Sections", ""))
        if isinstance(ipc_raw, str):
            ipc_sections = [s.strip() for s in ipc_raw.split(",") if s.strip()]
        else:
            ipc_sections = list(ipc_raw) if ipc_raw else []
        
        legal_issues = case.get("legal_issues", case.get("Legal Issues", ""))
        summary = case.get("summary", case.get("Summary", ""))
        full_text = case.get("text", case.get("judgment_text", case.get("Full Text", "")))
        
        # Build searchable text
        if not full_text:
            full_text = f"{summary}\n\nLegal Issues: {legal_issues}"
        
        return BailJudgmentChunk(
            case_id=case_id,
            bail_outcome=bail_outcome,
            ipc_sections=ipc_sections,
            legal_issues=legal_issues,
            summary=summary,
            full_text=full_text[:5000],  # Truncate for embedding
            metadata={
                "source": "IndianBailJudgments-1200",
                "law_type": "Bail Judgment",
                "indexed_at": datetime.now().isoformat()
            }
        )
    
    def create_collection(self):
        """Create Qdrant collection for bail judgments"""
        from qdrant_client.models import VectorParams, Distance
        
        collection = self.COLLECTION_NAME
        
        # Check if exists
        collections = self.client.get_collections().collections
        exists = any(c.name == collection for c in collections)
        
        if exists:
            logger.warning(f"Collection {collection} exists, recreating...")
            self.client.delete_collection(collection)
        
        self.client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=self.EMBEDDING_DIM,
                distance=Distance.COSINE
            )
        )
        logger.info(f"✅ Created collection: {collection}")
    
    def embed_and_index(self, chunks: List[BailJudgmentChunk], batch_size: int = 50):
        """Generate embeddings and index into Qdrant"""
        from qdrant_client.models import PointStruct
        
        logger.info(f"Indexing {len(chunks)} bail judgments...")
        
        points = []
        for i, chunk in enumerate(chunks):
            try:
                # Create embedding from summary + legal issues
                text_for_embedding = f"Bail {chunk.bail_outcome}. IPC: {', '.join(chunk.ipc_sections)}. {chunk.summary[:1000]}"
                embedding = self.embedder.encode(text_for_embedding).tolist()
                
                # Create payload
                payload = {
                    "case_id": chunk.case_id,
                    "bail_outcome": chunk.bail_outcome,
                    "ipc_sections": chunk.ipc_sections,
                    "legal_issues": chunk.legal_issues,
                    "summary": chunk.summary,
                    "text": chunk.full_text,
                    "law_type": "Bail Judgment",
                    "section_num": f"Case-{chunk.case_id}"
                }
                
                points.append(PointStruct(
                    id=i,
                    vector=embedding,
                    payload=payload
                ))
                
                # Batch upsert
                if len(points) >= batch_size:
                    self.client.upsert(
                        collection_name=self.COLLECTION_NAME,
                        points=points
                    )
                    self.stats["indexed"] += len(points)
                    logger.info(f"Indexed {self.stats['indexed']}/{len(chunks)}")
                    points = []
                    
            except Exception as e:
                logger.error(f"Error processing case {chunk.case_id}: {e}")
                self.stats["errors"] += 1
        
        # Final batch
        if points:
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=points
            )
            self.stats["indexed"] += len(points)
        
        logger.info(f"✅ Indexed {self.stats['indexed']} bail judgments")
    
    def run(self):
        """Run the full ingestion pipeline"""
        start_time = datetime.now()
        logger.info("=" * 60)
        logger.info("BAIL JUDGMENTS INGESTION PIPELINE")
        logger.info("=" * 60)
        
        # Initialize
        self._init_clients()
        
        # Download
        cases = self.download_dataset()
        
        # Process
        logger.info("Processing cases...")
        chunks = []
        for i, case in enumerate(cases):
            chunk = self.process_case(case, i)
            chunks.append(chunk)
        
        # Create collection
        self.create_collection()
        
        # Index
        self.embed_and_index(chunks)
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info("=" * 60)
        logger.info("INGESTION COMPLETE")
        logger.info(f"Total cases: {self.stats['total_cases']}")
        logger.info(f"Indexed: {self.stats['indexed']}")
        logger.info(f"Errors: {self.stats['errors']}")
        logger.info(f"Time: {elapsed:.1f}s")
        logger.info("=" * 60)
        
        return self.stats


def main():
    """Main entry point"""
    if not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_API_KEY"):
        logger.error("Missing QDRANT_URL or QDRANT_API_KEY environment variables")
        sys.exit(1)
    
    ingester = BailJudgmentsIngester()
    stats = ingester.run()
    
    print(f"\n✅ Successfully indexed {stats['indexed']} bail judgments")
    print(f"Collection: neethi-bail-judgments")


if __name__ == "__main__":
    main()

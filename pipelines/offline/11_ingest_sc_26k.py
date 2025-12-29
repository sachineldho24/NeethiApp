"""
SC Judgments 26K Ingestion Pipeline

Reads extracted JSONL from 10_extract_sc_26k.py, embeds with InLegalBERT,
and bulk uploads to Qdrant with doc_url in payload.

Usage:
    python pipelines/offline/11_ingest_sc_26k.py
    python pipelines/offline/11_ingest_sc_26k.py --batch-size 200

Input:
    data/processed/sc_judgments_26k.jsonl
"""

import os
import json
import uuid
import time
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Configure logging
log_path = Path("logs")
log_path.mkdir(exist_ok=True)
log_file = log_path / f"sc_ingest_26k_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger.add(str(log_file), rotation="50 MB", level="INFO")


COLLECTION_NAME = "neethi-judgments"
EMBEDDING_MODEL = "law-ai/InLegalBERT"
VECTOR_SIZE = 768
INPUT_FILE = Path("data/processed/sc_judgments_26k.jsonl")


def retry_with_backoff(func, max_retries=3, base_delay=2):
    """Retry with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            delay = base_delay * (2 ** attempt)
            logger.warning(f"Retry {attempt+1}: {e}. Waiting {delay}s...")
            time.sleep(delay)


def run_ingestion(batch_size: int = 100):
    """Run bulk ingestion"""
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
    from sentence_transformers import SentenceTransformer
    
    start_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info("SC JUDGMENTS 26K INGESTION")
    logger.info("=" * 60)
    
    # Check input file
    if not INPUT_FILE.exists():
        logger.error(f"Input file not found: {INPUT_FILE}")
        logger.error("Run 10_extract_sc_26k.py first!")
        return
    
    # Count total chunks
    total_chunks = sum(1 for _ in open(INPUT_FILE, encoding="utf-8"))
    logger.info(f"Total chunks to process: {total_chunks}")
    
    # Initialize Qdrant
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url:
        logger.error("QDRANT_URL not set!")
        return
    
    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=120)
    logger.info("✅ Qdrant connected")
    
    # Check/create collection
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        logger.info(f"✅ Created collection: {COLLECTION_NAME}")
    else:
        info = qdrant.get_collection(COLLECTION_NAME)
        logger.info(f"Collection exists with {info.points_count} vectors")
    
    # Load embedder
    logger.info(f"Loading {EMBEDDING_MODEL}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("✅ Embedder loaded")
    
    # Process in batches
    indexed = 0
    batch_texts = []
    batch_payloads = []
    
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            chunk = json.loads(line)
            
            batch_texts.append(chunk["text"])
            batch_payloads.append({
                "case_id": chunk["case_id"],
                "filename": chunk["filename"],
                "doc_url": chunk["doc_url"],  # Indian Kanoon URL
                "case_name": chunk["case_name"],
                "year": chunk["year"],
                "chunk_idx": chunk["chunk_idx"],
                "text": chunk["text"],
                "section": chunk["section"],
                "pages": chunk.get("pages", 0),
                "law_type": "SC Judgment",
                "section_num": chunk["case_id"],
                "source": "SC Judgment"
            })
            
            # Process batch
            if len(batch_texts) >= batch_size:
                # Embed
                embeddings = embedder.encode(batch_texts, show_progress_bar=False)
                
                # Create points
                points = [
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=emb.tolist(),
                        payload=payload
                    )
                    for emb, payload in zip(embeddings, batch_payloads)
                ]
                
                # Upload with retry
                def do_upsert():
                    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
                
                retry_with_backoff(do_upsert)
                indexed += len(points)
                
                # Progress
                elapsed = (datetime.now() - start_time).total_seconds() / 60
                rate = indexed / elapsed if elapsed > 0 else 0
                eta = (total_chunks - indexed) / rate if rate > 0 else 0
                
                if indexed % 5000 == 0 or indexed == len(points):
                    logger.info(
                        f"Indexed: {indexed}/{total_chunks} | "
                        f"Rate: {rate:.0f}/min | ETA: {eta:.0f}m"
                    )
                
                # Reset batch
                batch_texts = []
                batch_payloads = []
    
    # Final batch
    if batch_texts:
        embeddings = embedder.encode(batch_texts, show_progress_bar=False)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload=payload
            )
            for emb, payload in zip(embeddings, batch_payloads)
        ]
        
        def do_upsert():
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
        
        retry_with_backoff(do_upsert)
        indexed += len(points)
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    
    # Verify
    info = qdrant.get_collection(COLLECTION_NAME)
    
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info(f"Total Indexed: {indexed}")
    logger.info(f"Collection now has: {info.points_count} vectors")
    logger.info(f"Time: {elapsed:.1f} minutes")
    logger.info(f"Collection: {COLLECTION_NAME}")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SC Judgments 26K Ingestion")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size")
    args = parser.parse_args()
    
    run_ingestion(batch_size=args.batch_size)

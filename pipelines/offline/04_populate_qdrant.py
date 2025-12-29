"""
Step 4: Qdrant Vector Database Population

This script embeds chunks and populates the Qdrant vector database.

Run: python pipelines/offline/04_populate_qdrant.py

Prerequisites:
- Chunks created (run 02_chunking.py first)
- Qdrant Cloud account or local Docker instance
- QDRANT_URL and QDRANT_API_KEY in environment

Input: data/processed/chunks.jsonl
Output: Vectors indexed in Qdrant collection
"""

import os
import json
import yaml
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger
from datetime import datetime
import time

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance, VectorParams, PointStruct,
        Filter, FieldCondition, MatchValue,
        OptimizersConfigDiff, HnswConfigDiff
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger.add(
    "logs/qdrant_populate_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO"
)


@dataclass
class QdrantConfig:
    """Qdrant connection configuration"""
    url: str
    api_key: str
    collection_name: str = "neethi-legal-kb"
    vector_size: int = 768
    distance: str = "Cosine"


class QdrantPopulator:
    """
    Populates Qdrant vector database with legal document embeddings.
    """
    
    def __init__(
        self,
        config: QdrantConfig,
        embedding_model: str = "law-ai/InLegalBERT",
        batch_size: int = 32
    ):
        self.config = config
        self.batch_size = batch_size
        
        # Initialize Qdrant client
        if not QDRANT_AVAILABLE:
            raise ImportError("qdrant-client not installed. Run: pip install qdrant-client")
        
        logger.info(f"Connecting to Qdrant at {config.url}")
        self.client = QdrantClient(
            url=config.url,
            api_key=config.api_key,
            timeout=60
        )
        
        # Initialize embedding model
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedder = SentenceTransformer(embedding_model)
        
        # Verify vector size matches model
        test_embedding = self.embedder.encode("test")
        actual_size = len(test_embedding)
        if actual_size != config.vector_size:
            logger.warning(f"Vector size mismatch: config={config.vector_size}, model={actual_size}")
            self.config.vector_size = actual_size
        
        self.stats = {
            "total_chunks": 0,
            "indexed_chunks": 0,
            "failed_chunks": 0,
            "batches_processed": 0,
            "start_time": None,
            "end_time": None
        }
    
    def create_collection(self, recreate: bool = False) -> bool:
        """
        Create the Qdrant collection with proper settings.
        
        Args:
            recreate: If True, delete existing collection first
        """
        collection = self.config.collection_name
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == collection for c in collections)
        
        if exists:
            if recreate:
                logger.warning(f"Deleting existing collection: {collection}")
                self.client.delete_collection(collection)
            else:
                logger.info(f"Collection exists: {collection}")
                return True
        
        # Create collection
        logger.info(f"Creating collection: {collection}")
        
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT
        }
        
        self.client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(
                size=self.config.vector_size,
                distance=distance_map.get(self.config.distance, Distance.COSINE)
            ),
            # Optimize for search performance
            hnsw_config=HnswConfigDiff(
                m=16,
                ef_construct=100
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=20000
            )
        )
        
        logger.success(f"Collection created: {collection}")
        return True
    
    def load_chunks(self, chunks_file: str = "data/processed/chunks.jsonl") -> List[Dict]:
        """Load chunks from JSONL file"""
        chunks = []
        filepath = Path(chunks_file)
        
        if not filepath.exists():
            logger.error(f"Chunks file not found: {filepath}")
            return chunks
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    chunk = json.loads(line.strip())
                    chunks.append(chunk)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Loaded {len(chunks)} chunks from {filepath}")
        return chunks
    
    def embed_chunks(self, chunks: List[Dict]) -> List[List[float]]:
        """Generate embeddings for chunks"""
        texts = [c.get("text", "") for c in chunks]
        
        logger.info(f"Embedding {len(texts)} chunks...")
        embeddings = self.embedder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings.tolist()
    
    def index_chunks(
        self,
        chunks: List[Dict],
        embeddings: List[List[float]]
    ) -> int:
        """
        Index chunks with embeddings into Qdrant.
        
        Returns: Number of successfully indexed chunks
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunk/embedding count mismatch: {len(chunks)} vs {len(embeddings)}")
        
        collection = self.config.collection_name
        indexed = 0
        
        # Process in batches
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i:i + self.batch_size]
            batch_embeddings = embeddings[i:i + self.batch_size]
            
            points = []
            for j, (chunk, embedding) in enumerate(zip(batch_chunks, batch_embeddings)):
                point_id = i + j + 1  # 1-indexed
                
                # Create payload (metadata)
                payload = {
                    "chunk_id": chunk.get("chunk_id", f"chunk_{point_id}"),
                    "text": chunk.get("text", ""),
                    "law_type": chunk.get("law_type", ""),
                    "section_num": chunk.get("section_num"),
                    "case_name": chunk.get("case_name"),
                    "section_type": chunk.get("section_type"),
                    "source_id": chunk.get("source_id", ""),
                    "word_count": chunk.get("word_count", 0),
                }
                
                # Add any additional metadata
                if chunk.get("metadata"):
                    payload.update(chunk["metadata"])
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                ))
            
            try:
                self.client.upsert(
                    collection_name=collection,
                    points=points,
                    wait=True
                )
                indexed += len(points)
                self.stats["batches_processed"] += 1
                
                if self.stats["batches_processed"] % 10 == 0:
                    logger.info(f"Progress: {indexed}/{len(chunks)} chunks indexed")
                    
            except Exception as e:
                logger.error(f"Batch {i//self.batch_size} failed: {e}")
                self.stats["failed_chunks"] += len(batch_chunks)
        
        return indexed
    
    def create_payload_indexes(self) -> None:
        """Create indexes on payload fields for filtering"""
        collection = self.config.collection_name
        
        # Index common filter fields
        filter_fields = ["law_type", "section_num", "section_type"]
        
        for field in filter_fields:
            try:
                self.client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema="keyword"
                )
                logger.info(f"Created index on: {field}")
            except Exception as e:
                logger.debug(f"Index may already exist for {field}: {e}")
    
    def run(
        self,
        chunks_file: str = "data/processed/chunks.jsonl",
        recreate_collection: bool = False
    ) -> Dict:
        """
        Run the full population pipeline.
        
        Returns: Statistics dictionary
        """
        self.stats["start_time"] = datetime.now()
        
        # Step 1: Create collection
        self.create_collection(recreate=recreate_collection)
        
        # Step 2: Load chunks
        chunks = self.load_chunks(chunks_file)
        self.stats["total_chunks"] = len(chunks)
        
        if not chunks:
            logger.error("No chunks to index!")
            return self.stats
        
        # Step 3: Generate embeddings
        embeddings = self.embed_chunks(chunks)
        
        # Step 4: Index into Qdrant
        indexed = self.index_chunks(chunks, embeddings)
        self.stats["indexed_chunks"] = indexed
        
        # Step 5: Create payload indexes
        self.create_payload_indexes()
        
        self.stats["end_time"] = datetime.now()
        
        # Get collection info
        collection_info = self.client.get_collection(self.config.collection_name)
        self.stats["vectors_count"] = getattr(collection_info, 'points_count', 
                                              getattr(collection_info, 'vectors_count', 0))
        
        return self.stats
    
    def test_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Test search functionality"""
        logger.info(f"Test search: {query}")
        
        query_embedding = self.embedder.encode(query).tolist()
        
        try:
            # Try newer API first
            results = self.client.query_points(
                collection_name=self.config.collection_name,
                query=query_embedding,
                limit=top_k,
                with_payload=True
            )
            points = results.points
        except AttributeError:
            # Fallback to older API
            points = self.client.search(
                collection_name=self.config.collection_name,
                query_vector=query_embedding,
                limit=top_k
            )
        
        return [
            {
                "score": r.score,
                "chunk_id": r.payload.get("chunk_id"),
                "law_type": r.payload.get("law_type"),
                "section_num": r.payload.get("section_num"),
                "text": r.payload.get("text", "")[:200] + "..."
            }
            for r in points
        ]


def load_config() -> QdrantConfig:
    """Load Qdrant config from file or environment"""
    
    # Try loading from config file
    config_path = Path("configs/qdrant_config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
            qdrant = config_data.get("qdrant", {})
    else:
        qdrant = {}
    
    # Environment variables override config file
    url = os.getenv("QDRANT_URL") or qdrant.get("url", "")
    api_key = os.getenv("QDRANT_API_KEY") or qdrant.get("api_key", "")
    
    # Handle placeholder syntax in config
    if url.startswith("${"):
        url = os.getenv(url[2:-1], "")
    if api_key.startswith("${"):
        api_key = os.getenv(api_key[2:-1], "")
    
    if not url or not api_key:
        raise ValueError(
            "Qdrant credentials not found!\n"
            "Set QDRANT_URL and QDRANT_API_KEY environment variables,\n"
            "or create .env file with these values."
        )
    
    return QdrantConfig(
        url=url,
        api_key=api_key,
        collection_name=qdrant.get("collection_name", "neethi-legal-kb"),
        vector_size=qdrant.get("vector_size", 768),
        distance=qdrant.get("distance_metric", "Cosine")
    )


def main():
    """Main entry point for Qdrant population"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Populate Qdrant with embeddings")
    parser.add_argument("--model", type=str, default="law-ai/InLegalBERT",
                        help="Embedding model (default: law-ai/InLegalBERT)")
    parser.add_argument("--recreate", action="store_true",
                        help="Delete and recreate collection")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("NEETHI APP - QDRANT POPULATION PIPELINE (Step 4)")
    print("="*60 + "\n")
    
    # Check dependencies
    if not QDRANT_AVAILABLE:
        print("❌ qdrant-client not installed")
        print("   Run: pip install qdrant-client")
        return
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("❌ sentence-transformers not installed")
        print("   Run: pip install sentence-transformers")
        return
    
    # Load configuration
    try:
        config = load_config()
        print(f"✅ Qdrant URL: {config.url[:50]}...")
        print(f"✅ Collection: {config.collection_name}")
        print(f"✅ Model: {args.model}")
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Initialize populator
    try:
        populator = QdrantPopulator(config, embedding_model=args.model)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Run population
    print("\nStarting population pipeline...")
    stats = populator.run(recreate_collection=args.recreate)
    
    # Print results
    print("\n" + "="*50)
    print("POPULATION SUMMARY")
    print("="*50)
    print(f"Total chunks:    {stats['total_chunks']}")
    print(f"Indexed:         {stats['indexed_chunks']}")
    print(f"Failed:          {stats['failed_chunks']}")
    print(f"Batches:         {stats['batches_processed']}")
    
    if stats.get('start_time') and stats.get('end_time'):
        duration = (stats['end_time'] - stats['start_time']).total_seconds()
        print(f"Duration:        {duration:.1f} seconds")
    
    if stats.get('vectors_count'):
        print(f"Vectors in DB:   {stats['vectors_count']}")
    print("="*50)
    
    # Test search
    if stats['indexed_chunks'] > 0:
        print("\n📝 Testing search...")
        test_results = populator.test_search("What is the punishment for theft?")
        
        print("\nTop results for 'What is the punishment for theft?':")
        for i, r in enumerate(test_results, 1):
            print(f"\n{i}. [{r['law_type']} {r['section_num'] or ''}] (score: {r['score']:.3f})")
            print(f"   {r['text'][:100]}...")
    
    print("\n✅ Qdrant population complete!")
    print("Next step: Run the API server with: uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()

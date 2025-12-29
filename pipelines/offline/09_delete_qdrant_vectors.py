"""
Qdrant Vector Deletion Script

Deletes all vectors from the neethi-judgments collection
and recreates it with the same configuration.

Usage:
    python pipelines/offline/09_delete_qdrant_vectors.py
"""

import os
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "neethi-judgments"
VECTOR_SIZE = 768


def delete_all_vectors():
    """Delete all vectors from Qdrant collection"""
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance
    
    # Connect to Qdrant
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("QDRANT_API_KEY")
    
    if not qdrant_url:
        logger.error("QDRANT_URL not set!")
        return False
    
    logger.info(f"Connecting to Qdrant: {qdrant_url}")
    client = QdrantClient(url=qdrant_url, api_key=qdrant_key, timeout=120)
    logger.info("✅ Connected to Qdrant")
    
    # Check if collection exists
    collections = [c.name for c in client.get_collections().collections]
    
    if COLLECTION_NAME in collections:
        # Get current count
        info = client.get_collection(COLLECTION_NAME)
        current_count = info.points_count
        logger.info(f"Collection '{COLLECTION_NAME}' has {current_count} vectors")
        
        # Delete collection
        logger.info(f"Deleting collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
        logger.info("✅ Collection deleted")
    else:
        logger.info(f"Collection '{COLLECTION_NAME}' does not exist")
    
    # Recreate empty collection
    logger.info(f"Creating fresh collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    logger.info("✅ Empty collection created")
    
    # Verify
    info = client.get_collection(COLLECTION_NAME)
    logger.info(f"Verification: Collection now has {info.points_count} vectors")
    
    return True


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("QDRANT VECTOR DELETION")
    logger.info("=" * 60)
    
    success = delete_all_vectors()
    
    if success:
        logger.info("✅ All vectors deleted successfully")
    else:
        logger.error("❌ Deletion failed")

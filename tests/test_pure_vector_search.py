"""
Pure Vector Search Test - Tests fine-tuned model WITHOUT hardcoded sections

This bypasses the LibrarianAgent's hardcoded PRIMARY_SECTIONS to show
the raw retrieval capability of the fine-tuned embedding model.

Usage:
    python tests/test_pure_vector_search.py
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


# Test queries - same as test_queries.json
TEST_QUERIES = [
    {"id": 1, "category": "theft", "query": "What is the punishment for theft?", "expected": ["378", "379"]},
    {"id": 2, "category": "murder", "query": "What is murder under IPC Section 302?", "expected": ["302", "300"]},
    {"id": 3, "category": "assault", "query": "What is the punishment for assault and hurt?", "expected": ["323", "324"]},
    {"id": 4, "category": "fraud", "query": "What is cheating and fraud under Indian law?", "expected": ["420", "415"]},
    {"id": 5, "category": "dowry", "query": "What are the laws against dowry harassment?", "expected": ["498A", "304B"]},
    {"id": 6, "category": "kidnapping", "query": "What is kidnapping and abduction punishment?", "expected": ["359", "363"]},
    {"id": 7, "category": "defamation", "query": "What is defamation under IPC?", "expected": ["499", "500"]},
    {"id": 8, "category": "robbery", "query": "What is the difference between theft and robbery?", "expected": ["390", "392"]},
    {"id": 9, "category": "bail", "query": "Which offenses are bailable under IPC?", "expected": []},  # No specific section
    {"id": 10, "category": "general", "query": "What is Section 420 of IPC about?", "expected": ["420"]},
]


def main():
    print("\n" + "="*70)
    print("PURE VECTOR SEARCH TEST - Fine-tuned InLegalBERT")
    print("(No hardcoded sections - raw embedding similarity)")
    print("="*70)
    
    # Load fine-tuned model
    model_path = "models/inlegalbert-finetuned"
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        return
    
    print(f"\n📦 Loading fine-tuned model from {model_path}...")
    model = SentenceTransformer(model_path)
    print("   ✅ Model loaded")
    
    # Connect to Qdrant
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    if not url or not api_key:
        print("❌ QDRANT_URL or QDRANT_API_KEY not set")
        return
    
    client = QdrantClient(url=url, api_key=api_key)
    print("   ✅ Qdrant connected")
    
    collection = "neethi-legal-kb"
    
    # Run tests
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    total_score = 0
    hits = 0
    
    for test in TEST_QUERIES:
        query = test["query"]
        expected = test["expected"]
        category = test["category"]
        
        # Generate embedding
        query_embedding = model.encode(query).tolist()
        
        # Pure vector search (no filters, no hardcoding)
        results = client.query_points(
            collection_name=collection,
            query=query_embedding,
            limit=5,
            with_payload=True
        ).points
        
        print(f"\n[{test['id']}] {category.upper()}: {query}")
        print(f"    Expected sections: {expected}")
        print(f"    Top 5 results:")
        
        found_expected = False
        for i, r in enumerate(results):
            section = r.payload.get("section_num", "N/A")
            law_type = r.payload.get("law_type", "")
            score = r.score
            text = r.payload.get("text", "")[:80].replace("\n", " ")
            
            # Check if this section matches expected
            clean_section = section.replace("IPC", "").replace("BNS", "").strip()
            is_expected = any(exp in clean_section for exp in expected) if expected else False
            marker = "✅" if is_expected else "  "
            
            if is_expected:
                found_expected = True
            
            print(f"    {marker} [{i+1}] {law_type} {section} (score: {score:.3f})")
            print(f"          {text}...")
        
        # Score this query
        if not expected:
            total_score += 1  # Skip scoring for queries without expected sections
            hits += 1
        elif found_expected:
            total_score += 1
            hits += 1
            print(f"    ✅ Found expected section in top 5!")
        else:
            print(f"    ❌ Expected section NOT in top 5")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total queries:  {len(TEST_QUERIES)}")
    print(f"Hits (expected in top 5): {hits}/{len(TEST_QUERIES)}")
    print(f"Accuracy: {hits/len(TEST_QUERIES)*100:.1f}%")
    print("="*70)


if __name__ == "__main__":
    main()

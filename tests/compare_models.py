"""
Compare Base InLegalBERT vs Fine-tuned Model

Tests retrieval quality with sample legal queries and compares:
- Similarity scores
- Top-K retrieved chunks
- Response relevance

Usage:
    python tests/compare_models.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import numpy as np


# Test queries for Indian legal domain
TEST_QUERIES = [
    "What is the punishment for theft under IPC?",
    "How to file an FIR?",
    "What is anticipatory bail?",
    "What are the rights of an arrested person?",
    "What is the procedure for divorce in India?",
    "What is Section 420 IPC about cheating?",
    "Can police arrest without warrant?",
    "What is the punishment for murder?",
]


def load_model(model_path: str, name: str):
    """Load a sentence transformer model"""
    print(f"\n📦 Loading {name}...")
    try:
        model = SentenceTransformer(model_path)
        print(f"   ✅ {name} loaded")
        return model
    except Exception as e:
        print(f"   ❌ Failed to load {name}: {e}")
        return None


def get_qdrant_client():
    """Get Qdrant client from environment"""
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    
    if not url or not api_key:
        print("❌ QDRANT_URL or QDRANT_API_KEY not set")
        return None
    
    try:
        client = QdrantClient(url=url, api_key=api_key)
        return client
    except Exception as e:
        print(f"❌ Failed to connect to Qdrant: {e}")
        return None


def search_with_model(client, model, query: str, collection: str = "neethi-legal-kb", top_k: int = 3):
    """Search Qdrant using a specific model for embedding"""
    from qdrant_client.models import models
    
    # Encode query
    query_vector = model.encode(query).tolist()
    
    # Search using query_points (newer API)
    try:
        results = client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )
        return results.points
    except AttributeError:
        # Fallback to older API
        results = client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )
        return results


def compare_results(base_results, finetuned_results, query: str):
    """Compare search results from both models"""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")
    
    print(f"\n📊 BASE MODEL (law-ai/InLegalBERT):")
    for i, r in enumerate(base_results[:3]):
        score = r.score
        text = r.payload.get('text', '')[:150] + '...'
        print(f"   [{i+1}] Score: {score:.4f}")
        print(f"       {text}")
    
    print(f"\n🚀 FINE-TUNED MODEL:")
    for i, r in enumerate(finetuned_results[:3]):
        score = r.score
        text = r.payload.get('text', '')[:150] + '...'
        print(f"   [{i+1}] Score: {score:.4f}")
        print(f"       {text}")
    
    # Compare scores
    base_avg = sum(r.score for r in base_results[:3]) / 3 if base_results else 0
    ft_avg = sum(r.score for r in finetuned_results[:3]) / 3 if finetuned_results else 0
    
    diff = ft_avg - base_avg
    if diff > 0:
        print(f"\n   ✅ Fine-tuned is {diff:.4f} better (avg score)")
    else:
        print(f"\n   ⚠️ Base is {abs(diff):.4f} better (avg score)")
    
    return base_avg, ft_avg


def run_embedding_comparison(base_model, finetuned_model):
    """Compare embeddings directly without Qdrant"""
    print("\n" + "="*60)
    print("EMBEDDING SIMILARITY COMPARISON (No Qdrant needed)")
    print("="*60)
    
    # Sample legal texts for comparison
    texts = [
        "Section 379 IPC: Punishment for theft. Whoever commits theft shall be punished with imprisonment.",
        "Section 302 IPC: Punishment for murder. Whoever commits murder shall be punished with death.",
        "Anticipatory bail under Section 438 CrPC allows a person to seek bail in anticipation of arrest.",
        "FIR or First Information Report is the first step in criminal proceedings.",
    ]
    
    for query in TEST_QUERIES[:4]:
        print(f"\n🔍 Query: {query}")
        
        # Encode with both models
        base_query_emb = base_model.encode(query)
        ft_query_emb = finetuned_model.encode(query)
        
        base_text_embs = base_model.encode(texts)
        ft_text_embs = finetuned_model.encode(texts)
        
        # Compute cosine similarities
        print(f"\n   {'Text (truncated)':<50} {'Base':>8} {'FT':>8} {'Diff':>8}")
        print("   " + "-"*74)
        
        for i, text in enumerate(texts):
            base_sim = np.dot(base_query_emb, base_text_embs[i]) / (
                np.linalg.norm(base_query_emb) * np.linalg.norm(base_text_embs[i])
            )
            ft_sim = np.dot(ft_query_emb, ft_text_embs[i]) / (
                np.linalg.norm(ft_query_emb) * np.linalg.norm(ft_text_embs[i])
            )
            diff = ft_sim - base_sim
            
            text_short = text[:47] + "..." if len(text) > 50 else text
            print(f"   {text_short:<50} {base_sim:>8.4f} {ft_sim:>8.4f} {diff:>+8.4f}")


def main():
    print("\n" + "="*60)
    print("MODEL COMPARISON: Base vs Fine-tuned InLegalBERT")
    print("="*60)
    
    # Load models
    base_model = load_model("law-ai/InLegalBERT", "Base InLegalBERT")
    finetuned_model = load_model("models/inlegalbert-finetuned", "Fine-tuned Model")
    
    if not base_model or not finetuned_model:
        print("\n❌ Could not load one or both models. Exiting.")
        return
    
    # Run embedding comparison (no Qdrant needed)
    run_embedding_comparison(base_model, finetuned_model)
    
    # Try Qdrant comparison if available
    client = get_qdrant_client()
    
    if client:
        print("\n\n" + "="*60)
        print("QDRANT SEARCH COMPARISON")
        print("="*60)
        
        base_scores = []
        ft_scores = []
        
        for query in TEST_QUERIES:
            try:
                base_results = search_with_model(client, base_model, query)
                ft_results = search_with_model(client, finetuned_model, query)
                
                base_avg, ft_avg = compare_results(base_results, ft_results, query)
                base_scores.append(base_avg)
                ft_scores.append(ft_avg)
            except Exception as e:
                print(f"⚠️ Error searching for '{query}': {e}")
        
        # Summary
        if base_scores and ft_scores:
            print("\n" + "="*60)
            print("SUMMARY")
            print("="*60)
            print(f"   Base Model Avg Score:      {sum(base_scores)/len(base_scores):.4f}")
            print(f"   Fine-tuned Avg Score:      {sum(ft_scores)/len(ft_scores):.4f}")
            diff = sum(ft_scores)/len(ft_scores) - sum(base_scores)/len(base_scores)
            if diff > 0:
                print(f"   🚀 Fine-tuned is {diff:.4f} better overall!")
            else:
                print(f"   ⚠️ Base model scored {abs(diff):.4f} higher")
    else:
        print("\n⚠️ Qdrant not available - only embedding comparison shown")
    
    print("\n✅ Comparison complete!")


if __name__ == "__main__":
    main()

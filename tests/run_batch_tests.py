"""
Run batch test queries against the Neethi API

Usage:
    python tests/run_batch_tests.py [--base-url http://localhost:8000]
"""

import json
import time
import argparse
import requests
from pathlib import Path


def run_tests(base_url: str = "http://localhost:8000"):
    """Run all test queries and display results"""
    
    # Load test queries
    test_file = Path(__file__).parent / "test_queries.json"
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    queries = data.get("test_queries", [])
    
    print("\n" + "="*70)
    print("NEETHI API BATCH TEST")
    print("="*70)
    print(f"Base URL: {base_url}")
    print(f"Queries: {len(queries)}")
    print("="*70 + "\n")
    
    results = []
    total_time = 0
    
    for q in queries:
        query_id = q.get("id")
        category = q.get("category")
        query = q.get("query")
        
        print(f"\n[{query_id}] {category.upper()}")
        print(f"    Query: {query}")
        
        try:
            start = time.time()
            response = requests.post(
                f"{base_url}/api/v1/query",
                json={"query": query},
                timeout=60
            )
            elapsed = time.time() - start
            total_time += elapsed
            
            if response.status_code == 200:
                result = response.json()
                advice = result.get("advice", "")[:200]
                sources_count = len(result.get("sources", []))
                proc_time = result.get("processing_time_ms", 0)
                
                print(f"    ✅ Status: 200 | Time: {elapsed:.2f}s | Sources: {sources_count}")
                print(f"    Response: {advice}...")
                
                results.append({
                    "id": query_id,
                    "category": category,
                    "status": "success",
                    "time_s": elapsed,
                    "sources": sources_count
                })
            else:
                print(f"    ❌ Status: {response.status_code}")
                print(f"    Error: {response.text[:200]}")
                results.append({
                    "id": query_id,
                    "category": category,
                    "status": "error",
                    "error": response.text[:100]
                })
                
        except requests.exceptions.Timeout:
            print(f"    ⏰ Timeout after 60s")
            results.append({"id": query_id, "status": "timeout"})
        except requests.exceptions.ConnectionError:
            print(f"    ❌ Connection error - is the API running?")
            results.append({"id": query_id, "status": "connection_error"})
            break
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({"id": query_id, "status": "error", "error": str(e)})
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    success = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - success
    
    print(f"Total queries:  {len(queries)}")
    print(f"Successful:     {success}")
    print(f"Failed:         {failed}")
    print(f"Total time:     {total_time:.2f}s")
    print(f"Avg time:       {total_time/len(queries):.2f}s" if queries else "N/A")
    print("="*70)
    
    # Save results
    output_file = Path(__file__).parent / "test_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "base_url": base_url,
            "total_queries": len(queries),
            "successful": success,
            "failed": failed,
            "total_time_s": total_time,
            "results": results
        }, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch tests")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000",
                        help="API base URL")
    args = parser.parse_args()
    
    run_tests(args.base_url)

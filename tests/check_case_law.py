import requests
import json

r = requests.post('http://localhost:8000/api/v1/query', json={'query': 'What is murder under IPC 302?'})
d = r.json()
print('Sources:', len(d.get('sources',[])))
print('Case Law:', len(d.get('case_law',[])))
print()
print('=== CASE LAW DETAILS ===')
for i, c in enumerate(d.get('case_law',[]), 1):
    print(f"\n[{i}] {c.get('case_type', '')}")
    print(f"    Case: {c.get('case_name', 'N/A')}")
    print(f"    Year: {c.get('year', 'N/A')}")
    print(f"    URL: {c.get('doc_url', 'N/A')}")
    summary = c.get('summary', '')[:200]
    print(f"    Summary: {summary}...")

import sys
sys.path.insert(0, '/Users/pengyixuan/Desktop/macro_platform')

from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

# Test health
print("Testing /health...")
r = client.get("/health")
print(f"  Status: {r.status_code}")
print(f"  Response: {r.json()}\n")

# Test context
print("Testing /context...")
r = client.get("/context")
print(f"  Status: {r.status_code}")
print(f"  Keys: {r.json().keys()}\n")

# Test indicators
print("Testing /indicators...")
r = client.get("/indicators")
print(f"  Status: {r.status_code}")
print(f"  Found {len(r.json()['indicators'])} indicators\n")

# Test interpret (full interpretation)
print("Testing /interpret...")
try:
    r = client.post("/interpret")
    print(f"  Status: {r.status_code}")
    if r.status_code == 200:
        data = r.json()
        print(f"  Regime: {data.get('regime_assessment', {}).get('regime')}")
        print(f"  Is stub: {data.get('is_stub')}\n")
    else:
        print(f"  Error: {r.text}\n")
except Exception as e:
    print(f"  Error: {e}\n")

print("✓ All tests completed!")

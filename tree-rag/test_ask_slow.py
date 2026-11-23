"""Time one /ask call with a long timeout to tell 'slow' from 'stuck'."""
import time
import requests

BASE_URL = "http://127.0.0.1:8000"

try:
    with open(".token", encoding="utf-8") as file:
        token = file.read().strip()
except FileNotFoundError:
    token = ""

headers = {"Authorization": f"Bearer {token}"} if token else {}

print("health:", requests.get(f"{BASE_URL}/health", timeout=10).text[:300])

start = time.time()
try:
    r = requests.post(
        f"{BASE_URL}/ask",
        json={"question": "Which trees have high salinity tolerance?"},
        headers=headers,
        timeout=120,
    )
    print(f"status={r.status_code} time={time.time() - start:.1f}s")
    print(r.text[:1200])
except requests.exceptions.Timeout:
    print(f"STILL STUCK after {time.time() - start:.0f}s")
except Exception as exc:
    print(f"ERROR: {type(exc).__name__}: {exc}")
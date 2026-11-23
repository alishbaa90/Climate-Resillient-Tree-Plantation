import json
import sys

import requests

BASE_URL = "http://127.0.0.1:8000"
TOKEN_FILE = ".token"
POINTS = [
    ("Thatta town (inside)", 24.7476, 67.9236),
    ("Lahore (outside, expected 422)", 31.5204, 74.3587),
    ("Mandi Bahauddin (outside, expected 422)", 32.5849, 73.4905),
]


def main():
    try:
        with open(TOKEN_FILE, encoding="utf-8") as file:
            token = file.read().strip()
    except OSError:
        print("No .token file. Run: py get_token.py")
        sys.exit(1)
    headers = {"Authorization": f"Bearer {token}"}

    r = requests.get(f"{BASE_URL}/environment", params={"lat": 24.75, "lon": 67.92}, timeout=60)
    print(f"without a token: HTTP {r.status_code} (expected 401)")

    for label, lat, lon in POINTS:
        print("\n" + "=" * 70)
        print(f"{label}  ({lat}, {lon})")
        r = requests.get(f"{BASE_URL}/environment", params={"lat": lat, "lon": lon}, headers=headers, timeout=90)
        print(f"HTTP {r.status_code}")
        print(json.dumps(r.json(), indent=2)[:2500])


if __name__ == "__main__":
    main()
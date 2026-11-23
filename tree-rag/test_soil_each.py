"""Time each SoilGrids property separately for Thatta (one request at a time)."""
import time
import requests

LAT, LON = 24.75, 67.92
TIMEOUT = 25
PROPERTIES = ["phh2o", "clay", "sand", "silt", "soc"]

for i, prop in enumerate(PROPERTIES):
    if i > 0:
        time.sleep(15)  # avoid rate limiting
    start = time.time()
    try:
        r = requests.get(
            "https://rest.isric.org/soilgrids/v2.0/properties/query",
            params={
                "lon": LON,
                "lat": LAT,
                "property": prop,
                "depth": "0-5cm",
                "value": "mean",
            },
            timeout=TIMEOUT,
        )
        took = time.time() - start
        print(f"[{prop}] status={r.status_code} time={took:.1f}s size={len(r.content)} bytes")
        print(f"[{prop}] body: {r.text[:600]}")
    except requests.exceptions.Timeout:
        print(f"[{prop}] TIMEOUT after {TIMEOUT}s")
    except Exception as exc:
        print(f"[{prop}] ERROR: {type(exc).__name__}: {exc}")
    print()
"""Measure how often SoilGrids answers (and how fast) for the Thatta town point."""
import time

import requests

URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
LAT, LON = 24.7476, 67.9236
PARAMS = [("lon", LON), ("lat", LAT), ("value", "mean")]
PARAMS += [("property", p) for p in ["phh2o", "clay", "sand", "silt", "soc"]]
PARAMS += [("depth", d) for d in ["0-5cm", "5-15cm", "15-30cm"]]

ok = 0
for i in range(1, 7):
    start = time.time()
    try:
        r = requests.get(URL, params=PARAMS, timeout=(5, 40))
        print(f"try {i}: status={r.status_code} time={time.time() - start:.1f}s")
        ok += r.ok
    except requests.RequestException as exc:
        print(f"try {i}: FAILED after {time.time() - start:.1f}s ({type(exc).__name__})")
    time.sleep(10)  # stay polite: ISRIC rate-limits

print(f"\n{ok}/6 requests succeeded")
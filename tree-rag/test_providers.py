"""Quick check: which external provider is slow or down (NASA POWER vs SoilGrids)."""
import time
import requests

LAT, LON = 24.75, 67.92  # Thatta (inside coverage)
TIMEOUT = 20


def timed_get(name, url, params=None):
    start = time.time()
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        took = time.time() - start
        print(f"[{name}] status={r.status_code} time={took:.1f}s size={len(r.content)} bytes")
        print(f"[{name}] first 200 chars: {r.text[:200]!r}")
    except requests.exceptions.Timeout:
        print(f"[{name}] TIMEOUT after {TIMEOUT}s")
    except Exception as exc:
        print(f"[{name}] ERROR: {type(exc).__name__}: {exc}")
    print()


# 1) NASA POWER climatology
timed_get(
    "NASA POWER",
    "https://power.larc.nasa.gov/api/temporal/climatology/point",
    {
        "parameters": "T2M,PRECTOTCORR,RH2M",
        "community": "AG",
        "longitude": LON,
        "latitude": LAT,
        "format": "JSON",
    },
)

# 2) SoilGrids REST (topsoil pH, 0-5 cm)
timed_get(
    "SoilGrids",
    "https://rest.isric.org/soilgrids/v2.0/properties/query",
    {
        "lon": LON,
        "lat": LAT,
        "property": "phh2o",
        "depth": "0-5cm",
        "value": "mean",
    },
)
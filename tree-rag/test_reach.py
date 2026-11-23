"""Check which data hosts are reachable from this network."""
import time

import requests

HOSTS = {
    "ISRIC SoilGrids": "https://rest.isric.org/soilgrids/v2.0/properties/query?lon=67.92&lat=24.75&property=clay&depth=0-5cm&value=mean",
    "NASA POWER": "https://power.larc.nasa.gov/api/temporal/climatology/point?parameters=T2M&community=AG&longitude=67.92&latitude=24.75&format=JSON",
    "Google": "https://www.google.com",
}

for name, url in HOSTS.items():
    start = time.time()
    try:
        r = requests.get(url, timeout=(5, 15))
        print(f"{name}: status={r.status_code} time={time.time() - start:.1f}s")
    except requests.RequestException as exc:
        print(f"{name}: FAILED after {time.time() - start:.1f}s ({type(exc).__name__})")
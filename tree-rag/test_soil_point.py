"""Show the raw SoilGrids response for the exact point that returned 'no soil data'."""
import json
import requests

LAT, LON = 24.7476, 67.9236
params = [("lon", LON), ("lat", LAT), ("value", "mean")]
params += [("property", p) for p in ["phh2o", "clay", "sand", "silt", "soc"]]
params += [("depth", d) for d in ["0-5cm", "5-15cm", "15-30cm"]]

r = requests.get(
    "https://rest.isric.org/soilgrids/v2.0/properties/query",
    params=params,
    timeout=(5, 40),
)
print("status:", r.status_code)
try:
    print(json.dumps(r.json(), indent=2)[:4000])
except ValueError:
    print(r.text[:1500])
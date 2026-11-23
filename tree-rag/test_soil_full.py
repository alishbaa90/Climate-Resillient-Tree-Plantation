"""Print the full SoilGrids response for Thatta to see whether values are null."""
import requests

LAT, LON = 24.75, 67.92
r = requests.get(
    "https://rest.isric.org/soilgrids/v2.0/properties/query",
    params={
        "lon": LON,
        "lat": LAT,
        "property": ["phh2o", "clay", "sand", "silt", "soc"],
        "depth": "0-5cm",
        "value": "mean",
    },
    timeout=20,
)
print("status:", r.status_code)
print(r.text)
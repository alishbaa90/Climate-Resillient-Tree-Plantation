import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone

from dotenv import load_dotenv

import nasa_power
import soilgrids
from nasa_power import ProviderError

load_dotenv()
log = logging.getLogger("tree-api")

# Put the district boundary in this GeoJSON file (Polygon or MultiPolygon) to get an exact coverage check.
BOUNDARY_FILE = os.getenv("COVERAGE_GEOJSON", os.path.join("data", "thatta_boundary.geojson"))
# Until that file exists, a rough bounding box is used: min_lon,min_lat,max_lon,max_lat.
# THIS BOX IS ONLY AN APPROXIMATION AROUND THE LOWER INDUS DELTA. Replace it with the real boundary.
ROUGH_BBOX = [float(v) for v in os.getenv("COVERAGE_BBOX", "66.5,23.5,69.0,25.8").split(",")]

CLIMATE_TTL = 24 * 3600    # climate normals do not change
SOIL_TTL = 7 * 24 * 3600   # soil maps do not change
FAILURE_TTL = 300          # a real failure (for example "no data") is not asked again for 5 minutes
TRANSIENT_FAILURE_TTL = 20 # a slow or rate-limited provider may be asked again after 20 seconds
OVERALL_DEADLINE = 48      # seconds the endpoint waits for the requested providers together
ALL_PARTS = ("climate", "soil")
_CACHE = {}
_INFLIGHT = {}             # provider key -> running future, so the same point is not requested twice at once
_POOL = ThreadPoolExecutor(max_workers=8)

NOTES = [
    "These values are estimates from gridded datasets. They are not field measurements.",
    "Soil salinity (EC) is not available from these datasets. Ask the assistant about salinity: it answers only from the documents.",
]


class OutsideCoverage(ValueError):
    pass


def _polygons(geo):
    """Yield the rings of every polygon in a GeoJSON object (FeatureCollection, Feature, Polygon, MultiPolygon)."""
    kind = geo.get("type")
    if kind == "FeatureCollection":
        for feature in geo.get("features", []):
            yield from _polygons(feature)
    elif kind == "Feature":
        yield from _polygons(geo.get("geometry") or {})
    elif kind == "Polygon":
        yield geo["coordinates"]
    elif kind == "MultiPolygon":
        yield from geo["coordinates"]


def _in_ring(lon, lat, ring):
    inside, j = False, len(ring) - 1
    for i in range(len(ring)):
        xi, yi, xj, yj = ring[i][0], ring[i][1], ring[j][0], ring[j][1]
        if (yi > lat) != (yj > lat) and lon < (xj - xi) * (lat - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _load_polygons():
    if not os.path.exists(BOUNDARY_FILE):
        return None
    with open(BOUNDARY_FILE, encoding="utf-8") as file:
        return list(_polygons(json.load(file)))


_POLYGONS = _load_polygons()


def coverage_mode():
    return "boundary file" if _POLYGONS else "rough bounding box (not the real district boundary)"


def in_coverage(lat, lon):
    if _POLYGONS:
        return any(_in_ring(lon, lat, rings[0]) and not any(_in_ring(lon, lat, h) for h in rings[1:]) for rings in _POLYGONS)
    min_lon, min_lat, max_lon, max_lat = ROUGH_BBOX
    return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat


def _cached(key, fetch, ok_ttl):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and hit[0] > now:
        if hit[2] is not None:
            raise hit[2]
        return hit[1]
    try:
        value = fetch()
    except ProviderError as error:
        ttl = TRANSIENT_FAILURE_TTL if getattr(error, "transient", False) else FAILURE_TTL
        _CACHE[key] = (time.time() + ttl, None, error)
        raise
    _CACHE[key] = (time.time() + ok_ttl, value, None)
    return value


def _submit(key, module, lat, lon, ttl):
    """Run one provider in the background. If the same point is already being fetched, reuse that request."""
    running = _INFLIGHT.get(key)
    if running is not None and not running.done():
        return running
    future = _POOL.submit(_cached, key, lambda: module.fetch(lat, lon), ttl)
    _INFLIGHT[key] = future
    return future


def get_environment(lat, lon, parts=ALL_PARTS):
    """Climate and/or topsoil data for one point. Each requested provider runs in parallel and is independent:
    if one fails or is too slow, the other still answers and the failure is reported in result['errors']."""
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Latitude must be between -90 and 90 and longitude between -180 and 180.")
    if not in_coverage(lat, lon):
        raise OutsideCoverage("This point is outside the area covered by the knowledge base (Thatta).")

    if len(_CACHE) > 2000:
        _CACHE.clear()
    result = {
        "point": {"lat": round(lat, 5), "lon": round(lon, 5)},
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "coverage_check": coverage_mode(),
        "climate": None,
        "soil": None,
        "errors": {},
        "notes": NOTES,
    }
    providers = (
        ("climate", nasa_power, 2, CLIMATE_TTL),   # cache key rounded to ~1 km: the climate grid is far coarser
        ("soil", soilgrids, 3, SOIL_TTL),          # ~100 m, close to the 250 m soil grid
    )
    futures = []
    for name, module, decimals, ttl in providers:
        if name not in parts:
            continue
        key = (name, round(lat, decimals), round(lon, decimals))
        futures.append((name, _submit(key, module, lat, lon, ttl)))

    deadline = time.time() + OVERALL_DEADLINE
    for name, future in futures:
        remaining = max(0.1, deadline - time.time())
        try:
            result[name] = future.result(timeout=remaining)
        except ProviderError as error:
            result["errors"][name] = str(error)
        except FutureTimeout:
            result["errors"][name] = (
                f"The {name} service is taking too long. Try the same point again in a minute: "
                "the answer is saved once it arrives."
            )
        except Exception:
            log.exception("Unexpected failure in the %s provider", name)
            result["errors"][name] = f"The {name} service failed unexpectedly."
    return result
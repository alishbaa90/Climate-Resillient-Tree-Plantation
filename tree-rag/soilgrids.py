import math
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from nasa_power import ProviderError

URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPERTIES = {"phh2o": "ph", "clay": "clay_pct", "sand": "sand_pct", "silt": "silt_pct", "soc": "organic_carbon_g_kg"}
DEPTHS = {"0-5cm": 5, "5-15cm": 10, "15-30cm": 15}  # label -> thickness in cm; the average covers 0-30 cm
TIMEOUT = (5, 18)         # (connect, read) seconds for the exact point
NEARBY_TIMEOUT = (5, 15)  # (connect, read) seconds for each nearby-pixel attempt
ATTEMPTS = 2              # the REST gateway is sometimes slow at random, so one retry often works
RETRY_PAUSE = 1.0
NEARBY_STEP_DEG = 0.003   # about 300 m; the soil grid is 250 m
NEARBY_OFFSETS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # E, N, W, S (lon, lat) multipliers


def _transient(message, cause=None):
    """A ProviderError that marks a temporary problem (slow gateway, rate limit), not missing data."""
    error = ProviderError(message)
    error.transient = True
    if cause is not None:
        error.__cause__ = cause
    return error


def _request(params, timeout=TIMEOUT, attempts=ATTEMPTS):
    """One GET with a retry on timeouts and connection errors. Returns the response."""
    last_error = None
    for attempt in range(attempts):
        if attempt:
            time.sleep(RETRY_PAUSE)
        try:
            return requests.get(URL, params=params, timeout=timeout)
        except requests.RequestException as error:
            last_error = error
    raise _transient(
        "SoilGrids is too slow or could not be reached right now (its REST service is in beta and can be paused).",
        last_error,
    )


def _params(lat, lon):
    params = [("lon", lon), ("lat", lat), ("value", "mean")]
    params += [("property", p) for p in PROPERTIES]
    params += [("depth", d) for d in DEPTHS]
    return params


def _parse(response):
    """Return a dict of topsoil values, or an empty dict if every value is null at this point."""
    if response.status_code == 429:
        raise _transient("SoilGrids is limiting requests right now. Try again in a few minutes.")
    if not response.ok:
        raise _transient(f"SoilGrids returned HTTP {response.status_code} (its REST service can be paused).")
    try:
        layers = response.json()["properties"]["layers"]
    except (ValueError, KeyError, TypeError):
        raise ProviderError("SoilGrids sent an unexpected response.")

    result = {}
    for layer in layers:
        key = PROPERTIES.get(layer.get("name"))
        if not key:
            continue
        factor = (layer.get("unit_measure") or {}).get("d_factor") or 1
        total = weight = 0.0
        for depth in layer.get("depths", []):
            thickness = DEPTHS.get(depth.get("label"))
            mean = (depth.get("values") or {}).get("mean")
            if thickness and mean is not None:
                total += (mean / factor) * thickness
                weight += thickness
        if weight:
            result[key] = round(total / weight, 1)
    return result


def _distance_m(lat1, lon1, lat2, lon2):
    """Haversine distance in metres."""
    radius = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def _try_nearby(lat, lon, dlon, dlat):
    """Fetch one nearby pixel. Returns (lat, lon, values_or_empty, error_or_None)."""
    near_lat, near_lon = lat + dlat * NEARBY_STEP_DEG, lon + dlon * NEARBY_STEP_DEG
    try:
        values = _parse(_request(_params(near_lat, near_lon), timeout=NEARBY_TIMEOUT))
    except ProviderError as error:
        return near_lat, near_lon, {}, error
    return near_lat, near_lon, values, None


def fetch(lat, lon):
    """Topsoil (0-30 cm) properties from the ISRIC SoilGrids REST API, averaged over the three layers.
    If the exact pixel has no data (water, built-up land), the nearest valid pixel within about 300 m
    is used and the distance is reported. A slow or failed nearby request is reported as 'too slow',
    never as 'no data'. ISRIC documents that this REST service is in beta and can be paused."""
    result = _parse(_request(_params(lat, lon)))
    if result:
        result["label"] = "SoilGrids 250 m topsoil (0-30 cm) estimate, not a field measurement"
        return result

    with ThreadPoolExecutor(max_workers=len(NEARBY_OFFSETS)) as pool:
        attempts = list(pool.map(lambda off: _try_nearby(lat, lon, off[0], off[1]), NEARBY_OFFSETS))

    for near_lat, near_lon, values, _error in attempts:  # offsets order: E, N, W, S
        if values:
            metres = round(_distance_m(lat, lon, near_lat, near_lon))
            values["nearest_valid_pixel_m"] = metres
            values["label"] = (
                f"SoilGrids 250 m topsoil (0-30 cm) estimate from the nearest valid pixel, "
                f"about {metres} m from the selected point (the exact point has no data). Not a field measurement"
            )
            return values

    failed = [error for *_rest, error in attempts if error is not None]
    if failed:
        raise _transient(
            "SoilGrids is too slow right now to check the pixels around this point. Try again in a minute."
        )
    raise ProviderError(
        "SoilGrids has no soil data for this point or within about 300 m of it (for example water or built-up land)."
    )
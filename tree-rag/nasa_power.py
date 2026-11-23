import requests

URL = "https://power.larc.nasa.gov/api/temporal/climatology/point"
PARAMETERS = "T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M"
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
TIMEOUT = 30


class ProviderError(Exception):
    """A data provider could not give a usable answer. The message is safe to show to the user."""


def _num(value):
    """POWER marks missing values with -999."""
    return None if value is None or value <= -990 else float(value)


def fetch(lat, lon):
    """Long-term monthly climate normals for one point from NASA POWER (a coarse grid, about 50 km)."""
    try:
        response = requests.get(
            URL,
            params={"parameters": PARAMETERS, "community": "AG", "longitude": lon, "latitude": lat, "format": "JSON"},
            timeout=TIMEOUT,
        )
    except requests.RequestException as error:
        raise ProviderError("NASA POWER could not be reached.") from error
    if response.status_code == 429:
        raise ProviderError("NASA POWER is limiting requests right now. Try again in a few minutes.")
    if not response.ok:
        raise ProviderError(f"NASA POWER returned HTTP {response.status_code}.")
    try:
        data = response.json()["properties"]["parameter"]
    except (ValueError, KeyError):
        raise ProviderError("NASA POWER sent an unexpected response.")

    def series(name):
        values = data.get(name, {})
        return {key: _num(values.get(key)) for key in MONTHS + ["ANN"]}

    temp, temp_max, temp_min, rain, humidity = (
        series("T2M"), series("T2M_MAX"), series("T2M_MIN"), series("PRECTOTCORR"), series("RH2M")
    )
    # PRECTOTCORR is in mm per day, so a monthly total is the daily mean times the days of the month.
    monthly_rain = {m: round(rain[m] * d, 1) for m, d in zip(MONTHS, DAYS) if rain[m] is not None}
    if len(monthly_rain) < 12 or temp["ANN"] is None:
        raise ProviderError("NASA POWER returned incomplete data for this point.")

    hottest = max(MONTHS, key=lambda m: temp_max[m] if temp_max[m] is not None else -999)
    wettest = max(monthly_rain, key=monthly_rain.get)
    return {
        "temperature_mean_c": round(temp["ANN"], 1),
        "temperature_max_c": {"month": hottest, "value": round(temp_max[hottest], 1)} if temp_max[hottest] is not None else None,
        "temperature_min_c": round(min(v for k, v in temp_min.items() if k != "ANN" and v is not None), 1)
        if any(v is not None for k, v in temp_min.items() if k != "ANN") else None,
        "rainfall_annual_mm": round(sum(monthly_rain.values())),
        "rainfall_wettest_month": {"month": wettest, "mm": monthly_rain[wettest]},
        "rainfall_monthly_mm": monthly_rain,
        "humidity_percent": round(humidity["ANN"], 1) if humidity["ANN"] is not None else None,
        "label": "NASA POWER long-term monthly averages (coarse grid, not a local weather station)",
    }
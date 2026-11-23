import json
import logging
import os
from typing import Optional

import requests
from fastapi import APIRouter, Header, HTTPException

import environment
import supabase_client as db

log = logging.getLogger("tree-api")
router = APIRouter()
SPECIES_FILE = os.getenv("SPECIES_JSON", os.path.join("data", "species.json"))
_FILE_CACHE = {}  # path -> (file modified time, parsed JSON)


def _error(code, message, status):
    return HTTPException(status_code=status, detail={"code": code, "message": message})


def _require_login(authorization):
    if not db.configured():
        raise _error("auth_not_configured", "Login is not configured on the server.", 503)
    if not authorization or not authorization.lower().startswith("bearer "):
        raise _error("missing_token", "Please sign in first.", 401)
    try:
        return db.verify_token(authorization[7:].strip())
    except db.AuthError:
        raise _error("invalid_token", "Your session has expired. Please sign in again.", 401)
    except requests.RequestException:
        raise _error("auth_unreachable", "The login service could not be reached.", 502)


def _json_file(path):
    """Parsed JSON of a data file, re-read only when the file changes."""
    modified = os.path.getmtime(path)
    cached = _FILE_CACHE.get(path)
    if cached is None or cached[0] != modified:
        with open(path, encoding="utf-8") as file:
            cached = (modified, json.load(file))
        _FILE_CACHE[path] = cached
    return cached[1]


def _parse_parts(parts):
    """'climate', 'soil' or 'climate,soil' (the default is both)."""
    if not parts:
        return environment.ALL_PARTS
    wanted = tuple(dict.fromkeys(p.strip() for p in parts.split(",") if p.strip()))
    if not wanted or any(p not in environment.ALL_PARTS for p in wanted):
        raise _error("invalid_parts", "parts must be 'climate', 'soil' or 'climate,soil'.", 422)
    return wanted


@router.get("/boundary")
def district_boundary(authorization: Optional[str] = Header(default=None)):
    """The district boundary (GeoJSON) that the map draws and that /environment uses for its coverage check."""
    _require_login(authorization)
    path = environment.BOUNDARY_FILE
    if not os.path.exists(path):
        raise _error("boundary_missing", "The district boundary file is not installed on the server.", 404)
    return _json_file(path)


@router.get("/species")
def species_profiles(authorization: Optional[str] = Header(default=None)):
    """Species profiles built from the knowledge-base tables by build_species.py (no language model)."""
    _require_login(authorization)
    if not os.path.exists(SPECIES_FILE):
        raise _error("species_missing", "The species file has not been built yet. Run: py build_species.py", 404)
    return _json_file(SPECIES_FILE)


@router.get("/environment")
def environment_for_point(lat: float, lon: float, parts: Optional[str] = None,
                          authorization: Optional[str] = Header(default=None)):
    """Climate (NASA POWER) and/or topsoil (SoilGrids) for a point inside the covered area.
    Use parts=climate or parts=soil to ask for one of them: the map loads climate first, soil after."""
    _require_login(authorization)
    wanted = _parse_parts(parts)
    try:
        result = environment.get_environment(lat, lon, wanted)
    except environment.OutsideCoverage as error:
        raise _error("outside_coverage", str(error), 422)
    except ValueError as error:
        raise _error("invalid_point", str(error), 422)

    # Asking for both and getting neither is a server-side failure. A single failed part is returned
    # with status 200 and its reason in result["errors"], so the page can show it next to the other part.
    if len(wanted) == len(environment.ALL_PARTS) and result["climate"] is None and result["soil"] is None:
        log.error("Environment providers failed: %s", result["errors"])
        raise _error("environment_unavailable", "Both data services failed: " + " ".join(result["errors"].values()), 502)
    log.info("environment | %.4f,%.4f | %s", lat, lon,
             " ".join(f"{name}={'ok' if result[name] else 'failed'}" for name in wanted))
    return result
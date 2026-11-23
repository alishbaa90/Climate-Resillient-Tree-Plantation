import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
TIMEOUT = 15

# A verified token is remembered for a short time, so that every request does not call Supabase Auth.
_TOKEN_TTL_SECONDS = 60
_TOKEN_CACHE = {}

# Columns of "messages" that must never be sent as NULL. Missing values are filled with an empty list.
_JSON_LIST_COLUMNS = ("sources", "warnings")


class AuthError(Exception):
    """The token is missing, expired or not valid."""


class DatabaseError(Exception):
    """Supabase refused or failed a database request."""


def configured():
    return bool(SUPABASE_URL and SUPABASE_ANON_KEY)


def verify_token(token):
    """Ask Supabase Auth who owns this access token. Works for email, Google and any other login.
    Returns {"id": ..., "email": ...}. Raises AuthError for a bad token and requests exceptions
    if Supabase cannot be reached."""
    now = time.time()
    cached = _TOKEN_CACHE.get(token)
    if cached and cached[0] > now:
        return cached[1]

    response = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={"apikey": SUPABASE_ANON_KEY, "Authorization": f"Bearer {token}"},
        timeout=TIMEOUT,
    )
    if response.status_code in (400, 401, 403):
        raise AuthError("invalid or expired token")
    response.raise_for_status()

    data = response.json()
    user = {"id": data["id"], "email": data.get("email")}
    if len(_TOKEN_CACHE) > 500:
        _TOKEN_CACHE.clear()
    _TOKEN_CACHE[token] = (now + _TOKEN_TTL_SECONDS, user)
    return user


def _rest(method, table, token, params=None, payload=None, return_rows=True):
    """Call the Supabase REST API as the signed-in user. Row Level Security decides what is visible."""
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if method == "POST" and return_rows:
        headers["Prefer"] = "return=representation"
    response = requests.request(
        method,
        f"{SUPABASE_URL}/rest/v1/{table}",
        headers=headers,
        params=params,
        json=payload,
        timeout=TIMEOUT,
    )
    if not response.ok:
        raise DatabaseError(f"{method} {table} failed with {response.status_code}: {response.text[:300]}")
    return response.json() if response.text else None


def create_conversation(token, title):
    rows = _rest("POST", "conversations", token, payload={"title": title})
    return rows[0]


def get_conversation(token, conversation_id):
    rows = _rest("GET", "conversations", token, params={"id": f"eq.{conversation_id}", "select": "id,title,created_at"})
    return rows[0] if rows else None


def list_conversations(token, limit=50):
    return _rest(
        "GET", "conversations", token,
        params={"select": "id,title,created_at", "order": "created_at.desc", "limit": str(limit)},
    )


def add_messages(token, rows):
    """Insert several messages in one request.
    A bulk insert needs the same columns in every row. When rows have different columns, missing values
    would be sent as NULL and break the NOT NULL columns (sources, warnings). So every row gets every
    column, with an empty list or None where the caller gave no value."""
    columns = set()
    for row in rows:
        columns.update(row.keys())
    uniform = [
        {
            column: row.get(column, [] if column in _JSON_LIST_COLUMNS else None)
            for column in columns
        }
        for row in rows
    ]
    return _rest("POST", "messages", token, payload=uniform, return_rows=False)


def list_messages(token, conversation_id):
    return _rest(
        "GET", "messages", token,
        params={
            "conversation_id": f"eq.{conversation_id}",
            "select": "role,content,answer_type,reliable,sources,warnings,top_score,config,created_at",
            "order": "seq.asc",
        },
    )


def delete_conversation(token, conversation_id):
    _rest("DELETE", "conversations", token, params={"id": f"eq.{conversation_id}"}, return_rows=False)
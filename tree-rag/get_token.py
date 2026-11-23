import getpass
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
TOKEN_FILE = ".token"


def main():
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        print("Set SUPABASE_URL and SUPABASE_ANON_KEY in the .env file first.")
        sys.exit(1)

    email = input("Email of the test user: ").strip()
    password = getpass.getpass("Password (hidden): ")

    response = requests.post(
        f"{SUPABASE_URL}/auth/v1/token",
        params={"grant_type": "password"},
        headers={"apikey": SUPABASE_ANON_KEY, "Content-Type": "application/json"},
        json={"email": email, "password": password},
        timeout=20,
    )
    if not response.ok:
        print(f"Sign-in failed ({response.status_code}): {response.text[:300]}")
        sys.exit(1)

    token = response.json()["access_token"]
    with open(TOKEN_FILE, "w", encoding="utf-8") as file:
        file.write(token)
    print(f"Signed in. The access token was saved to {TOKEN_FILE} (it expires after about an hour).")
    print("Never share this file or commit it to GitHub.")


if __name__ == "__main__":
    main()
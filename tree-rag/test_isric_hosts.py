"""Diagnose why rest.isric.org does not connect, and whether files.isric.org does."""
import socket
import time

import requests

for host in ["rest.isric.org", "files.isric.org"]:
    print(f"--- {host}")
    try:
        ips = sorted({info[4][0] for info in socket.getaddrinfo(host, 443)})
        print("DNS ok:", ips)
    except OSError as exc:
        print("DNS FAILED:", exc)
        continue
    start = time.time()
    try:
        with socket.create_connection((host, 443), timeout=8):
            print(f"TCP 443 connect ok in {time.time() - start:.1f}s")
    except OSError as exc:
        print(f"TCP 443 connect FAILED after {time.time() - start:.1f}s: {exc}")
        continue
    try:
        r = requests.get(f"https://{host}/", timeout=(8, 15))
        print("HTTPS status:", r.status_code)
    except requests.RequestException as exc:
        print("HTTPS FAILED:", type(exc).__name__)
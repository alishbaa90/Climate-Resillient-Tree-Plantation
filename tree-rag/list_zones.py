import os
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION = "tree_evidence"
GUIDELINE = "silvo_ecological_zones_2023.pdf"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def main():
    region_filter = sys.argv[1] if len(sys.argv) > 1 else None

    zones = {}  # (region, zone title) -> {"pages": set, "rows": int}
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in batch:
            payload = point.payload
            if payload["source"] != GUIDELINE or payload["content_type"] != "table":
                continue
            if region_filter and payload["region"] != region_filter:
                continue
            for line in payload["text"].split("\n"):
                if not line.startswith("Zone: "):
                    continue
                title = line[len("Zone: "):].split(";")[0].strip()
                entry = zones.setdefault((payload["region"], title), {"pages": set(), "rows": 0})
                entry["pages"].add(payload["page"])
                entry["rows"] += 1
        if offset is None:
            break

    if not zones:
        print("No zone rows found.")
        return

    ordered = sorted(zones.items(), key=lambda item: (item[0][0], min(item[1]["pages"]), item[0][1]))
    print(f"{'region':<20} {'pages':<8} {'rows':<5} zone")
    for (region, title), info in ordered:
        pages = ",".join(str(p) for p in sorted(info["pages"]))
        print(f"{region:<20} {pages:<8} {info['rows']:<5} {title}")


if __name__ == "__main__":
    main()
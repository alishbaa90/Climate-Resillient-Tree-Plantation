import os
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION = "tree_evidence"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def main():
    if len(sys.argv) != 3:
        print("Usage: py inspect_page.py <pdf_file_name> <pdf_page_number>")
        print("Example: py inspect_page.py silvo_ecological_zones_2023.pdf 46")
        return

    source = sys.argv[1]
    page = int(sys.argv[2])

    chunks = []
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
            if point.payload["source"] == source and point.payload["page"] == page:
                chunks.append(point.payload)
        if offset is None:
            break

    if not chunks:
        print(f"No stored chunks found for {source}, page {page}.")
        return

    chunks.sort(key=lambda p: p["chunk_index"])
    for payload in chunks:
        print("=" * 70)
        print(f"chunk_index={payload['chunk_index']}  type={payload['content_type']}  "
              f"region={payload['region']}  length={len(payload['text'])} chars")
        print(f"title: {payload['title']}")
        print("-" * 70)
        print(payload["text"])


if __name__ == "__main__":
    main()
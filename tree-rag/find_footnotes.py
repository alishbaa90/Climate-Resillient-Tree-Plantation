import os

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

offset = None
count = 0
while True:
    batch, offset = client.scroll(
        collection_name="tree_evidence",
        limit=256,
        offset=offset,
        with_payload=True,
        with_vectors=False,
    )
    for point in batch:
        payload = point.payload
        for line in payload["text"].split("\n"):
            if line.startswith("Table footnote:"):
                count += 1
                print(f"{payload['source']} p{payload['page']} ({payload['region']}): {line[:220]}")
    if offset is None:
        break
print(f"\n{count} footnote(s) found.")
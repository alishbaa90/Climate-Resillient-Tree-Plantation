import os
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


def test_jina():
    print("Testing Jina...")
    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": "jina-embeddings-v3",
            "task": "retrieval.passage",
            "input": ["Shisham is a drought tolerant tree."],
        },
        timeout=30,
    )
    response.raise_for_status()
    vector = response.json()["data"][0]["embedding"]
    print(f"Jina OK. Embedding size: {len(vector)}")


def test_qdrant():
    print("Testing Qdrant...")
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    collections = client.get_collections()
    print(f"Qdrant OK. Existing collections: {len(collections.collections)}")


if __name__ == "__main__":
    test_jina()
    test_qdrant()
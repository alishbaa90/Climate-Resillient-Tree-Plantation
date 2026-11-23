import os
import re

import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny
from rank_bm25 import BM25Okapi

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION = "tree_evidence"
EMBED_MODEL = "jina-embeddings-v3"
RERANK_MODEL = "jina-reranker-v2-base-multilingual"

DENSE_K = 20            # candidates from vector search
BM25_K = 20             # candidates from keyword search
RERANK_CANDIDATES = 20  # how many fused candidates are sent to the reranker
RERANK_TOP_N = 10       # results returned (answer.py then picks a balanced subset)
RRF_K = 60              # standard constant for Reciprocal Rank Fusion

# Common names and scientific names that appear in the knowledge base documents.
# Used for query expansion: asking for "Shisham" also searches for "Dalbergia sissoo", and vice versa.
# Extend this list whenever a species is missed because the question used a different name.
SPECIES_ALIASES = [
    (["shisham", "sheesham", "tali", "tahli", "talhi"], ["Dalbergia sissoo"]),
    (["kikar", "babul", "babur"], ["Vachellia nilotica", "Acacia nilotica"]),
    (["neem"], ["Azadirachta indica"]),
    (["siris", "sirin", "sareehan"], ["Albizia lebbeck"]),
    (["jand", "kandi", "kahur"], ["Prosopis cineraria"]),
    (["ber", "bair"], ["Ziziphus mauritiana"]),
    (["peelu", "pilu", "jar"], ["Salvadora oleoides", "Salvadora persica"]),
    (["farash", "ghaz"], ["Tamarix aphylla"]),
    (["jaman", "jamun"], ["Syzygium cumini"]),
    (["arjun"], ["Terminalia arjuna"]),
    (["kumbhat", "kumbhut"], ["Senegalia senegal", "Acacia senegal"]),
    (["rohira", "rohida"], ["Tecomella undulata"]),
    (["sukh chain", "sukhchain"], ["Millettia pinnata"]),
]

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def expand_query(question):
    """Append the other names of any species mentioned in the question."""
    lowered = question.lower()
    extra = []
    for common_names, scientific_names in SPECIES_ALIASES:
        common_mentioned = any(re.search(rf"\b{re.escape(name)}\b", lowered) for name in common_names)
        scientific_mentioned = any(name.lower() in lowered for name in scientific_names)
        if common_mentioned and not scientific_mentioned:
            extra.extend(scientific_names)
        elif scientific_mentioned and not common_mentioned:
            extra.extend(name.title() for name in common_names)
    if not extra:
        return question
    return f"{question} {' '.join(extra)}"


def region_filter(regions):
    if not regions:
        return None
    return Filter(must=[FieldCondition(key="region", match=MatchAny(any=regions))])


def load_corpus(regions):
    """Load all chunks (without vectors) for the given regions so BM25 can index them."""
    points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=region_filter(regions),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break
    return points


def embed_query(text):
    response = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={"model": EMBED_MODEL, "task": "retrieval.query", "input": [text]},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]


def rerank(question, documents, top_n):
    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": RERANK_MODEL,
            "query": question,
            "documents": documents,
            "top_n": top_n,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["results"]


class HybridRetriever:
    def __init__(self, regions=None):
        self.regions = regions
        self.points = load_corpus(regions)
        if not self.points:
            raise RuntimeError("No chunks found. Run ingest.py first or check the region names.")
        self.by_id = {str(p.id): p for p in self.points}
        # Title is included in the BM25 text, matching what was embedded during ingestion.
        tokenized = [tokenize(f"{p.payload['title']} {p.payload['text']}") for p in self.points]
        self.bm25 = BM25Okapi(tokenized)
        print(f"Retriever ready: {len(self.points)} chunks (regions={regions})")

    def dense_search(self, query):
        vector = embed_query(query)
        result = client.query_points(
            collection_name=COLLECTION,
            query=vector,
            query_filter=region_filter(self.regions),
            limit=DENSE_K,
            with_payload=False,
        )
        return [str(p.id) for p in result.points]

    def bm25_search(self, query):
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:BM25_K]
        return [str(self.points[i].id) for i in ranked if scores[i] > 0]

    @staticmethod
    def fuse(dense_ids, bm25_ids):
        """Reciprocal Rank Fusion: combine two ranked lists into one."""
        scores = {}
        for ranking in (dense_ids, bm25_ids):
            for rank, point_id in enumerate(ranking, start=1):
                scores[point_id] = scores.get(point_id, 0.0) + 1.0 / (RRF_K + rank)
        return sorted(scores, key=scores.get, reverse=True)

    def search(self, question):
        query = expand_query(question)
        dense_ids = self.dense_search(query)
        bm25_ids = self.bm25_search(query)
        fused_ids = self.fuse(dense_ids, bm25_ids)[:RERANK_CANDIDATES]

        documents = []
        for point_id in fused_ids:
            payload = self.by_id[point_id].payload
            documents.append(f"{payload['title']}\n\n{payload['text']}")

        reranked = rerank(query, documents, RERANK_TOP_N)

        results = []
        for item in reranked:
            point_id = fused_ids[item["index"]]
            payload = self.by_id[point_id].payload
            results.append(
                {
                    "rerank_score": item["relevance_score"],
                    "in_dense": point_id in dense_ids,
                    "in_bm25": point_id in bm25_ids,
                    "source": payload["source"],
                    "page": payload["page"],
                    "region": payload["region"],
                    "doc_type": payload["doc_type"],
                    "content_type": payload.get("content_type", "text"),
                    "text": payload["text"],
                }
            )
        return results
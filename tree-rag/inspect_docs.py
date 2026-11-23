import os
import re

import pymupdf
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION = "tree_evidence"
DATA_DIR = "data"
GUIDELINE = "silvo_ecological_zones_2023.pdf"
GUIDELINE_REGION = "pakistan"

# Part 1: check how Annex 3 was stored after re-ingestion.
INSPECT_PAGE = 77
PREVIEW_CHARS = 900

# Part 2: where the Sindh section of the guideline is (printed page 35 -> PDF page ~45).
SINDH_PAGES = range(44, 52)
KEYWORDS = ["dalbergia sissoo", "shisham", "sissoo"]
PROVINCES = ["Sindh", "Balochistan", "Punjab", "AJK", "Khyber", "Gilgit"]
HEADING_PATTERN = re.compile(r"^\s*\d+(\.\d+)*\.?\s+[A-Z][A-Za-z ,&\-]{3,80}$")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def part1_stored_chunks():
    print("=" * 70)
    print(f"PART 1: stored chunks for {GUIDELINE}, page {INSPECT_PAGE}")
    print("=" * 70)

    matches = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="region", match=MatchValue(value=GUIDELINE_REGION))]
            ),
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in batch:
            payload = point.payload
            if payload["source"] == GUIDELINE and payload["page"] == INSPECT_PAGE:
                matches.append(payload)
        if offset is None:
            break

    matches.sort(key=lambda p: p["chunk_index"])
    for payload in matches:
        print(f"\n--- chunk_index={payload['chunk_index']}  type={payload['content_type']}  "
              f"length={len(payload['text'])} chars ---")
        print(payload["text"][:PREVIEW_CHARS])
        if len(payload["text"]) > PREVIEW_CHARS:
            print("   [... shortened ...]")


def part2_shisham_in_guideline():
    print("\n" + "=" * 70)
    print(f"PART 2: Shisham mentions in {GUIDELINE}, grouped by page")
    print("=" * 70)

    doc = pymupdf.open(os.path.join(DATA_DIR, GUIDELINE))
    for page_number, page in enumerate(doc, start=1):
        text = " ".join(page.get_text().split())
        lowered = text.lower()
        found = [k for k in KEYWORDS if k in lowered]
        if not found:
            continue
        provinces = [p for p in PROVINCES if p.lower() in lowered]
        position = lowered.find(found[0])
        snippet = text[max(0, position - 60):position + 100]
        print(f"\npage {page_number} | provinces on page: {provinces or 'none'}")
        print(f"   starts with: {text[:110]}...")
        print(f"   snippet: ...{snippet}...")
    doc.close()


def part3_sindh_section():
    print("\n" + "=" * 70)
    print(f"PART 3: headings and Shisham on PDF pages {SINDH_PAGES.start}-{SINDH_PAGES.stop - 1}")
    print("=" * 70)

    doc = pymupdf.open(os.path.join(DATA_DIR, GUIDELINE))
    for page_number in SINDH_PAGES:
        if page_number > len(doc):
            break
        text = doc[page_number - 1].get_text()
        headings = [line.strip() for line in text.split("\n") if HEADING_PATTERN.match(line)]
        has_shisham = any(k in text.lower() for k in KEYWORDS)
        print(f"\npage {page_number} | shisham mentioned: {has_shisham}")
        print(f"   headings: {headings[:8]}")
    doc.close()


if __name__ == "__main__":
    part1_stored_chunks()
    part2_shisham_in_guideline()
    part3_sindh_section()
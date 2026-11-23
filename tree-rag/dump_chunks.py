"""Write the knowledge-base chunks to text files so the species parser can be built from the real format.
Read-only: it does not change Qdrant or any frozen file. Contains document text only (no API keys)."""
from retrieval import load_corpus

REGIONS = ["thatta", "sindh", "pakistan"]  # same as ALLOWED_REGIONS in answer.py

points = load_corpus(REGIONS)
points.sort(key=lambda p: (p.payload["source"], p.payload["page"]))

with open("chunk_overview.txt", "w", encoding="utf-8") as overview:
    for i, p in enumerate(points, start=1):
        pl = p.payload
        first_line = " ".join(pl["text"].split())[:90]
        overview.write(
            f"{i:>3} | {pl['source']} p{pl['page']} | region={pl['region']} | "
            f"{pl.get('content_type', 'text')} | {len(pl['text'])} chars | {first_line}\n"
        )

tables = [p for p in points if p.payload.get("content_type") == "table"]
with open("chunk_tables.txt", "w", encoding="utf-8") as out:
    for p in tables:
        pl = p.payload
        out.write(f"===== {pl['source']} | page {pl['page']} | region={pl['region']} | title={pl['title']}\n")
        out.write(pl["text"] + "\n\n")

print(f"{len(points)} chunks in total, {len(tables)} table chunks")
print("Wrote chunk_overview.txt and chunk_tables.txt")
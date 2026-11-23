"""Write the full text of the text chunks needed to check the held-out questions.
Read-only: it only reads Qdrant and writes chunk_text.txt (document text, no keys)."""
from retrieval import load_corpus

GUIDELINE = "silvo_ecological_zones_2023.pdf"
KALERI = "thatta_halophytes_kaleri_2017.pdf"
RAJPAR = "thar_salinity_drought_rajpar_2018.pdf"

points = load_corpus(["thatta", "sindh", "pakistan"])
points.sort(key=lambda p: (p.payload["source"], p.payload["page"]))


def wanted(p):
    source, page = p.payload["source"], p.payload["page"]
    return ((source == GUIDELINE and page in (15, 16, 17, 77))
            or source == KALERI
            or (source == RAJPAR and page in (1, 2, 3, 4)))


count = 0
with open("chunk_text.txt", "w", encoding="utf-8") as out:
    for p in points:
        payload = p.payload
        if wanted(p) and payload.get("content_type", "text") == "text":
            out.write(f"===== {payload['source']} | page {payload['page']}\n{payload['text']}\n\n")
            count += 1
print(f"Wrote chunk_text.txt with {count} text chunks")
import os
import re
import time
import uuid
from collections import Counter

import pymupdf
import requests
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, PointStruct, VectorParams

load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION = "tree_evidence"
VECTOR_SIZE = 1024
DATA_DIR = "data"
MAX_CHARS = 1000          # limit for normal text chunks
TABLE_MAX_CHARS = 4500    # limit for table chunks, so a small table stays in ONE chunk
MIN_CHARS = 100
BATCH_SIZE = 16
CAPTION_MAX_CHARS = 400
CAPTION_BLOCKS = 3        # how many text blocks above a table are used as its caption
FOOTNOTE_MAX_GAP = 80     # a footnote must start within this many points below its table
FOOTNOTE_MAX_CHARS = 500
FOOTNOTE_START = re.compile(r"^\s*[*\u2020\u2021]\s*\S")

# If True, the collection is deleted and rebuilt on every run.
# Keep it True once after changing the chunking logic, then set it back to False.
RESET_COLLECTION = True

# Metadata for each file. When you add a new PDF, add an entry here too.
# "page_sections" lets one file contain parts that belong to different regions:
# (first_pdf_page, last_pdf_page, region, section label).
# Guideline: PDF page = printed page + 10 (printed 35 -> PDF 45). Ranges come from its contents page.
# Pages not covered by a section (general chapter 1 and the annexes) keep the file's default region.
DOCS = {
    "silvo_ecological_zones_2023.pdf": {
        "title": "Guidelines and Lists of Suitable Tree Species for Plantation in Different Silvo-Ecological Zones of Pakistan (2023)",
        "region": "pakistan",
        "doc_type": "guideline",
        "skip_refs": True,
        "page_sections": [
            (19, 26, "balochistan", "Balochistan province zones section"),
            (27, 34, "khyber_pakhtunkhwa", "Khyber Pakhtunkhwa province zones section"),
            (35, 44, "punjab", "Punjab province zones section"),
            (45, 50, "sindh", "Sindh province zones section"),
            (51, 56, "ajk", "Azad Jammu and Kashmir zones section"),
            (57, 62, "gilgit_baltistan", "Gilgit-Baltistan zones section"),
        ],
    },
    "thatta_halophytes_kaleri_2017.pdf": {
        "title": "Distribution of Halophyte Plants in Relation to Properties of Salt-Affected Soils of District Thatta (Kaleri et al., 2017)",
        "region": "thatta",
        "doc_type": "research",
        "skip_refs": True,
    },
    "thar_salinity_drought_rajpar_2018.pdf": {
        "title": "Salinity and Drought Stress Tolerance of Some Plant Species of Thar Desert, Pakistan (Rajpar et al., 2018)",
        "region": "sindh",
        "doc_type": "research",
        "skip_refs": True,
    },
    "punjab_forest_species.pdf": {
        "title": "Forest and Environment Species (Punjab Forest Department)",
        "region": "punjab",
        "doc_type": "species_list",
        "skip_refs": False,
    },
}
DEFAULT_META = {"title": "Unknown document", "region": "unknown", "doc_type": "other", "skip_refs": False}

REF_PATTERN = re.compile(r"^\s*(references|literature cited)\s*$", re.IGNORECASE | re.MULTILINE)

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def ensure_collection():
    if RESET_COLLECTION and client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
        print(f"Collection '{COLLECTION}' deleted (reset).")

    if not client.collection_exists(COLLECTION):
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        # Indexes so we can filter by region and document type during search.
        client.create_payload_index(
            collection_name=COLLECTION, field_name="region", field_schema=PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name=COLLECTION, field_name="doc_type", field_schema=PayloadSchemaType.KEYWORD
        )
        print(f"Collection '{COLLECTION}' created.")
    else:
        print(f"Collection '{COLLECTION}' already exists.")


def region_and_title_for_page(meta, page_number):
    """Return (region, title) for a page. Pages inside a configured section get that section's
    region and a title that names the section, so searches can tell the sections apart."""
    for first, last, region, label in meta.get("page_sections", []):
        if first <= page_number <= last:
            return region, f"{meta['title']} - {label}"
    return meta["region"], meta["title"]


def clean_cell(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def table_rows_by_words(page, table):
    """Rebuild table rows from the words found inside each row, assigning every word to a column
    by its horizontal position. This restores the spaces between words and splits merged header
    cells correctly. Raises an exception if the table structure is not usable."""
    rows = table.rows
    if not rows:
        raise ValueError("table has no rows")

    # Use the row with the most real cells as the reference for column boundaries.
    reference = max(rows, key=lambda r: sum(1 for c in r.cells if c is not None))
    bounds = sorted((c[0], c[2]) for c in reference.cells if c is not None)
    if len(bounds) < 2:
        raise ValueError("not enough columns")

    def column_index(x_center):
        for i, (x0, x1) in enumerate(bounds):
            if x0 <= x_center <= x1:
                return i
        return min(range(len(bounds)), key=lambda i: min(abs(x_center - bounds[i][0]), abs(x_center - bounds[i][1])))

    result = []
    for row in rows:
        rect = pymupdf.Rect(row.bbox)
        # Shrink the clip slightly so words from neighbouring rows are not picked up.
        clip = pymupdf.Rect(rect.x0, rect.y0 + 1, rect.x1, rect.y1 - 1)
        if clip.is_empty:
            clip = rect
        words = page.get_text("words", clip=clip)
        cells = [[] for _ in bounds]
        for word in words:
            cells[column_index((word[0] + word[2]) / 2)].append(word)
        result.append(
            [" ".join(w[4] for w in sorted(cell, key=lambda w: (w[5], w[6], w[7]))) for cell in cells]
        )
    return result


def table_to_lines(rows):
    """Turn table rows into self-contained lines such as
    'Zone: 2. Subtropical Irrigated Plains (Perennial); Category: Main Species; Scientific Name: ...; Local Name: ...'.
    - Columns that are empty in every row (merged-cell artifacts) are dropped.
    - A title row above the real header row (zone tables) becomes the 'Zone'.
    - Rows like 'Main Species' / 'Subsidiary Species' become the 'Category' of the rows below them.
    - Empty cells are written as (blank) so values can never shift into the wrong column."""
    rows = [[clean_cell(cell) for cell in row] for row in rows]
    rows = [row for row in rows if any(row)]
    if len(rows) < 2:
        return []

    # Pad rows to the same width, then drop columns that are empty in every row.
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    keep = [i for i in range(width) if any(row[i] for row in rows)]
    rows = [[row[i] for i in keep] for row in rows]
    if len(keep) < 2:
        return []

    # Zone tables start with a title row before the real header row.
    header_index = 0
    for i, row in enumerate(rows[:3]):
        if any("scientific name" in cell.lower() for cell in row):
            header_index = i
            break
    title = " ".join(cell for row in rows[:header_index] for cell in row if cell)
    headers = [h if h else f"column {i + 1}" for i, h in enumerate(rows[header_index])]

    lines = []
    category = ""
    for row in rows[header_index + 1:]:
        non_empty = [cell for cell in row if cell]
        # Section rows such as "Main Species" label the rows below them.
        if len(non_empty) == 1 and row[0] and "species" in row[0].lower() and len(row[0]) < 40:
            category = row[0]
            continue
        parts = []
        if title:
            parts.append(f"Zone: {title}")
        if category:
            parts.append(f"Category: {category}")
        parts.extend(f"{header}: {value if value else '(blank)'}" for header, value in zip(headers, row))
        lines.append("; ".join(parts))
    return lines


def extract_pages(pdf_path, skip_refs):
    """Return a list of pages. Each page is a dict with:
    page (number), text (text outside tables), tables (list of {caption, footnote, lines})."""
    doc = pymupdf.open(pdf_path)
    pages = []
    for page_number, page in enumerate(doc, start=1):
        # Find tables on this page.
        tables = []
        try:
            for table in page.find_tables().tables:
                try:
                    rows = table_rows_by_words(page, table)
                except Exception:
                    rows = table.extract()  # fallback to the default extraction
                if len(rows) >= 2 and table.col_count >= 2:
                    tables.append({"rect": pymupdf.Rect(table.bbox), "rows": rows, "footnote": ""})
        except Exception as error:
            print(f"  (table detection failed on page {page_number}: {error})")

        text_blocks = [b for b in page.get_text("blocks", sort=True) if b[6] == 0 and b[4].strip()]

        def in_table(block):
            block_rect = pymupdf.Rect(block[:4])
            return any(block_rect.intersects(t["rect"]) for t in tables)

        outside_blocks = [b for b in text_blocks if not in_table(b)]

        # Footnotes: a text block that starts with * directly below a table belongs to that table.
        footnote_blocks = []
        for block in outside_blocks:
            if not FOOTNOTE_START.match(block[4]):
                continue
            owners = [t for t in tables if t["rect"].y1 <= block[1] + 2 and block[1] - t["rect"].y1 <= FOOTNOTE_MAX_GAP]
            if not owners:
                continue
            owner = max(owners, key=lambda t: t["rect"].y1)
            note = re.sub(r"\s+", " ", block[4]).strip()
            owner["footnote"] = f"{owner['footnote']} {note}".strip()[:FOOTNOTE_MAX_CHARS]
            footnote_blocks.append(block)
        outside_blocks = [b for b in outside_blocks if not any(b is f for f in footnote_blocks)]

        # Caption for each table: the nearest text blocks directly above it.
        for table in tables:
            above = [b for b in outside_blocks if b[3] <= table["rect"].y0 + 2]
            above.sort(key=lambda b: b[3])
            caption = " ".join(re.sub(r"\s+", " ", b[4]).strip() for b in above[-CAPTION_BLOCKS:])
            table["caption"] = caption[-CAPTION_MAX_CHARS:]
            table["lines"] = table_to_lines(table["rows"])

        text = "\n\n".join(b[4].strip() for b in outside_blocks)

        reached_references = False
        if skip_refs:
            match = REF_PATTERN.search(text)
            if match:
                text = text[: match.start()]
                reached_references = True
                tables = []  # ignore tables on and after the references page

        text = text.strip()
        usable_tables = [t for t in tables if t["lines"]]
        if text or usable_tables:
            pages.append({"page": page_number, "text": text, "tables": usable_tables})
        if reached_references:
            break
    doc.close()
    return pages


def clean_text(text):
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def hard_split(text):
    """Last resort: split a very long piece of text by words."""
    parts = []
    current = ""
    for word in text.split(" "):
        if current and len(current) + len(word) + 1 > MAX_CHARS:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        parts.append(current)
    return parts


def split_long_block(block):
    """Split a block longer than MAX_CHARS by lines first, then by sentences, then by words."""
    pieces = []
    for line in block.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) <= MAX_CHARS:
            pieces.append(line)
            continue
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            if len(sentence) <= MAX_CHARS:
                pieces.append(sentence)
            else:
                pieces.extend(hard_split(sentence))
    return pieces


def pack_pieces(pieces, max_chars=MAX_CHARS):
    """Merge small pieces into chunks of at most max_chars."""
    chunks = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) + 1 > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def chunk_text(text):
    blocks = [b.strip() for b in clean_text(text).split("\n\n") if b.strip()]
    pieces = []
    for block in blocks:
        if len(block) <= MAX_CHARS:
            pieces.append(block)
        else:
            pieces.extend(split_long_block(block))
    chunks = pack_pieces(pieces)
    return [c for c in chunks if len(c) >= MIN_CHARS]


def chunk_table(table):
    """Group table row-lines into chunks. Every chunk starts with the table's caption and ends with the
    table's footnote (if it has one). Small tables stay in one chunk so that list-style questions see
    every row."""
    prefix = f"Table context: {table['caption']}\n" if table["caption"] else "Table:\n"
    footnote = f"\nTable footnote: {table['footnote']}" if table.get("footnote") else ""
    return [prefix + chunk + footnote for chunk in pack_pieces(table["lines"], TABLE_MAX_CHARS)]


def embed_passages(texts):
    for attempt in range(3):
        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "jina-embeddings-v3",
                "task": "retrieval.passage",
                "input": texts,
            },
            timeout=60,
        )
        if response.status_code == 429:
            # Rate limited: wait and retry.
            time.sleep(5 * (attempt + 1))
            continue
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]
    raise RuntimeError("Jina rate limit reached. Please try again in a few minutes.")


def ingest_pdf(pdf_path):
    filename = os.path.basename(pdf_path)
    meta = DOCS.get(filename, DEFAULT_META)
    print(f"Reading {filename} (default region={meta['region']}, type={meta['doc_type']}) ...")

    records = []
    table_count = 0
    footnote_count = 0
    for page in extract_pages(pdf_path, meta["skip_refs"]):
        page_number = page["page"]
        region, title = region_and_title_for_page(meta, page_number)
        chunk_index = 0

        for chunk in chunk_text(page["text"]):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}-{page_number}-text-{chunk_index}"))
            records.append(
                {"id": point_id, "text": chunk, "page": page_number, "chunk_index": chunk_index,
                 "content_type": "text", "region": region, "title": title}
            )
            chunk_index += 1

        for table in page["tables"]:
            table_count += 1
            if table.get("footnote"):
                footnote_count += 1
            for chunk in chunk_table(table):
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{filename}-{page_number}-table-{chunk_index}"))
                records.append(
                    {"id": point_id, "text": chunk, "page": page_number, "chunk_index": chunk_index,
                     "content_type": "table", "region": region, "title": title}
                )
                chunk_index += 1

    if not records:
        print("  0 chunks created. The PDF may be a scanned image.")
        return

    table_chunks = sum(1 for r in records if r["content_type"] == "table")
    average = sum(len(r["text"]) for r in records) // len(records)
    print(f"  {len(records)} chunks created (average {average} chars): "
          f"{len(records) - table_chunks} text, {table_chunks} table chunks from {table_count} tables "
          f"({footnote_count} with a footnote).")
    print(f"  Chunks per region: {dict(Counter(r['region'] for r in records))}")

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        # The (section) title is prepended only for embedding.
        # The payload stores the original chunk text without the title.
        embed_inputs = [f"{r['title']}\n\n{r['text']}" for r in batch]
        vectors = embed_passages(embed_inputs)
        points = [
            PointStruct(
                id=r["id"],
                vector=vector,
                payload={
                    "text": r["text"],
                    "source": filename,
                    "title": r["title"],
                    "region": r["region"],
                    "doc_type": meta["doc_type"],
                    "content_type": r["content_type"],
                    "page": r["page"],
                    "chunk_index": r["chunk_index"],
                },
            )
            for r, vector in zip(batch, vectors)
        ]
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"  Stored {min(start + BATCH_SIZE, len(records))}/{len(records)}")


if __name__ == "__main__":
    ensure_collection()
    pdf_files = sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf"))
    if not pdf_files:
        print("No PDFs found in the data folder.")
    for pdf_file in pdf_files:
        ingest_pdf(os.path.join(DATA_DIR, pdf_file))
    print("Done.")
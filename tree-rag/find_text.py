import os
import sys

import pymupdf

DATA_DIR = "data"
SNIPPET_CHARS = 220
MAX_HITS_PER_FILE = 12


def main():
    keywords = [k.lower() for k in sys.argv[1:]]
    if not keywords:
        print("Usage: py find_text.py <keyword> [<keyword> ...]")
        print("Example: py find_text.py Thatta Badin")
        return

    pdf_files = sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf"))
    for filename in pdf_files:
        doc = pymupdf.open(os.path.join(DATA_DIR, filename))
        hits = []
        for page_number, page in enumerate(doc, start=1):
            text = " ".join(page.get_text().split())
            lowered = text.lower()
            for keyword in keywords:
                start = 0
                while True:
                    position = lowered.find(keyword, start)
                    if position == -1:
                        break
                    snippet_start = max(0, position - SNIPPET_CHARS // 2)
                    hits.append((page_number, keyword, text[snippet_start:snippet_start + SNIPPET_CHARS]))
                    start = position + len(keyword)
        doc.close()

        print(f"\n{filename}: {len(hits)} hit(s)")
        for page_number, keyword, snippet in hits[:MAX_HITS_PER_FILE]:
            print(f"  page {page_number} [{keyword}]: ...{snippet}...")
        if len(hits) > MAX_HITS_PER_FILE:
            print(f"  ... and {len(hits) - MAX_HITS_PER_FILE} more")


if __name__ == "__main__":
    main()
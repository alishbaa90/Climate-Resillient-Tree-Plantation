"""Build data/species.json from the knowledge-base table chunks.
No LLM and no web calls: it only reads the chunks from Qdrant (read-only), so the output is the same every run.
Every fact keeps the page it came from. Names that are not a plain 'Genus species' are skipped and listed."""
import json
import os
import re
from datetime import datetime, timezone

from answer import clean_footnote
from retrieval import load_corpus

REGIONS = ["thatta", "sindh", "pakistan"]  # same as ALLOWED_REGIONS in answer.py (the Punjab document is not used)
GUIDELINE = "silvo_ecological_zones_2023.pdf"
SURVEY = "thatta_halophytes_kaleri_2017.pdf"
OUT_FILE = os.path.join("data", "species.json")
BINOMIAL = re.compile(r"^[A-Z][a-z]+ [a-z]+$")

ZONE_KEYS = ["Zone", "Category", "Scientific Name", "Local Name", "Indigenous/Exotic"]
ANNEX_KEYS = ["Scientific Name", "Local Name", "Native/Exotic", "Salinity Tolerance", "Fruit/Nectar"]
LANDSCAPE_KEYS = ["Species", "Local Names", "Landscape Characterization Suitable Habitat",
                  "Landscape Characterization", "Suitable Habitat"]
SYNONYM_KEYS = ["Table 2. List of valid", "species name with their", "synonyms"]
SURVEY_KEYS = ["Halophytes", "EC (dS m -1)", "pH", "OM (%)"]

S_ZONES = "Sindh zone tables"
S_SALINE = "Annex 3 (saline and waterlogged soils)"
S_BIRDS = "Annex 2 (bird food and pollinators)"
S_LANDSCAPE = "Urban landscaping tables"
S_SYNONYMS = "Table 2 (valid names)"


def parse_row(line, keys):
    """Split 'Key: value; Key: value' into a dict. It splits only on the known keys, because values
    (for example local names or descriptions) can contain ';' themselves. '(blank)' becomes None."""
    alternation = "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))
    parts = re.split(rf"(?:^|;\s+)({alternation}): ", line)
    if len(parts) < 3 or parts[0].strip():
        return None
    row = {}
    for key, value in zip(parts[1::2], parts[2::2]):
        value = value.strip().rstrip(";,").strip()
        row[key] = None if value in ("", "(blank)") else value
    return row


def plain_name(raw, where, page, skipped):
    """Return (name, had_footnote_star). Names that are not a plain 'Genus species' are skipped and reported."""
    if not raw:
        return None, False
    starred = raw.strip().endswith("*")
    name = re.sub(r"\s+", " ", raw.replace("*", "")).strip()
    if BINOMIAL.match(name):
        return name, starred
    skipped.append(f"{name} ({where}, p.{page})")
    return None, False


def new_entry(name):
    return {"scientific_name": name, "_local_names": {}, "_origin": {}, "sindh_zones": [], "salinity": [],
            "bird_pollinator": [], "landscape": [], "table2_notes": [], "thatta_survey": []}


def note(entry, field, key, page):
    entry["_" + field].setdefault(key, set()).add(page)


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def main():
    points = load_corpus(REGIONS)
    points.sort(key=lambda p: (p.payload["source"], p.payload["page"]))
    species, skipped, synonym_rows, survey_rows = {}, [], [], []

    def get(name):
        return species.setdefault(name, new_entry(name))

    for point in points:
        payload = point.payload
        source, page = payload["source"], payload["page"]
        lines = payload["text"].split("\n")

        if source == GUIDELINE and 46 <= page <= 48:  # Sindh zone tables
            pending, footnote = [], None
            for line in lines:
                if line.startswith("Table footnote:"):
                    footnote = clean_footnote(line[len("Table footnote:"):]).lstrip("*").strip()
                elif line.startswith("Zone: "):
                    row = parse_row(line, ZONE_KEYS)
                    name, starred = plain_name(row and row.get("Scientific Name"), S_ZONES, page, skipped)
                    if not name:
                        continue
                    entry = get(name)
                    same = next((z for z in entry["sindh_zones"]
                                 if z["zone"] == row["Zone"] and z["category"] == row["Category"]), None)
                    if same:
                        same["listed_twice_in_source"] = True
                        continue
                    zone = {"zone": row["Zone"], "category": row["Category"], "status": row["Indigenous/Exotic"],
                            "page": page, "footnote": None, "listed_twice_in_source": False}
                    entry["sindh_zones"].append(zone)
                    if starred:
                        pending.append(zone)
                    if row.get("Local Name"):
                        note(entry, "local_names", (row["Local Name"], S_ZONES), page)
                    if row.get("Indigenous/Exotic"):
                        note(entry, "origin", (row["Indigenous/Exotic"], S_ZONES), page)
            for zone in pending:
                zone["footnote"] = footnote

        elif source == GUIDELINE and page in (76, 77):  # Annex 2 and Annex 3
            for line in lines:
                if not line.startswith("Scientific Name: "):
                    continue
                row = parse_row(line, ANNEX_KEYS)
                section = S_SALINE if page == 77 else S_BIRDS
                name, _ = plain_name(row and row.get("Scientific Name"), section, page, skipped)
                if not name:
                    continue
                entry = get(name)
                if row.get("Local Name"):
                    note(entry, "local_names", (row["Local Name"], section), page)
                if row.get("Native/Exotic"):
                    note(entry, "origin", (row["Native/Exotic"], section), page)
                if page == 77 and row.get("Salinity Tolerance"):
                    entry["salinity"].append({"level": row["Salinity Tolerance"], "page": page})
                if page == 76 and row.get("Fruit/Nectar"):
                    entry["bird_pollinator"].append({"type": row["Fruit/Nectar"], "page": page})

        elif source == GUIDELINE and 64 <= page <= 75:  # urban landscaping tables
            for line in lines:
                if not line.startswith("Species: "):
                    continue
                row = parse_row(line, LANDSCAPE_KEYS)
                name, _ = plain_name(row and row.get("Species"), S_LANDSCAPE, page, skipped)
                if not name:
                    continue
                entry = get(name)
                if row.get("Local Names"):
                    note(entry, "local_names", (row["Local Names"], S_LANDSCAPE), page)
                description = row.get("Landscape Characterization") or row.get("Landscape Characterization Suitable Habitat")
                entry["landscape"].append({"description": description, "habitat": row.get("Suitable Habitat"), "page": page})

        elif source == GUIDELINE and page == 14:  # Table 2: valid names and synonyms
            for line in lines:
                if line.startswith("Table 2. List of valid: "):
                    row = parse_row(line, SYNONYM_KEYS)
                    if row and row.get("Table 2. List of valid") and row["Table 2. List of valid"] != "Valid Name":
                        synonym_rows.append((row["Table 2. List of valid"], row.get("species name with their"), page))

        elif source == SURVEY and page == 4:  # Table 7: soil where each halophyte was collected
            for line in lines:
                if line.startswith("Halophytes: ") and "EC (dS m -1)" in line:
                    row = parse_row(line, SURVEY_KEYS)
                    if row and row.get("Halophytes") and row["Halophytes"] not in ("Average", "Maximum", "Minimum"):
                        survey_rows.append((row, page))

    # Table 2 is only attached to species that already exist from the other tables.
    for valid, synonym, page in synonym_rows:
        if not synonym:
            continue
        text = f"Table 2 lists {synonym} as a synonym of {valid}."
        if valid in species:
            species[valid]["table2_notes"].append({"text": text, "page": page})
        elif synonym in species:
            species[synonym]["table2_notes"].append({"text": text, "page": page})

    survey_only = []
    for row, page in survey_rows:
        record = {"plant": row["Halophytes"], "soil_ec_ds_m": number(row.get("EC (dS m -1)")),
                  "soil_ph": number(row.get("pH")), "soil_om_pct": number(row.get("OM (%)")), "page": page}
        base = row["Halophytes"].split("/")[0].strip()  # 'Tamarix aphylla/gallica' -> 'Tamarix aphylla'
        if base in species:
            species[base]["thatta_survey"].append(record)
        else:
            survey_only.append(record)

    cards = []
    for name in sorted(species):
        entry = species[name]
        entry["local_names"] = [{"text": k[0], "section": k[1], "pages": sorted(v)}
                                for k, v in sorted(entry.pop("_local_names").items())]
        entry["origin"] = [{"value": k[0], "section": k[1], "pages": sorted(v)}
                           for k, v in sorted(entry.pop("_origin").items())]
        entry["id"] = re.sub(r"[^a-z]+", "-", name.lower()).strip("-")
        cards.append(entry)

    data = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "species": cards, "survey_plants": survey_only, "skipped_rows": sorted(set(skipped))}
    os.makedirs("data", exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=1)

    print(f"Wrote {OUT_FILE}")
    print(f"Species cards: {len(cards)}")
    print(f"  with a Sindh zone listing: {sum(1 for c in cards if c['sindh_zones'])}")
    print(f"  with a salinity level (Annex 3): {sum(1 for c in cards if c['salinity'])}")
    print(f"  with Thatta field-survey soil data: {sum(1 for c in cards if c['thatta_survey'])}")
    print(f"Survey plants without a species card: {len(survey_only)}")
    print(f"Skipped rows (not a plain 'Genus species' name): {len(set(skipped))}")
    for item in sorted(set(skipped)):
        print("   ", item)


if __name__ == "__main__":
    main()
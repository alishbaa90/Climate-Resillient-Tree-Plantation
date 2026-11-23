import hashlib
import os
import re
import time

import requests
from dotenv import load_dotenv

from retrieval import HybridRetriever, expand_query

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# Groq retires models regularly. Override with GROQ_MODEL in .env if this one stops working.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Lower reasoning effort uses fewer tokens (the free tier has tokens-per-minute and tokens-per-day limits).
# Set GROQ_REASONING_EFFORT to low, medium or high in .env, or to an empty value to disable it.
REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")
# Set SHOW_RETRIEVAL=0 in .env to hide the retrieval trace.
SHOW_RETRIEVAL = os.getenv("SHOW_RETRIEVAL", "1") != "0"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS_URL = "https://api.groq.com/openai/v1/models"
MAX_RETRIES = 4

# Bump this whenever the code (not the prompt) changes how answers are produced,
# so that old evaluation results are recognised as outdated.
CODE_VERSION = "5"

ALLOWED_REGIONS = ["thatta", "sindh", "pakistan"]

# Evidence gate. On the 45-question development set the unanswerable questions scored <= 0.272
# (except one near-miss that the LLM refused) and the lowest answerable question scored 0.367.
# The threshold sits in that gap. Re-check it on a held-out question set.
EVIDENCE_THRESHOLD = 0.32
# Chunks below this score are not sent to the LLM. Must not be higher than EVIDENCE_THRESHOLD.
MIN_CHUNK_SCORE = 0.30
MAX_CONTEXT_CHUNKS = 5
# Limits so that one document or one page cannot crowd out the others.
MAX_PER_SOURCE = 4
MAX_PER_PAGE = 2
# Diversity: the best chunk of every (document, region) group is included first, as long as its
# score is at least this fraction of the top score.
DIVERSITY_RATIO = 0.75

EVIDENCE_UNAVAILABLE = (
    "Evidence unavailable: the current knowledge base does not contain enough "
    "information to answer this question reliably."
)

# Tells the LLM how far each passage's evidence reaches geographically.
SCOPE_LABELS = {
    "thatta": "measured in District Thatta",
    "sindh": "research from the Sindh region (not necessarily Thatta)",
    "pakistan": "Pakistan-wide guideline (not specific to Thatta or Sindh)",
    "punjab": "Punjab document",
}

# What kind of document a source is. It is shown in every passage header AND added to every answer
# that cites the source, so the answer never depends on the model remembering it.
# Add an entry whenever a new research document is added to the knowledge base.
SOURCE_NOTES = {
    "thar_salinity_drought_rajpar_2018.pdf":
        "pot experiments on seedlings of four Thar species under controlled conditions "
        "(a study, not a planting recommendation)",
    "thatta_halophytes_kaleri_2017.pdf":
        "field survey of halophyte plants and soil samples in District Thatta "
        "(measurements, not a planting recommendation)",
}

SYSTEM_PROMPT = """You are a tree plantation assistant for Thatta, Sindh, Pakistan.

Rules:
1. Answer ONLY from the numbered evidence passages. Do not use outside knowledge, and never invent
   numbers, species or sources.
2. Cite every factual claim with the passage number in square brackets, like [1] or [2][3]. Use no
   other citation format and put nothing else inside square brackets.
3. Choose exactly one answer type, in this order:
   a) Direct answer (no prefix): the passages fully answer what was asked.
   b) "Partial evidence:" followed by what the passages say (with citations) and then what is
      missing. Use it when at least one passage mentions the main subject of the question (the
      species, soil property, method or place that was asked about) but does not answer the whole
      question. This includes the case where a species is listed for a larger area (a province, a
      zone or the whole country) but the question is about a smaller place (a district) that no
      passage mentions.
      Example: the question is "Is Shisham suitable for Thatta?" and a passage lists Dalbergia
      sissoo (Shisham) as a main species of a zone in Sindh but says nothing about Thatta. The
      correct answer starts with "Partial evidence:", says where Shisham is listed, and says that
      the passages do not mention Thatta.
   c) "Evidence unavailable: " plus one sentence saying what is missing. Use it ONLY when no passage
      says anything about the main subject of the question, or when the question asks for one
      specific fact that no passage gives.
      Example: the question is "Which zone does Thatta belong to?" and no passage says which zone
      Thatta is in. The correct answer starts with "Evidence unavailable:".
   Before you choose type c), check every passage: if any of them mentions the species or topic
   that was asked about, use type b) instead.
4. Scope: each passage header states how far its evidence reaches. If you rely on a Pakistan-wide or
   Sindh-wide passage, say so. Do not claim something is suitable for Thatta unless a passage says
   so for Thatta. Never assume that Thatta belongs to a zone unless a passage says so.
5. Report only what the passages state. Add no conclusions or recommendations of your own: do not
   write "therefore", "this suggests", "this indicates" or "not well suited". Never claim that a
   list is the only one, or complete ("the only", "no other", "all of"): you see only a few search
   results.
6. Tables: rows look like "Header: value; Header: value". "(blank)" means the cell is empty in the
   source. Never fill a blank cell and never combine values from different rows.
7. For list questions, include EVERY matching row from the passages, grouped by the relevant value.
   For a table chunk with N rows, go through all N rows in order and check at the end that none is
   missing. End a list with one sentence saying it is based on the retrieved passages and may not
   be exhaustive.
8. Zone table rows ("Zone: ...; Category: ...; Scientific Name: ...; Local Name: ...;
   Indigenous/Exotic: ...") say nothing about salinity. When you report species from them, name the
   zone, say whether each species is a main or a subsidiary species, and give its Indigenous/Exotic
   value. If that value is "Extinct", say that the table marks the species as extinct. A name that
   ends with * refers to a table footnote; the footnote is added automatically below your answer,
   so do not repeat it. For salinity questions do not list zone-table species.
9. The table "Tree Species Suitable for Saline and Waterlogged Soils" lists species able to grow in
   saline and waterlogged soils, with a salinity tolerance of Low, Medium or High. Report a species
   from it as "listed as able to grow in saline and waterlogged soils, with Low (or Medium or High)
   salinity tolerance". Never turn a Low level into "not suitable". The levels are not defined in
   numbers, so never convert them into dS/m and never compare them with measured soil EC.
10. Passage headers say what kind of document a passage comes from. When you report findings from a
   study or survey, name its kind (for example "in a pot experiment on seedlings" or "in a field
   survey in Thatta"). An experiment shows which species were TESTED and how they ranked. Give the
   ranking, and do not call the species "suitable" or "recommended" because of it.
11. If a guideline passage and a study or survey passage both mention the same species or topic,
   give each source its own sentence. If they differ, say that the sources differ. Use every
   relevant passage.
12. Copy numbers, units and names exactly, including scientific names. Do not shorten or invent
   abbreviations of species names; use only the forms that the passage itself uses (for example
   "S. oleoides"). If a number or a sentence is cut off at the start or the end of a passage, say
   that the passage is incomplete instead of guessing the missing part.
13. Do not mix different methods or steps. If passages describe separate methods (for example one
   for saline soils and one for waterlogged soils), report each separately and say which applies to
   what.
14. For suitability questions, do not begin with a bare "Yes" or "No". Begin with what the evidence
   lists and where (province, zone, document). Never state how many zones, species or categories a
   province or document has; report only the ones you can see.
15. Write complete sentences or list lines that name the species or subject. Never answer with a bare
   fragment such as a single word or number. Keep the answer short and practical.
16. If a passage gives different rankings for different conditions (for example one ranking for
   salinity and another for drought), report each ranking separately and name its condition. Never
   merge them into one ranking. Say that a species tolerates a condition only if the passage says so
   for that species."""

# Characters that the model sometimes writes instead of a normal space or hyphen.
UNICODE_REPLACEMENTS = {
    "\u202f": " ", "\u00a0": " ", "\u2009": " ", "\u200a": " ", "\u2002": " ", "\u2003": " ",
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
}

# Sentences that claim a list is complete are replaced by one standard caveat (see the guard below).
COMPLETENESS_PATTERNS = [
    r"\bthe only\b", r"\bno other\b", r"\bthese are all\b", r"\bare all the\b", r"\ball the rows\b",
]
HEDGE_WORDS = re.compile(r"\b(not|may|might|could|cannot)\b")
EXHAUSTIVE_NOTE = "This list is based on the retrieved passages and may not be exhaustive."

BINOMIAL = re.compile(r"\b([A-Z][a-z]{3,}) ([a-z]{4,})\b")
_NAME_INDEX = {}


class DailyLimitError(RuntimeError):
    """Raised when the Groq tokens-per-day limit is used up. Waiting a few seconds does not help."""


def config_hash():
    """Short fingerprint of everything that changes the answers (prompt, source notes, thresholds,
    reasoning effort, model and code version). Saved with every evaluation result so old results
    can be recognised."""
    parts = [
        SYSTEM_PROMPT,
        repr(sorted(SOURCE_NOTES.items())),
        str(EVIDENCE_THRESHOLD),
        str(MIN_CHUNK_SCORE),
        str(REASONING_EFFORT),
        GROQ_MODEL,
        CODE_VERSION,
    ]
    return hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()[:8]


def list_active_models():
    """Return the model IDs currently available to this API key."""
    response = requests.get(
        GROQ_MODELS_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        timeout=30,
    )
    response.raise_for_status()
    return sorted(m["id"] for m in response.json()["data"])


def scope_label(item):
    if item["doc_type"] == "guideline" and item["region"] == "sindh":
        return "Sindh province section of the national guideline (this passage does not say which district belongs to which zone)"
    return SCOPE_LABELS.get(item["region"], item["region"])


def describe_kind(item):
    """Short description of a chunk for the passage header (tables get their row count)."""
    if item.get("content_type") == "table":
        rows = [
            line for line in item["text"].split("\n")
            if line.strip() and not line.startswith(("Table context:", "Table footnote:", "Table:"))
        ]
        return f"table chunk with {len(rows)} rows"
    return "text"


def select_context(results):
    """Choose up to MAX_CONTEXT_CHUNKS chunks for the LLM.
    Pass 1: the best chunk of each (document, region) group, if it is close enough to the top score.
            This keeps a strong match from one document (for example a salinity list) from being
            pushed out by several similar chunks of another document.
    Pass 2: fill the remaining slots in rank order, with per-source and per-page limits.
    The result is returned in rank order."""
    candidates = [r for r in results if r["rerank_score"] >= MIN_CHUNK_SCORE]
    if not candidates:
        return []

    top_score = candidates[0]["rerank_score"]
    position = {id(item): i for i, item in enumerate(candidates)}
    chosen = []
    per_source = {}
    per_page = {}

    def add(item):
        chosen.append(item)
        per_source[item["source"]] = per_source.get(item["source"], 0) + 1
        page_key = (item["source"], item["page"])
        per_page[page_key] = per_page.get(page_key, 0) + 1

    seen_groups = set()
    for item in candidates:
        group = (item["source"], item["region"])
        if group in seen_groups:
            continue
        seen_groups.add(group)
        if item["rerank_score"] >= DIVERSITY_RATIO * top_score and len(chosen) < MAX_CONTEXT_CHUNKS:
            add(item)

    for item in candidates:
        if len(chosen) >= MAX_CONTEXT_CHUNKS:
            break
        if any(item is picked for picked in chosen):
            continue
        page_key = (item["source"], item["page"])
        if per_source.get(item["source"], 0) >= MAX_PER_SOURCE or per_page.get(page_key, 0) >= MAX_PER_PAGE:
            continue
        add(item)

    return sorted(chosen, key=lambda item: position[id(item)])


def build_context(results):
    blocks = []
    for number, item in enumerate(results, start=1):
        note = SOURCE_NOTES.get(item["source"])
        document_kind = f"; document kind: {note}" if note else ""
        header = (
            f"[{number}] (source: {item['source']}, page {item['page']}; "
            f"scope: {scope_label(item)}; {describe_kind(item)}{document_kind})"
        )
        blocks.append(f"{header}\n{item['text']}")
    return "\n\n".join(blocks)


def retry_wait_seconds(error_text):
    """Read 'try again in 10.4s' (or '1m2.3s') from a Groq rate limit error."""
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", error_text)
    if not match:
        return 15.0
    minutes = int(match.group(1)) if match.group(1) else 0
    return min(minutes * 60 + float(match.group(2)) + 1.0, 60.0)


def retry_hint(error_text):
    match = re.search(r"try again in ([\dhms.]+)", error_text, flags=re.IGNORECASE)
    return f"Try again in about {match.group(1)}." if match else "Try again later."


def call_groq(question, context):
    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Evidence passages:\n\n{context}\n\nQuestion: {question}"},
        ],
    }
    if REASONING_EFFORT:
        payload["reasoning_effort"] = REASONING_EFFORT

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    response = None
    for attempt in range(MAX_RETRIES):
        response = requests.post(GROQ_URL, headers=headers, json=payload, timeout=60)
        if response.ok:
            break
        if response.status_code == 429:
            if "per day" in response.text.lower():
                raise DailyLimitError(
                    f"Groq daily token limit reached for model '{GROQ_MODEL}'. {retry_hint(response.text)}"
                )
            if attempt < MAX_RETRIES - 1:
                wait = retry_wait_seconds(response.text)
                print(f"  (Groq rate limit reached, waiting {wait:.0f}s and retrying ...)")
                time.sleep(wait)
                continue
        if response.status_code == 400 and "reasoning_effort" in payload and "reasoning" in response.text.lower():
            # This model does not accept reasoning_effort: retry without it.
            payload.pop("reasoning_effort")
            continue
        break

    if not response.ok:
        message = f"Groq error {response.status_code} (model='{GROQ_MODEL}'): {response.text}"
        if response.status_code == 404:
            try:
                message += f"\nAvailable models: {', '.join(list_active_models())}"
            except Exception:
                pass
            message += "\nSet GROQ_MODEL in your .env file to one of the available models."
        raise RuntimeError(message)

    return response.json()["choices"][0]["message"]["content"].strip()


def clean_unicode(text):
    """Replace special spaces and hyphens that the model sometimes writes with plain ones."""
    for old, new in UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def normalize_citations(text):
    """Convert citation styles like 【1†L1-L4】, 【2】 or [T1] into plain [1] / [2], and remove
    bracketed phrases such as [pH range reported in the data] that are not citations."""
    text = re.sub(r"【\s*(\d+)[^】]*】", r"[\1]", text)
    text = re.sub(r"\[[A-Za-z]+(\d+)\]", r"[\1]", text)
    return re.sub(r"\[(?!\d+\])[^\[\]]{6,}\]", "", text)


def soften_completeness_claims(text):
    """Guard: the model sees only a few search results, so it cannot know that a list is complete.
    Sentences such as 'These are the only species ...' are removed. If the answer has no caveat
    about exhaustiveness yet, one standard caveat is added in their place.
    Sentences that already hedge ('may not be a complete list') are kept.
    Returns (new_text, changed)."""
    already_hedged = "exhaustive" in text.lower()
    note_added = False
    changed = False
    output_lines = []
    for line in text.split("\n"):
        kept = []
        for sentence in re.split(r"(?<=[.!?])\s+", line):
            lowered = sentence.lower()
            claims_completeness = any(re.search(p, lowered) for p in COMPLETENESS_PATTERNS)
            if claims_completeness and not HEDGE_WORDS.search(lowered):
                changed = True
                if not already_hedged and not note_added:
                    kept.append(EXHAUSTIVE_NOTE)
                    note_added = True
                continue
            kept.append(sentence)
        rebuilt = " ".join(s for s in kept if s.strip())
        if rebuilt.strip():
            output_lines.append(rebuilt)
        elif not line.strip():
            output_lines.append(line)
    return "\n".join(output_lines), changed


def build_name_index(retriever):
    """Index of the scientific names in the knowledge base (built once per retriever)."""
    key = id(retriever)
    if key not in _NAME_INDEX:
        corpus = " ".join(" ".join(p.payload["text"].split()) for p in retriever.points)
        table_names = set(re.findall(r"Scientific Name: ([A-Z][a-z]+ [a-z]+)", corpus))
        by_epithet = {}
        for name in table_names:
            by_epithet.setdefault(name.split()[1], set()).add(name)
        genera = {name.split()[0] for name in table_names}
        _NAME_INDEX[key] = (corpus.lower(), by_epithet, genera)
    return _NAME_INDEX[key]


def check_species_names(text, retriever):
    """Guard against invented or misspelled scientific names.
    - A Latin name that appears in the knowledge base is accepted.
    - A name that is not in the knowledge base but whose species word (epithet) belongs to exactly one
      species in the species tables is corrected (for example 'Salicaria oleoides' -> 'Salvadora oleoides').
    - Any other name whose genus is a known table genus is reported as unverified.
    Returns (text, corrections, unverified_names)."""
    corpus_lower, by_epithet, genera = build_name_index(retriever)
    corrections = {}
    unverified = []
    for genus, epithet in sorted(set(BINOMIAL.findall(text))):
        name = f"{genus} {epithet}"
        if name.lower() in corpus_lower:
            continue
        candidates = by_epithet.get(epithet, set())
        if len(candidates) == 1:
            corrections[name] = next(iter(candidates))
        elif genus in genera:
            unverified.append(name)
    for wrong, right in corrections.items():
        text = re.sub(rf"\b{re.escape(wrong)}\b", right, text)
    return text, corrections, unverified


def group_numbers(numbers):
    return "".join(f"[{n}]" for n in numbers)


def provenance_footer(cited_numbers, context_chunks):
    """Add the kind of every cited research document, taken from SOURCE_NOTES (not from the model).
    Passages with the same note are grouped, for example '[1][3][4] pot experiments ...'."""
    grouped = {}
    for number in cited_numbers:
        if 1 <= number <= len(context_chunks):
            note = SOURCE_NOTES.get(context_chunks[number - 1]["source"])
            if note:
                grouped.setdefault(note, []).append(number)
    if not grouped:
        return ""
    parts = [f"{group_numbers(numbers)} {note}" for note, numbers in grouped.items()]
    return "\n\nSource types: " + "; ".join(parts)


def clean_footnote(note):
    """Tidy a footnote taken from a table chunk: collapse spaces and drop a dangling label such as
    'Publisher:' that was cut off when the PDF text was extracted."""
    note = re.sub(r"\s+", " ", note).strip()
    note = re.sub(r"[\s,;]*\b[A-Za-z]+:\s*$", "", note)
    return note.rstrip(" ,;")


def footnote_footer(cited_numbers, context_chunks):
    """Add the footnotes of the cited table chunks (for example the '*' note under the mangrove table),
    taken from the chunk text and not from the model."""
    grouped = {}
    for number in cited_numbers:
        if not 1 <= number <= len(context_chunks):
            continue
        for line in context_chunks[number - 1]["text"].split("\n"):
            if line.startswith("Table footnote:"):
                note = clean_footnote(line[len("Table footnote:"):])
                if note:
                    grouped.setdefault(note, []).append(number)
    if not grouped:
        return ""
    parts = [f"{group_numbers(numbers)} {note}" for note, numbers in grouped.items()]
    return "\n\nTable footnotes: " + "; ".join(parts)


def build_trace(results, context_chunks):
    """List every retrieved passage and whether it was sent to the LLM (and as which number)."""
    numbers = {id(item): n for n, item in enumerate(context_chunks, start=1)}
    return [
        {
            "rank": rank,
            "score": item["rerank_score"],
            "source": item["source"],
            "page": item["page"],
            "region": item["region"],
            "content_type": item["content_type"],
            "context_number": numbers.get(id(item)),
        }
        for rank, item in enumerate(results, start=1)
    ]


def unavailable_result(top_score, trace):
    return {
        "answer": EVIDENCE_UNAVAILABLE,
        "raw_answer": EVIDENCE_UNAVAILABLE,
        "softened": False,
        "name_corrections": {},
        "unverified_names": [],
        "sources": [],
        "top_score": top_score,
        "used_llm": False,
        "trace": trace,
    }


def answer_question(retriever, question):
    results = retriever.search(question)
    top_score = results[0]["rerank_score"] if results else 0.0

    if not results or top_score < EVIDENCE_THRESHOLD:
        return unavailable_result(top_score, build_trace(results, []))

    context_chunks = select_context(results)
    if not context_chunks:
        # Never call the LLM with an empty context.
        return unavailable_result(top_score, build_trace(results, []))

    raw_answer = call_groq(question, build_context(context_chunks))
    answer = normalize_citations(clean_unicode(raw_answer))
    answer, softened = soften_completeness_claims(answer)
    answer, corrections, unverified = check_species_names(answer, retriever)

    # Show only the sources the answer actually cites.
    cited_numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    sources = [
        {
            "number": n,
            "source": context_chunks[n - 1]["source"],
            "page": context_chunks[n - 1]["page"],
            "score": context_chunks[n - 1]["rerank_score"],
        }
        for n in cited_numbers
        if 1 <= n <= len(context_chunks)
    ]
    if not answer.lower().lstrip(" *#>_-").startswith("evidence unavailable"):
        answer += footnote_footer(cited_numbers, context_chunks)
        answer += provenance_footer(cited_numbers, context_chunks)

    return {
        "answer": answer,
        "raw_answer": raw_answer,
        "softened": softened,
        "name_corrections": corrections,
        "unverified_names": unverified,
        "sources": sources,
        "top_score": top_score,
        "used_llm": True,
        "trace": build_trace(results, context_chunks),
    }


if __name__ == "__main__":
    retriever = HybridRetriever(regions=ALLOWED_REGIONS)
    print(f"Using Groq model: {GROQ_MODEL}")
    print("Ask a question (type 'exit' to quit):")
    while True:
        question = input("\n> ").strip()
        if question.lower() in ("exit", "quit", ""):
            break
        expanded = expand_query(question)
        if expanded != question:
            print(f"[search query expanded to: {expanded}]")
        try:
            result = answer_question(retriever, question)
        except RuntimeError as error:
            print(f"\n{error}")
            continue

        if SHOW_RETRIEVAL:
            print("\nRetrieved passages:")
            for t in result["trace"]:
                used = f"sent to LLM as [{t['context_number']}]" if t["context_number"] else "not used"
                print(f"  {t['rank']:>2}. {t['score']:.3f}  {t['source']} p{t['page']}  "
                      f"region={t['region']}  {t['content_type']}  -> {used}")

        print(f"\n[top rerank score: {result['top_score']:.3f} | LLM used: {result['used_llm']}]")
        print(f"\n{result['answer']}")
        if result["softened"]:
            print("\n(note: a completeness claim in the model's answer was replaced by a caveat)")
        if result["name_corrections"]:
            print(f"(note: species names corrected: {result['name_corrections']})")
        if result["unverified_names"]:
            print(f"(warning: species names not found in the knowledge base: {result['unverified_names']})")
        if result["sources"]:
            print("\nSources:")
            for s in result["sources"]:
                print(f"  [{s['number']}] {s['source']} (page {s['page']}, score {s['score']:.3f})")
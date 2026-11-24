# 🌳 Tree Plantation Evidence System — Thatta, Sindh

An evidence-grounded assistant for tree planting questions in Thatta, Sindh. It answers **only from a small set of source documents**, cites the document and page for every claim, and says so plainly when the documents do not contain the answer. Built end-to-end: document ingestion, hybrid retrieval with reranking, grounded generation with code-level guards, a REST API with login and chat history, a live map with climate and soil data, and a species explorer.

**Live demo:** https://climate-resillient-tree-plantation-1.onrender.com

> Sign-ups are closed to protect free-tier API quotas. A demo login is available on request. The backend runs on a free Render instance, so after idle time the first request can take about a minute while it wakes up.

---

## Overview

Most RAG demos answer everything, even when the documents say nothing. This system is built around one rule: **no evidence, no answer.**

**Pipeline:**

```
Question
   → Hybrid Retrieval (dense Qdrant search + BM25 → Reciprocal Rank Fusion)
   → Jina Reranker (top 20 re-ranked, top 10 returned)
   → Relevance Gate (top score ≥ 0.32, otherwise no LLM call)
   → Grounded Generation (Groq LLM, answers only from retrieved passages, [n] citations)
   → Code-level Guards (citation check, species-name check, footnotes, completeness caveats)
   → Answer + sources + warnings + reliability flag
```

Alongside the assistant, a **Map Explorer** shows estimated climate (NASA POWER) and topsoil (ISRIC SoilGrids) for any point inside the Thatta district boundary, and a **Tree Species** page shows 132 species profiles built directly from the document tables. Live map data is deliberately kept separate from the language model (see Known Limitations).

---

## Features

- **Three answer types** — *Evidence-backed*, *Partial evidence* (subject is in the documents but the full answer is not), and *Evidence unavailable*. The assistant never fills the gap with a guess
- **Citations on every claim** — each answer shows the document and page number as source chips
- **Hybrid retrieval** — dense embeddings and BM25 keyword search fused with Reciprocal Rank Fusion, then reranked
- **Relevance gate** — a minimum evidence score stops the LLM from being called on weak context
- **Code-level guards** — citation formats normalised to `[n]`, unverified species names flagged, "the only / no other" claims replaced by a caveat, table footnotes and source types (pot experiment, field survey) appended by code, not by the model
- **Retry on serious warnings** — if an answer has no citation, mixes answer types, or contains an unverified species, the API retries once (max 2 attempts) and marks the answer unreliable if it still fails
- **Login and chat history** — Supabase email and Google login, chat history stored per user with Row Level Security
- **Map Explorer** — Leaflet map with the Thatta boundary, climate loaded first and soil after, Retry buttons, and a clear message when a point is outside the district
- **Tree Species explorer** — 132 species cards built from document tables by a deterministic script (no LLM), with search, three filters, and "Not listed in current evidence" wherever the documents are silent
- **Resilient live data** — NASA POWER and SoilGrids run in parallel, results are cached, and a failed or slow provider is reported honestly instead of being shown as "no data"
- **Structured evaluation** — a 45-question development set and a separate 20-question held-out set, with failure diagnosis tools
- **Deployed** — FastAPI backend and static frontend on Render, with CORS limited to the frontend origin

---

## Tech Stack

| Layer | Tools |
|---|---|
| Embeddings | Jina `jina-embeddings-v3` (1024 dimensions) |
| Reranker | Jina `jina-reranker-v2-base-multilingual` |
| Vector database | Qdrant Cloud |
| Keyword search | rank_bm25 |
| LLM | Groq `openai/gpt-oss-120b` (temperature 0.1) |
| Backend | FastAPI, Uvicorn, Pydantic, Python 3.11 |
| Auth and chat history | Supabase (Auth, Postgres, Row Level Security) |
| Live data | NASA POWER, ISRIC SoilGrids, geoBoundaries |
| Frontend | Plain HTML/JS, Leaflet, supabase-js |
| Hosting | Render (web service + static site) |
| Core | Python, requests, python-dotenv |

> All models are pre-trained and used as-is. **No model was trained in this project.** The work is the retrieval and grounded-generation pipeline around them.

---

## Knowledge Base

121 chunks from three documents:

| Document | Chunks |
|---|---|
| National silvo-ecological zones guideline (2023): zone tables, urban landscaping, Annexes 2 and 3 | 66 |
| Kaleri et al. (2017), halophytes of District Thatta, *J. Basic and Applied Sciences* 13, 85-90 | 34 |
| Rajpar et al. (2018), salinity and drought in four Thar species, *Pak. J. Bot.* 50(2), 465-469 | 21 |

The source PDFs are not included in this repository (two are journal articles). A Punjab document is ingested but excluded from Thatta answers by region filtering.

---

## Evaluation

Two separate sets are reported. They must not be merged or described as "accuracy".

### Development set (45 questions)

Questions were written while looking at system outputs, and some expectations were changed after seeing failures. **This is a development set, not a test set.**

| Retrieval (36 questions with gold pages, k = 10) | Recall@10 | MRR |
|---|---|---|
| Dense only | 0.944 | 0.679 |
| BM25 only | 1.000 | 0.885 |
| Hybrid (RRF) | 0.972 | 0.840 |
| Hybrid + rerank | 1.000 | 0.912 |

End to end: 44/45 passed the pre-written checks, and 9/9 unanswerable questions returned *Evidence unavailable*.

### Held-out set (20 questions)

Written before looking at any system answer, with expectations taken from the PDF text and two decision rules fixed in advance. The system was **not** tuned on this set.

- **16/20 passed** the pre-written checks (80%; 95% confidence interval roughly 58% to 92%, because n = 20 is small)
- **5/5 unanswerable questions were refused** — all five by the LLM, none by the relevance gate
- Retrieval on the 15 questions with gold pages: hybrid + rerank Recall@10 1.000, MRR 0.933 (dense only: MRR 0.655)

The 4 failures were not caused by missing evidence: in all of them the right pages had been retrieved. One was a checker error, two were the model leaving a row out of a long table list, and one was the relevance gate refusing a short species-name question (top score 0.242, under the 0.32 gate).

All questions, expectations and results are in `tree-rag/eval/` (`heldout_questions.json`, `heldout_expectations.json`, `heldout_results.json`).

---

## Known Limitations

- **The gate is a coarse filter.** An unanswerable question about a topic that *is* in a paper can score high (up to 0.693), while a short species-name query can score low (0.242). No single threshold separates them, so refusing correctly is mostly done by the LLM and the prompt
- **List questions** can lose a row from a long table. Planned fixes: higher reasoning effort for list questions or a structured lookup from `species.json`, plus an entity-aware gate
- **The guideline does not say which Sindh zone Thatta belongs to** and never mentions Thatta, Sujawal or Badin, so the assistant answers that this is not in the evidence
- **Salinity levels** (Low, Medium, High) come from Annex 3 with no numeric definition. Low does not mean "unsuitable"
- **Source inconsistencies are kept, not hidden:** Kaleri's text and Table 7 give slightly different soil EC ranges, and the Rajpar tables are garbled in extraction (no numeric questions should be asked from them)
- **Live climate and soil values are not given to the LLM.** The documents contain no per-species rainfall or temperature ranges for Thatta, and combining them would need new rules, guards and a new evaluation
- **SoilGrids (ISRIC) can be slow or unavailable.** The map shows a Retry button
- **District boundary** is the geoBoundaries ADM2 polygon for Thatta and may include part of Sujawal
- The held-out set is small (20 questions), so its numbers are indicative only

---

## Project Structure

```text
Climate-Resillient-Tree-Plantation/
├── tree-rag/
│   ├── ingest.py               # Document ingestion into Qdrant (frozen)
│   ├── retrieval.py            # Dense + BM25 + RRF + Jina reranker (frozen)
│   ├── answer.py               # Prompt, gate, grounded generation, guards (frozen)
│   ├── api_guards.py           # Answer checks, warnings, reliable flag
│   ├── main.py                 # FastAPI app (/ask, /chat, /conversations, /health)
│   ├── supabase_client.py      # Token verification and chat history (RLS)
│   ├── env_api.py              # /boundary, /species, /environment
│   ├── environment.py          # Coverage check, parallel providers, caching
│   ├── nasa_power.py           # Climate data (NASA POWER)
│   ├── soilgrids.py            # Topsoil data (ISRIC SoilGrids)
│   ├── build_species.py        # Builds data/species.json from knowledge-base tables
│   ├── make_boundary.py        # Builds the Thatta polygon from geoBoundaries
│   ├── evaluate.py             # Development retrieval evaluation
│   ├── evaluate_answers.py     # End-to-end evaluation
│   ├── evaluate_heldout.py     # Held-out retrieval evaluation
│   ├── diagnose_failures.py    # Read-only trace of which passages reached the LLM
│   ├── eval/                   # Questions, expectations and results (dev and held-out)
│   ├── data/                   # species.json, thatta_boundary.geojson
│   ├── test_*.py               # Connection, API and provider test scripts
│   └── requirements.txt
└── frontend/
    ├── config.js               # API URL and wake-up ping
    ├── login.html              # Sign in, create account, Google login
    ├── home.html               # Landing page
    ├── assistant.html          # Chat with sources and history
    ├── map.html                # Map Explorer
    ├── species.html            # Tree Species explorer
    ├── sources.html            # Knowledge sources
    └── about.html              # How it works and limits
```

> `ingest.py`, `retrieval.py` and `answer.py` are frozen: any change alters the configuration hash (`7a8d3019`) and invalidates the evaluation numbers above.

---

## Setup

```bash
# Clone and enter the project
git clone https://github.com/alishbaa90/Climate-Resillient-Tree-Plantation.git
cd Climate-Resillient-Tree-Plantation/tree-rag

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file inside `tree-rag/` (never commit it):

```env
JINA_API_KEY=your_key_here
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
```

Optional settings: `PUBLIC_ASK`, `CORS_ORIGINS`, `GROQ_MODEL`, `GROQ_REASONING_EFFORT`, `ASK_MAX_ATTEMPTS`, `COVERAGE_GEOJSON`, `SPECIES_JSON`.

> **Note on ingestion:** the Qdrant collection is already built for the deployed system. Rebuilding it needs the source PDFs in `tree-rag/data/` plus the Qdrant and Jina keys.

---

## Usage

**Backend (terminal 1):**

```bash
cd tree-rag
py -m uvicorn main:app
```

Interactive docs: http://localhost:8000/docs

**Frontend (terminal 2):**

```bash
cd frontend
py -m http.server 5500
```

Open http://localhost:5500/login.html (do not open the pages with `file://`). On localhost the pages call `http://127.0.0.1:8000`; anywhere else they use the deployed backend set in `frontend/config.js`.

**Rebuild the species data (no LLM, no API quota used):**

```bash
py build_species.py
```

**Run the evaluations:**

```bash
py evaluate_heldout.py        # held-out retrieval (Jina only)
py diagnose_failures.py 4 5 15   # which passages reached the LLM for given questions
```

> The full end-to-end evaluation uses about 140,000 Groq tokens. Run it once per day at most.

---

## Deployment

- **Backend:** Render web service, root directory `tree-rag`, build command `pip install -r requirements.txt`, start command `uvicorn main:app --host 0.0.0.0 --port $PORT`, health check `/health`, `PYTHON_VERSION=3.11.9`, `PUBLIC_ASK=0`, and `CORS_ORIGINS` set to the frontend URL. A healthy start reports 121 chunks and configuration `7a8d3019`
- **Frontend:** Render static site, root directory `frontend`, no build command
- Row Level Security is enabled on `conversations` and `messages`. The Supabase anon key in the frontend is public by design, and no service key is used anywhere

---

## Data and Credits

Climate: NASA POWER. Topsoil: ISRIC SoilGrids. Boundary: geoBoundaries. Photos: Wikipedia (not part of the knowledge base). Map tiles: OpenStreetMap contributors. Source documents belong to their authors and publishers.

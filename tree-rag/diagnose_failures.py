"""Read-only diagnosis of the held-out failures: which passages were retrieved and which were sent to the LLM.
It uses Jina (embedding and reranker) only. It does not call Groq and it changes nothing."""
import json
import sys

import answer as pipeline
from retrieval import HybridRetriever

IDS = [int(a) for a in sys.argv[1:]] or [4, 5, 15]
with open("eval/heldout_questions.json", encoding="utf-8") as file:
    questions = {q["id"]: q for q in json.load(file)}

retriever = HybridRetriever(regions=pipeline.ALLOWED_REGIONS)
print(f"configuration {pipeline.config_hash()} | evidence threshold {pipeline.EVIDENCE_THRESHOLD}")
for qid in IDS:
    question = questions[qid]["question"]
    results = retriever.search(question)
    context = pipeline.select_context(results)
    print("\n" + "=" * 70)
    print(f"#{qid} {question}")
    print(f"expanded query: {pipeline.expand_query(question)}")
    for t in pipeline.build_trace(results, context):
        used = f"sent to LLM as [{t['context_number']}]" if t["context_number"] else "not sent"
        print(f"  {t['rank']:>2}. {t['score']:.3f}  {t['source']} p{t['page']}  {t['content_type']}  -> {used}")
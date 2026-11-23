import json
import os
import sys
import time

import requests

from answer import ALLOWED_REGIONS, EVIDENCE_THRESHOLD
from retrieval import RERANK_CANDIDATES, HybridRetriever, expand_query, rerank

QUESTIONS_FILE = os.path.join("eval", "questions.json")
RESULTS_FILE = os.path.join("eval", "results.json")
K = 10                      # number of results evaluated per question
PAUSE_SECONDS = 1.0         # pause between questions to stay inside the Jina rate limit
CONFIGS = ["dense", "bm25", "hybrid", "hybrid+rerank"]
THRESHOLDS = [round(0.20 + 0.05 * i, 2) for i in range(11)]  # 0.20 ... 0.70


def with_retry(function, *args):
    """Call a function that talks to the Jina API and retry on rate limits (HTTP 429)."""
    for attempt in range(4):
        try:
            return function(*args)
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else None
            if status == 429 and attempt < 3:
                wait = 10 * (attempt + 1)
                print(f"  (rate limit, waiting {wait}s ...)")
                time.sleep(wait)
                continue
            raise


def rank_question(retriever, question):
    """Return the ranked chunk ids of every configuration plus the top reranker score."""
    query = expand_query(question)
    dense_ids = with_retry(retriever.dense_search, query)
    bm25_ids = retriever.bm25_search(query)
    fused_ids = retriever.fuse(dense_ids, bm25_ids)
    candidate_ids = fused_ids[:RERANK_CANDIDATES]

    documents = []
    for point_id in candidate_ids:
        payload = retriever.by_id[point_id].payload
        documents.append(f"{payload['title']}\n\n{payload['text']}")
    reranked = with_retry(rerank, query, documents, K)

    rankings = {
        "dense": dense_ids[:K],
        "bm25": bm25_ids[:K],
        "hybrid": fused_ids[:K],
        "hybrid+rerank": [candidate_ids[item["index"]] for item in reranked],
    }
    top_score = reranked[0]["relevance_score"] if reranked else 0.0
    return rankings, top_score


def matches(payload, gold_item):
    return payload["source"] == gold_item["source"] and payload["page"] in gold_item["pages"]


def score_ranking(ranked_ids, gold_items, by_id):
    payloads = [by_id[point_id].payload for point_id in ranked_ids]
    relevant_flags = [any(matches(p, g) for g in gold_items) for p in payloads]
    first_hit = next((i + 1 for i, flag in enumerate(relevant_flags) if flag), None)
    items_found = sum(1 for g in gold_items if any(matches(p, g) for p in payloads))
    return {
        "recall": items_found / len(gold_items),
        "hit": 1.0 if first_hit else 0.0,
        "mrr": 1.0 / first_hit if first_hit else 0.0,
        "precision": sum(relevant_flags) / len(payloads) if payloads else 0.0,
        "first_hit": first_hit,
    }


def average(values):
    return sum(values) / len(values) if values else 0.0


def main():
    with open(QUESTIONS_FILE, encoding="utf-8") as file:
        questions = json.load(file)
    if len(sys.argv) > 1:
        questions = questions[: int(sys.argv[1])]

    retriever = HybridRetriever(regions=ALLOWED_REGIONS)
    by_id = retriever.by_id

    results = []
    for number, item in enumerate(questions, start=1):
        print(f"[{number}/{len(questions)}] {item['question']}")
        rankings, top_score = rank_question(retriever, item["question"])
        entry = {"id": item["id"], "question": item["question"], "label": item["label"], "top_score": top_score}
        if item["gold"]:
            entry["scores"] = {name: score_ranking(ranked, item["gold"], by_id) for name, ranked in rankings.items()}
            entry["retrieved"] = [
                f"{by_id[i].payload['source']} p{by_id[i].payload['page']}" for i in rankings["hybrid+rerank"]
            ]
        results.append(entry)
        time.sleep(PAUSE_SECONDS)

    with open(RESULTS_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    # ---------- Retrieval metrics (questions that have gold pages) ----------
    scored = [r for r in results if "scores" in r]
    print("\n" + "=" * 78)
    print(f"RETRIEVAL METRICS at k={K} ({len(scored)} questions with gold pages)")
    print("=" * 78)
    print(f"{'configuration':<16}{'Recall@' + str(K):>12}{'HitRate@' + str(K):>12}{'MRR':>10}{'Precision@' + str(K):>14}")
    for name in CONFIGS:
        rows = [r["scores"][name] for r in scored]
        print(f"{name:<16}{average([x['recall'] for x in rows]):>12.3f}{average([x['hit'] for x in rows]):>12.3f}"
              f"{average([x['mrr'] for x in rows]):>10.3f}{average([x['precision'] for x in rows]):>14.3f}")

    print("\nRecall@k = share of gold pages found in the top results. HitRate = share of questions with at least")
    print("one gold page found. MRR = average of 1/rank of the first gold page. Precision = share of returned")
    print("chunks that come from a gold page.")

    # ---------- Misses of the full pipeline ----------
    misses = [r for r in scored if r["scores"]["hybrid+rerank"]["first_hit"] is None]
    print("\n" + "=" * 78)
    print(f"QUESTIONS WHERE hybrid+rerank FOUND NO GOLD PAGE: {len(misses)}")
    print("=" * 78)
    for r in misses:
        print(f"\n#{r['id']} {r['question']}")
        print("   retrieved: " + ", ".join(r["retrieved"][:5]))
    if misses:
        print("\nOpen the PDF for each miss: either the gold page in questions.json is wrong, or retrieval failed.")

    # ---------- Evidence gate ----------
    wanted = [r for r in results if r["label"] in ("answerable", "partial")]
    blocked = [r for r in results if r["label"] == "unanswerable"]
    print("\n" + "=" * 78)
    print(f"EVIDENCE GATE ({len(wanted)} answerable/partial, {len(blocked)} unanswerable)")
    print("=" * 78)
    print("Lowest top scores among answerable/partial questions:")
    for r in sorted(wanted, key=lambda r: r["top_score"])[:5]:
        print(f"  {r['top_score']:.3f}  #{r['id']} {r['question']}")
    print("Highest top scores among unanswerable questions:")
    for r in sorted(blocked, key=lambda r: -r["top_score"])[:5]:
        print(f"  {r['top_score']:.3f}  #{r['id']} {r['question']}")

    print(f"\n{'threshold':>10}{'answerable passed':>20}{'unanswerable blocked':>24}{'balanced accuracy':>20}")
    best = None
    for threshold in THRESHOLDS:
        pass_rate = average([1.0 if r["top_score"] >= threshold else 0.0 for r in wanted])
        block_rate = average([1.0 if r["top_score"] < threshold else 0.0 for r in blocked])
        balanced = (pass_rate + block_rate) / 2
        marker = "  <- current" if abs(threshold - EVIDENCE_THRESHOLD) < 1e-9 else ""
        print(f"{threshold:>10.2f}{pass_rate:>20.3f}{block_rate:>24.3f}{balanced:>20.3f}{marker}")
        if best is None or balanced > best[1]:
            best = (threshold, balanced)
    print(f"\nBest balanced accuracy on this set: threshold {best[0]:.2f} ({best[1]:.3f}).")
    print("If an unanswerable question still passes the gate, the LLM prompt has to say 'Evidence unavailable'.")
    print(f"\nPer-question details saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
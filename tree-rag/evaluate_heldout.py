"""Retrieval metrics on the held-out question set.
It uses the same metric code as evaluate.py (imported), but it reads eval/heldout_questions.json and writes
eval/heldout_retrieval_results.json. It never touches eval/results.json (the development-set results).
It uses Jina only (embedding and reranker), no Groq. Informational: do not tune retrieval or the threshold on this set."""
import json
import os
import time

from answer import ALLOWED_REGIONS, EVIDENCE_THRESHOLD
from evaluate import CONFIGS, K, PAUSE_SECONDS, THRESHOLDS, average, rank_question, score_ranking
from retrieval import HybridRetriever

QUESTIONS_FILE = os.path.join("eval", "heldout_questions.json")
RESULTS_FILE = os.path.join("eval", "heldout_retrieval_results.json")


def main():
    with open(QUESTIONS_FILE, encoding="utf-8") as file:
        questions = json.load(file)

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

    scored = [r for r in results if "scores" in r]
    print("\n" + "=" * 78)
    print(f"HELD-OUT RETRIEVAL METRICS at k={K} ({len(scored)} questions with gold pages)")
    print("=" * 78)
    print(f"{'configuration':<16}{'Recall@' + str(K):>12}{'HitRate@' + str(K):>12}{'MRR':>10}{'Precision@' + str(K):>14}")
    for name in CONFIGS:
        rows = [r["scores"][name] for r in scored]
        print(f"{name:<16}{average([x['recall'] for x in rows]):>12.3f}{average([x['hit'] for x in rows]):>12.3f}"
              f"{average([x['mrr'] for x in rows]):>10.3f}{average([x['precision'] for x in rows]):>14.3f}")
    print("\nThe corpus has 121 chunks, so the top 10 are about 8% of it: Recall@10 is an easy target and MRR says more.")

    misses = [r for r in scored if r["scores"]["hybrid+rerank"]["first_hit"] is None]
    print("\n" + "=" * 78)
    print(f"QUESTIONS WHERE hybrid+rerank FOUND NO GOLD PAGE: {len(misses)}")
    print("=" * 78)
    for r in misses:
        print(f"\n#{r['id']} {r['question']}")
        print("   retrieved: " + ", ".join(r["retrieved"][:5]))

    wanted = [r for r in results if r["label"] in ("answerable", "partial")]
    blocked = [r for r in results if r["label"] == "unanswerable"]
    print("\n" + "=" * 78)
    print(f"EVIDENCE GATE, informational only ({len(wanted)} answerable/partial, {len(blocked)} unanswerable)")
    print("=" * 78)
    print("Lowest top scores among answerable/partial questions:")
    for r in sorted(wanted, key=lambda r: r["top_score"])[:5]:
        print(f"  {r['top_score']:.3f}  #{r['id']} {r['question']}")
    print("Highest top scores among unanswerable questions:")
    for r in sorted(blocked, key=lambda r: -r["top_score"])[:5]:
        print(f"  {r['top_score']:.3f}  #{r['id']} {r['question']}")
    print(f"\n{'threshold':>10}{'answerable passed':>20}{'unanswerable blocked':>24}")
    for threshold in THRESHOLDS:
        pass_rate = average([1.0 if r["top_score"] >= threshold else 0.0 for r in wanted])
        block_rate = average([1.0 if r["top_score"] < threshold else 0.0 for r in blocked])
        marker = "  <- current" if abs(threshold - EVIDENCE_THRESHOLD) < 1e-9 else ""
        print(f"{threshold:>10.2f}{pass_rate:>20.3f}{block_rate:>24.3f}{marker}")
    print("\nDo not change the threshold because of this table: that would tune the system on the held-out set.")
    print(f"\nPer-question details saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
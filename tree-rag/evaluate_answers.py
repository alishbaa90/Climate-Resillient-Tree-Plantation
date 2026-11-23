import argparse
import json
import os
import re
import time

import requests

import answer as pipeline
from retrieval import HybridRetriever

QUESTIONS_FILE = os.path.join("eval", "questions.json")
EXPECTATIONS_FILE = os.path.join("eval", "expectations.json")
RESULTS_FILE = os.path.join("eval", "answer_results.json")
GUIDELINE = "silvo_ecological_zones_2023.pdf"
PAUSE_SECONDS = 20.0  # pause after each question that used the LLM (Groq tokens-per-minute limit)

CLASSES = ["answer", "partial", "unavailable"]
DEFAULT_EXPECT = {"answerable": "answer", "partial": "partial", "unanswerable": "unavailable"}
FORBIDDEN_PHRASES = ["the only", "no other", "therefore", "this suggests", "this indicates", "not well suited"]
MIN_ANSWER_WORDS = 6  # shorter answers are flagged as fragments (a warning only)


def classify(text):
    cleaned = re.sub(r"^[\s*#>_\-]+", "", text).lower()
    if cleaned.startswith("evidence unavailable"):
        return "unavailable"
    if cleaned.startswith("partial evidence"):
        return "partial"
    return "answer"


def contains_item(text_lower, item):
    """True if one of the alternatives (separated by '|') appears as a whole word or phrase.
    A plural ending (s / es) is accepted, so 'ridge' matches 'ridges' but 'pot' does not match 'potential'."""
    for option in item.split("|"):
        option = option.strip().lower()
        if not option:
            continue
        plural = r"(?:s|es)?" if option[-1].isalpha() else ""
        pattern = r"(?<![a-z0-9])" + re.escape(option) + plural + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            return True
    return False


def table_rows(retriever):
    """All rows of the guideline tables as dictionaries (header -> value)."""
    rows = []
    for point in retriever.points:
        payload = point.payload
        if payload["source"] != GUIDELINE or payload["content_type"] != "table":
            continue
        for line in payload["text"].split("\n"):
            fields = dict(part.split(": ", 1) for part in line.split("; ") if ": " in part)
            if "Scientific Name" in fields:
                rows.append(fields)
    return rows


def zone_species(rows, spec):
    """Scientific names of one zone table. spec = {"zone": "6. Subtropical Arid Desert",
    "category": "Main Species" (optional), "indigenous": "Exotic" (optional)}."""
    names = set()
    for fields in rows:
        if fields.get("Zone") != spec["zone"]:
            continue
        if spec.get("category") and fields.get("Category") != spec["category"]:
            continue
        if spec.get("indigenous") and fields.get("Indigenous/Exotic") != spec["indigenous"]:
            continue
        name = fields["Scientific Name"].replace("*", "").strip()
        if name:
            names.add(name)
    return sorted(names)


def extinct_species(rows):
    return sorted(
        {f["Scientific Name"].replace("*", "").strip() for f in rows if f.get("Indigenous/Exotic", "").lower() == "extinct"}
    )


def run_question(retriever, question):
    for attempt in range(4):
        try:
            return pipeline.answer_question(retriever, question)
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else None
            if status == 429 and attempt < 3:
                wait = 10 * (attempt + 1)
                print(f"  (Jina rate limit, waiting {wait}s ...)")
                time.sleep(wait)
                continue
            raise


def evaluate_one(rows, extinct_names, expectation, result):
    text = pipeline.clean_unicode(result["answer"])
    lowered = text.lower()
    status = classify(text)
    accepted = expectation.get("accept") or [expectation["expect"]]
    status_ok = status in accepted

    must_include = expectation.get("must_include", [])
    missing = [m for m in must_include if not contains_item(lowered, m)]
    found_fraction = (len(must_include) - len(missing)) / len(must_include) if must_include else 1.0
    wrongly_present = [x for x in expectation.get("must_exclude", []) if contains_item(lowered, x)]

    # Row coverage: every row of the named zone table(s) must appear in the answer.
    specs = expectation.get("zone_rows", [])
    if isinstance(specs, dict):
        specs = [specs]
    expected_rows = []
    unknown_zones = []
    for spec in specs:
        names = zone_species(rows, spec)
        if not names:
            unknown_zones.append(spec["zone"])
        expected_rows.extend(names)
    missing_rows = [n for n in expected_rows if n.lower() not in lowered]

    # A species that the table marks as extinct must not be listed without saying so.
    extinct_unmentioned = []
    if status != "unavailable" and "extinct" not in lowered:
        extinct_unmentioned = [n for n in extinct_names if n.lower() in lowered]

    content_ok = (
        found_fraction >= expectation.get("min_fraction", 1.0)
        and not wrongly_present
        and not missing_rows
        and not unknown_zones
        and not extinct_unmentioned
    )

    citations_ok = True
    bad_citations = []
    if status != "unavailable":
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        valid = {t["context_number"] for t in result["trace"] if t["context_number"]}
        bad_citations = sorted(cited - valid)
        citations_ok = bool(cited) and not bad_citations

    unverified = result.get("unverified_names", [])
    corrections = result.get("name_corrections", {})
    warnings = [f"forbidden phrase '{p}'" for p in FORBIDDEN_PHRASES if p in lowered]
    for wrong, right in corrections.items():
        warnings.append(f"the model wrote the species name '{wrong}'; corrected to '{right}'")
    word_count = len(re.sub(r"\[\d+\]", "", text).split())
    if status == "answer" and word_count < MIN_ANSWER_WORDS:
        warnings.append(f"very short answer ({word_count} words)")

    reasons = []
    if not status_ok:
        reasons.append(f"status '{status}' (accepted: {accepted})")
    if missing:
        reasons.append(f"missing: {missing}")
    if wrongly_present:
        reasons.append(f"should not appear: {wrongly_present}")
    if unknown_zones:
        reasons.append(f"zone not found in the knowledge base (check the name in expectations.json): {unknown_zones}")
    if missing_rows:
        reasons.append(f"table rows missing from the answer ({len(missing_rows)} of {len(expected_rows)}): {missing_rows}")
    if extinct_unmentioned:
        reasons.append(f"lists species that the table marks as extinct without saying so: {extinct_unmentioned}")
    if not citations_ok:
        reasons.append("no citation" if not bad_citations else f"citation numbers not in context: {bad_citations}")
    if unverified:
        reasons.append(f"species names not found in the knowledge base: {unverified}")

    return {
        "status": status,
        "expected": expectation["expect"],
        "passed": status_ok and content_ok and citations_ok and not unverified,
        "reasons": reasons,
        "warnings": warnings,
        "softened": result.get("softened", False),
        "name_corrections": corrections,
        "used_llm": result["used_llm"],
        "top_score": result["top_score"],
        "answer": text,
        "config_hash": pipeline.config_hash(),
    }


def load_records(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as file:
            return {r["id"]: r for r in json.load(file)}
    except (OSError, ValueError, KeyError):
        return {}


def main():
    parser = argparse.ArgumentParser(description="End-to-end answer evaluation")
    parser.add_argument("ids", nargs="*", type=int, help="question ids to run (default: all)")
    parser.add_argument("--threshold", type=float, default=None, help="override EVIDENCE_THRESHOLD for this run")
    parser.add_argument("--effort", choices=["low", "medium", "high"], default=None, help="override the Groq reasoning effort")
    parser.add_argument("--resume", action="store_true", help="skip questions already answered with the same configuration")
    parser.add_argument("--pause", type=float, default=PAUSE_SECONDS, help="seconds to wait after each LLM call")
    parser.add_argument("--questions", default=QUESTIONS_FILE, help="questions file (default: eval/questions.json)")
    parser.add_argument("--expectations", default=EXPECTATIONS_FILE, help="expectations file")
    parser.add_argument("--out", default=RESULTS_FILE, help="where to save the full answers")
    args = parser.parse_args()

    if args.threshold is not None:
        pipeline.EVIDENCE_THRESHOLD = args.threshold
        pipeline.MIN_CHUNK_SCORE = min(pipeline.MIN_CHUNK_SCORE, args.threshold)
    if args.effort:
        pipeline.REASONING_EFFORT = args.effort
    config = pipeline.config_hash()
    print(f"Evidence threshold: {pipeline.EVIDENCE_THRESHOLD} | minimum chunk score: {pipeline.MIN_CHUNK_SCORE} "
          f"| reasoning effort: {pipeline.REASONING_EFFORT or 'default'} | configuration: {config}")

    with open(args.questions, encoding="utf-8") as file:
        questions = json.load(file)
    expectations = {}
    if os.path.exists(args.expectations):
        with open(args.expectations, encoding="utf-8") as file:
            expectations = json.load(file)
    if args.ids:
        questions = [q for q in questions if q["id"] in args.ids]

    previous = load_records(args.out)
    retriever = HybridRetriever(regions=pipeline.ALLOWED_REGIONS)
    rows = table_rows(retriever)
    extinct_names = extinct_species(rows)
    print(f"Species marked extinct in the guideline tables: {extinct_names}")

    records = []
    stopped_message = None
    for number, item in enumerate(questions, start=1):
        old = previous.get(item["id"])
        if args.resume and old and "error" not in old and old.get("config_hash") == config:
            records.append(old)
            print(f"\n[{number}/{len(questions)}] #{item['id']} (already done with this configuration, skipped)")
            continue

        expectation = expectations.get(str(item["id"])) or {"expect": DEFAULT_EXPECT[item["label"]]}
        print(f"\n[{number}/{len(questions)}] #{item['id']} {item['question']}")
        try:
            result = run_question(retriever, item["question"])
        except pipeline.DailyLimitError as error:
            stopped_message = str(error)
            print(f"  STOPPED: {stopped_message}")
            break
        except (RuntimeError, requests.HTTPError) as error:
            print(f"  ERROR: {error}")
            records.append({"id": item["id"], "question": item["question"], "error": str(error)})
            continue

        record = evaluate_one(rows, extinct_names, expectation, result)
        record.update({"id": item["id"], "question": item["question"], "label": item["label"]})
        records.append(record)

        verdict = "PASS" if record["passed"] else "FAIL"
        print(f"  {verdict}  status={record['status']}  top_score={record['top_score']:.3f}  LLM used={record['used_llm']}")
        for reason in record["reasons"]:
            print(f"    - {reason}")
        for warning in record["warnings"]:
            print(f"    (warning: {warning})")
        if record["softened"]:
            print("    (note: the guard replaced a completeness claim in the model's answer)")
        if record["used_llm"]:
            time.sleep(args.pause)

    merged = dict(previous)
    merged.update({r["id"]: r for r in records})
    with open(args.out, "w", encoding="utf-8") as file:
        json.dump([merged[key] for key in sorted(merged)], file, indent=2)

    scored = [r for r in records if "error" not in r]
    errors = [r for r in records if "error" in r]
    not_run = len(questions) - len(records)
    print("\n" + "=" * 78)
    print(f"SUMMARY: {sum(r['passed'] for r in scored)}/{len(scored)} passed, {len(errors)} errors, {not_run} not run")
    print("=" * 78)
    if stopped_message or errors or not_run:
        print("WARNING: this is NOT a complete result. Run again later with --resume to finish the remaining questions.")
        if stopped_message:
            print(stopped_message)

    print("\nExpected vs actual answer type (rows = expected, columns = actual):")
    print(f"{'':<14}" + "".join(f"{c:>13}" for c in CLASSES))
    for expected in CLASSES:
        row = [sum(1 for r in scored if r["expected"] == expected and r["status"] == actual) for actual in CLASSES]
        print(f"{expected:<14}" + "".join(f"{n:>13}" for n in row))

    unanswerable = [r for r in scored if r["label"] == "unanswerable"]
    refused = [r for r in unanswerable if r["status"] == "unavailable"]
    print(f"\nCore principle: {len(refused)}/{len(unanswerable)} unanswerable questions got 'Evidence unavailable'.")
    leaks = [r for r in unanswerable if r["used_llm"]]
    if leaks:
        print("Unanswerable questions that passed the gate and reached the LLM:")
        for r in leaks:
            print(f"  #{r['id']} (top score {r['top_score']:.3f}) -> LLM said: {r['status']}")

    guarded = [r["id"] for r in scored if r["softened"]]
    corrected = {r["id"]: r["name_corrections"] for r in scored if r.get("name_corrections")}
    warned = {r["id"]: r["warnings"] for r in scored if r["warnings"]}
    print(f"\nAnswers where the completeness guard rewrote a sentence: {guarded}")
    print(f"Answers where a species name was corrected: {corrected if corrected else 'none'}")
    print(f"Answers with warnings: {warned if warned else 'none'}")

    failures = [r for r in scored if not r["passed"]]
    if failures:
        print("\nFAILED QUESTIONS (read each answer yourself, the checks are only a signal):")
        for r in failures:
            print(f"\n#{r['id']} {r['question']}")
            for reason in r["reasons"]:
                print(f"   - {reason}")
            print("   answer: " + r["answer"][:600].replace("\n", " "))
    print(f"\nFull answers saved to {args.out}")


if __name__ == "__main__":
    main()
import json
import os
import sys

RESULTS_FILE = os.path.join("eval", "answer_results.json")


def main():
    wanted = {int(arg) for arg in sys.argv[1:]}
    with open(RESULTS_FILE, encoding="utf-8") as file:
        records = json.load(file)

    for record in records:
        if wanted and record["id"] not in wanted:
            continue
        print("=" * 78)
        print(f"#{record['id']} {record['question']}")
        if "error" in record:
            print(f"ERROR: {record['error'][:200]}")
            continue
        verdict = "PASS" if record["passed"] else "FAIL"
        print(f"{verdict} | status={record['status']} | top_score={record['top_score']:.3f} "
              f"| config={record.get('config_hash', 'old')}")
        print("-" * 78)
        print(record["answer"])


if __name__ == "__main__":
    main()
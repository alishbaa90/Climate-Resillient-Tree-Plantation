import json
import sys

import requests

BASE_URL = "http://127.0.0.1:8000"
QUESTIONS = [
    "Which trees have high salinity tolerance?",
    "Is Shisham suitable for Thatta?",
    "What is the annual rainfall in Thatta?",
]


def main():
    questions = sys.argv[1:] or QUESTIONS
    print(requests.get(f"{BASE_URL}/health", timeout=10).json())
    for question in questions:
        print("\n" + "=" * 78)
        print(f"Q: {question}")
        response = requests.post(f"{BASE_URL}/ask", json={"question": question}, timeout=180)
        if response.status_code != 200:
            print(f"HTTP {response.status_code}: {json.dumps(response.json(), indent=2)}")
            continue
        data = response.json()
        print(f"type={data['answer_type']} reliable={data['reliable']} attempts={data['attempts']} "
              f"top_score={data['top_score']} {data['latency_ms']}ms")
        print("-" * 78)
        print(data["answer"])
        if data["sources"]:
            print("\nSources:")
            for s in data["sources"]:
                print(f"  [{s['number']}] {s['source']} (page {s['page']}, score {s['score']})")
        for w in data["warnings"]:
            print(f"WARNING ({w['severity']}): {w['code']} - {w['message']}")


if __name__ == "__main__":
    main()
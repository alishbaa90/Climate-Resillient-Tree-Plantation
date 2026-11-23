import json
import sys

import requests

BASE_URL = "http://127.0.0.1:8000"
TOKEN_FILE = ".token"


def load_token():
    try:
        with open(TOKEN_FILE, encoding="utf-8") as file:
            return file.read().strip()
    except OSError:
        print("No .token file. Run: py get_token.py")
        sys.exit(1)


def show(label, response):
    print(f"\n--- {label}: HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError:
        print(response.text[:300])
        return None


def main():
    token = load_token()
    headers = {"Authorization": f"Bearer {token}"}

    # 1) Without a token the API must refuse.
    r = requests.post(f"{BASE_URL}/chat", json={"question": "Which trees have high salinity tolerance?"}, timeout=30)
    show("chat without a token (expected 401)", r)

    # 2) First question creates a conversation.
    r = requests.post(f"{BASE_URL}/chat", headers=headers, timeout=180,
                      json={"question": "Which trees have high salinity tolerance?"})
    first = show("first question", r)
    if not first or r.status_code != 200:
        return
    print(f"type={first['answer_type']} reliable={first['reliable']} saved={first['saved']}")
    print(first["answer"])
    conversation_id = first["conversation_id"]
    print(f"conversation_id: {conversation_id}")

    # 3) Second question in the same conversation.
    r = requests.post(f"{BASE_URL}/chat", headers=headers, timeout=180,
                      json={"question": "Is Shisham suitable for Thatta?", "conversation_id": conversation_id})
    second = show("second question, same conversation", r)
    if second and r.status_code == 200:
        print(f"type={second['answer_type']} saved={second['saved']}")

    # 4) The chat list and the saved messages.
    r = requests.get(f"{BASE_URL}/conversations", headers=headers, timeout=30)
    listing = show("conversations", r)
    if listing:
        for row in listing[:5]:
            print(f"  {row['id']}  {row['title']}")

    r = requests.get(f"{BASE_URL}/conversations/{conversation_id}/messages", headers=headers, timeout=30)
    messages = show("messages of the first conversation", r)
    if messages:
        for m in messages:
            preview = m["content"][:90].replace("\n", " ")
            print(f"  {m['role']:<9}{m['answer_type'] or '':<12}{preview}")

    # 5) A random conversation id must give 404 (and use no Groq tokens).
    r = requests.post(f"{BASE_URL}/chat", headers=headers, timeout=30,
                      json={"question": "Which trees have low salinity tolerance?",
                            "conversation_id": "00000000-0000-0000-0000-000000000000"})
    show("chat in an unknown conversation (expected 404)", r)


if __name__ == "__main__":
    main()
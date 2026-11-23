from retrieval import HybridRetriever

# Only search evidence that is relevant to Thatta, Sindh and Pakistan-wide guidelines.
ALLOWED_REGIONS = ["thatta", "sindh", "pakistan"]

if __name__ == "__main__":
    retriever = HybridRetriever(regions=ALLOWED_REGIONS)
    print("Type a question (type 'exit' to quit):")
    while True:
        question = input("\n> ").strip()
        if question.lower() in ("exit", "quit", ""):
            break
        for rank, item in enumerate(retriever.search(question), start=1):
            found_by = []
            if item["in_dense"]:
                found_by.append("dense")
            if item["in_bm25"]:
                found_by.append("bm25")
            text = item["text"].replace("\n", " ")
            print(f"\n#{rank}  rerank={item['rerank_score']:.3f}  {item['source']}  (page {item['page']})")
            print(f"    region={item['region']}  type={item['doc_type']}  found_by={'+'.join(found_by)}")
            print(f"    {text[:300]}...")
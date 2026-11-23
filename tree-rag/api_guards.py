import re

# Warnings with severity "serious" make the answer "not reliable" and trigger one more attempt.
SERIOUS = "serious"
INFO = "info"

LOW_CONFIDENCE_MARGIN = 0.10  # an answer whose top score is within this margin of the gate gets an info note


def classify_answer(text):
    """Return 'answer', 'partial' or 'unavailable' from the way the answer begins."""
    cleaned = re.sub(r"^[\s*#>_\-]+", "", text).lower()
    if cleaned.startswith("evidence unavailable"):
        return "unavailable"
    if cleaned.startswith("partial evidence"):
        return "partial"
    return "answer"


def check_answer(result, evidence_threshold):
    """Inspect one result of answer.answer_question() and return (answer_type, warnings, reliable).
    This layer never changes the answer text. It only reports problems, so that the app (and the user)
    can see them. Warnings are dictionaries with 'code', 'severity' and 'message'."""
    text = result["answer"]
    lowered = text.lower()
    answer_type = classify_answer(text)
    warnings = []

    def add(code, severity, message):
        warnings.append({"code": code, "severity": severity, "message": message})

    if answer_type != "unavailable":
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        valid = {t["context_number"] for t in result["trace"] if t["context_number"]}
        if not cited:
            add("no_citation", SERIOUS, "The answer contains no source citation [n].")
        elif cited - valid:
            add("invalid_citation", SERIOUS,
                f"The answer cites passages that were not provided: {sorted(cited - valid)}.")

    if "partial evidence:" in lowered and "evidence unavailable:" in lowered:
        add("mixed_answer_types", SERIOUS,
            "The answer mixes 'Partial evidence' and 'Evidence unavailable' in one reply.")

    if result.get("unverified_names"):
        add("unverified_species", SERIOUS,
            f"Species names not found in the knowledge base: {result['unverified_names']}.")

    if result.get("name_corrections"):
        add("species_name_corrected", INFO,
            f"The model wrote species names that were corrected automatically: {result['name_corrections']}.")

    if result.get("softened"):
        add("completeness_claim_removed", INFO,
            "A claim that a list is complete was replaced by a standard caveat.")

    if result["used_llm"] and result["top_score"] < evidence_threshold + LOW_CONFIDENCE_MARGIN:
        add("low_retrieval_confidence", INFO,
            "The best matching passage scored close to the evidence threshold. Treat this answer with extra care.")

    reliable = not any(w["severity"] == SERIOUS for w in warnings)
    return answer_type, warnings, reliable
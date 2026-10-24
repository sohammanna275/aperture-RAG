import re


POLICY_WORDS = {"nimbus", "starter", "pro", "sso"}

POLICY_PHRASES = (
    "audit logs",
    "single sign on",
    "company policy",
)

COMPARISON_WORDS = {
    "compare",
    "comparison",
    "difference",
    "differences",
    "versus",
    "vs",
}


def choose_route(question):
    if not question.strip():
        raise ValueError("The routing question must not be empty.")

    words = re.findall(r"[a-z0-9]+", question.casefold())
    tokens = set(words)
    normalized_question = " ".join(words)

    has_policy_word = bool(tokens & POLICY_WORDS)

    has_policy_phrase = any(
        phrase in normalized_question
        for phrase in POLICY_PHRASES
    )

    needs_documents = has_policy_word or has_policy_phrase

    if not needs_documents:
        return {
            "route": "direct",
            "reason": "No configured policy-related terms were detected.",
        }

    mentions_both_plans = {"starter", "pro"}.issubset(tokens)
    asks_for_comparison = bool(tokens & COMPARISON_WORDS)

    if mentions_both_plans or asks_for_comparison:
        return {
            "route": "deep",
            "reason": "Policy-related question involving both plans or a comparison.",
        }

    return {
        "route": "standard",
        "reason": "Policy-related question without a detected comparison.",
    }
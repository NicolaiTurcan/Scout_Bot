"""Compare chunk size/overlap settings on labelled questions with the real embedder.

A question counts as answered when the retrieved chunk contains its answer phrase. Per
setting: answer@1 / answer@3, the weakest top score among real questions, the best top
score among out-of-corpus ones, and the gap between them.

Usage:
    uv run python compare_chunking.py
"""
from scoutbot.duckdb_store import DuckDBKnowledgeBase
from scoutbot.knowledge import fastembed_embedder, load_corpus

#: (question, phrase the answering chunk must contain); ``None`` = not in the corpus.
QUESTIONS = [
    ("How long does a refund to a bank transfer take?", "10 business days"),
    ("Can I get my money back for an activation key I already revealed?", "activation keys"),
    ("Do you deliver to Switzerland?", "outside the EU"),
    ("My parcel says delivered but it is not here.", "neighbours"),
    ("My headphones broke by themselves eight months in. Will you sort it out at no cost?",
     "two-year warranty"),
    ("I opened the device to fix it myself, am I still covered?", "void the warranty"),
    ("What happens if my card payment for the renewal fails?", "retry it three times"),
    ("I see two charges for the same order.", "authorisation hold"),
    ("I never got the password reset email.", "spam folder"),
    ("Can you combine my two accounts into one?", "cannot merge"),
    ("What is the capital of France?", None),
    ("Can you recommend a good pasta recipe?", None),
]

SETTINGS = [(size, overlap) for size in (30, 60, 120) for overlap in (0, size // 4, size // 2)]


def evaluate(kb: DuckDBKnowledgeBase) -> dict[str, float]:
    """Run every question against ``kb``.

    Args:
        kb: An indexed store.

    Returns:
        ``answer@1``, ``answer@3``, ``min_real`` (weakest real top score), ``max_out``
        (best out-of-corpus top score) and ``gap`` (``min_real - max_out``).
    """
    at1 = at3 = 0
    real, out = [], []
    for question, phrase in QUESTIONS:
        hits = kb.search(question, k=3)
        if phrase is None:
            out.append(hits[0].score)
            continue
        real.append(hits[0].score)
        at1 += phrase in hits[0].text
        at3 += any(phrase in hit.text for hit in hits)
    return {
        "answer@1": at1 / len(real),
        "answer@3": at3 / len(real),
        "min_real": min(real),
        "max_out": max(out),
        "gap": min(real) - max(out),
    }


def main() -> None:
    """Index the corpus with each setting and print one result row per setting."""
    embedder = fastembed_embedder()
    docs = load_corpus()
    columns = ["chunks", "answer@1", "answer@3", "min_real", "max_out", "gap"]
    print(f"{'size':>4} {'overlap':>7} " + " ".join(f"{c:>8}" for c in columns))
    for size, overlap in SETTINGS:
        with DuckDBKnowledgeBase(embedder) as kb:
            kb.index(docs, size=size, overlap=overlap)
            r = evaluate(kb)
            print(
                f"{size:>4} {overlap:>7} {len(kb):>8} {r['answer@1']:>8.0%} "
                f"{r['answer@3']:>8.0%} {r['min_real']:>8.3f} {r['max_out']:>8.3f} "
                f"{r['gap']:>8.3f}"
            )


if __name__ == "__main__":
    main()

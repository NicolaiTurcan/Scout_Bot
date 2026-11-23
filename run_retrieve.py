"""Index ``knowledge_base/`` into ``knowledge.db``, reopen it without re-indexing, and print
the top passages for a few questions, including one the corpus cannot answer.

Usage:
    uv run python run_retrieve.py
"""
import re
import textwrap

from scoutbot import config
from scoutbot.knowledge import KnowledgeBase, fastembed_embedder, load_corpus

TOP_K = 3

#: (label, question). The warranty one shares no content words with warranty.md; the last
#: one is outside the corpus.
QUESTIONS = [
    ("refund", "How long does a refund to a bank transfer take?"),
    ("shipping", "Do you deliver to Switzerland?"),
    (
        "warranty, no shared keywords",
        "My headphones broke by themselves eight months in. Will you sort it out at no cost?",
    ),
    ("out of corpus", "What is the capital of France?"),
]

#: Common English words ignored when reporting keyword overlap.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "to", "of", "on", "in", "my", "i", "you", "do", "does",
    "how", "what", "that", "them", "it", "after", "long", "for", "and", "or", "can", "be",
    "by", "at", "no", "out", "will", "we", "your", "our",
}


def _content_words(text: str) -> set[str]:
    """Extract content words for the keyword-overlap report.

    Args:
        text: Any text.

    Returns:
        Lower-cased words of ``text`` without stopwords.
    """
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOPWORDS}


def main() -> None:
    """Build and reopen the store, print top hits per question and the in/out-of-corpus gap."""
    embedder = fastembed_embedder()
    docs = load_corpus()
    db_path = config.DEFAULT_KNOWLEDGE_DB_PATH

    with KnowledgeBase(embedder, db_path) as kb:
        indexed = kb.build(docs)
    print(f"Indexed {indexed} chunks from {len(docs)} documents into {db_path.name}\n")

    # A brand-new store on the same file: no build() call, the vectors are already on disk.
    best_scores = {}
    with KnowledgeBase(embedder, db_path) as kb:
        print(f"Reopened {db_path.name}: {len(kb)} chunks, no re-indexing\n")
        for label, question in QUESTIONS:
            hits = kb.search(question, k=TOP_K)
            best_scores[label] = hits[0].score if hits else 0.0
            print(f"=== [{label}] {question}")
            for hit in hits:
                shared = sorted(_content_words(question) & _content_words(hit.text))
                where = f"{hit.source}#{hit.ordinal}"
                print(f"  {hit.score:.3f}  {where}  shared words: {shared or '-'}")
                print(textwrap.indent(textwrap.fill(hit.text, 96), "         "))
            print()

    in_corpus = [score for label, score in best_scores.items() if label != "out of corpus"]
    out_of_corpus = best_scores["out of corpus"]
    print("=== Top score per question")
    for label, score in best_scores.items():
        print(f"  {score:.3f}  {label}")
    print(
        f"\nGap between the weakest real hit ({min(in_corpus):.3f}) and the out-of-corpus "
        f"question ({out_of_corpus:.3f}): {min(in_corpus) - out_of_corpus:.3f}"
    )


if __name__ == "__main__":
    main()

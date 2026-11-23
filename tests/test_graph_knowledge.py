"""Offline tests for the ticket graph with a knowledge base wired in."""
from fakes import HIGH_TRIAGE, LOW_TRIAGE, triage_response
from langgraph.checkpoint.memory import InMemorySaver

from scoutbot.graph import build_ticket_graph, initial_state, thread_config
from scoutbot.graph.nodes import make_retrieve_policies_node
from scoutbot.knowledge import Hit

REFUND_HIT = Hit("refunds.md", 1, "Refunds to a bank transfer take up to 10 business days.", 0.9)
WEAK_HIT = Hit("shipping.md", 0, "Standard shipping is free over 50 euros.", 0.3)


class FakeKnowledgeBase:
    """Returns fixed hits and records the queries it receives."""

    def __init__(self, hits):
        self.hits = hits
        self.queries = []

    def search(self, query, k=3):
        """Record ``query`` and return the first ``k`` fixed hits."""
        self.queries.append(query)
        return self.hits[:k]


def test_retrieve_node_searches_with_ticket_text_and_drops_weak_hits():
    kb = FakeKnowledgeBase([REFUND_HIT, WEAK_HIT])
    node = make_retrieve_policies_node(kb, min_score=0.5)

    update = node(initial_state("When is my refund coming?", "1001", "T-1"))

    assert kb.queries == ["When is my refund coming?"]
    assert update["policies"] == [REFUND_HIT._asdict()]
    assert update["notes"] == ["retrieved policies: refunds.md#1 (0.90)"]


def test_triage_prompt_cites_retrieved_policies(messages_create):
    messages_create.return_value = triage_response(LOW_TRIAGE)
    graph = build_ticket_graph(knowledge_base=FakeKnowledgeBase([REFUND_HIT]))

    final = graph.invoke(initial_state("When is my refund coming?", "1001", "T-1"))

    prompt = messages_create.call_args.kwargs["messages"][0]["content"]
    assert "Relevant policy passages:" in prompt
    assert "[refunds.md#1] Refunds to a bank transfer" in prompt
    assert final["outcome"] == "handled"
    assert final["notes"][1].startswith("retrieved policies:")


def test_escalation_payload_shows_policies_to_the_human(messages_create):
    messages_create.return_value = triage_response(HIGH_TRIAGE)
    graph = build_ticket_graph(
        checkpointer=InMemorySaver(), knowledge_base=FakeKnowledgeBase([REFUND_HIT])
    )

    paused = graph.invoke(initial_state("Refund now!", "1001", "T-2"), thread_config("T-2"))

    assert paused["__interrupt__"][0].value["policies"] == [REFUND_HIT._asdict()]


def test_graph_without_knowledge_base_has_no_retrieval_step(messages_create):
    messages_create.return_value = triage_response(LOW_TRIAGE)

    graph = build_ticket_graph()
    final = graph.invoke(initial_state("Hi", "1001"))

    assert "retrieve_policies" not in graph.get_graph().nodes
    assert "policies" not in final
    prompt = messages_create.call_args.kwargs["messages"][0]["content"]
    assert "Relevant policy passages" not in prompt

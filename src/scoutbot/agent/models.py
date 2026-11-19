"""Data classes describing an agent loop run."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoopStep:
    """One Thought/Action/Observation entry in the ReAct trace.

    Attributes:
        thought: The model's reasoning text for this step (or a fixed note for server tool
            steps and the step-cap step).
        action: Name of the tool called, or ``None`` for a final answer / cap step.
        action_input: Arguments the model passed to the tool.
        observation: What the tool returned (client tool) or a summary of what Anthropic
            already ran (server tool).
    """

    thought: str
    action: Optional[str] = None
    action_input: Optional[dict] = None
    observation: Optional[dict] = None


@dataclass
class AgentResult:
    """Outcome of :func:`scoutbot.agent.loop.run_agent_loop` for one ticket.

    Attributes:
        final_answer: The model's final plain-text answer, or a fixed escalation message if
            the step cap was hit.
        steps: The full trace, in order.
        escalated: ``True`` if the loop hit its step cap without a final answer.
    """

    final_answer: str
    steps: list[LoopStep] = field(default_factory=list)
    escalated: bool = False

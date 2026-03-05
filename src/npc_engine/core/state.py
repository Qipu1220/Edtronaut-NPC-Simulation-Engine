"""
Simulation State Definition for LangGraph.

Defines the state schema that flows through the simulation graph,
including conversation history, trust metrics, and emotional tracking.
"""
from typing import List, Dict, TypedDict, Optional
from langchain_core.messages import BaseMessage


class EmotionalVector(TypedDict):
    """Tracks NPC emotional state on three axes."""
    engagement: float      # 0.0 (disengaged) to 1.0 (highly engaged)
    skepticism: float      # 0.0 (trusting) to 1.0 (very skeptical)
    openness: float        # 0.0 (closed) to 1.0 (receptive)


class SimulationState(TypedDict):
    """
    Complete state for the NPC simulation.

    Attributes:
        messages: List of conversation turns (HumanMessage, AIMessage)
        trust_score: NPC's trust in user (0-100), starts at 80
        emotional_vector: Current emotional state of NPC
        session_id: Unique identifier for this conversation
        meeting_status: "active", "warning", or "ended"
        turn_count: Number of conversation turns
        rag_context: Latest RAG retrieval for debugging
        director_hint: Hidden hint injected by Director (if any)
    """
    messages: List[BaseMessage]
    trust_score: int
    emotional_vector: EmotionalVector
    session_id: str
    meeting_status: str
    turn_count: int
    rag_context: str
    director_hint: Optional[str]


# Constants for initial state
INITIAL_TRUST_SCORE = 80
INITIAL_EMOTIONAL_VECTOR: EmotionalVector = {
    "engagement": 0.6,
    "skepticism": 0.3,
    "openness": 0.5
}


def create_initial_state(session_id: str) -> SimulationState:
    """
    Create a fresh simulation state for a new session.

    Args:
        session_id: Unique identifier for this conversation

    Returns:
        Initial SimulationState with default values
    """
    return SimulationState(
        messages=[],
        trust_score=INITIAL_TRUST_SCORE,
        emotional_vector=INITIAL_EMOTIONAL_VECTOR.copy(),
        session_id=session_id,
        meeting_status="active",
        turn_count=0,
        rag_context="",
        director_hint=None
    )

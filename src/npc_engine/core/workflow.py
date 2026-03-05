"""
LangGraph Workflow Definition for Edtronaut NPC Simulation.

Defines the state graph that orchestrates the conversation flow,
including the NPC agent node and conditional routing.
"""
from typing import Dict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from langchain_core.language_models import BaseChatModel

from .state import SimulationState, create_initial_state
from ..agents.npc_agent import create_npc_node
from ..agents.director import DirectorLayer, MeetingStatus


def should_continue(state: SimulationState) -> Literal["continue", "end"]:
    """
    Conditional edge to determine if simulation should continue.

    Returns:
        "continue" if meeting is active, "end" otherwise
    """
    if state["meeting_status"] == MeetingStatus.ENDED.value:
        return "end"
    return "continue"


def create_simulation_graph(
    llm: BaseChatModel,
    persona: Dict,
    director: DirectorLayer
) -> StateGraph:
    """
    Create the LangGraph simulation workflow.

    Args:
        llm: LangChain chat model
        persona: NPC persona configuration
        director: DirectorLayer instance

    Returns:
        Compiled StateGraph
    """
    # Create the graph
    graph = StateGraph(SimulationState)

    # Add NPC agent node
    npc_node = create_npc_node(llm, persona, director)
    graph.add_node("npc_agent", npc_node)

    # Set entry point
    graph.set_entry_point("npc_agent")

    # Add conditional edge for meeting continuation
    graph.add_conditional_edges(
        "npc_agent",
        should_continue,
        {
            "continue": END,  # Wait for next user input
            "end": END        # Meeting ended
        }
    )

    return graph.compile()


class SimulationEngine:
    """
    Main engine for running NPC simulations.

    Manages sessions, processes messages, and maintains state.
    """

    def __init__(self, llm: BaseChatModel, persona: Dict):
        """
        Initialize simulation engine.

        Args:
            llm: LangChain chat model
            persona: NPC persona configuration
        """
        self.llm = llm
        self.persona = persona
        self.director = DirectorLayer(persona)
        self.graph = create_simulation_graph(llm, persona, self.director)
        self.sessions: Dict[str, SimulationState] = {}

    def create_session(self, session_id: str) -> SimulationState:
        """
        Create a new simulation session.

        Args:
            session_id: Unique session identifier

        Returns:
            Initial simulation state
        """
        state = create_initial_state(session_id)
        self.sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> SimulationState:
        """
        Get existing session or create new one.

        Args:
            session_id: Session identifier

        Returns:
            Simulation state
        """
        if session_id not in self.sessions:
            return self.create_session(session_id)
        return self.sessions[session_id]

    def process_message(
        self,
        session_id: str,
        user_input: str
    ) -> Dict:
        """
        Process a user message and return NPC response.

        Args:
            session_id: Session identifier
            user_input: User's message

        Returns:
            Dict with response and updated state
        """
        # Get or create session
        state = self.get_session(session_id)

        # Check if meeting already ended
        if state["meeting_status"] == MeetingStatus.ENDED.value:
            return {
                "response": "This meeting has concluded. Please start a new session.",
                "state": state,
                "meeting_ended": True
            }

        # Add user message to state
        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        # Execute graph
        result = self.graph.invoke(state)

        # Update stored session
        self.sessions[session_id] = result

        # Extract response
        last_message = result["messages"][-1]
        response = last_message.content if hasattr(last_message, 'content') else str(last_message)

        return {
            "response": response,
            "state": result,
            "meeting_ended": result["meeting_status"] == MeetingStatus.ENDED.value
        }

    def get_debug_state(self, session_id: str) -> Dict:
        """
        Get internal state for debugging.

        Args:
            session_id: Session identifier

        Returns:
            Dict with debug information
        """
        state = self.get_session(session_id)

        return {
            "trust_score": state["trust_score"],
            "emotional_vector": state["emotional_vector"],
            "turn_count": state["turn_count"],
            "meeting_status": state["meeting_status"],
            "director_hint": state.get("director_hint"),
            "rag_context": state.get("rag_context", "")[:200] + "..." if state.get("rag_context") else None
        }

    def end_session(self, session_id: str) -> None:
        """Remove a session from memory."""
        if session_id in self.sessions:
            del self.sessions[session_id]

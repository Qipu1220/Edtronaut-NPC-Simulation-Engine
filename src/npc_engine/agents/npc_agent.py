"""
NPC Agent - LangGraph node that generates CEO responses using LLM.

This agent:
- Uses persona configuration to drive response style
- Incorporates RAG context for knowledge-grounded responses
- Adjusts behavior based on emotional state and trust score
- Follows Director's behavioral directives
"""
from typing import Dict, Optional, Callable
import logging

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.language_models import BaseChatModel

from ..core.state import SimulationState
from ..utils.rag_mock import mock_rag_retrieve
from .director import DirectorLayer, MeetingStatus

logger = logging.getLogger(__name__)

# Maximum messages to include in chat context
MAX_CHAT_HISTORY = 6


class NPCAgent:
    """
    LLM-powered NPC agent that generates CEO responses.

    The agent uses a persona-driven prompt template and adjusts
    its responses based on the current simulation state.
    """

    def __init__(self, llm: BaseChatModel, persona: Dict, director: DirectorLayer):
        """
        Initialize NPC agent.

        Args:
            llm: LangChain chat model instance
            persona: NPC persona configuration dict
            director: DirectorLayer instance for behavioral guidance
        """
        self.llm = llm
        self.persona = persona
        self.director = director
        self.prompt_template = self._build_prompt_template()

    def _build_prompt_template(self) -> ChatPromptTemplate:
        """Build the prompt template from persona configuration."""
        personality = self.persona.get("personality", {})
        core_values = self.persona.get("core_values", [])

        system_prompt = f"""You are {self.persona.get('name', 'the CEO')}, {self.persona.get('role', 'a business executive')}.

PERSONALITY TRAITS:
{self._format_traits(personality.get('traits', []))}

COMMUNICATION STYLE: {personality.get('communication_style', 'professional')}
PATIENCE LEVEL: {personality.get('patience_level', 'medium')}

CORE VALUES:
{self._format_values(core_values)}

CURRENT EMOTIONAL STATE:
- Engagement: {{engagement:.2f}} (0=disengaged, 1=highly engaged)
- Skepticism: {{skepticism:.2f}} (0=trusting, 1=very skeptical)
- Openness: {{openness:.2f}} (0=closed, 1=receptive)

BEHAVIORAL DIRECTIVE:
- Tone: {{behavioral_tone}}
- Pace: {{behavioral_pace}}
- Elaboration Level: {{elaboration_level}}

{template_prefix}

{rag_context}

{director_hint}

REJECTION RULES:
- Firmly decline questions about personal matters, office gossip, or non-business topics
- Redirect to strategic business discussions
- If the user seems unprepared, express professional disappointment

RESPONSE GUIDELINES:
1. Stay in character as a decisive, brand-protective CEO
2. Match the current emotional state and behavioral directive
3. Be concise when trust is low, elaborate when trust is high
4. Never break character or acknowledge being an AI
5. Use knowledge base context when relevant

Chat History:
{{chat_history}}

Human: {{user_input}}
AI:"""

        return ChatPromptTemplate.from_template(system_prompt)

    def _format_traits(self, traits: list) -> str:
        """Format personality traits for prompt."""
        return "\n".join(f"- {trait}" for trait in traits)

    def _format_values(self, values: list) -> str:
        """Format core values for prompt."""
        formatted = []
        for value in values:
            formatted.append(f"- {value.get('name', 'Value')}: {value.get('description', '')}")
        return "\n".join(formatted)

    def generate_response(self, state: SimulationState) -> str:
        """
        Generate NPC response based on current state.

        Args:
            state: Current simulation state

        Returns:
            NPC response string
        """
        user_input = state["messages"][-1].content if state["messages"] else ""
        emotional_vector = state["emotional_vector"]
        trust_score = state["trust_score"]

        # Get behavioral directive from Director
        directive = self.director.get_behavioral_directive(trust_score, emotional_vector)

        # Get RAG context (single retrieval, reused)
        rag_context = state.get("rag_context") or mock_rag_retrieve(user_input, state["session_id"])

        # Get director hint if available
        director_hint = state.get("director_hint")
        hint_section = ""
        if director_hint:
            hint_section = f"[DIRECTOR GUIDANCE - Incorporate subtly: {director_hint}]"

        # Format chat history (last N messages)
        chat_history = self._format_chat_history(state["messages"][:-1])

        # Build prompt
        formatted_prompt = self.prompt_template.format_messages(
            engagement=emotional_vector["engagement"],
            skepticism=emotional_vector["skepticism"],
            openness=emotional_vector["openness"],
            behavioral_tone=directive["tone"],
            behavioral_pace=directive["pace"],
            elaboration_level=directive["elaboration_level"],
            template_prefix=directive.get("template_prefix", ""),
            rag_context=rag_context,
            director_hint=hint_section,
            chat_history=chat_history,
            user_input=user_input
        )

        try:
            response = self.llm.invoke(formatted_prompt)
            ai_response = response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            ai_response = "I apologize, but I'm having trouble processing that right now. Could you rephrase?"

        return ai_response

    def _format_chat_history(self, messages: list) -> str:
        """Format message history for context."""
        if not messages:
            return "No previous conversation."

        # Limit to last N messages
        recent = messages[-MAX_CHAT_HISTORY:]
        formatted = []

        for msg in recent:
            role = "Human" if isinstance(msg, HumanMessage) else "CEO"
            formatted.append(f"{role}: {msg.content}")

        return "\n".join(formatted)


def create_npc_node(llm: BaseChatModel, persona: Dict, director: DirectorLayer) -> Callable:
    """
    Factory function to create an NPC agent node for LangGraph.

    Returns:
        Callable node function for use in StateGraph
    """
    agent = NPCAgent(llm, persona, director)

    def npc_node(state: SimulationState) -> Dict:
        """
        LangGraph node that processes user input and generates NPC response.

        Args:
            state: Current simulation state

        Returns:
            Dict with updated state fields
        """
        user_input = state["messages"][-1].content if state["messages"] else ""

        # Classify user intent and get trust/emotion deltas
        intent_type, trust_delta, emotion_deltas = director.classify_user_intent(user_input)

        # Update trust score
        new_trust = director.update_trust_score(state["trust_score"], trust_delta)

        # Update emotional vector
        new_emotions = director.update_emotional_vector(
            state["emotional_vector"],
            emotion_deltas
        )

        # Determine meeting status
        new_status, status_msg = director.determine_meeting_status(
            new_trust,
            state["meeting_status"]
        )

        # Check for struggling user
        hint = director.check_struggling(
            user_input,
            new_trust,
            state["turn_count"]
        )

        # Get RAG context (single retrieval)
        rag_context = mock_rag_retrieve(user_input, state["session_id"])

        # Create updated state for response generation
        updated_state = {
            **state,
            "trust_score": new_trust,
            "emotional_vector": new_emotions,
            "meeting_status": new_status,
            "director_hint": hint,
            "rag_context": rag_context
        }

        # Generate NPC response
        if new_status == MeetingStatus.ENDED.value:
            response = "I'm afraid this meeting is over. We clearly don't see eye to eye on what's important for this brand."
        else:
            response = agent.generate_response(updated_state)

        # Return state updates
        return {
            "messages": [AIMessage(content=response)],
            "trust_score": new_trust,
            "emotional_vector": new_emotions,
            "meeting_status": new_status,
            "turn_count": state["turn_count"] + 1,
            "rag_context": rag_context,
            "director_hint": hint
        }

    return npc_node

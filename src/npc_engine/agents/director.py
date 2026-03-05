"""
Director Layer - Monitors simulation state and provides guidance.

The Director is responsible for:
- Monitoring trust score and triggering behavioral changes
- Detecting if user is struggling and injecting hints
- Managing meeting status transitions
- Providing meta-guidance for NPC responses
"""
from typing import Tuple, Dict, Optional
from enum import Enum


class MeetingStatus(str, Enum):
    """Meeting status values for type safety."""
    ACTIVE = "active"
    WARNING = "warning"
    ENDED = "ended"


class DirectorLayer:
    """
    The Director monitors the simulation and adjusts NPC behavior.

    Responsibilities:
    - Monitor trust score and trigger behavioral changes
    - Detect if user is struggling and inject hints
    - Manage meeting status transitions
    - Provide meta-guidance for NPC responses
    """

    # Trust score thresholds
    TRUST_HIGH = 70
    TRUST_MEDIUM = 50
    TRUST_LOW = 30
    TRUST_CRITICAL = 15

    # Emotional vector bounds
    EMOTION_MIN = 0.0
    EMOTION_MAX = 1.0

    # Keyword lists for intent classification (class-level for efficiency)
    STRATEGIC_KEYWORDS = frozenset([
        "strategy", "growth", "market", "revenue", "brand",
        "innovation", "digital", "expansion", "investment",
        "competitive", "opportunity", "sustainability"
    ])
    BRAND_KEYWORDS = frozenset(["heritage", "craftsmanship", "luxury", "quality", "excellence"])

    # Hints for struggling users
    HINTS = [
        "Consider asking about our strategic priorities for the coming year.",
        "You might want to explore how digital transformation affects our luxury positioning.",
        "Think about what makes our brand unique in the competitive landscape.",
        "Perhaps discuss how sustainability integrates with our business model."
    ]

    def __init__(self, persona: Dict):
        """
        Initialize Director with NPC persona configuration.

        Args:
            persona: Dict containing persona config from JSON
        """
        self.persona = persona
        self.trust_modifiers = persona.get("trust_modifiers", {})
        self.red_lines = persona.get("red_lines", {})
        self.hint_triggers = persona.get("hint_triggers", {})

    def classify_user_intent(self, user_input: str) -> Tuple[str, int, Dict]:
        """
        Classify the user's message intent and calculate impact.

        Returns:
            Tuple of (intent_type, trust_delta, emotion_deltas)
        """
        user_lower = user_input.lower()

        # Check for red line triggers (most severe)
        for trigger in self.red_lines.get("trigger_phrases", []):
            if trigger.lower() in user_lower:
                return (
                    "red_line_violation",
                    self.trust_modifiers.get("poor_preparation", {}).get("trust_delta", -15),
                    {"engagement": -0.2, "skepticism": 0.25, "openness": -0.15}
                )

        # Check for rejected topics
        for topic in self.red_lines.get("reject_topics", []):
            topic_keywords = topic.split()
            if all(kw in user_lower for kw in topic_keywords):
                return (
                    "irrelevant_topic",
                    self.trust_modifiers.get("irrelevant_questions", {}).get("trust_delta", -10),
                    {"engagement": -0.1, "skepticism": 0.2, "openness": -0.1}
                )

        # Check for strategic questions (positive)
        strategic_count = sum(1 for kw in self.STRATEGIC_KEYWORDS if kw in user_lower)
        if strategic_count >= 2:
            return (
                "strategic_inquiry",
                self.trust_modifiers.get("strategic_questions", {}).get("trust_delta", 5),
                {"engagement": 0.1, "skepticism": -0.05, "openness": 0.1}
            )

        # Check for brand alignment
        if any(kw in user_lower for kw in self.BRAND_KEYWORDS):
            return (
                "brand_alignment",
                self.trust_modifiers.get("brand_alignment", {}).get("trust_delta", 8),
                {"engagement": 0.15, "skepticism": -0.1, "openness": 0.15}
            )

        # Check if user is struggling
        struggling_indicators = self.hint_triggers.get("struggling_indicators", [])
        for indicator in struggling_indicators:
            if indicator.lower() in user_lower:
                return (
                    "user_struggling",
                    -3,
                    {"engagement": -0.05, "skepticism": 0.1, "openness": 0}
                )

        return ("neutral", 0, {"engagement": 0, "skepticism": 0, "openness": 0})

    def update_trust_score(self, current_score: int, delta: int) -> int:
        """Update and bound trust score to valid range."""
        return max(0, min(100, current_score + delta))

    def update_emotional_vector(
        self,
        current_vector: Dict,
        deltas: Dict
    ) -> Dict:
        """Update and bound emotional vector values."""
        return {
            "engagement": max(
                self.EMOTION_MIN,
                min(self.EMOTION_MAX, current_vector["engagement"] + deltas.get("engagement", 0))
            ),
            "skepticism": max(
                self.EMOTION_MIN,
                min(self.EMOTION_MAX, current_vector["skepticism"] + deltas.get("skepticism", 0))
            ),
            "openness": max(
                self.EMOTION_MIN,
                min(self.EMOTION_MAX, current_vector["openness"] + deltas.get("openness", 0))
            )
        }

    def determine_meeting_status(
        self,
        trust_score: int,
        current_status: str
    ) -> Tuple[str, str]:
        """
        Determine meeting status based on trust score.

        Returns:
            Tuple of (new_status, status_message)
        """
        if trust_score <= self.TRUST_CRITICAL:
            return (MeetingStatus.ENDED.value, "Meeting terminated due to lack of strategic alignment.")

        if trust_score <= self.TRUST_LOW:
            return (MeetingStatus.WARNING.value, "CEO is visibly impatient. This is your final warning.")

        if trust_score <= self.TRUST_MEDIUM:
            if current_status != MeetingStatus.WARNING.value:
                return (MeetingStatus.WARNING.value, "CEO's patience is wearing thin.")

        if trust_score >= self.TRUST_HIGH and current_status == MeetingStatus.WARNING.value:
            return (MeetingStatus.ACTIVE.value, "CEO seems more engaged now.")

        return (current_status, "")

    def check_struggling(self, user_input: str, trust_score: int, turn_count: int) -> Optional[str]:
        """
        Check if user needs guidance and return a hint.

        Returns:
            Hint string or None
        """
        # Check for explicit help-seeking
        struggling_indicators = self.hint_triggers.get("struggling_indicators", [])
        user_lower = user_input.lower()

        if any(indicator.lower() in user_lower for indicator in struggling_indicators):
            hint_idx = turn_count % len(self.HINTS)
            return self.HINTS[hint_idx]

        # Check for low engagement over multiple turns
        if trust_score < self.TRUST_MEDIUM and turn_count >= 3:
            return f"[DIRECTOR HINT: {self.HINTS[turn_count % len(self.HINTS)]}]"

        return None

    def get_behavioral_directive(self, trust_score: int, emotional_vector: Dict) -> Dict:
        """
        Generate behavioral directives for the NPC based on current state.

        Returns:
            Dict with tone, pace, elaboration_level, and willingness_to_share
        """
        templates = self.persona.get("response_templates", {})

        if trust_score >= self.TRUST_HIGH:
            return {
                "tone": "collaborative and open",
                "pace": "relaxed",
                "template_prefix": templates.get("high_trust", ""),
                "elaboration_level": "high",
                "willingness_to_share": 0.9
            }

        if trust_score >= self.TRUST_MEDIUM:
            return {
                "tone": "professional but guarded",
                "pace": "measured",
                "template_prefix": templates.get("medium_trust", ""),
                "elaboration_level": "medium",
                "willingness_to_share": 0.6
            }

        if trust_score >= self.TRUST_LOW:
            return {
                "tone": "terse and impatient",
                "pace": "brisk",
                "template_prefix": templates.get("low_trust", ""),
                "elaboration_level": "low",
                "willingness_to_share": 0.3
            }

        # Critical trust level
        return {
            "tone": "dismissive",
            "pace": "ending",
            "template_prefix": templates.get("meeting_end", ""),
            "elaboration_level": "minimal",
            "willingness_to_share": 0.1
        }

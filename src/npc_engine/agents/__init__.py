"""Agents module."""
from .director import DirectorLayer, MeetingStatus

# Import npc_agent lazily to avoid dependency issues
def get_npc_agent():
    from .npc_agent import NPCAgent, create_npc_node
    return NPCAgent, create_npc_node

__all__ = ["DirectorLayer", "MeetingStatus", "get_npc_agent"]

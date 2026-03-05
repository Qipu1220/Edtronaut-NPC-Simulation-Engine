"""Core module for state and workflow."""
from .state import SimulationState, create_initial_state

# Import workflow lazily to avoid circular imports when dependencies aren't installed
def get_workflow():
    from .workflow import SimulationEngine, create_simulation_graph
    return SimulationEngine, create_simulation_graph

__all__ = ["SimulationState", "create_initial_state", "get_workflow"]

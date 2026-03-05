# Edtronaut NPC Simulation Engine

An AI-powered NPC simulation engine for job simulation platform, built with Python, FastAPI, and LangGraph.

## Features

- **Stateful Conversations**: LangGraph-powered state management with trust scores and emotional vectors
- **Director Layer**: Monitors user performance and adjusts NPC behavior dynamically
- **Persona-Driven NPCs**: JSON-configurable NPC personalities (includes Gucci CEO persona)
- **Trust Score Decay**: Strategic questions increase trust; irrelevant questions decrease it
- **Hidden Hints**: Director provides subtle guidance when users struggle
- **Mock RAG**: Placeholder knowledge base for Gucci-related information

## Project Structure

```
Edtronaut-NPC-Simulation-Engine/
├── main.py                      # FastAPI entry point
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment configuration template
├── src/
│   └── npc_engine/
│       ├── __init__.py
│       ├── api.py               # FastAPI endpoints
│       ├── core/
│       │   ├── state.py         # LangGraph state definitions
│       │   └── workflow.py      # Graph workflow and engine
│       ├── agents/
│       │   ├── npc_agent.py     # LLM-powered NPC agent
│       │   └── director.py      # Director layer logic
│       ├── utils/
│       │   └── rag_mock.py      # Mock RAG implementation
│       └── data/
│           └── persona_ceo.json # CEO persona configuration
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Usage

### Start the Server

```bash
python main.py
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### POST /chat
Send a message to the NPC CEO:

```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_input": "What is your strategy for digital growth?", "session_id": "test-session-1"}'
```

#### POST /session/new
Create a new session:

```bash
curl -X POST "http://localhost:8000/session/new"
```

#### GET /session/{session_id}/debug
Get internal state for debugging

## Trust Score Mechanics

| User Behavior | Trust Impact | Engagement | Skepticism |
|--------------|--------------|------------|------------|
| Strategic questions | +5 | +0.1 | -0.05 |
| Brand alignment | +8 | +0.15 | -0.1 |
| Irrelevant questions | -10 | -0.1 | +0.2 |
| Red line triggers | -15 | -0.2 | +0.25 |

### Trust Thresholds

- **High (70+)**: CEO is collaborative and open
- **Medium (50-69)**: CEO is professional but guarded
- **Low (30-49)**: CEO becomes terse and impatient
- **Critical (<30)**: Meeting ends

## License

MIT License

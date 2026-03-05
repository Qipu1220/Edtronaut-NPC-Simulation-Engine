"""
FastAPI Endpoints for Edtronaut NPC Simulation Engine.

Provides REST API endpoints for:
- POST /chat: Send message to NPC
- POST /session/new: Create new session
- GET /session/{id}/debug: Get debug state
- DELETE /session/{id}: End session
"""
import os
import uuid
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from .core.workflow import SimulationEngine
from .core.state import SimulationState

logger = logging.getLogger(__name__)


# Request/Response Models
class ChatRequest(BaseModel):
    """Request model for /chat endpoint."""
    user_input: str
    session_id: str


class ChatResponse(BaseModel):
    """Response model for /chat endpoint."""
    response: str
    session_id: str
    meeting_ended: bool
    debug_state: dict


class NewSessionResponse(BaseModel):
    """Response model for /session/new endpoint."""
    session_id: str
    message: str


class DebugResponse(BaseModel):
    """Response model for /session/{id}/debug endpoint."""
    session_id: str
    state: dict


# Global engine instance
_engine: Optional[SimulationEngine] = None


def load_persona() -> dict:
    """Load CEO persona from JSON file."""
    persona_path = Path(__file__).parent / "data" / "persona_ceo.json"
    with open(persona_path, "r") as f:
        return json.load(f)


def get_llm():
    """Get LLM instance based on environment configuration."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-pro"),
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
    else:
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
            api_key=os.getenv("OPENAI_API_KEY")
        )


def get_engine() -> SimulationEngine:
    """Get or create simulation engine instance."""
    global _engine
    if _engine is None:
        llm = get_llm()
        persona = load_persona()
        _engine = SimulationEngine(llm, persona)
        logger.info("Simulation engine initialized")
    return _engine


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Edtronaut NPC Simulation Engine",
        description="AI-powered NPC simulation for job training",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        """
        Send a message to the NPC CEO.

        The NPC will respond based on:
        - Current trust score
        - Emotional state
        - Director's behavioral directives
        - RAG context from knowledge base
        """
        try:
            engine = get_engine()
            result = engine.process_message(request.session_id, request.user_input)

            return ChatResponse(
                response=result["response"],
                session_id=request.session_id,
                meeting_ended=result["meeting_ended"],
                debug_state=engine.get_debug_state(request.session_id)
            )
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/session/new", response_model=NewSessionResponse)
    async def new_session():
        """
        Create a new simulation session.

        Returns a unique session ID for subsequent requests.
        """
        try:
            engine = get_engine()
            session_id = str(uuid.uuid4())
            engine.create_session(session_id)

            return NewSessionResponse(
                session_id=session_id,
                message="New session created. Trust score starts at 80."
            )
        except Exception as e:
            logger.error(f"Session creation error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/session/{session_id}/debug", response_model=DebugResponse)
    async def get_debug(session_id: str):
        """
        Get internal state for debugging.

        Returns trust score, emotional vector, and other metrics.
        """
        try:
            engine = get_engine()
            debug_state = engine.get_debug_state(session_id)

            return DebugResponse(
                session_id=session_id,
                state=debug_state
            )
        except Exception as e:
            logger.error(f"Debug error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/session/{session_id}")
    async def end_session(session_id: str):
        """
        End a simulation session.

        Removes session state from memory.
        """
        try:
            engine = get_engine()
            engine.end_session(session_id)

            return {"message": f"Session {session_id} ended"}
        except Exception as e:
            logger.error(f"Session end error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "service": "edtronaut-npc-engine"}

    return app

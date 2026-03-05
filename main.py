"""
Edtronaut NPC Simulation Engine - FastAPI Entry Point

Main entry point for the AI Co-worker Engine that simulates
interactions with NPC characters like the Gucci Group CEO.
"""
import os
import logging
from dotenv import load_dotenv

from src.npc_engine.api import create_app

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = create_app()


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))

    logger.info(f"Starting Edtronaut NPC Engine on {host}:{port}")
    uvicorn.run("main:app", host=host, port=port, reload=True)

"""
TripleTriple Gateway Server — Application Composition

Thin entry point that creates the FastAPI app and includes
the HTTP routes and WebSocket routers. All initialization
logic lives in `startup.py`.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .startup import lifespan
from .routes import router as http_router
from .websocket import router as ws_router
from ..version import get_version

# Also re-export key objects for backward compatibility
from .startup import agent, session_manager, dock, model_selector  # noqa: F401

app = FastAPI(
    title="TripleTriple Gateway",
    description="AI Agent Gateway — Control Plane",
    version=get_version(),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(http_router)
app.include_router(ws_router)

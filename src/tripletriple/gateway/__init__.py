"""
TripleTriple Gateway Package

Exports the FastAPI application instance.
"""

from .server import app  # noqa: F401

__all__ = ["app"]

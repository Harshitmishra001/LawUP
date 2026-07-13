"""LawUP API — FastAPI application entry point."""

from fastapi import FastAPI

app = FastAPI(
    title="LawUP",
    description="Agentic AI contract analysis system",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}

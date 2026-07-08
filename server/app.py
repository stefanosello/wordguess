"""The wordguess FastAPI application.

Right now the app is deliberately tiny: a single health-check endpoint so
we can prove the server boots and responds. Puzzle generation, rollout, and
scoring endpoints are added in later milestones (see the README API contract).

Run it with:

    uvicorn server.app:app --reload
"""

from fastapi import FastAPI

app = FastAPI(title="wordguess")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe: returns ``{"status": "ok"}`` when the server is up."""
    return {"status": "ok"}

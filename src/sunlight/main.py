"""Application entrypoint for the sunlight project."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from sunlight.web.routes import router as web_router


def create_app() -> FastAPI:
    """Build and configure the FastAPI application."""
    app = FastAPI(title="sunlight")
    app.mount("/static", StaticFiles(directory="src/sunlight/static"), name="static")
    app.include_router(web_router)
    return app


app = create_app()

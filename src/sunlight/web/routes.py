"""Web routes for the sunlight app."""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()
templates = Jinja2Templates(directory="src/sunlight/templates")


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Render the main visualization page."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"page_title": "Sunlight Scene"},
    )

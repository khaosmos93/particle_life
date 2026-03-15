# sunlight

Minimal first implementation for a Python + web 3D visualization project.

## What this step includes

- FastAPI backend serving a single web page.
- Three.js frontend scene rendering.
- A visible Sun object and a black wireframe Earth sphere.
- Orbit camera controls for scene inspection.

## Project layout

- `src/sunlight/main.py`: FastAPI app factory + static mounting.
- `src/sunlight/web/routes.py`: Web route for the landing page.
- `src/sunlight/templates/index.html`: Page shell.
- `src/sunlight/static/css/styles.css`: Basic dark theme and layout.
- `src/sunlight/static/js/scene.js`: Three.js scene construction and animation.
- `src/sunlight/simulation/scene.py`: Reusable scene object definitions for future modules.

## Run locally / in GitHub Codespaces

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src uvicorn sunlight.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open `http://localhost:8000` (or the forwarded Codespaces port URL).

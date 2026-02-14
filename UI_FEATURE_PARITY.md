# UI Feature Parity Checklist

## Source audit (`particle-life-app/`)
- `particle-life-app/` exists locally but is empty in this repository snapshot (no UI files, scripts, or controls to mirror).
- Because there are no reference UI assets in that folder, parity is based on currently active runtime features in this app's config API (`/api/config`) and simulation endpoints.

## Implemented feature map

| Feature | Implemented in our app | Notes |
|---|---|---|
| Dynamic control sections from backend schema (`simulation`, `render`, `random/seed`) | `src/particle_life/static/main.js` (`bindControl`, `buildUI`) | UI now renders from API metadata as single source of truth. |
| All numeric/select/toggle controls in config schema | `src/particle_life/static/main.js` | Includes world size, repel radius, max speed, point opacity, show HUD, seed, etc. |
| Preset load | `src/particle_life/static/main.js` + `/api/config/preset` | Uses backend preset list dynamically. |
| Reset and randomize seed actions | `src/particle_life/static/main.js` + `/api/config/reset`, `/api/config/randomize` | Fully wired and immediate. |
| Interaction matrix full editing (all cells) | `src/particle_life/static/main.js` (`Interaction Matrix` section) | Inline numeric cell editing and drag adjustment. |
| Matrix quick actions | `src/particle_life/static/main.js` | Presets: fully random, zero, identity, symmetric random. |
| Matrix copy/paste | `src/particle_life/static/main.js` | Validates dimensions and finite values. |
| Atomic matrix commit path | `src/particle_life/static/main.js` + `/api/config/update` | Matrix edits are drafted locally and committed as one update. |
| Validation / clamping of matrix and params | `src/particle_life/realtime.py` | Handles NaN/inf and clamps values to bounds. |
| Dev runtime assertions for matrix coherence | `src/particle_life/static/main.js` | Enabled via `?dev` query flag. |

## Deliberate omissions
- Placeholder menu bar entries (`File/View/Help`) removed because they were not connected to functionality.
- Removed dead/placeholder presets UI for `positions` and `types` that had no backend behavior.
- No extra tools (row/column paint helpers, import/export files) added since they are not present in local reference folder and not backed by API.

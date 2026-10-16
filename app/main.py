"""FastAPI web service for On-Device Crash Test."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from app.state import STORE, load_default_state

app = FastAPI(title="On-Device Crash Test")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class RunRequest(BaseModel):
    scenario_id: str = "robot_vacuum"
    conditions: list[str] = Field(default_factory=lambda: ["night"])
    n_images: int = 20
    mode: str = "live"


def _load_scenario_profile(scenario_id: str) -> dict[str, Any]:
    path = Path("scenarios") / f"{scenario_id}.yaml"
    if not path.exists():
        return {"id": scenario_id, "name": scenario_id, "status": "not yet run", "conditions": []}
    return yaml.safe_load(path.read_text())


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request, scenario: str = "robot_vacuum") -> Any:
    profile = _load_scenario_profile(scenario)
    fridge_profile = _load_scenario_profile("fridge_interior")
    fridge_showcase_exists = (Path("showcase") / "fridge_interior" / "state.json").exists()
    state = load_default_state(scenario)
    return TEMPLATES.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "scenario": profile,
            "fridge_available": fridge_showcase_exists,
            "fridge_status": fridge_profile.get("status", "not yet run"),
            "state": state,
        },
    )


@app.get("/state")
def get_state(run_id: str | None = None, scenario: str = "robot_vacuum") -> JSONResponse:
    if run_id:
        return JSONResponse(STORE.get_run(run_id))
    active_id = STORE.get_active_run_id()
    if active_id:
        return JSONResponse(STORE.get_run(active_id))
    return JSONResponse(load_default_state(scenario))


@app.post("/run")
def start_run(req: RunRequest) -> JSONResponse:
    active = STORE.get_active_run_id()
    if active:
        cur = STORE.get_run(active)
        if cur.get("phase") in ("evaluating", "diagnosing", "repairing", "re-evaluating"):
            raise HTTPException(status_code=409, detail="A live run is already in progress.")

    run_id = f"run_{uuid.uuid4().hex[:8]}"
    initial_state = load_default_state(req.scenario_id)
    initial_state.update(
        {
            "run_id": run_id,
            "scenario_id": req.scenario_id,
            "phase": "not yet run",
            "mode": req.mode,
            "n_images": req.n_images,
        }
    )
    STORE.set_run(run_id, initial_state, active=False)
    return JSONResponse({"run_id": run_id, "phase": initial_state["phase"]})


@app.get("/artifacts/{file_path:path}")
def get_artifact(file_path: str) -> FileResponse:
    allowed_roots = [Path("data").resolve(), Path("showcase").resolve(), Path("models").resolve()]
    candidate = (Path.cwd() / file_path).resolve()
    if not any(str(candidate).startswith(str(root)) for root in allowed_roots):
        raise HTTPException(status_code=403, detail="Access denied")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(str(candidate))

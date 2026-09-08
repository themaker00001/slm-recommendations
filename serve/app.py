"""Local test frontend for the relevance pipeline: a small FastAPI backend
serving a single-page UI, backed entirely by local processes -- the trained
bi-encoder checkpoint (serve/predict.py's RelevanceServer) for scoring, and a
local Ollama model for the "what would the teacher say" comparison. Nothing
in this file calls out to any cloud API.

Run:
    python serve/app.py
    open http://localhost:8000
"""
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve.predict import RelevanceServer, RELEVANCE_LABELS  # noqa: E402
from teacher.label_with_ollama import call_ollama, check_ollama_reachable  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "bi_encoder.pt"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="SLM Relevance Tester")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_server: RelevanceServer | None = None


def get_server() -> RelevanceServer | None:
    global _server
    if _server is None and CHECKPOINT_PATH.exists():
        _server = RelevanceServer(str(CHECKPOINT_PATH))
    return _server


class Item(BaseModel):
    item_id: str
    item_text: str


class ScoreRequest(BaseModel):
    query: str
    items: list[Item]


class TeacherRequest(BaseModel):
    query: str
    items: list[Item]
    model: str = "qwen3:14b"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status():
    server = get_server()
    return {
        "checkpoint_loaded": server is not None,
        "checkpoint_path": str(CHECKPOINT_PATH),
        "ollama_reachable": check_ollama_reachable(),
    }


@app.post("/api/score")
def score(req: ScoreRequest):
    server = get_server()
    if server is None:
        return {"error": f"No trained checkpoint at {CHECKPOINT_PATH}. "
                          f"Run model/train.py first."}
    results = []
    for item in req.items:
        label, relevance = server.score(req.query, item.item_id, item.item_text)
        results.append({
            "item_id": item.item_id,
            "item_text": item.item_text,
            "predicted_label": label,
            "label_name": RELEVANCE_LABELS[label],
            "relevance_score": relevance,
            "kept": label >= 1,
        })
    results.sort(key=lambda r: r["relevance_score"], reverse=True)
    return {"query": req.query, "results": results}


@app.post("/api/teacher")
def teacher(req: TeacherRequest):
    if not check_ollama_reachable():
        return {"error": "Ollama isn't reachable at localhost:11434. Run `ollama serve`."}
    results = []
    for item in req.items:
        try:
            out = call_ollama(req.model, req.query, item.item_text)
            results.append({
                "item_id": item.item_id,
                "item_text": item.item_text,
                "teacher_label": out["label"],
                "label_name": RELEVANCE_LABELS[out["label"]],
                "teacher_reason": out["reason"],
            })
        except Exception as exc:  # noqa: BLE001 -- surface the failure per-item, keep going
            results.append({
                "item_id": item.item_id,
                "item_text": item.item_text,
                "error": str(exc),
            })
    return {"query": req.query, "model": req.model, "results": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

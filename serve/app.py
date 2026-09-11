"""Local test frontend for the relevance pipeline: a small FastAPI backend
serving a single-page UI, backed entirely by local processes -- the trained
bi-encoder checkpoint (serve/predict.py's RelevanceServer) for scoring, and a
local Ollama model for the "what would the teacher say" comparison. Nothing
in this file calls out to any cloud API.

Run:
    python serve/app.py
    open http://localhost:8000
"""
import json
import re
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from serve.predict import RelevanceServer, RELEVANCE_LABELS  # noqa: E402
from teacher.label_with_ollama import call_ollama, check_ollama_reachable  # noqa: E402
from retrieval.semantic_retrieve import SemanticRetriever  # noqa: E402
from personalization.model import PersonalizationModel  # noqa: E402
from personalization.dataset import item_features, PERSONA_IDS  # noqa: E402
from personalization.simulate_users import PERSONAS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "bi_encoder.pt"
PERSONALIZATION_CHECKPOINT_PATH = ROOT / "checkpoints" / "personalization.pt"
STATIC_DIR = Path(__file__).resolve().parent / "static"
CATALOG_PATH = ROOT / "data" / "samples" / "demo_catalog.json"

app = FastAPI(title="SLM Relevance Tester")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_server: RelevanceServer | None = None
_semantic_retriever: SemanticRetriever | None = None
_personalization_model: PersonalizationModel | None = None
CATALOG = json.loads(CATALOG_PATH.read_text())
_CATALOG_PRICES = [item["price"] for item in CATALOG]
_PRICE_MIN, _PRICE_MAX = min(_CATALOG_PRICES), max(_CATALOG_PRICES)


def get_server() -> RelevanceServer | None:
    global _server
    if _server is None and CHECKPOINT_PATH.exists():
        _server = RelevanceServer(str(CHECKPOINT_PATH))
    return _server


def get_semantic_retriever() -> SemanticRetriever:
    global _semantic_retriever
    if _semantic_retriever is None:
        _semantic_retriever = SemanticRetriever(CATALOG)
    return _semantic_retriever


def get_personalization_model() -> PersonalizationModel | None:
    global _personalization_model
    if _personalization_model is None and PERSONALIZATION_CHECKPOINT_PATH.exists():
        import torch
        ckpt = torch.load(PERSONALIZATION_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        model = PersonalizationModel(ckpt["num_personas"], ckpt["num_categories"])
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _personalization_model = model
    return _personalization_model


def personalization_score(model: PersonalizationModel, persona_id: str, item: dict) -> float:
    import torch
    persona_idx = torch.tensor([PERSONA_IDS.index(persona_id)])
    features = torch.tensor([item_features(item["group"], item["price"], _PRICE_MIN, _PRICE_MAX)])
    return model.engagement_prob(persona_idx, features).item()


def item_text(item: dict) -> str:
    return f"{item['title']} | Brand: {item['brand']}"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def keyword_retrieve(query: str, limit: int = 12) -> list[dict]:
    """Stage 1 of the funnel: naive keyword retrieval, optimized for recall
    the way the blog describes -- it matches on shared word stems and doesn't
    know anything about intent, so it happily pulls in "salt & vinegar chips"
    for a "salt" query. This is the *un-filtered* candidate set that the SLM
    relevance model (stage 2) then has to clean up."""
    q_tokens = {t for t in tokenize(query) if len(t) >= 3}
    scored = []
    for item in CATALOG:
        i_tokens = [t for t in tokenize(f"{item['title']} {item['brand']} {item['category']}")
                    if len(t) >= 3]
        # Match in either direction (item token starts with query token, or
        # vice versa) so plural/singular mismatches work ("fruits" query vs
        # "Fruit" category, "salt" query vs "salted" item) -- the length>=3
        # floor on BOTH sides is what keeps this safe: without it, short
        # fragments like the "s" in "Meyer's" matched every query trivially.
        matches = sum(1 for qt in q_tokens for it in i_tokens
                       if it.startswith(qt) or qt.startswith(it))
        if matches > 0:
            scored.append((matches, item))
    scored.sort(key=lambda pair: -pair[0])
    return [item for _, item in scored[:limit]]


def hybrid_retrieve(query: str) -> list[dict]:
    """Stage 1 of the funnel, recall-oriented: unions naive keyword matches
    with dense semantic matches (retrieval/semantic_retrieve.py), so an item
    with no shared word stem -- "Paper Towels" for a "cleaning" query -- can
    still surface via meaning instead of being invisible to the pipeline
    before the relevance model ever gets a chance to judge it. Each result
    carries which method(s) found it, purely for the UI to show its work."""
    keyword_hits = {item["item_id"]: item for item in keyword_retrieve(query)}
    semantic_hits = {item["item_id"]: item for item, _ in get_semantic_retriever().retrieve(query)}

    merged = {}
    for item_id, item in {**keyword_hits, **semantic_hits}.items():
        found_by = []
        if item_id in keyword_hits:
            found_by.append("keyword")
        if item_id in semantic_hits:
            found_by.append("semantic")
        merged[item_id] = {**item, "found_by": found_by}
    return list(merged.values())


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


class SearchRequest(BaseModel):
    query: str
    use_filter: bool = True
    persona: str | None = None


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


@app.get("/api/catalog")
def catalog():
    groups = sorted({item["group"] for item in CATALOG})
    return {"catalog": CATALOG, "suggested_queries": groups}


@app.get("/api/personas")
def personas():
    return {
        "personas": [
            {"id": pid, "label": p["label"], "blurb": p["blurb"]}
            for pid, p in PERSONAS.items()
        ],
        "loaded": get_personalization_model() is not None,
    }


@app.post("/api/search")
def search(req: SearchRequest):
    """The full funnel in one call: hybrid (keyword + semantic) retrieval
    over the demo catalog, then (if a checkpoint is trained) the SLM
    relevance model scores and, when use_filter is on, filters/ranks the
    candidates -- the same pre-auction gate serve/predict.py demonstrates
    from the CLI. If a persona is given and the personalization model is
    trained, the surviving (relevant) results are additionally re-ranked by
    predicted engagement for that persona -- relevance decides what's
    eligible, personalization decides the order, matching the blog's own
    "relevance gates, a separate signal ranks" split."""
    candidates = hybrid_retrieve(req.query)
    server = get_server()
    persona_model = get_personalization_model() if req.persona else None

    results = []
    for item in candidates:
        text = item_text(item)
        entry = {**item, "item_text": text}
        if server is not None:
            label, relevance = server.score(req.query, item["item_id"], text)
            entry.update(predicted_label=label, label_name=RELEVANCE_LABELS[label],
                         relevance_score=relevance, kept=label >= 1)
        else:
            entry.update(predicted_label=None, label_name=None,
                         relevance_score=None, kept=True)

        if persona_model is not None and req.persona in PERSONA_IDS:
            entry["personalization_score"] = personalization_score(persona_model, req.persona, item)
        else:
            entry["personalization_score"] = None

        results.append(entry)

    if persona_model is not None and req.persona in PERSONA_IDS:
        # Relevance already decided eligibility; personalization only
        # reorders within that -- irrelevant items never get promoted back
        # by a persona liking their category.
        results.sort(key=lambda r: (r["kept"], r["personalization_score"] or 0), reverse=True)
    elif server is not None and req.use_filter:
        results.sort(key=lambda r: r["relevance_score"], reverse=True)

    return {
        "query": req.query,
        "use_filter": req.use_filter,
        "persona": req.persona,
        "checkpoint_loaded": server is not None,
        "personalization_loaded": persona_model is not None,
        "results": results,
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

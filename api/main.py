import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

load_dotenv()

from agent import agent
from agent.agent import ALLOWED_MODELS, DEFAULT_MODEL
from rag import retriever
from tools import salary_tool
from tools.constants import KNOWN_AREA_TYPES, KNOWN_CITIES
from tools.planner import DEFAULT_AREA_TYPE, compute_plan

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    salary_tool.is_loaded()
    retriever.warmup()
    yield


app = FastAPI(title="AI-Powered Financial Dream & Goal Planner", lifespan=lifespan)

METADATA_PATH = Path(__file__).parent.parent / "models" / "model_metadata.json"

MAX_GOAL_YEARS = 60


class GoalInput(BaseModel):
    goal_type: Literal["Marriage", "Car", "Home"]
    years: int = Field(ge=0, le=MAX_GOAL_YEARS)


class PlanRequest(BaseModel):
    age: int = Field(ge=15, le=70)
    city: str
    area_type: str = DEFAULT_AREA_TYPE
    education: str
    job_role: str
    savings_percentage: float = Field(ge=0, le=100)
    goals: list[GoalInput] = Field(min_length=1)

    @field_validator("city")
    @classmethod
    def city_must_be_known(cls, v):
        if v not in KNOWN_CITIES:
            raise ValueError(f"Unknown city: {v}")
        return v

    @field_validator("area_type")
    @classmethod
    def area_type_must_be_known(cls, v):
        if v not in KNOWN_AREA_TYPES:
            raise ValueError(f"Unknown area type: {v}")
        return v


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[dict] = Field(default_factory=list)
    model_id: str = DEFAULT_MODEL

    @field_validator("model_id")
    @classmethod
    def model_must_be_allowed(cls, v):
        if v not in ALLOWED_MODELS:
            raise ValueError(f"Unknown model: {v}. Choose one of {ALLOWED_MODELS}")
        return v


def _run_plan(req: PlanRequest):
    return compute_plan(
        age=req.age,
        city=req.city,
        education=req.education,
        job_role=req.job_role,
        savings_percentage=req.savings_percentage,
        area_type=req.area_type,
        goals=[g.model_dump() for g in req.goals],
    )


@app.post("/plan")
def plan(req: PlanRequest):
    return _run_plan(req)


@app.post("/recalculate")
def recalculate(req: PlanRequest):
    return _run_plan(req)


@app.post("/chat")
def chat(req: ChatRequest):
    return agent.run(req.message, req.history, req.model_id)


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    def generate():
        for event in agent.stream(req.message, req.history, req.model_id):
            yield json.dumps(event) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": salary_tool.is_loaded(),
        "vector_store_loaded": retriever.is_loaded(),
    }


@app.get("/models/metadata")
def models_metadata():
    if not METADATA_PATH.exists():
        return JSONResponse(status_code=404, content={"error": "model_metadata.json not found - run models/train_and_compare.py"})
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@app.exception_handler(ValueError)
def value_error_handler(request, exc):
    return JSONResponse(status_code=422, content={"error": str(exc)})


@app.exception_handler(Exception)
def generic_error_handler(request, exc):
    logger.exception(exc)
    return JSONResponse(status_code=500, content={"error": "Internal error"})

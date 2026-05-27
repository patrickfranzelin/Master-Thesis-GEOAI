import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text


PG_CONN = os.environ["PG_CONN"]
SCHEMA = os.environ.get("EVAL_SCHEMA", "src_google")
engine = create_engine(PG_CONN)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SemanticSentenceEvaluation(BaseModel):
    building_id: int
    sentence_quality: str


def ensure_table():
    with engine.begin() as conn:
        conn.execute(
            text(
                f"""
                create table if not exists {SCHEMA}.semantic_description_evaluation (
                    id serial primary key,
                    building_id integer not null,
                    sentence_quality text not null,
                    created_at timestamp default current_timestamp
                )
                """
            )
        )
        conn.execute(
            text(
                f"""
                create index if not exists idx_semantic_description_eval_building
                on {SCHEMA}.semantic_description_evaluation(building_id)
                """
            )
        )


@app.post("/save")
def save(e: SemanticSentenceEvaluation):
    try:
        ensure_table()
        with engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    insert into {SCHEMA}.semantic_description_evaluation (
                        building_id,
                        sentence_quality
                    )
                    values (
                        :building_id,
                        :sentence_quality
                    )
                    """
                ),
                {
                    "building_id": e.building_id,
                    "sentence_quality": e.sentence_quality,
                },
            )
        return {"status": "ok"}
    except Exception as exc:
        print("ERROR:", exc)
        return {"status": "error", "detail": str(exc)}


@app.get("/health")
def health():
    return {"status": "ok", "schema": SCHEMA}

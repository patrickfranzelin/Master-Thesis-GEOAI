import os
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from fastapi.middleware.cors import CORSMiddleware

PG_CONN = os.environ["PG_CONN"]
engine = create_engine(PG_CONN)

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Evaluation(BaseModel):
    building_id: int
    exists_label: bool | None = None
    improvement: str | None = None
    quality: str | None = None
    has_post: bool

@app.post("/save")
def save(e: Evaluation):
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO src.evaluation (
                        building_id,
                        exists_label,
                        improvement,
                        quality,
                        has_post
                    )
                    VALUES (
                        :building_id,
                        :exists_label,
                        :improvement,
                        :quality,
                        :has_post
                    )
                """),
                e.dict()
            )
        return {"status": "ok"}

    except Exception as ex:
        print("ERROR:", ex)
        return {"status": "error", "detail": str(ex)}
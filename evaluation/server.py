import os
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from fastapi.middleware.cors import CORSMiddleware
import json
from typing import List, Optional

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
    original: str
    original_error: Optional[List[str]] = None  # ← was str | None
    post_vs_sam: str
    post: str
    post_error: Optional[List[str]] = None       # ← was str | None
    has_post: bool

@app.post("/save")
def save(e: Evaluation):
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                     INSERT INTO src.evaluation (building_id,
                                                 original,
                                                 sam,
                                                 post,
                                                 tags,
                                                 has_post)
                     VALUES (:building_id,
                             :original,
                             :sam,
                             :post,
                             :tags,
                             :has_post)
                     """),
                {
                    "building_id": e.building_id,
                    "original": e.original,
                    "sam": e.post_vs_sam,  # map correctly
                    "post": e.post,
                    "tags": json.dumps({
                        "original_errors": e.original_error,
                        "post_errors": e.post_error
                    }),
                    "has_post": e.has_post
                }
            )
        return {"status": "ok"}

    except Exception as ex:
        print("ERROR:", ex)
        return {"status": "error", "detail": str(ex)}
#!/usr/bin/env python3
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import numpy as np, cv2, json, uvicorn
from src.mllm.internvl_client import InternVL3Points

app = FastAPI(title="InternVL Server", version="1.0")

# Load heavy model once
print(" Loading InternVL model ...")
model = InternVL3Points("OpenGVLab/InternVL2-8B", device="cuda", max_new_tokens=256)
print("✅ Model ready.")

@app.post("/infer_points")
async def infer_points(image: UploadFile, poly_json: str = Form(...)):
    poly_xy = json.loads(poly_json)["poly_xy"]
    img_bytes = await image.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    rgb = cv2.cvtColor(cv2.imdecode(nparr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    res = model.infer_points(rgb, poly_xy)
    return JSONResponse(content=res or {"error": "no result"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)


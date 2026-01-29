from openai import OpenAI
import base64
import json
import re

class MLLMClient:
    def __init__(self, base_url: str, api_key: str = "EMPTY"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate_points(self, img_b64: str) -> dict:
        prompt = """Precise inspector. See WHITE GRID (50px) + RED STAR = target HOUSE.
Output 4 INSIDE ROOF + 4 OUTSIDE points (GRID-snapped).
JSON ONLY: {"inside": [[x1,y1],...], "outside": [[x5,y5],...]}"""
        resp = self.client.chat.completions.create(
            model="internvl8b",
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt},
                                                  {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]}],
            temperature=0.0, max_tokens=256, response_format={"type": "json_object"}
        )
        return self._parse_safe(resp.choices[0].message.content)

    def _parse_safe(self, raw: str) -> dict:
        try:
            return json.loads(raw)
        except:
            cleaned = re.sub(r'```json?\s*|\s*```', '', raw).strip()
            try:
                return json.loads(cleaned)
            except:
                return {"inside": [], "outside": []}

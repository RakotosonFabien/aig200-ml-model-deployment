import os, json
from pathlib import Path
import torch
from fastapi import FastAPI, Header, HTTPException, Depends
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

load_dotenv(BASE_DIR / ".env")
APP_API_KEY = os.environ.get("API_KEY", "").strip()

def verify_api_key(x_api_key: str = Header(default="", alias="X-API-Key")):
    provided_key = x_api_key.strip()
    if not APP_API_KEY:
        raise HTTPException(status_code=500, detail="Server API key is not configured")
    if provided_key != APP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

class PredictIn(BaseModel):
    text: str

app = FastAPI(title="Sentiment API")

tokenizer = None
model = None
id2label = None

@app.on_event("startup")
def load_artifacts():
    global tokenizer, model, id2label
    with open(ARTIFACTS_DIR / "label_map.json", "r", encoding="utf-8") as f:
        maps = json.load(f)
    id2label = {int(k): v for k, v in maps["id2label"].items()} if isinstance(next(iter(maps["id2label"].keys())), str) else maps["id2label"]

    transformer_dir = str(ARTIFACTS_DIR / "transformer")
    tokenizer = AutoTokenizer.from_pretrained(transformer_dir)
    model = AutoModelForSequenceClassification.from_pretrained(transformer_dir)
    model.eval()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/predict", dependencies=[Depends(verify_api_key)])
def predict(payload: PredictIn):
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must be non-empty")

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs.pop("token_type_ids", None)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

    pred_id = int(torch.argmax(logits, dim=-1).item())
    return {
        "label_id": pred_id,
        "label": id2label[pred_id],
        "probabilities": probs
    }

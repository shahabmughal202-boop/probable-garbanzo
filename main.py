from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
model = pipeline("sentiment-analysis")

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running"}

@app.post("/predict")
def predict(text: str):
    result = model(text)
    return {"result": result}
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
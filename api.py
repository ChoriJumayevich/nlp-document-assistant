from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import pipeline
from capstone.modules.config import settings
from loguru import logger

app = FastAPI(title=settings.PROJECT_NAME)

# Load sentiment model once at startup
# Using a multilingual sentiment model
logger.info("Loading sentiment analysis model...")
sentiment_pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    """Predicts sentiment of the given text."""
    logger.info(f"Received sentiment request for text: {request.text[:50]}...")
    
    results = sentiment_pipe(request.text)
    if not results:
        return SentimentResponse(sentiment="unknown", confidence=0.0)
        
    res = results[0]
    # Map the model output to something readable
    # xlm-roberta returns Label_0 (neg), Label_1 (neu), Label_2 (pos)
    label_map = {"Label_0": "negative", "Label_1": "neutral", "Label_2": "positive"}
    label = label_map.get(res["label"], res["label"])
    
    return SentimentResponse(
        sentiment=label,
        confidence=res["score"]
    )

@app.get("/")
def home():
    return {"message": "Uzbek Document Assistant API is running"}

if __name__ == "__main__":
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
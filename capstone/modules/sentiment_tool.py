import requests
from loguru import logger
from capstone.modules.config import settings

class SentimentTool:
    """Tool for sentiment analysis that communicates with the FastAPI service."""
    
    def __init__(self):
        self.endpoint = f"http://{settings.API_HOST}:{settings.API_PORT}/predict"

    def run(self, text: str) -> str:
        """Calls the sentiment API and returns a formatted string."""
        logger.info(f"Running SentimentTool for text: {text[:50]}...")
        
        try:
            response = requests.post(self.endpoint, json={"text": text}, timeout=5)
            if response.status_code == 200:
                data = response.json()
                sentiment = data.get("sentiment", "unknown")
                confidence = data.get("confidence", 0.0)
                return f"Sentiment: {sentiment}, Confidence: {confidence:.2f}"
            else:
                logger.error(f"API returned status code {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to connect to Sentiment API: {e}")
            
        # Offline Fallback logic
        if settings.OFFLINE_FALLBACK:
            logger.warning("Offline Fallback: Returning mock sentiment.")
            return "Sentiment: neutral (fallback), Confidence: 0.50"
            
        return "Error: Could not determine sentiment."

sentiment_tool = SentimentTool()
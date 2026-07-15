import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Project configuration settings."""
    
    PROJECT_NAME: str = "Uzbek Document Assistant"
    
    # Offline fallback mode
    # If True, uses local models only and skips external API calls if they fail
    OFFLINE_FALLBACK: bool = True
    
    # Model configuration
    # Using a multilingual model suitable for Uzbek
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    LLM_MODEL_NAME: str = "google/gemma-2b-it" 
    
    # Vector DB configuration
    FAISS_INDEX_PATH: str = "capstone/vector_store/faiss_index"
    TOP_K: int = 3
    
    # FastAPI configuration
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    
    # Data paths
    KB_PATH: str = "capstone/data/uz_kb_mini.jsonl"

    class Config:
        env_file = ".env"

settings = Settings()
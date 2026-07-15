import os
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from loguru import logger
from capstone.modules.config import settings
from capstone.modules.kb_loader import kb_loader

class VectorStoreManager:
    """Manages the FAISS vector database for the Uzbek assistant."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=settings.EMBEDDING_MODEL_NAME)
        self.vector_db: Optional[FAISS] = None

    def index_kb(self, file_path: str = settings.KB_PATH):
        """Creates a FAISS index from the JSONL knowledge base."""
        raw_data = kb_loader.load_jsonl(file_path)
        if not raw_data:
            logger.warning("No data found to index.")
            return

        documents = [
            Document(page_content=item["text"], metadata={"category": item.get("category", "general")})
            for item in raw_data
        ]
        
        logger.info("Creating FAISS index...")
        self.vector_db = FAISS.from_documents(documents, self.embeddings)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(settings.FAISS_INDEX_PATH), exist_ok=True)
        self.vector_db.save_local(settings.FAISS_INDEX_PATH)
        logger.info(f"FAISS index saved to {settings.FAISS_INDEX_PATH}")

    def load_index(self):
        """Loads the FAISS index from disk."""
        if os.path.exists(settings.FAISS_INDEX_PATH):
            logger.info(f"Loading FAISS index from {settings.FAISS_INDEX_PATH}")
            self.vector_db = FAISS.load_local(
                settings.FAISS_INDEX_PATH, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            logger.warning("FAISS index not found on disk. Initializing empty.")
            # Optionally trigger index_kb() here if path exists
            self.index_kb()

    def search(self, query: str, k: int = settings.TOP_K) -> List[Document]:
        """Performs semantic search."""
        if not self.vector_db:
            self.load_index()
        
        return self.vector_db.similarity_search(query, k=k)

vector_store_manager = VectorStoreManager()
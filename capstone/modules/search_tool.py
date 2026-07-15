from typing import List
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from capstone.modules.vector_store import vector_store_manager
from capstone.modules.prompts import prompts
from capstone.modules.config import settings
from loguru import logger

class SearchTool:
    """Tool for semantic search and answer generation from documents."""
    
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.LLM_MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            temperature=0.1,
            top_p=0.9
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

    def run(self, query: str) -> str:
        """Executes the search and generates an answer."""
        logger.info(f"Running SearchTool for query: {query}")
        docs = vector_store_manager.search(query)
        context = "\n".join([doc.page_content for doc in docs])
        
        prompt_text = prompts.SEARCH_PROMPT.format(context=context, question=query)
        response = self.llm.invoke(prompt_text)
        
        # Simple cleanup if LLM returns the prompt
        if prompt_text in response:
            response = response.replace(prompt_text, "").strip()
            
        return response

search_tool = SearchTool()
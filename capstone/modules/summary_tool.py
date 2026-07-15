from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from capstone.modules.prompts import prompts
from capstone.modules.config import settings
from loguru import logger

class SummaryTool:
    """Tool for summarizing Uzbek text."""
    
    def __init__(self):
        # In a real production system, we'd share the LLM model instance
        # For simplicity in this capstone, we re-initialize or assume access
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
            max_new_tokens=150
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

    def run(self, text: str) -> str:
        """Summarizes the input text."""
        logger.info(f"Running SummaryTool")
        prompt_text = prompts.SUMMARY_PROMPT.format(text=text)
        response = self.llm.invoke(prompt_text)
        
        if prompt_text in response:
            response = response.replace(prompt_text, "").strip()
            
        return response

summary_tool = SummaryTool()
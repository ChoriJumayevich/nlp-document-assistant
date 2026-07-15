from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
from capstone.modules.prompts import prompts
from capstone.modules.config import settings
from loguru import logger

class IntentRouter:
    """Routes user input to the correct tool."""
    
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
            max_new_tokens=20
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)

    def route(self, user_input: str) -> str:
        """Determines the best tool for the input."""
        logger.info(f"Routing intent for input: {user_input}")
        prompt_text = prompts.ROUTER_PROMPT.format(user_input=user_input)
        response = self.llm.invoke(prompt_text).strip()
        
        if prompt_text in response:
            response = response.replace(prompt_text, "").strip()
            
        # Extract the first matching tool name
        for tool in ["SearchTool", "SummaryTool", "SentimentTool"]:
            if tool.lower() in response.lower():
                return tool
                
        return "SearchTool"  # Default

intent_router = IntentRouter()
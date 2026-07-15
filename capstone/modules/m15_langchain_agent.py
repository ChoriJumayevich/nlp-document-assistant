from typing import List
from capstone.modules.router import intent_router
from capstone.modules.search_tool import search_tool
from capstone.modules.summary_tool import summary_tool
from capstone.modules.sentiment_tool import sentiment_tool
from capstone.modules.utils import format_trace_step
from loguru import logger

class DocumentAssistantAgent:
    """The main agent that coordinates tools to answer Uzbek document queries."""
    
    def __init__(self):
        self.history: List[str] = []
        self.tools = {
            "SearchTool": search_tool,
            "SummaryTool": summary_tool,
            "SentimentTool": sentiment_tool
        }

    def run(self, user_message: str) -> str:
        """Processes the user message and returns the response."""
        logger.info(f"Agent received message: {user_message}")
        self.history = []  # Reset trace for new run
        self.history.append(format_trace_step("Input", user_message))
        
        # 1. Routing
        selected_tool_name = intent_router.route(user_message)
        self.history.append(format_trace_step("Router", f"Selected {selected_tool_name}"))
        
        # 2. Tool Execution
        tool = self.tools.get(selected_tool_name, search_tool)
        
        try:
            if selected_tool_name == "SearchTool":
                self.history.append(format_trace_step("Retriever", "FAISS similarity search"))
            
            result = tool.run(user_message)
            self.history.append(format_trace_step(selected_tool_name, "Execution successful"))
            self.history.append(format_trace_step("LLM", "Generating final response"))
            
            return result
        except Exception as e:
            logger.error(f"Error executing tool {selected_tool_name}: {e}")
            self.history.append(format_trace_step("Error", str(e)))
            return "Kechirasiz, so'rovingizni bajarishda xatolik yuz berdi."

    def last_trace(self) -> List[str]:
        """Returns the trace of the last execution."""
        return self.history

assistant_agent = DocumentAssistantAgent()
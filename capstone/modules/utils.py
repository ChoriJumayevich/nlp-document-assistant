import logging
import sys
from loguru import logger

def setup_logging():
    """Sets up global logging using Loguru."""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )

def format_trace_step(step_name: str, details: str = "") -> str:
    """Formats a single step in the execution trace."""
    if details:
        return f"[{step_name}] -> {details}"
    return f"[{step_name}]"

setup_logging()
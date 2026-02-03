"""Q&A module for document question answering with citations."""

from .retriever import Retriever
from .generator import (
    LLMGenerator,
    OllamaGenerator,
    OpenAIGenerator,
    AnthropicGenerator,
    GeminiGenerator,
    create_generator,
    get_default_generator
)
from .qa_service import QAService, create_qa_service
from .prompts import build_qa_prompt, build_summary_prompt
from .summary_service import SummaryService, get_summary_service

__all__ = [
    "Retriever",
    "LLMGenerator",
    "OllamaGenerator",
    "OpenAIGenerator",
    "AnthropicGenerator",
    "GeminiGenerator",
    "create_generator",
    "get_default_generator",
    "QAService",
    "create_qa_service",
    "build_qa_prompt",
    "build_summary_prompt",
    "SummaryService",
    "get_summary_service"
]

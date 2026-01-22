# vision_utils.py
"""
Vision model utilities for generating descriptions of images and tables.
Supports multiple backends:
- OpenAI GPT-4V
- Anthropic Claude
- Google Gemini (Cloud API)

"""
import base64
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class VisionModel(ABC):
    """Abstract base class for vision models."""

    # Default prompts for summary generation
    IMAGE_SUMMARY_PROMPT = (
        "Generate a summary of this image.\n\n"
        "Your summary should:\n"
        "1. Start with what type of visual this is (chart, graph, diagram, photo, etc.)\n"
        "2. Describe the main subject or topic in one sentence\n"
        "3. Extract key data points, numbers, or statistics visible\n"
        "4. Note any important text, labels, titles, or legends\n"
        "5. Highlight the main insight or takeaway\n\n"
        "Format as a clear summary in 3-5 sentences. Be specific with numbers."
    )

    TABLE_SUMMARY_PROMPT = (
        "Generate a summary of this table.\n\n"
        "Include:\n"
        "1. TABLE PURPOSE: What is this table about?\n"
        "2. STRUCTURE: Column headers and what they represent\n"
        "3. KEY DATA: Important values with specific numbers and units\n"
        "4. EXTREMES: Maximum and minimum values\n"
        "5. PATTERNS: Notable trends or comparisons\n"
        "6. TOTALS: Any summary rows (totals, averages)\n\n"
        "Format as a structured summary. Be precise with all numbers."
    )

    @abstractmethod
    def describe_image(self, image_path: str, prompt: str = None) -> str:
        """Generate a description for an image."""
        pass

    @abstractmethod
    def describe_table(self, table_image_path: str, prompt: str = None) -> str:
        """Generate a description/summary for a table image."""
        pass

    def summarize_image(self, image_path: str) -> str:
        """Generate a concise summary of an image."""
        return self.describe_image(image_path, self.IMAGE_SUMMARY_PROMPT)

    def summarize_table(self, table_image_path: str) -> str:
        """Generate a concise summary of a table."""
        return self.describe_table(table_image_path, self.TABLE_SUMMARY_PROMPT)


class OpenAIVisionModel(VisionModel):
    """OpenAI GPT-4V vision model."""

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.openai.com/v1/chat/completions"

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _call_api(self, image_path: str, prompt: str) -> str:
        """Call OpenAI API with image."""
        base64_image = self._encode_image(image_path)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 500
        }

        response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def describe_image(self, image_path: str, prompt: str = None) -> str:
        if prompt is None:
            prompt = (
                "Describe this image in detail. Include:\n"
                "1. What type of image it is (chart, diagram, photo, etc.)\n"
                "2. The main content and any text visible\n"
                "3. Key insights or data points if applicable\n"
                "Keep the description concise but informative (2-4 sentences)."
            )
        return self._call_api(image_path, prompt)

    def describe_table(self, table_image_path: str, prompt: str = None) -> str:
        if prompt is None:
            prompt = (
                "Analyze this table image and provide a summary. Include:\n"
                "1. What the table represents\n"
                "2. Column headers and their meaning\n"
                "3. Key data points or trends\n"
                "4. Any notable values (highest, lowest, totals)\n"
                "Keep the summary concise but capture all important information."
            )
        return self._call_api(table_image_path, prompt)


class AnthropicVisionModel(VisionModel):
    """Anthropic Claude vision model."""

    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model
        self.api_url = "https://api.anthropic.com/v1/messages"

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_media_type(self, image_path: str) -> str:
        """Get media type from file extension."""
        ext = Path(image_path).suffix.lower()
        media_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        return media_types.get(ext, "image/png")

    def _call_api(self, image_path: str, prompt: str) -> str:
        """Call Anthropic API with image."""
        base64_image = self._encode_image(image_path)
        media_type = self._get_media_type(image_path)

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": self.model,
            "max_tokens": 500,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image
                            }
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        }

        response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["content"][0]["text"]

    def describe_image(self, image_path: str, prompt: str = None) -> str:
        if prompt is None:
            prompt = (
                "Describe this image in detail. Include:\n"
                "1. What type of image it is (chart, diagram, photo, etc.)\n"
                "2. The main content and any text visible\n"
                "3. Key insights or data points if applicable\n"
                "Keep the description concise but informative (2-4 sentences)."
            )
        return self._call_api(image_path, prompt)

    def describe_table(self, table_image_path: str, prompt: str = None) -> str:
        if prompt is None:
            prompt = (
                "Analyze this table image and provide a summary. Include:\n"
                "1. What the table represents\n"
                "2. Column headers and their meaning\n"
                "3. Key data points or trends\n"
                "4. Any notable values (highest, lowest, totals)\n"
                "Keep the summary concise but capture all important information."
            )
        return self._call_api(table_image_path, prompt)


class GeminiVisionModel(VisionModel):
    """
    Google Gemini vision model (Cloud API).

    Supported models:
    - gemini-2.0-flash (default, fast and capable)
    - gemini-1.5-pro (more capable, slower)
    - gemini-1.5-flash (balanced)
    - gemini-2.0-flash-lite (fastest, lightweight)

    Requires: GOOGLE_API_KEY environment variable
    Get API key: https://makersuite.google.com/app/apikey
    """

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self.api_key = api_key
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_mime_type(self, image_path: str) -> str:
        """Get MIME type from file extension."""
        ext = Path(image_path).suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".heic": "image/heic",
            ".heif": "image/heif"
        }
        return mime_types.get(ext, "image/png")

    def _call_api(self, image_path: str, prompt: str) -> str:
        """Call Gemini API with image."""
        base64_image = self._encode_image(image_path)
        mime_type = self._get_mime_type(image_path)

        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_image
                            }
                        },
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "topK": 32,
                "topP": 1,
                "maxOutputTokens": 1024,
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }

        response = requests.post(
            f"{self.api_url}?key={self.api_key}",
            headers=headers,
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()

        # Extract text from Gemini response
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if parts and "text" in parts[0]:
                    return parts[0]["text"]

        raise ValueError(f"Unexpected Gemini API response format: {result}")

    def describe_image(self, image_path: str, prompt: str = None) -> str:
        if prompt is None:
            # Gemini-optimized prompt for image description
            prompt = (
                "You are analyzing an image from a PDF document. "
                "Provide a detailed but concise description that includes:\n\n"
                "1. IMAGE TYPE: Identify what kind of visual this is "
                "(chart, graph, diagram, photograph, illustration, screenshot, etc.)\n"
                "2. CONTENT: Describe the main subject matter and any visible text, labels, or annotations\n"
                "3. KEY DATA: If this contains data (charts/graphs), extract the key numbers, trends, or comparisons\n"
                "4. CONTEXT: What information does this image convey in a document context?\n\n"
                "Format your response as a flowing paragraph of 3-5 sentences that would help someone "
                "understand this image without seeing it. Be specific about numbers and text when visible."
            )
        return self._call_api(image_path, prompt)

    def describe_table(self, table_image_path: str, prompt: str = None) -> str:
        if prompt is None:
            # Gemini-optimized prompt for table analysis
            prompt = (
                "You are analyzing a table image extracted from a PDF document. "
                "Provide a comprehensive summary that includes:\n\n"
                "1. TABLE PURPOSE: What does this table represent or compare?\n"
                "2. STRUCTURE: List the column headers and describe what each column contains\n"
                "3. ROW CATEGORIES: What are the row labels or categories?\n"
                "4. KEY FINDINGS:\n"
                "   - Highest/lowest values and where they occur\n"
                "   - Notable patterns or trends\n"
                "   - Any totals, averages, or summary rows\n"
                "5. DATA EXTRACTION: List 3-5 specific data points with their exact values\n\n"
                "Format as a structured paragraph that captures all essential information. "
                "Be precise with numbers and include units where visible."
            )
        return self._call_api(table_image_path, prompt)


class OllamaVisionModel(VisionModel):
    """
    Local vision model via Ollama.

    Supported open-source vision models:
    - llava (LLaVA 7B - default, good balance)
    - llava:13b (LLaVA 13B - more capable)
    - llava:34b (LLaVA 34B - most capable)
    - llava-llama3 (LLaVA with Llama 3 base)
    - bakllava (BakLLaVA - fine-tuned LLaVA)
    - moondream (Moondream2 - lightweight, fast)
    - minicpm-v (MiniCPM-V - efficient)

    Setup:
    1. Install Ollama: https://ollama.ai
    2. Start server: ollama serve
    3. Pull model: ollama pull llava
    """

    def __init__(
        self,
        model: str = "llava",
        base_url: str = "http://localhost:11434",
        timeout: int = 120
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_url = f"{self.base_url}/api/generate"

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _call_api(self, image_path: str, prompt: str) -> str:
        """Call Ollama API with image."""
        base64_image = self._encode_image(image_path)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 512
            }
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Ensure Ollama is running: 'ollama serve'"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Ollama request timed out after {self.timeout}s. "
                f"Try a smaller/faster model like 'moondream'"
            )

    def describe_image(self, image_path: str, prompt: str = None) -> str:
        if prompt is None:
            # Optimized prompt for local models (shorter, clearer)
            prompt = (
                "Describe this image from a PDF document.\n"
                "Include: image type, main content, any visible text, and key data points.\n"
                "Be concise but informative (3-4 sentences)."
            )
        return self._call_api(image_path, prompt)

    def describe_table(self, table_image_path: str, prompt: str = None) -> str:
        if prompt is None:
            # Optimized prompt for local models
            prompt = (
                "Analyze this table image from a PDF.\n"
                "Include: what the table shows, column headers, key values, and notable patterns.\n"
                "Extract specific numbers when visible. Be concise but complete."
            )
        return self._call_api(table_image_path, prompt)

    @staticmethod
    def list_available_models(base_url: str = "http://localhost:11434") -> List[str]:
        """List vision-capable models available in Ollama."""
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=10)
            response.raise_for_status()
            models = response.json().get("models", [])

            # Filter for known vision models
            vision_models = []
            vision_keywords = ["llava", "bakllava", "moondream", "minicpm", "cogvlm", "qwen"]

            for model in models:
                name = model.get("name", "").lower()
                if any(kw in name for kw in vision_keywords):
                    vision_models.append(model.get("name"))

            return vision_models
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            return []


class Gemma3VisionModel(VisionModel):
    """
    Gemma 3 4B Vision Model - supports both local (Ollama) and API modes.

    Switch between modes with a single parameter:
    - mode="local": Uses Ollama (free, runs on your machine)
    - mode="api": Uses Google AI API (requires API key)

    Setup for LOCAL mode:
        1. Install Ollama: https://ollama.ai
        2. Pull model: ollama pull gemma3:4b
        3. Start server: ollama serve

    Setup for API mode:
        1. Get API key: https://makersuite.google.com/app/apikey
        2. Pass api_key parameter

    Usage:
        # Local mode (default)
        model = Gemma3VisionModel(mode="local")

        # API mode
        model = Gemma3VisionModel(mode="api", api_key="your-google-api-key")
    """

    def __init__(
        self,
        mode: str = "local",
        api_key: str = None,
        ollama_base_url: str = "http://localhost:11434",
        timeout: int = 120
    ):
        """
        Initialize Gemma 3 4B vision model.

        Args:
            mode: "local" for Ollama, "api" for Google AI API
            api_key: Google API key (required for api mode)
            ollama_base_url: Ollama server URL (for local mode)
            timeout: Request timeout in seconds
        """
        self.mode = mode.lower()
        self.timeout = timeout

        if self.mode == "api":
            if not api_key:
                raise ValueError("api_key is required for API mode")
            self.api_key = api_key
            self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemma-3-4b-it:generateContent"
            logger.info("Gemma3VisionModel: Using Google AI API mode")
        elif self.mode == "local":
            self.ollama_base_url = ollama_base_url.rstrip("/")
            self.ollama_api_url = f"{self.ollama_base_url}/api/generate"
            self.model_name = "gemma3:4b"
            logger.info(f"Gemma3VisionModel: Using local Ollama mode ({self.model_name})")
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'local' or 'api'")

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get_mime_type(self, image_path: str) -> str:
        """Get MIME type from file extension."""
        ext = Path(image_path).suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        return mime_types.get(ext, "image/png")

    def _call_local(self, image_path: str, prompt: str) -> str:
        """Call Ollama API with image (local mode)."""
        base64_image = self._encode_image(image_path)

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [base64_image],
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 512
            }
        }

        try:
            response = requests.post(
                self.ollama_api_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.ollama_base_url}. "
                f"Ensure Ollama is running: 'ollama serve' and model is pulled: 'ollama pull gemma3:4b'"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(f"Ollama request timed out after {self.timeout}s")

    def _call_api(self, image_path: str, prompt: str) -> str:
        """Call Google AI API with image (API mode)."""
        base64_image = self._encode_image(image_path)
        mime_type = self._get_mime_type(image_path)

        headers = {"Content-Type": "application/json"}

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_image
                            }
                        },
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "topK": 32,
                "topP": 1,
                "maxOutputTokens": 1024
            }
        }

        response = requests.post(
            f"{self.api_url}?key={self.api_key}",
            headers=headers,
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()
        result = response.json()

        # Extract text from response
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if parts and "text" in parts[0]:
                    return parts[0]["text"]

        raise ValueError(f"Unexpected API response format: {result}")

    def _call(self, image_path: str, prompt: str) -> str:
        """Route to appropriate backend based on mode."""
        if self.mode == "local":
            return self._call_local(image_path, prompt)
        else:
            return self._call_api(image_path, prompt)

    # Gemma 3 optimized prompts (more detailed for better results)
    IMAGE_SUMMARY_PROMPT = (
        "Generate a summary of this image.\n\n"
        "Your summary should:\n"
        "1. Start with what type of visual this is (chart, graph, diagram, photo, illustration, etc.)\n"
        "2. Describe the main subject or topic in one sentence\n"
        "3. Extract and list any key data points, numbers, or statistics visible\n"
        "4. Note any important text, labels, titles, or legends\n"
        "5. Highlight the main insight or takeaway from this image\n\n"
        "Format your response as a clear, readable summary in 3-5 sentences. "
        "Focus on information that would be useful for someone who cannot see the image. "
        "Be specific with numbers and percentages when visible."
    )

    TABLE_SUMMARY_PROMPT = (
        "Generate a summary of this table.\n\n"
        "Your summary should include:\n"
        "1. TABLE TITLE/PURPOSE: What is this table about? (one sentence)\n"
        "2. STRUCTURE: Number of columns and rows, column headers\n"
        "3. KEY DATA POINTS: List the most important values with their context\n"
        "   - Include specific numbers, percentages, and units\n"
        "   - Highlight maximum and minimum values\n"
        "4. PATTERNS & TRENDS: Any notable patterns, comparisons, or trends\n"
        "5. SUMMARY ROW: If there are totals or averages, include them\n\n"
        "Format as a structured summary that captures all essential information. "
        "Someone reading this summary should understand the table's content without seeing it. "
        "Be precise with all numerical values."
    )

    def describe_image(self, image_path: str, prompt: str = None) -> str:
        if prompt is None:
            prompt = self.IMAGE_SUMMARY_PROMPT
        return self._call(image_path, prompt)

    def describe_table(self, table_image_path: str, prompt: str = None) -> str:
        if prompt is None:
            prompt = self.TABLE_SUMMARY_PROMPT
        return self._call(table_image_path, prompt)

    def summarize_image(self, image_path: str) -> str:
        """Generate a concise summary of an image."""
        return self._call(image_path, self.IMAGE_SUMMARY_PROMPT)

    def summarize_table(self, table_image_path: str) -> str:
        """Generate a concise summary of a table."""
        return self._call(table_image_path, self.TABLE_SUMMARY_PROMPT)


class FallbackVisionModel(VisionModel):
    """
    Fallback model when no vision API is available.
    Uses basic metadata and markdown content for tables.
    """

    def describe_image(self, image_path: str, prompt: str = None) -> str:
        """Return basic description based on filename."""
        path = Path(image_path)
        filename = path.stem

        if "full" in filename:
            return f"Full page scan image from {filename}"
        elif "table" in filename:
            return f"Table image extracted from document: {filename}"
        else:
            return f"Image extracted from document: {filename}"

    def describe_table(self, table_image_path: str, markdown_content: str = None, prompt: str = None) -> str:
        """Use markdown content as description if available."""
        if markdown_content:
            # Create a summary from markdown
            lines = markdown_content.strip().split("\n")
            if lines:
                header = lines[0] if lines else ""
                row_count = len([l for l in lines if l.strip() and not l.startswith("|--")])
                return f"Table with approximately {row_count} rows. Headers: {header[:200]}"

        path = Path(table_image_path)
        return f"Table extracted from document: {path.stem}"


class VisionProcessor:
    """
    High-level processor for handling vision tasks.
    Manages model selection and batch processing.

    Supported backends:
    - Gemma 3 4B (local via Ollama OR cloud API) - RECOMMENDED
    - Anthropic Claude (cloud)
    - OpenAI GPT-4V (cloud)
    - Google Gemini (cloud)
    - Ollama (local - LLaVA, etc.)
    - Fallback (no API)
    """

    def __init__(
        self,
        model: VisionModel = None,
        # Gemma 3 configuration (recommended)
        use_gemma3: bool = False,
        gemma3_mode: str = "local",  # "local" or "api"
        # Other API keys
        openai_api_key: str = None,
        anthropic_api_key: str = None,
        google_api_key: str = None,
        ollama_model: str = None,
        ollama_base_url: str = "http://localhost:11434"
    ):
        """
        Initialize vision processor with preferred model.

        Priority (when no explicit model provided):
        1. Explicitly provided model
        2. Gemma 3 4B (if use_gemma3=True)
        3. Anthropic Claude (if ANTHROPIC_API_KEY)
        4. OpenAI GPT-4V (if OPENAI_API_KEY)
        5. Google Gemini (if GOOGLE_API_KEY)
        6. Ollama local (if ollama_model specified)
        7. Fallback (no API needed)

        Args:
            model: Explicit VisionModel instance
            use_gemma3: Use Gemma 3 4B model
            gemma3_mode: "local" (Ollama) or "api" (Google AI)
            openai_api_key: OpenAI API key
            anthropic_api_key: Anthropic API key
            google_api_key: Google API key (for Gemini or Gemma3 API mode)
            ollama_model: Ollama model name (e.g., "llava", "moondream")
            ollama_base_url: Ollama server URL
        """
        if model:
            self.model = model
            self.model_name = "custom"
        elif use_gemma3:
            # Gemma 3 4B - single parameter to switch between local and API
            self.model = Gemma3VisionModel(
                mode=gemma3_mode,
                api_key=google_api_key if gemma3_mode == "api" else None,
                ollama_base_url=ollama_base_url
            )
            self.model_name = f"gemma3:{gemma3_mode}"
            print(f"VisionProcessor: Using Gemma 3 4B vision model ({gemma3_mode} mode)")
        elif anthropic_api_key:
            self.model = AnthropicVisionModel(anthropic_api_key)
            self.model_name = "anthropic"
            print("VisionProcessor: Using Anthropic Claude vision model")
        elif openai_api_key:
            self.model = OpenAIVisionModel(openai_api_key)
            self.model_name = "openai"
            print("VisionProcessor: Using OpenAI GPT-4V vision model")
        elif google_api_key:
            self.model = GeminiVisionModel(google_api_key)
            self.model_name = "gemini"
            print("VisionProcessor: Using Google Gemini vision model")
        elif ollama_model:
            self.model = OllamaVisionModel(
                model=ollama_model,
                base_url=ollama_base_url
            )
            self.model_name = f"ollama:{ollama_model}"
            print(f"VisionProcessor: Using Ollama local model ({ollama_model})")
        else:
            self.model = FallbackVisionModel()
            self.model_name = "fallback"
            print("VisionProcessor: Using fallback model (no vision API configured)")

    def process_image(self, image_path: str) -> str:
        """Process a single image and return description."""
        try:
            return self.model.describe_image(image_path)
        except Exception as e:
            print(f"Warning: Vision API failed for {image_path}: {e}")
            return FallbackVisionModel().describe_image(image_path)

    def process_table(self, table_image_path: str, markdown_fallback: str = None) -> str:
        """Process a table image and return description."""
        try:
            return self.model.describe_table(table_image_path)
        except Exception as e:
            print(f"Warning: Vision API failed for {table_image_path}: {e}")
            return FallbackVisionModel().describe_table(table_image_path, markdown_fallback)

    def summarize_image(self, image_path: str) -> str:
        """Generate a summary of an image."""
        try:
            return self.model.summarize_image(image_path)
        except Exception as e:
            print(f"Warning: Vision API failed for {image_path}: {e}")
            return FallbackVisionModel().describe_image(image_path)

    def summarize_table(self, table_image_path: str) -> str:
        """Generate a summary of a table."""
        try:
            return self.model.summarize_table(table_image_path)
        except Exception as e:
            print(f"Warning: Vision API failed for {table_image_path}: {e}")
            return FallbackVisionModel().describe_table(table_image_path)

    def process_blocks(self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process all blocks and add vision-generated descriptions.
        Modifies blocks in place and returns them.
        """
        for block in blocks:
            if block["type"] == "image" and block.get("image_link"):
                if not block.get("content"):
                    block["content"] = self.process_image(block["image_link"])

            elif block["type"] == "table" and block.get("table_link"):
                # Use vision model to describe table, with markdown as fallback
                markdown_fallback = block.get("content")
                block["content"] = self.process_table(
                    block["table_link"],
                    markdown_fallback
                )

        return blocks

# vision_utils.py
"""
Vision model utilities for generating descriptions of images and tables.
Supports multiple backends: OpenAI GPT-4V, Anthropic Claude, or local models.
"""
import base64
import requests
from pathlib import Path
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod


class VisionModel(ABC):
    """Abstract base class for vision models."""

    @abstractmethod
    def describe_image(self, image_path: str, prompt: str = None) -> str:
        """Generate a description for an image."""
        pass

    @abstractmethod
    def describe_table(self, table_image_path: str, prompt: str = None) -> str:
        """Generate a description/summary for a table image."""
        pass


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
    """

    def __init__(
        self,
        model: VisionModel = None,
        openai_api_key: str = None,
        anthropic_api_key: str = None
    ):
        """
        Initialize vision processor with preferred model.

        Priority:
        1. Explicitly provided model
        2. Anthropic (if API key provided)
        3. OpenAI (if API key provided)
        4. Fallback (no API needed)
        """
        if model:
            self.model = model
        elif anthropic_api_key:
            self.model = AnthropicVisionModel(anthropic_api_key)
            print("VisionProcessor: Using Anthropic Claude vision model")
        elif openai_api_key:
            self.model = OpenAIVisionModel(openai_api_key)
            print("VisionProcessor: Using OpenAI GPT-4V vision model")
        else:
            self.model = FallbackVisionModel()
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

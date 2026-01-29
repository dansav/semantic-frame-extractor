import base64
import re
from abc import ABC, abstractmethod

import cv2
import numpy as np
from openai import OpenAI


class BaseMatcher(ABC):
    """Abstract base class for frame matchers."""

    @abstractmethod
    def match(self, image: np.ndarray, query: str) -> float:
        """
        Check if an image matches the query.

        Args:
            image: BGR image as numpy array (OpenCV format)
            query: Text description to match against

        Returns:
            Confidence score between 0.0 and 1.0
        """
        pass

    @abstractmethod
    def match_batch(self, images: list[np.ndarray], query: str) -> list[float]:
        """
        Check if multiple images match the query.

        Args:
            images: List of BGR images as numpy arrays
            query: Text description to match against

        Returns:
            List of confidence scores between 0.0 and 1.0
        """
        pass


def encode_image_base64(image: np.ndarray, format: str = ".jpg") -> str:
    """Convert a numpy image array to base64 string."""
    _, buffer = cv2.imencode(format, image)
    return base64.b64encode(buffer).decode("utf-8")


class GenerationMatcher(BaseMatcher):
    """
    Matcher using vision LLM generation (chat completions).

    Works with LM Studio's OpenAI-compatible API.
    """

    SYSTEM_PROMPT = """You are an image analysis assistant. Your task is to determine if an image matches a given description.

Analyze the image carefully and respond with ONLY a confidence score between 0 and 100, where:
- 0 means the described object/scene is definitely NOT present
- 100 means the described object/scene is definitely present and clearly visible

Respond with just the number, nothing else."""

    BATCH_SYSTEM_PROMPT = """You are an image analysis assistant. Your task is to determine if each image matches a given description.

Analyze each image carefully and respond with ONLY confidence scores between 0 and 100 for each image, where:
- 0 means the described object/scene is definitely NOT present
- 100 means the described object/scene is definitely present and clearly visible

Respond with just the numbers separated by commas, in the same order as the images. Example: 85,0,42,100"""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "qwen/qwen3-vl-4b",
        api_key: str = "lm-studio",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def match(self, image: np.ndarray, query: str) -> float:
        """Check if a single image matches the query."""
        image_b64 = encode_image_base64(image)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                        },
                        {
                            "type": "text",
                            "text": f"Does this image contain: {query}",
                        },
                    ],
                },
            ],
            max_tokens=10,
            temperature=0.1,
        )

        return self._parse_score(response.choices[0].message.content)

    def match_batch(self, images: list[np.ndarray], query: str) -> list[float]:
        """Check if multiple images match the query in a single request."""
        if not images:
            return []

        if len(images) == 1:
            return [self.match(images[0], query)]

        # Build content with multiple images
        content = []
        for i, image in enumerate(images):
            image_b64 = encode_image_base64(image)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                }
            )

        content.append(
            {
                "type": "text",
                "text": f"For each of the {len(images)} images above (in order), does it contain: {query}",
            }
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.BATCH_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            max_tokens=50,
            temperature=0.1,
        )

        return self._parse_batch_scores(
            response.choices[0].message.content, len(images)
        )

    def _parse_score(self, text: str | None) -> float:
        """Parse a single confidence score from LLM response."""
        if not text:
            return 0.0

        # Extract first number from response
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        if numbers:
            score = float(numbers[0])
            # Normalize to 0-1 range
            return min(1.0, max(0.0, score / 100.0))

        return 0.0

    def _parse_batch_scores(self, text: str | None, expected_count: int) -> list[float]:
        """Parse multiple confidence scores from LLM response."""
        if not text:
            return [0.0] * expected_count

        # Extract all numbers from response
        numbers = re.findall(r"\d+(?:\.\d+)?", text)
        scores = []

        for num_str in numbers[:expected_count]:
            score = float(num_str)
            scores.append(min(1.0, max(0.0, score / 100.0)))

        # Pad with zeros if we didn't get enough scores
        while len(scores) < expected_count:
            scores.append(0.0)

        return scores


class EmbeddingMatcher(BaseMatcher):
    """
    Matcher using vision-language embeddings.

    Requires a model that supports multimodal embeddings (e.g., Qwen3-VL-Embedding).
    This is typically faster than generation-based matching.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "qwen.qwen3-vl-embedding-2b",
        api_key: str = "lm-studio",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self._query_cache: dict[str, list[float]] = {}

    def _get_text_embedding(self, text: str) -> list[float]:
        """Get embedding for text query (cached)."""
        if text not in self._query_cache:
            response = self.client.embeddings.create(model=self.model, input=text)
            self._query_cache[text] = response.data[0].embedding
        return self._query_cache[text]

    def _get_image_embedding(self, image: np.ndarray) -> list[float]:
        """Get embedding for an image."""
        image_b64 = encode_image_base64(image)

        # Note: This API format may need adjustment based on LM Studio's
        # actual multimodal embedding endpoint
        response = self.client.embeddings.create(
            model=self.model,
            input=[{"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_b64}"}],
        )
        return response.data[0].embedding

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        similarity = dot_product / (norm_a * norm_b)
        # Convert from [-1, 1] to [0, 1] range
        return (similarity + 1) / 2

    def match(self, image: np.ndarray, query: str) -> float:
        """Check if an image matches the query using embeddings."""
        text_emb = self._get_text_embedding(query)
        image_emb = self._get_image_embedding(image)
        return self._cosine_similarity(text_emb, image_emb)

    def match_batch(self, images: list[np.ndarray], query: str) -> list[float]:
        """Check multiple images against the query."""
        if not images:
            return []

        text_emb = self._get_text_embedding(query)
        scores = []

        for image in images:
            image_emb = self._get_image_embedding(image)
            scores.append(self._cosine_similarity(text_emb, image_emb))

        return scores


class TransformersEmbeddingMatcher(BaseMatcher):
    """
    Matcher using Qwen3-VL-Embedding directly via transformers.

    Downloads the model from Hugging Face on first use.
    Much faster than API-based approaches after initial load.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        max_pixels: int = 1_000_000,
    ):
        from PIL import Image
        from .vendor.qwen3_vl_embedding import Qwen3VLEmbedder

        self.Image = Image
        self.embedder = Qwen3VLEmbedder(model_name_or_path=model_name, max_pixels=max_pixels)
        self._query_cache: dict[str, np.ndarray] = {}

    def _numpy_to_pil(self, image: np.ndarray):
        """Convert BGR numpy array to PIL Image."""
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.Image.fromarray(rgb)

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text query (cached)."""
        if text not in self._query_cache:
            embeddings = self.embedder.process([{"text": text}])
            self._query_cache[text] = embeddings[0].cpu().numpy()
        return self._query_cache[text]

    def match(self, image: np.ndarray, query: str) -> float:
        """Check if an image matches the query using embeddings."""
        text_emb = self._get_text_embedding(query)
        pil_image = self._numpy_to_pil(image)

        # Get image embedding
        image_emb = self.embedder.process([{"image": pil_image}])[0].cpu().numpy()

        # Embeddings are already normalized, so dot product = cosine similarity
        similarity = float(np.dot(text_emb, image_emb))
        # Convert from [-1, 1] to [0, 1] range
        return (similarity + 1) / 2

    def match_batch(self, images: list[np.ndarray], query: str) -> list[float]:
        """Check multiple images against the query."""
        if not images:
            return []

        text_emb = self._get_text_embedding(query)
        pil_images = [self._numpy_to_pil(img) for img in images]

        # Process all images in one batch
        image_inputs = [{"image": img} for img in pil_images]
        image_embeddings = self.embedder.process(image_inputs).cpu().numpy()

        # Compute similarities (embeddings are normalized)
        similarities = image_embeddings @ text_emb
        # Convert from [-1, 1] to [0, 1] range
        return [(float(s) + 1) / 2 for s in similarities]
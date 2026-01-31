import numpy as np
from openai import OpenAI

from .base import BaseMatcher, encode_image_base64


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
        api_key: str = "not-needed",
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

        # Note: This API format may need adjustment based on the backend's
        # multimodal embedding endpoint implementation
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

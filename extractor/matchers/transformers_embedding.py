import numpy as np
from PIL import Image

from .base import BaseMatcher


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
        from ..vendor.qwen3_vl_embedding import Qwen3VLEmbedder

        self.embedder = Qwen3VLEmbedder(model_name_or_path=model_name, max_pixels=max_pixels)
        self._query_cache: dict[str, np.ndarray] = {}

    def _numpy_to_pil(self, image: np.ndarray):
        """Convert RGB numpy array to PIL Image."""
        return Image.fromarray(image)

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

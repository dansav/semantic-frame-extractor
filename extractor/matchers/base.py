import base64
import io
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class BaseMatcher(ABC):
    """Abstract base class for frame matchers."""

    @abstractmethod
    def match(self, image: np.ndarray, query: str) -> float:
        """
        Check if an image matches the query.

        Args:
            image: RGB image as numpy array
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
            images: List of RGB images as numpy arrays
            query: Text description to match against

        Returns:
            List of confidence scores between 0.0 and 1.0
        """
        pass


def encode_image_base64(image: np.ndarray) -> str:
    """Convert an RGB numpy image array to base64 JPEG string."""
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

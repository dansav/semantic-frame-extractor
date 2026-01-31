import re

import numpy as np
from openai import OpenAI, APIStatusError

from .base import BaseMatcher, encode_image_base64


class ImageProcessingError(Exception):
    """Raised when the backend fails to process images."""
    pass


class GenerationMatcher(BaseMatcher):
    """
    Matcher using vision LLM generation (chat completions).

    Works with any OpenAI-compatible API.
    """

    SYSTEM_PROMPT = """You are an image analysis assistant. Your task is to determine if an image matches a given description.

Analyze the image carefully and respond with ONLY a confidence score between 0 and 100, where:
- 0 means the described object/scene is definitely NOT present
- 100 means the described object/scene is definitely present and clearly visible

Respond with just the number, nothing else."""

    BATCH_SYSTEM_PROMPT = """You are an image analysis assistant. Evaluate EACH image INDEPENDENTLY to determine if it matches a given description.

For each image, respond with a confidence score between 0 and 100:
- 0 means the described object/scene is definitely NOT present
- 100 means the described object/scene is definitely present and clearly visible

IMPORTANT: If NONE of the images match, respond with all zeros. Do NOT pick a "best match" - only give high scores to images that actually contain what is described.

Respond with just the numbers separated by commas, in the same order as the images.
Examples: 85,0,42,100 or 0,0,0,0,0 or 0,0,75,0,0"""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "qwen/qwen3-vl-4b",
        api_key: str = "not-needed",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def match(self, image: np.ndarray, query: str) -> float:
        """Check if a single image matches the query."""
        image_b64 = encode_image_base64(image)

        try:
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
        except APIStatusError as e:
            self._handle_api_error(e, 1)

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

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.BATCH_SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                max_tokens=50,
                temperature=0.1,
            )
        except APIStatusError as e:
            self._handle_api_error(e, len(images))

        return self._parse_batch_scores(
            response.choices[0].message.content, len(images)
        )

    def _handle_api_error(self, error: APIStatusError, batch_size: int) -> None:
        """Handle API errors with helpful suggestions."""
        error_body = getattr(error, "body", None)
        if isinstance(error_body, dict):
            error_msg = error_body.get("error", str(error))
        else:
            error_msg = str(error_body or error)

        if "failed to process image" in str(error_msg).lower():
            suggestions = [
                "Try reducing --batch-size (currently {})".format(batch_size),
                "Increase context length in the LLM backend settings",
                "Try a smaller image resolution with --max-pixels",
            ]
            raise ImageProcessingError(
                f"Backend failed to process image(s). Suggestions:\n"
                + "\n".join(f"  - {s}" for s in suggestions)
            ) from error

        # Re-raise other errors as-is
        raise

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

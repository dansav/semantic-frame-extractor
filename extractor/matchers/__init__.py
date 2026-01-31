# Matchers are imported lazily to avoid loading heavy dependencies (torch, transformers)
# Import specific matchers directly from their modules:
#   from extractor.matchers.base import BaseMatcher
#   from extractor.matchers.generation import GenerationMatcher
#   from extractor.matchers.embedding import EmbeddingMatcher
#   from extractor.matchers.transformers_embedding import TransformersEmbeddingMatcher

# Matchers are imported lazily to avoid loading heavy dependencies (torch, transformers)
# Import specific matchers directly from their modules:
#   from extractor.matchers.base import BaseMatcher
#   from extractor.matchers.chat_api import ChatApiMatcher
#   from extractor.matchers.transformers_embedding import TransformersEmbeddingMatcher

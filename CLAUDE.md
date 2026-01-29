# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Semantic Frame Extractor - Extract video frames matching a text query using vision LLMs (via LM Studio).

## Development Setup

```bash
# Install dependencies
uv sync

# Run extraction
uv run python main.py "**/*.mp4" "A dark blue car" --mode quick
uv run python main.py "clip.mp4" "Person waving" --mode exhaustive --threshold 0.6
```

Requires LM Studio running locally with a vision embedding model (e.g., `qwen.qwen3-vl-embedding-2b`).

## Architecture

```
main.py                 # CLI entry point
extractor/
├── video.py            # VideoReader: frame extraction, seeking, binary search
├── matcher.py          # EmbeddingMatcher/GenerationMatcher: LM Studio integration
└── modes.py            # quick_extract() and exhaustive_extract() algorithms
```

## Extraction Modes

**Quick mode**: Samples frames at intervals (default 2s), batches to matcher, returns matches. Fast for finding representative frames.

**Exhaustive mode**: When a match is found during sampling (default 1s), binary searches backward to find the first matching frame, then captures every frame until no longer matching. Ensures complete segment capture.

## Matcher Types

- `TransformersEmbeddingMatcher` (default): Uses Qwen3-VL-Embedding directly via transformers. Downloads model from HuggingFace on first run. Fastest after initial load.
- `GenerationMatcher`: Uses LM Studio chat completions API. Better for complex reasoning queries.
- `EmbeddingMatcher`: Uses LM Studio embeddings API (limited support).

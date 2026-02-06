# AGENTS.md

This file provides guidance to Codex CLI and other coding agents when working in this repository.

## Project Overview

Semantic Frame Extractor: extract video frames matching a text query using vision LLMs.

## Development Setup

Use `uv` for all Python execution and dependency management.

- Do not run `python`, `pip`, or `pytest` directly.
- Use `uv run ...` for Python commands and scripts.
- Use `uv sync` to install dependencies.
- PyTorch setup is platform-sensitive; use `./install_pytorch.sh` when needed.

```bash
# Install dependencies
uv sync

# Run extraction
uv run main.py "**/*.mp4" "A dark blue car" --mode quick
uv run main.py "clip.mp4" "Person waving" --mode exhaustive --threshold 0.6
```

For API-based matchers, an OpenAI-compatible server must be running locally (for example LM Studio or vLLM).

## Architecture

```text
main.py                 # CLI entry point
extractor/
├── video.py            # VideoReader: frame extraction, seeking, binary search
├── matchers/           # Matcher implementations
│   ├── base.py         # BaseMatcher abstract class
│   ├── transformers_embedding.py  # Local HuggingFace model
│   └── chat_api.py     # OpenAI-compatible chat API
└── modes.py            # quick_extract() and exhaustive_extract() algorithms
```

## Extraction Modes

- Quick mode: samples frames at intervals (default `2s`), batches matcher calls, returns matches. Fast for representative frames.
- Exhaustive mode: samples frames (default `1s`), refines match boundaries with binary search, then captures all matching frames in matching segments.

## Matcher Types

- `TransformersEmbeddingMatcher` (default): uses Qwen3-VL-Embedding locally via `transformers`. Downloads from HuggingFace on first run.
- `ChatApiMatcher`: uses OpenAI-compatible chat completions API and can work better for complex reasoning queries.

## Agent Workflow Expectations

- Keep changes focused and minimal.
- Preserve existing CLI behavior unless the task explicitly changes it.
- Validate changes with repo scripts/commands via `uv run` where possible.
- If a command is expected to be long-running, state intent before running it.

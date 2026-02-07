# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Semantic Frame Extractor - Extract video frames matching a text query using vision LLMs.

## Development Setup

Use `uv` for all Python execution and dependency management. Except for getting pytorch which require some extra care to get platform specific setup (see `install_pytorch.sh`).

```bash
# Install dependencies
uv sync

# Run extraction
uv run main.py "**/*.mp4" "A dark blue car" --mode quick
uv run main.py "clip.mp4" "Person waving" --mode exhaustive --threshold 0.6
```

For API-based matchers, requires an OpenAI-compatible server running locally (e.g., LM Studio, vLLM).

## Architecture

```text
main.py                 # CLI entry point
extractor/
├── video.py            # VideoReader: frame extraction, seeking, binary search
├── matchers/           # Matcher implementations
│   ├── base.py         # BaseMatcher abstract class
│   ├── transformers_embedding.py  # Local HuggingFace model
│   └── chat_api.py     # OpenAI-compatible chat API
├── modes.py            # quick_extract() and exhaustive_extract() algorithms
└── tui.py              # Rich TUI progress display (ExtractionProgress, ExhaustiveProgress)
```

## Extraction Modes

**Quick mode**: Samples frames at intervals (default 2s), batches to matcher, returns matches. Fast for finding representative frames.

**Exhaustive mode**: When a match is found during sampling (default 1s), binary searches backward to find the first matching frame, then captures every frame until no longer matching. Ensures complete segment capture.

## Matcher Types

- `TransformersEmbeddingMatcher` (default): Uses Qwen3-VL-Embedding directly via transformers. Downloads model from HuggingFace on first run. Fastest after initial load.
- `ChatApiMatcher`: Uses OpenAI-compatible chat completions API. Better for complex reasoning queries.

## Output Modes

The tool has two output paths that must be kept in sync:

- **TUI mode** (default): Rich-based progress display in `extractor/tui.py`. Quick mode uses `ExtractionProgress`; exhaustive mode uses `ExhaustiveProgress`. Both share a common `_print_summary()` for final output.
- **Plain text mode** (`--no-tui`): Simple print-based output in `main.py` (`main_plain()`).

When adding or modifying features that affect progress reporting or summary output, update both output paths. The two modes display different metrics per extraction mode: quick mode shows matches/samples, exhaustive mode shows segments/frames extracted.

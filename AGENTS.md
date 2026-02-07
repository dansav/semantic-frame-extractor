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
├── modes.py            # quick_extract() and exhaustive_extract() algorithms
└── tui.py              # Rich TUI progress display (ExtractionProgress, ExhaustiveProgress)
```

## Extraction Modes

- Quick mode: samples frames at intervals (default `2s`), batches matcher calls, returns matches. Fast for representative frames.
- Exhaustive mode: samples frames (default `1s`), refines match boundaries with binary search, then captures all matching frames in matching segments.

## Matcher Types

- `TransformersEmbeddingMatcher` (default): uses Qwen3-VL-Embedding locally via `transformers`. Downloads from HuggingFace on first run.
- `ChatApiMatcher`: uses OpenAI-compatible chat completions API and can work better for complex reasoning queries.

## Output Modes

The tool has two output paths that must be kept in sync:

- **TUI mode** (default): Rich-based progress display in `extractor/tui.py`. Quick mode uses `ExtractionProgress`; exhaustive mode uses `ExhaustiveProgress`. Both share a common `_print_summary()` for final output.
- **Plain text mode** (`--no-tui`): Simple print-based output in `main.py` (`main_plain()`).

When adding or modifying features that affect progress reporting or summary output, update both output paths. The two modes display different metrics per extraction mode: quick mode shows matches/samples, exhaustive mode shows segments/frames extracted.

## Agent Workflow Expectations

- Keep changes focused and minimal.
- Preserve existing CLI behavior unless the task explicitly changes it.
- Validate changes with repo scripts/commands via `uv run` where possible.
- If a command is expected to be long-running, state intent before running it.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dynamic Frame Extractor - A Python 3.12+ project (currently in early development).

## Development Setup

This project uses `uv` for Python package management (indicated by pyproject.toml structure and .python-version file).

```bash
# Install dependencies
uv sync

# Run the main script
uv run python main.py
```

## Project Structure

- `main.py` - Entry point
- `pyproject.toml` - Project configuration and dependencies

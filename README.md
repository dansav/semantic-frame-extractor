# Semantic Frame Extractor

Identify and extract specific frames from videos using natural language queries. This tool leverages Vision LLMs (like Qwen3-VL) to "watch" your videos and save frames that match your description.

## Features

- **Natural Language Search**: Find "a red sports car", "person holding an umbrella", or "text saying 'Hello'" without training custom models.
- **Local & Private**: Runs entirely locally using HuggingFace Transformers or any OpenAI-compatible server (e.g. [LM Studio](https://lmstudio.ai/)). No video data leaves your machine.
- **Two IO Modes**:
  - **Quick**: Samples video at intervals (default 2s) to find representative keyframes.
  - **Exhaustive**: Locates the precise start and end of matching segments using binary search, capturing every matching frame.
- **Flexible Backends**:
  - `transformers`: (Default) Downloads and runs models like `Qwen/Qwen3-VL` directly. Fast and self-contained.
  - `generation`: Connects to any OpenAI-compatible Chat API (like LM Studio or vLLM) for complex reasoning tasks.
  - `embedding`: Vector-based matching (experimental).

## Installation

This project is built with Python 3.12 and uses `uv` for dependency management.

1. **Clone the repository**

   ```bash
   git clone git@github.com:dansav/semantic-frame-extractor.git
   cd semantic-frame-extractor
   ```

2. **Install dependencies**

   ```bash
   # Install uv if you haven't already
   pip install uv

   # Sync dependencies
   uv sync
   ```

  See <https://docs.astral.sh/uv/getting-started/installation/> for other ways to install uv.

## Usage

The basic syntax is `main.py [VIDEO_PATTERN] [QUERY]`.

### Quick Search (Default)

Best for finding a few examples of an object or event.

```bash
uv run main.py "**/*.mp4" "A golden retriever playing fetch"
```

### Exhaustive Segment Extraction

Best for clipping out entire scenes. This finds the exact start/end timestamps and saves every frame in between.

```bash
uv run main.py "vacation.mov" "People swimming in the pool" \
    --mode exhaustive \
    --threshold 0.6
```

### Using Custom Modes/Backends

**Use a different interval:**  
Sample every 0.5 seconds for finer granularity.

```bash
uv run main.py "video.mp4" "cat" --interval 0.5
```

**Use an OpenAI Compatible Server (e.g. LM Studio, vLLM):**  

1. Start your local server and load a Vision model (e.g., Qwen-VL).
2. Ensure the API is accessible (default `http://localhost:1234/v1`).
3. Run with the generation matcher:

```bash
uv run main.py "video.mp4" "cat" --matcher generation
```

## CLI Options

| Flag | Description | Default |
| ---- | ----------- | ------- |
| `pattern` | Glob pattern for video files (e.g. `*.mp4`, `**/*.mov`) | Required |
| `query` | Text description to search for | Required |
| `--mode`, `-m` | `quick` or `exhaustive` | `quick` |
| `--threshold`, `-t` | Confidence score cutoff (0.0 - 1.0) | `0.7` |
| `--output`, `-o` | Directory to save extracted frames | `./extracted_frames` |
| `--interval`, `-i` | Sampling interval in seconds | 2.0 (quick), 1.0 (exhaustive) |
| `--matcher` | Backend: `transformers`, `generation`, `embedding` | `transformers` |
| `--model` | Specific model name/path to use | Auto-detected |
| `--max-pixels` | Resolution limit for inference | 1,000,000 (1MP) |

## Tips

- **Thresholds**: If you aren't getting matches, try lowering the threshold to `0.5` or `0.6`.
- **Performance**: The `transformers` matcher is generally faster than `generation` as it batches images efficiently.
- **Memory**: Vision models can be VRAM heavy. If you run out of memory, try a smaller model variant or increase `--interval`.

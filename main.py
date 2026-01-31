#!/usr/bin/env python3
"""
Semantic Frame Extractor - Extract video frames matching a text query using vision LLMs.

Usage:
    uv run python main.py "**/*.mp4" "A dark blue car with a gray roof rack"
    uv run python main.py "**/*.mp4" "A person waving" --mode exhaustive --threshold 0.6
"""

import argparse
import re
import sys
from glob import glob
from pathlib import Path

from extractor import quick_extract, exhaustive_extract, VideoReader
from extractor.modes import save_frame


def parse_time_value(value: str) -> tuple[str, float]:
    """
    Parse a time value that can be either seconds or percentage.

    Args:
        value: Time string like "53.1s", "53.1", or "75%"

    Returns:
        Tuple of (type, value) where type is "seconds" or "percent"

    Raises:
        ValueError: If the format is invalid
    """
    value = value.strip()

    if value.endswith("%"):
        try:
            percent = float(value[:-1])
            if not 0 <= percent <= 100:
                raise ValueError(f"Percentage must be between 0 and 100, got {percent}")
            return ("percent", percent)
        except ValueError as e:
            raise ValueError(f"Invalid percentage format: {value}") from e

    # Remove optional 's' suffix for seconds
    if value.endswith("s"):
        value = value[:-1]

    try:
        seconds = float(value)
        if seconds < 0:
            raise ValueError(f"Time cannot be negative, got {seconds}")
        return ("seconds", seconds)
    except ValueError as e:
        raise ValueError(f"Invalid time format: {value}") from e


def resolve_time(time_spec: tuple[str, float] | None, duration: float) -> float | None:
    """
    Convert a time specification to seconds.

    Args:
        time_spec: Tuple from parse_time_value(), or None
        duration: Video duration in seconds

    Returns:
        Time in seconds, or None if time_spec is None
    """
    if time_spec is None:
        return None

    time_type, value = time_spec
    if time_type == "percent":
        return (value / 100.0) * duration
    else:
        return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract video frames matching a text query using vision LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "videos/*.mp4" "A red sports car"
  %(prog)s "**/*.mp4" "Person holding umbrella" --mode exhaustive
  %(prog)s "clip.mp4" "Cat sleeping" --threshold 0.7 --output ./matches
  %(prog)s "clip.mp4" "Car" --start 30s --end 120s
  %(prog)s "clip.mp4" "Car" --start 10%% --end 50%%
        """,
    )

    parser.add_argument(
        "pattern",
        help="Glob pattern for video files (e.g., '**/*.mp4')",
    )
    parser.add_argument(
        "query",
        help="Text description of frames to extract",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("./extracted_frames"),
        help="Output directory for matched frames (default: ./extracted_frames)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["quick", "exhaustive"],
        default="quick",
        help="Extraction mode: 'quick' samples keyframes, 'exhaustive' captures all matching frames (default: quick)",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.7,
        help="Confidence threshold 0.0-1.0 (default: 0.7)",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=None,
        help="Sample interval in seconds (default: 2.0 for quick, 1.0 for exhaustive)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=5,
        help="Batch size for quick mode (default: 5)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start time: seconds (e.g., '30s' or '30') or percentage (e.g., '10%%')",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End time: seconds (e.g., '120s' or '120') or percentage (e.g., '75%%')",
    )

    # API configuration (for generation/embedding matchers)
    parser.add_argument(
        "--api-url",
        default="http://localhost:1234/v1",
        help="OpenAI-compatible API URL (default: http://localhost:1234/v1)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (default: auto-detect based on matcher type)",
    )
    parser.add_argument(
        "--matcher",
        choices=["transformers", "generation", "embedding"],
        default="transformers",
        help="Matcher type: 'transformers' (local, fast), 'generation' (API chat completions), 'embedding' (API embeddings) (default: transformers)",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=1_000_000,
        help="Max pixels for image resizing during matching (default: 1000000 = 1MP). Lower = faster but less accurate.",
    )

    return parser.parse_args()


def create_matcher(args: argparse.Namespace):
    """Create the appropriate matcher based on arguments."""
    if args.matcher == "transformers":
        from extractor.matchers.transformers_embedding import TransformersEmbeddingMatcher

        model = args.model or "Qwen/Qwen3-VL-Embedding-2B"
        return TransformersEmbeddingMatcher(
            model_name=model, max_pixels=args.max_pixels
        )
    elif args.matcher == "embedding":
        from extractor.matchers.embedding import EmbeddingMatcher

        model = args.model or "qwen.qwen3-vl-embedding-2b"
        return EmbeddingMatcher(base_url=args.api_url, model=model)
    else:
        from extractor.matchers.generation import GenerationMatcher

        model = args.model or "qwen/qwen3-vl-4b"
        return GenerationMatcher(base_url=args.api_url, model=model)


def progress_callback(frame, confidence, is_match):
    """Print progress during extraction."""
    status = "MATCH" if is_match else "     "
    print(f"  [{status}] {frame.time_seconds:7.2f}s  conf={confidence:.3f}")


def main():
    args = parse_args()

    # Parse start/end time specifications
    try:
        start_spec = parse_time_value(args.start) if args.start else None
        end_spec = parse_time_value(args.end) if args.end else None
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Expand tilde in output path
    output_dir = args.output.expanduser()

    # Expand glob pattern
    # Handle tilde expansion (e.g. ~) and check for direct file match first
    # This prevents glob from interpreting characters like [] in filenames as patterns
    pattern_path = Path(args.pattern).expanduser()

    if pattern_path.is_file():
        video_files = [str(pattern_path)]
    else:
        video_files = sorted(glob(str(pattern_path), recursive=True))

    video_files = [Path(f) for f in video_files if Path(f).is_file()]

    if not video_files:
        print(f"Error: No video files found matching pattern: {args.pattern}")
        sys.exit(1)

    print(f"Found {len(video_files)} video file(s)")
    print(f"Query: {args.query}")
    print(f"Mode: {args.mode}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {output_dir}")
    print(f"Matcher: {args.matcher}")
    print()

    # Create matcher
    matcher = create_matcher(args)

    # Determine sample interval
    if args.interval is not None:
        interval = args.interval
    else:
        interval = 2.0 if args.mode == "quick" else 1.0

    total_matches = 0

    for video_path in video_files:
        print(f"Processing: {video_path}")
        video_name = video_path.stem

        try:
            # Get video duration to resolve percentage-based times
            with VideoReader(video_path) as video:
                duration = video.duration

            start_time = resolve_time(start_spec, duration)
            end_time = resolve_time(end_spec, duration)

            # Display time range if specified
            if start_time is not None or end_time is not None:
                start_str = f"{start_time:.1f}s" if start_time else "0s"
                end_str = f"{end_time:.1f}s" if end_time else f"{duration:.1f}s"
                print(f"  Time range: {start_str} - {end_str}")

            if args.mode == "quick":
                matches = quick_extract(
                    video_path=video_path,
                    query=args.query,
                    matcher=matcher,
                    threshold=args.threshold,
                    sample_interval=interval,
                    batch_size=args.batch_size,
                    callback=progress_callback,
                    start_time=start_time,
                    end_time=end_time,
                )
            else:
                matches = exhaustive_extract(
                    video_path=video_path,
                    query=args.query,
                    matcher=matcher,
                    threshold=args.threshold,
                    sample_interval=interval,
                    callback=progress_callback,
                    start_time=start_time,
                    end_time=end_time,
                )

            # Save matched frames
            video_matches = 0
            for matched in matches:
                output_path = save_frame(matched, output_dir, video_name)
                video_matches += 1
                print(f"    Saved: {output_path.name}")

            print(f"  -> {video_matches} frame(s) extracted")
            total_matches += video_matches

        except Exception as e:
            print(f"  Error processing {video_path}: {e}")
            continue

        print()

    print(f"Total: {total_matches} frame(s) extracted from {len(video_files)} video(s)")


if __name__ == "__main__":
    main()

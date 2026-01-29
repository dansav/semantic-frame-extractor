#!/usr/bin/env python3
"""
Semantic Frame Extractor - Extract video frames matching a text query using vision LLMs.

Usage:
    uv run python main.py "**/*.mp4" "A dark blue car with a gray roof rack"
    uv run python main.py "**/*.mp4" "A person waving" --mode exhaustive --threshold 0.6
"""

import argparse
import sys
from glob import glob
from pathlib import Path

from extractor import quick_extract, exhaustive_extract
from extractor.matcher import (
    EmbeddingMatcher,
    GenerationMatcher,
    TransformersEmbeddingMatcher,
)
from extractor.modes import save_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract video frames matching a text query using vision LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "videos/*.mp4" "A red sports car"
  %(prog)s "**/*.mp4" "Person holding umbrella" --mode exhaustive
  %(prog)s "clip.mp4" "Cat sleeping" --threshold 0.7 --output ./matches
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

    # LM Studio configuration
    parser.add_argument(
        "--api-url",
        default="http://localhost:1234/v1",
        help="LM Studio API URL (default: http://localhost:1234/v1)",
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
        help="Matcher type: 'transformers' (local, fast), 'generation' (LM Studio API), 'embedding' (LM Studio embeddings API) (default: transformers)",
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
        model = args.model or "Qwen/Qwen3-VL-Embedding-2B"
        return TransformersEmbeddingMatcher(
            model_name=model, max_pixels=args.max_pixels
        )
    elif args.matcher == "embedding":
        model = args.model or "qwen.qwen3-vl-embedding-2b"
        return EmbeddingMatcher(base_url=args.api_url, model=model)
    else:
        model = args.model or "qwen/qwen3-vl-4b"
        return GenerationMatcher(base_url=args.api_url, model=model)


def progress_callback(frame, confidence, is_match):
    """Print progress during extraction."""
    status = "MATCH" if is_match else "     "
    print(f"  [{status}] {frame.time_seconds:7.2f}s  conf={confidence:.3f}")


def main():
    args = parse_args()

    # Expand glob pattern
    video_files = sorted(glob(args.pattern, recursive=True))
    video_files = [Path(f) for f in video_files if Path(f).is_file()]

    if not video_files:
        print(f"Error: No video files found matching pattern: {args.pattern}")
        sys.exit(1)

    print(f"Found {len(video_files)} video file(s)")
    print(f"Query: {args.query}")
    print(f"Mode: {args.mode}")
    print(f"Threshold: {args.threshold}")
    print(f"Output: {args.output}")
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
            if args.mode == "quick":
                matches = quick_extract(
                    video_path=video_path,
                    query=args.query,
                    matcher=matcher,
                    threshold=args.threshold,
                    sample_interval=interval,
                    batch_size=args.batch_size,
                    callback=progress_callback,
                )
            else:
                matches = exhaustive_extract(
                    video_path=video_path,
                    query=args.query,
                    matcher=matcher,
                    threshold=args.threshold,
                    sample_interval=interval,
                    callback=progress_callback,
                )

            # Save matched frames
            video_matches = 0
            for matched in matches:
                output_path = save_frame(matched, args.output, video_name)
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

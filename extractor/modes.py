from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from PIL import Image

from .video import VideoReader, Frame
from .matchers.base import BaseMatcher


@dataclass
class MatchedFrame:
    """A frame that matched the query."""

    frame: Frame
    confidence: float
    video_path: Path


def save_frame(
    matched: MatchedFrame,
    output_dir: Path,
    video_name: str,
) -> Path:
    """Save a matched frame to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Format: videoname_time_confidence.png
    time_str = f"{matched.frame.time_seconds:.3f}".replace(".", "_")
    conf_str = f"{matched.confidence:.2f}".replace(".", "_")
    filename = f"{video_name}_{time_str}s_conf{conf_str}.png"

    output_path = output_dir / filename
    pil_image = Image.fromarray(matched.frame.image)
    pil_image.save(str(output_path), format="PNG")
    return output_path


def quick_extract(
    video_path: Path,
    query: str,
    matcher: BaseMatcher,
    threshold: float = 0.5,
    sample_interval: float = 2.0,
    batch_size: int = 5,
    callback: callable = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> Iterator[MatchedFrame]:
    """
    Quick extraction mode - samples frames at intervals and returns matches.

    Args:
        video_path: Path to video file
        query: Text description to match
        matcher: Matcher instance to use
        threshold: Minimum confidence to consider a match (0.0-1.0)
        sample_interval: Seconds between sampled frames
        batch_size: Number of frames to process in each batch
        callback: Optional callback(frame, confidence, is_match) for progress
        start_time: Start time in seconds (default: beginning of video)
        end_time: End time in seconds (default: end of video)

    Yields:
        MatchedFrame for each frame above threshold
    """
    with VideoReader(video_path) as video:
        batch_frames: list[Frame] = []

        for frame in video.sample_frames(
            interval_seconds=sample_interval,
            start_time=start_time,
            end_time=end_time,
        ):
            batch_frames.append(frame)

            if len(batch_frames) >= batch_size:
                # Process batch
                images = [f.image for f in batch_frames]
                scores = matcher.match_batch(images, query)

                for f, score in zip(batch_frames, scores):
                    is_match = score >= threshold
                    if callback:
                        callback(f, score, is_match)

                    if is_match:
                        yield MatchedFrame(
                            frame=f,
                            confidence=score,
                            video_path=video_path,
                        )

                batch_frames = []

        # Process remaining frames
        if batch_frames:
            images = [f.image for f in batch_frames]
            scores = matcher.match_batch(images, query)

            for f, score in zip(batch_frames, scores):
                is_match = score >= threshold
                if callback:
                    callback(f, score, is_match)

                if is_match:
                    yield MatchedFrame(
                        frame=f,
                        confidence=score,
                        video_path=video_path,
                    )


def _identify_segments(
    sample_results: list[tuple[Frame, float, bool]],
) -> list[tuple[int, int]]:
    """
    Find runs of consecutive matching samples.

    Returns:
        List of (start_sample_index, end_sample_index) tuples into sample_results.
    """
    segments = []
    i = 0
    while i < len(sample_results):
        _, _, is_match = sample_results[i]
        if not is_match:
            i += 1
            continue

        start = i
        end = i
        while end + 1 < len(sample_results) and sample_results[end + 1][2]:
            end += 1

        segments.append((start, end))
        i = end + 1

    return segments


def exhaustive_extract(
    video_path: Path,
    query: str,
    matcher: BaseMatcher,
    threshold: float = 0.5,
    sample_interval: float = 1.0,
    callback: callable = None,
    phase_callback: callable = None,
    start_time: float | None = None,
    end_time: float | None = None,
) -> Iterator[MatchedFrame]:
    """
    Exhaustive extraction mode - finds all frames in matching segments.

    Three-phase algorithm:
    1. Scanning: Sample at intervals, batch-check for matches
    2. Refining: Binary search at segment boundaries for exact start/end frames
    3. Extracting: Extract every frame within refined segment boundaries

    Args:
        video_path: Path to video file
        query: Text description to match
        matcher: Matcher instance to use
        threshold: Minimum confidence to consider a match (0.0-1.0)
        sample_interval: Seconds between samples when scanning
        callback: Optional callback(frame, confidence, is_match) for scanning progress
        phase_callback: Optional callback(phase, current, total) for phase transitions
            Phases: "scan_complete", "refining", "refine_complete", "extracting"
        start_time: Start time in seconds (default: beginning of video)
        end_time: End time in seconds (default: end of video)

    Yields:
        MatchedFrame for each frame in matching segments
    """
    with VideoReader(video_path) as video:
        # Convert start/end times to frame bounds for clamping
        start_frame_bound = video.time_to_frame(start_time) if start_time is not None else 0
        end_frame_bound = video.time_to_frame(end_time) if end_time is not None else video.frame_count

        # === PHASE 1: SCANNING ===
        # Sample at intervals and record which samples match (batched)
        sample_results: list[tuple[Frame, float, bool]] = []
        batch_frames: list[Frame] = []
        batch_size = 5

        for frame in video.sample_frames(
            interval_seconds=sample_interval,
            start_time=start_time,
            end_time=end_time,
        ):
            batch_frames.append(frame)

            if len(batch_frames) >= batch_size:
                images = [f.image for f in batch_frames]
                scores = matcher.match_batch(images, query)

                for f, score in zip(batch_frames, scores):
                    is_match = score >= threshold
                    sample_results.append((f, score, is_match))
                    if callback:
                        callback(f, score, is_match)

                batch_frames = []

        # Process remaining frames in batch
        if batch_frames:
            images = [f.image for f in batch_frames]
            scores = matcher.match_batch(images, query)

            for f, score in zip(batch_frames, scores):
                is_match = score >= threshold
                sample_results.append((f, score, is_match))
                if callback:
                    callback(f, score, is_match)

        if not sample_results:
            return

        # Identify matching segments from scan results
        segments = _identify_segments(sample_results)

        if phase_callback:
            phase_callback("scan_complete", len(segments), len(segments))

        if not segments:
            return

        # === PHASE 2: REFINING ===
        # Binary search to find exact boundaries for each segment
        def is_match_func(frame: Frame) -> bool:
            score = matcher.match(frame.image, query)
            return score >= threshold

        refined_segments: list[
            tuple[int, int, float]
        ] = []  # (start_frame, end_frame, avg_conf)

        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            first_sample_frame = sample_results[seg_start][0].frame_number
            last_sample_frame = sample_results[seg_end][0].frame_number

            # Binary search to find exact start of segment
            if seg_start > 0:
                prev_frame = sample_results[seg_start - 1][0].frame_number
                exact_start = video.binary_search_first_match(
                    prev_frame + 1,
                    first_sample_frame + 1,
                    is_match_func,
                )
                if exact_start is not None:
                    first_sample_frame = exact_start
            else:
                exact_start = video.binary_search_first_match(
                    start_frame_bound,
                    first_sample_frame + 1,
                    is_match_func,
                )
                if exact_start is not None:
                    first_sample_frame = exact_start

            # Binary search to find exact end of segment
            if seg_end + 1 < len(sample_results):
                next_frame = sample_results[seg_end + 1][0].frame_number
                exact_end = video.binary_search_last_match(
                    last_sample_frame,
                    next_frame,
                    is_match_func,
                )
                if exact_end is not None:
                    last_sample_frame = exact_end
            else:
                exact_end = video.binary_search_last_match(
                    last_sample_frame,
                    end_frame_bound,
                    is_match_func,
                )
                if exact_end is not None:
                    last_sample_frame = exact_end

            avg_confidence = sum(
                s[1] for s in sample_results[seg_start : seg_end + 1]
            ) / (seg_end - seg_start + 1)

            refined_segments.append(
                (first_sample_frame, last_sample_frame, avg_confidence)
            )

            if phase_callback:
                phase_callback("refining", seg_idx + 1, len(segments))

        # Calculate total frames to extract across all segments (clamped to bounds)
        total_extract_frames = sum(
            min(end_frame_bound - 1, end) - max(start_frame_bound, start) + 1
            for start, end, _ in refined_segments
        )

        if phase_callback:
            phase_callback(
                "refine_complete", total_extract_frames, total_extract_frames
            )

        # === PHASE 3: EXTRACTING ===
        # Extract all frames in refined segments WITHOUT querying the matcher
        extracted_count = 0

        for start_frame, end_frame, avg_confidence in refined_segments:
            clamped_start = max(start_frame_bound, start_frame)
            clamped_end = min(end_frame_bound - 1, end_frame)
            for frame_num in range(clamped_start, clamped_end + 1):
                f = video.get_frame_at_number(frame_num)
                if f:
                    extracted_count += 1

                    if phase_callback and extracted_count % 10 == 0:
                        phase_callback(
                            "extracting", extracted_count, total_extract_frames
                        )

                    yield MatchedFrame(
                        frame=f,
                        confidence=avg_confidence,
                        video_path=video_path,
                    )

        # Final extracting callback to ensure we reach 100%
        if phase_callback and extracted_count > 0:
            phase_callback("extracting", extracted_count, total_extract_frames)

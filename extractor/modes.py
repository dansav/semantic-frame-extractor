from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2

from .video import VideoReader, Frame
from .matcher import BaseMatcher


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
    cv2.imwrite(str(output_path), matched.frame.image)
    return output_path


def quick_extract(
    video_path: Path,
    query: str,
    matcher: BaseMatcher,
    threshold: float = 0.5,
    sample_interval: float = 2.0,
    batch_size: int = 5,
    callback: callable = None,
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

    Yields:
        MatchedFrame for each frame above threshold
    """
    with VideoReader(video_path) as video:
        batch_frames: list[Frame] = []

        for frame in video.sample_frames(interval_seconds=sample_interval):
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


def exhaustive_extract(
    video_path: Path,
    query: str,
    matcher: BaseMatcher,
    threshold: float = 0.5,
    sample_interval: float = 1.0,
    callback: callable = None,
) -> Iterator[MatchedFrame]:
    """
    Exhaustive extraction mode - finds all frames in matching segments.

    Optimized algorithm:
    1. Sample at intervals and check for matches
    2. If two consecutive samples both match, extract all frames between without querying
    3. Use binary search only at segment boundaries (match→no-match transitions)

    Args:
        video_path: Path to video file
        query: Text description to match
        matcher: Matcher instance to use
        threshold: Minimum confidence to consider a match (0.0-1.0)
        sample_interval: Seconds between samples when scanning
        callback: Optional callback(frame, confidence, is_match) for progress

    Yields:
        MatchedFrame for each frame in matching segments
    """
    with VideoReader(video_path) as video:
        # First pass: sample at intervals and record which samples match (batched)
        sample_results: list[tuple[Frame, float, bool]] = []
        batch_frames: list[Frame] = []
        batch_size = 5

        for frame in video.sample_frames(interval_seconds=sample_interval):
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

        def is_match_func(frame: Frame) -> bool:
            score = matcher.match(frame.image, query)
            return score >= threshold

        # Process samples to find segments and extract frames
        i = 0
        while i < len(sample_results):
            frame, score, is_match = sample_results[i]

            if not is_match:
                i += 1
                continue

            # Found a matching sample - determine segment boundaries
            segment_start_sample = i
            segment_end_sample = i

            # Find how far the matching segment extends
            while segment_end_sample + 1 < len(sample_results) and sample_results[segment_end_sample + 1][2]:
                segment_end_sample += 1

            # Get frame numbers for the segment boundaries
            first_sample_frame = sample_results[segment_start_sample][0].frame_number
            last_sample_frame = sample_results[segment_end_sample][0].frame_number

            # Binary search to find exact start of segment
            if segment_start_sample > 0:
                # Search between previous non-match and this match
                prev_frame = sample_results[segment_start_sample - 1][0].frame_number
                exact_start = video.binary_search_first_match(
                    prev_frame + 1,
                    first_sample_frame + 1,
                    is_match_func,
                )
                if exact_start is not None:
                    first_sample_frame = exact_start
            else:
                # First sample is a match - search from beginning
                exact_start = video.binary_search_first_match(
                    0,
                    first_sample_frame + 1,
                    is_match_func,
                )
                if exact_start is not None:
                    first_sample_frame = exact_start

            # Binary search to find exact end of segment
            if segment_end_sample + 1 < len(sample_results):
                # Search between last match and next non-match
                next_frame = sample_results[segment_end_sample + 1][0].frame_number
                exact_end = video.binary_search_last_match(
                    last_sample_frame,
                    next_frame,
                    is_match_func,
                )
                if exact_end is not None:
                    last_sample_frame = exact_end
            else:
                # Last sample is a match - search to end of video
                exact_end = video.binary_search_last_match(
                    last_sample_frame,
                    video.frame_count,
                    is_match_func,
                )
                if exact_end is not None:
                    last_sample_frame = exact_end

            # Extract all frames in the segment WITHOUT querying the matcher
            for frame_num in range(first_sample_frame, last_sample_frame + 1):
                f = video.get_frame_at_number(frame_num)
                if f:
                    # Use interpolated confidence (we don't query for intermediate frames)
                    # Just use the average of the boundary samples
                    avg_confidence = sum(s[1] for s in sample_results[segment_start_sample:segment_end_sample + 1]) / (segment_end_sample - segment_start_sample + 1)

                    yield MatchedFrame(
                        frame=f,
                        confidence=avg_confidence,
                        video_path=video_path,
                    )

            # Move past this segment
            i = segment_end_sample + 1

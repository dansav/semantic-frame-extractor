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

    # Format: videoname_time_confidence.jpg
    time_str = f"{matched.frame.time_seconds:.3f}".replace(".", "_")
    conf_str = f"{matched.confidence:.2f}".replace(".", "_")
    filename = f"{video_name}_{time_str}s_conf{conf_str}.jpg"

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
    search_margin_seconds: float = 5.0,
    callback: callable = None,
) -> Iterator[MatchedFrame]:
    """
    Exhaustive extraction mode - finds all frames in matching segments.

    When a match is detected during sampling:
    1. Binary search backward to find the first matching frame
    2. Extract all frames until no longer matching
    3. Resume sampling from end of segment

    Args:
        video_path: Path to video file
        query: Text description to match
        matcher: Matcher instance to use
        threshold: Minimum confidence to consider a match (0.0-1.0)
        sample_interval: Seconds between samples when scanning
        search_margin_seconds: How far back to search for segment start
        callback: Optional callback(frame, confidence, is_match) for progress

    Yields:
        MatchedFrame for each frame in matching segments
    """
    with VideoReader(video_path) as video:
        current_time = 0.0
        processed_up_to_frame = -1  # Track to avoid re-processing frames

        def is_match(frame: Frame) -> bool:
            """Helper to check if a frame matches."""
            score = matcher.match(frame.image, query)
            return score >= threshold

        def get_frame_score(frame: Frame) -> float:
            """Helper to get confidence score."""
            return matcher.match(frame.image, query)

        while current_time < video.duration:
            frame = video.get_frame_at_time(current_time)
            if not frame or frame.frame_number <= processed_up_to_frame:
                current_time += sample_interval
                continue

            score = get_frame_score(frame)
            is_frame_match = score >= threshold

            if callback:
                callback(frame, score, is_frame_match)

            if is_frame_match:
                # Found a match! Search for segment boundaries

                # Binary search backward to find segment start
                search_start_frame = max(
                    0,
                    processed_up_to_frame + 1,
                    video.time_to_frame(current_time - search_margin_seconds),
                )
                first_match_frame = video.binary_search_first_match(
                    search_start_frame,
                    frame.frame_number + 1,
                    is_match,
                )

                if first_match_frame is None:
                    first_match_frame = frame.frame_number

                # Now extract all frames from first match until no longer matching
                current_frame_num = first_match_frame

                while current_frame_num < video.frame_count:
                    f = video.get_frame_at_number(current_frame_num)
                    if not f:
                        break

                    f_score = get_frame_score(f)
                    f_is_match = f_score >= threshold

                    if callback and current_frame_num > frame.frame_number:
                        callback(f, f_score, f_is_match)

                    if f_is_match:
                        if current_frame_num > processed_up_to_frame:
                            yield MatchedFrame(
                                frame=f,
                                confidence=f_score,
                                video_path=video_path,
                            )
                        current_frame_num += 1
                    else:
                        # End of matching segment
                        break

                # Update position to resume sampling after segment
                processed_up_to_frame = current_frame_num - 1
                current_time = video.frame_to_time(current_frame_num) + sample_interval
            else:
                # No match, continue sampling
                current_time += sample_interval

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import cv2
import numpy as np


@dataclass
class Frame:
    """A video frame with its timestamp."""
    image: np.ndarray
    time_seconds: float
    frame_number: int


class VideoReader:
    """Handles video file operations and frame extraction."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._cap = cv2.VideoCapture(str(self.path))
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video: {self.path}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration = self._frame_count / self._fps if self._fps > 0 else 0

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self._duration

    def time_to_frame(self, time_seconds: float) -> int:
        """Convert time in seconds to frame number."""
        return int(time_seconds * self._fps)

    def frame_to_time(self, frame_number: int) -> float:
        """Convert frame number to time in seconds."""
        return frame_number / self._fps

    def get_frame_at_time(self, time_seconds: float) -> Frame | None:
        """Extract a single frame at the specified time."""
        frame_number = self.time_to_frame(time_seconds)
        return self.get_frame_at_number(frame_number)

    def get_frame_at_number(self, frame_number: int) -> Frame | None:
        """Extract a single frame by frame number."""
        if frame_number < 0 or frame_number >= self._frame_count:
            return None

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, image = self._cap.read()

        if not ret:
            return None

        return Frame(
            image=image,
            time_seconds=self.frame_to_time(frame_number),
            frame_number=frame_number,
        )

    def sample_frames(self, interval_seconds: float = 1.0) -> Iterator[Frame]:
        """Yield frames at regular intervals."""
        current_time = 0.0
        while current_time < self._duration:
            frame = self.get_frame_at_time(current_time)
            if frame:
                yield frame
            current_time += interval_seconds

    def get_frames_range(
        self, start_frame: int, end_frame: int, step: int = 1
    ) -> Iterator[Frame]:
        """Yield all frames in a range."""
        start_frame = max(0, start_frame)
        end_frame = min(self._frame_count, end_frame)

        for frame_number in range(start_frame, end_frame, step):
            frame = self.get_frame_at_number(frame_number)
            if frame:
                yield frame

    def get_keyframes(self) -> Iterator[Frame]:
        """
        Extract keyframes (I-frames) from the video.

        Note: OpenCV doesn't directly expose keyframe detection for all codecs.
        This implementation samples more densely and could be enhanced with
        scene change detection or ffprobe for true keyframe extraction.
        """
        # Simple approach: sample every 2 seconds as approximate keyframes
        # For true keyframe extraction, would need ffprobe or similar
        yield from self.sample_frames(interval_seconds=2.0)

    def binary_search_first_match(
        self,
        start_frame: int,
        end_frame: int,
        is_match: callable,
    ) -> int | None:
        """
        Binary search to find the first frame that matches.

        Args:
            start_frame: Lower bound frame number
            end_frame: Upper bound frame number
            is_match: Callable that takes a Frame and returns True if it matches

        Returns:
            Frame number of first match, or None if no match found
        """
        if start_frame >= end_frame:
            return None

        result = None
        low, high = start_frame, end_frame

        while low < high:
            mid = (low + high) // 2
            frame = self.get_frame_at_number(mid)

            if frame and is_match(frame):
                result = mid
                high = mid  # Keep searching left for earlier match
            else:
                low = mid + 1

        return result

    def binary_search_last_match(
        self,
        start_frame: int,
        end_frame: int,
        is_match: callable,
    ) -> int | None:
        """
        Binary search to find the last frame that matches.

        Args:
            start_frame: Lower bound frame number
            end_frame: Upper bound frame number
            is_match: Callable that takes a Frame and returns True if it matches

        Returns:
            Frame number of last match, or None if no match found
        """
        if start_frame >= end_frame:
            return None

        result = None
        low, high = start_frame, end_frame - 1

        while low <= high:
            mid = (low + high) // 2
            frame = self.get_frame_at_number(mid)

            if frame and is_match(frame):
                result = mid
                low = mid + 1  # Keep searching right for later match
            else:
                high = mid - 1

        return result

    def close(self):
        """Release video capture resources."""
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

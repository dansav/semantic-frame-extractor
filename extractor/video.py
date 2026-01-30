from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
import numpy as np


@dataclass
class Frame:
    """A video frame with its timestamp."""
    image: np.ndarray  # RGB format
    time_seconds: float
    frame_number: int


class VideoReader:
    """
    Handles video file operations and frame extraction using PyAV.

    Note: This class is not thread-safe. Create separate instances for
    concurrent access.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._container = av.open(str(self.path))
        self._stream = self._container.streams.video[0]

        # Set thread type for faster decoding
        self._stream.thread_type = "AUTO"

        self._fps = float(self._stream.average_rate) if self._stream.average_rate else 30.0

        # Frame count may not be available for all containers
        if self._stream.frames > 0:
            self._frame_count = self._stream.frames
        elif self._stream.duration and self._stream.time_base:
            # Estimate from duration
            duration_sec = float(self._stream.duration * self._stream.time_base)
            self._frame_count = int(duration_sec * self._fps)
        else:
            # Fallback: estimate from container duration
            if self._container.duration:
                duration_sec = self._container.duration / av.time_base
                self._frame_count = int(duration_sec * self._fps)
            else:
                self._frame_count = 0

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

    def _seek_to_time(self, time_seconds: float) -> None:
        """Seek to a specific time in the video."""
        # Convert to stream time base
        pts = int(time_seconds / self._stream.time_base)
        self._container.seek(pts, stream=self._stream, backward=True)

    def _seek_to_frame(self, frame_number: int) -> None:
        """Seek to a specific frame number."""
        time_seconds = self.frame_to_time(frame_number)
        self._seek_to_time(time_seconds)

    def get_frame_at_time(self, time_seconds: float) -> Frame | None:
        """Extract a single frame at the specified time."""
        if time_seconds < 0 or time_seconds > self._duration:
            return None

        target_frame = self.time_to_frame(time_seconds)
        return self.get_frame_at_number(target_frame)

    def get_frame_at_number(self, frame_number: int) -> Frame | None:
        """Extract a single frame by frame number."""
        if frame_number < 0 or (self._frame_count > 0 and frame_number >= self._frame_count):
            return None

        target_time = self.frame_to_time(frame_number)
        self._seek_to_time(target_time)

        # Decode frames until we reach or pass the target
        for frame in self._container.decode(video=0):
            frame_time = float(frame.pts * self._stream.time_base) if frame.pts else 0
            current_frame_num = self.time_to_frame(frame_time)

            if current_frame_num >= frame_number:
                return Frame(
                    image=frame.to_ndarray(format='rgb24'),
                    time_seconds=frame_time,
                    frame_number=current_frame_num,
                )

        return None

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
        if self._frame_count > 0:
            end_frame = min(self._frame_count, end_frame)

        for frame_number in range(start_frame, end_frame, step):
            frame = self.get_frame_at_number(frame_number)
            if frame:
                yield frame

    def get_keyframes(self) -> Iterator[Frame]:
        """
        Extract keyframes (I-frames) from the video.

        Uses PyAV's frame.key_frame property for true keyframe detection.
        """
        # Seek to beginning
        self._container.seek(0)

        for frame in self._container.decode(video=0):
            if frame.key_frame:
                frame_time = float(frame.pts * self._stream.time_base) if frame.pts else 0
                yield Frame(
                    image=frame.to_ndarray(format='rgb24'),
                    time_seconds=frame_time,
                    frame_number=self.time_to_frame(frame_time),
                )

    def get_keyframe_timestamps(self) -> list[float]:
        """
        Get timestamps of all keyframes without full decoding.

        This is faster than get_keyframes() when you only need timestamps.
        """
        timestamps = []
        self._container.seek(0)

        for packet in self._container.demux(video=0):
            if packet.is_keyframe and packet.pts is not None:
                timestamp = float(packet.pts * self._stream.time_base)
                timestamps.append(timestamp)

        return timestamps

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
        """Release video resources."""
        self._container.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

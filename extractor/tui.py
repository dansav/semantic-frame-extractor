"""
TUI (Text User Interface) components using Rich.

Provides progress bars, live statistics, and summary tables for video extraction.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    ProgressColumn,
    Task,
)
from rich.table import Table
from rich.text import Text


@dataclass
class VideoStats:
    """Statistics for a single video."""

    path: Path
    duration: float = 0.0
    frames_processed: int = 0
    total_frames: int = 0
    matches_found: int = 0
    processing_time: float = 0.0
    confidences: list[float] = field(default_factory=list)
    # Track which frame indices were matches for the timeline
    match_indices: list[int] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        """Percentage of frames that matched."""
        if self.frames_processed == 0:
            return 0.0
        return (self.matches_found / self.frames_processed) * 100

    @property
    def avg_confidence(self) -> float:
        """Average confidence of matches."""
        if not self.confidences:
            return 0.0
        return sum(self.confidences) / len(self.confidences)

    @property
    def max_confidence(self) -> float:
        """Maximum confidence seen."""
        if not self.confidences:
            return 0.0
        return max(self.confidences)


@dataclass
class ExtractionStats:
    """Overall extraction statistics."""

    videos: list[VideoStats] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    query: str = ""
    mode: str = "quick"
    threshold: float = 0.7
    matcher_type: str = ""

    @property
    def total_matches(self) -> int:
        return sum(v.matches_found for v in self.videos)

    @property
    def total_frames(self) -> int:
        return sum(v.frames_processed for v in self.videos)

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def frames_per_second(self) -> float:
        if self.elapsed_time == 0:
            return 0.0
        return self.total_frames / self.elapsed_time


class MatchAwareBarColumn(ProgressColumn):
    """
    A progress bar that shows matches as green segments and non-matches as default color.
    Expands to fill available space.
    """

    def __init__(self, console: Console | None = None):
        super().__init__()
        self._console = console

    def render(self, task: Task) -> RenderableType:
        """Render the progress bar with match highlighting."""
        total = task.total or 1
        completed = task.completed

        # Get match data from task fields
        match_set: set = task.fields.get("match_set", set())
        total_frames: int = task.fields.get("total_frames", int(total))

        # Calculate bar width based on terminal width
        # Reserve space for: spinner(2) + description(55) + percentage(5) + elapsed(8) + remaining(8) + padding(12)
        terminal_width = self._console.width if self._console else 120
        bar_width = max(20, terminal_width - 90)

        # Calculate how many bar characters to fill
        if total_frames > 0 and total > 0:
            filled_chars = int((completed / total) * bar_width)
        else:
            filled_chars = 0

        # Build the bar character by character
        bar = Text()

        for i in range(bar_width):
            # Which frame range does this character represent?
            frame_start = int((i / bar_width) * total_frames) if total_frames > 0 else 0
            frame_end = int(((i + 1) / bar_width) * total_frames) if total_frames > 0 else 0

            # Check if any frames in this range are matches
            has_match = any(f in match_set for f in range(frame_start, frame_end + 1))

            if i < filled_chars:
                # This part of the bar is filled (processed)
                if has_match:
                    bar.append("━", style="green bold")
                else:
                    bar.append("━", style="bar.complete")
            else:
                # This part is not yet processed
                bar.append("━", style="bar.back")

        return bar


class ExtractionProgress:
    """
    Rich-based progress display for video frame extraction.

    Usage:
        with ExtractionProgress(videos, query, mode, threshold, matcher) as progress:
            for video_path in videos:
                progress.start_video(video_path, duration, estimated_frames)
                for frame, confidence, is_match in process_frames():
                    progress.update(confidence, is_match)
                progress.finish_video()
    """

    def __init__(
        self,
        video_files: list[Path],
        query: str,
        mode: str,
        threshold: float,
        matcher_type: str,
    ):
        self.console = Console()
        self.stats = ExtractionStats(
            query=query,
            mode=mode,
            threshold=threshold,
            matcher_type=matcher_type,
        )
        self.video_files = video_files
        self._current_video_stats: VideoStats | None = None
        self._video_start_time: float = 0.0

        # Track match positions for the current video
        self._current_match_set: set[int] = set()

        # Progress bars - using custom match-aware bar for video progress
        # expand=True makes the progress bar use full terminal width
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}", justify="left"),
            MatchAwareBarColumn(console=self.console),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )

        # Task IDs
        self._overall_task = None
        self._video_task = None

        # Pre-created task IDs for each video (mapping: video index -> task ID)
        self._video_tasks: dict[int, int] = {}
        self._current_video_index: int = -1

        # Live display
        self._live: Live | None = None

        # Current frame info for display
        self._last_confidence: float = 0.0
        self._last_was_match: bool = False

        # Track if we're still running (for hiding stats panel when done)
        self._is_running: bool = True

        # Track overall frame progress (updated as we learn about each video)
        self._overall_frames_processed: int = 0
        self._overall_frames_total: int = 0
        self._overall_match_set: set[int] = set()  # Global frame indices for matches

    def _get_current_totals(self) -> tuple[int, int]:
        """Get total frames and matches including current video in progress."""
        total_frames = self.stats.total_frames
        total_matches = self.stats.total_matches

        # Add current video stats if processing
        if self._current_video_stats:
            total_frames += self._current_video_stats.frames_processed
            total_matches += self._current_video_stats.matches_found

        return total_frames, total_matches

    def _make_stats_panel(self) -> Panel:
        """Create a panel showing current statistics."""
        stats_table = Table.grid(padding=(0, 2))
        stats_table.add_column(style="cyan", justify="right")
        stats_table.add_column(style="white")
        stats_table.add_column(style="cyan", justify="right")
        stats_table.add_column(style="white")

        # Get totals including current video
        total_frames, total_matches = self._get_current_totals()

        # Current video stats
        if self._current_video_stats:
            match_style = "green bold" if self._last_was_match else "white"
        else:
            match_style = "white"

        stats_table.add_row(
            "Matches:",
            Text(str(total_matches), style="green bold"),
            "Frames:",
            str(total_frames),
        )

        conf_text = f"{self._last_confidence:.3f}"
        stats_table.add_row(
            "Last conf:",
            Text(conf_text, style=match_style),
            "Threshold:",
            f"{self.stats.threshold:.2f}",
        )

        if total_frames > 0:
            elapsed = self.stats.elapsed_time
            fps = total_frames / elapsed if elapsed > 0 else 0.0
            stats_table.add_row(
                "Speed:",
                f"{fps:.1f} frames/s",
                "Mode:",
                self.stats.mode,
            )

        return Panel(
            stats_table,
            title="[bold]Statistics[/bold]",
            border_style="blue",
            padding=(0, 1),
        )

    def _make_display(self) -> RenderableType:
        """Create the full display."""
        if self._is_running:
            # Show progress bars and stats panel during execution
            return Group(
                self._progress,
                self._make_stats_panel(),
            )
        else:
            # Only show progress bars when done (stats panel hidden)
            return self._progress

    def _truncate_name(self, name: str, max_len: int = 50) -> str:
        """Truncate a name to fit in the display."""
        if len(name) > max_len:
            return name[: max_len - 3] + "..."
        return name

    def __enter__(self):
        """Start the live display."""
        # Start with total=0, we'll update it as we learn about each video
        self._overall_task = self._progress.add_task(
            "[bold blue]Overall",
            total=0,
            match_set=self._overall_match_set,
            total_frames=0,
        )

        # Pre-create all video tasks (shown as pending/dimmed)
        for i, video_path in enumerate(self.video_files):
            video_name = self._truncate_name(video_path.name)
            task_id = self._progress.add_task(
                f"[dim]{video_name}[/dim]",
                total=0,  # Unknown until we start processing
                match_set=set(),
                total_frames=0,
                start=False,  # Don't start the task yet
            )
            self._video_tasks[i] = task_id

        self._live = Live(
            self._make_display(),
            console=self.console,
            refresh_per_second=4,
            transient=True,  # Clear the live display when done
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the live display and show summary."""
        self._is_running = False
        if self._live:
            self._live.__exit__(exc_type, exc_val, exc_tb)
        self.print_summary()

    def start_video(
        self,
        video_path: Path,
        duration: float,
        estimated_frames: int,
    ) -> None:
        """Start processing a new video."""
        self._current_video_index += 1
        self._current_video_stats = VideoStats(
            path=video_path,
            duration=duration,
            total_frames=estimated_frames,
        )
        self._video_start_time = time.time()
        self._current_match_set = set()

        # Update overall total with this video's frames
        self._overall_frames_total += estimated_frames
        if self._overall_task is not None:
            self._progress.update(
                self._overall_task,
                total=self._overall_frames_total,
                total_frames=self._overall_frames_total,
            )

        # Activate the pre-created video task
        video_name = self._truncate_name(video_path.name)
        self._video_task = self._video_tasks.get(self._current_video_index)

        if self._video_task is not None:
            self._progress.update(
                self._video_task,
                description=f"[cyan]{video_name}[/cyan]",
                total=estimated_frames,
                total_frames=estimated_frames,
                match_set=self._current_match_set,
            )
            # Start the task (makes it active)
            self._progress.start_task(self._video_task)

        if self._live:
            self._live.update(self._make_display())

    def update(self, confidence: float, is_match: bool) -> None:
        """Update progress with a processed frame."""
        if self._current_video_stats is None:
            return

        frame_index = self._current_video_stats.frames_processed
        self._current_video_stats.frames_processed += 1
        self._last_confidence = confidence
        self._last_was_match = is_match

        if is_match:
            self._current_video_stats.matches_found += 1
            self._current_video_stats.confidences.append(confidence)
            self._current_video_stats.match_indices.append(frame_index)
            self._current_match_set.add(frame_index)
            # Track in overall match set using global frame index
            self._overall_match_set.add(self._overall_frames_processed)

        # Update overall progress
        self._overall_frames_processed += 1
        if self._overall_task is not None:
            self._progress.update(
                self._overall_task,
                completed=self._overall_frames_processed,
                match_set=self._overall_match_set,
            )

        # Update video progress bar
        if self._video_task is not None:
            self._progress.update(
                self._video_task,
                advance=1,
                match_set=self._current_match_set,
            )

        if self._live:
            self._live.update(self._make_display())

    def finish_video(self) -> None:
        """Finish processing current video."""
        if self._current_video_stats is None:
            return

        self._current_video_stats.processing_time = time.time() - self._video_start_time
        self.stats.videos.append(self._current_video_stats)

        # Mark the video task as complete (change style to indicate it's done)
        if self._video_task is not None:
            video_name = self._truncate_name(self._current_video_stats.path.name)

            # Update description to show it's complete (use green for completed)
            self._progress.update(
                self._video_task,
                description=f"[green]{video_name}[/green]",
            )

            self._video_task = None

        self._current_video_stats = None
        self._current_match_set = set()

        if self._live:
            self._live.update(self._make_display())

    def log_saved_frame(self, output_path: Path, confidence: float) -> None:
        """Log a saved frame (shown below progress)."""
        # We'll collect these and show in summary
        pass

    def _get_timeline_width(self) -> int:
        """Calculate timeline width based on terminal width."""
        # Reserve space for other columns: Video(40) + Duration(10) + Frames(8) + Matches(8) + Rate(8) + AvgConf(10) + Time(8) + padding(20)
        terminal_width = self.console.width
        timeline_width = max(20, terminal_width - 120)
        return timeline_width

    def _make_timeline(self, video: VideoStats, width: int | None = None) -> Text:
        """Create a visual timeline showing where matches occurred."""
        if video.total_frames == 0:
            return Text("-")

        # Use provided width or calculate dynamically
        timeline_width = width if width is not None else self._get_timeline_width()

        timeline = Text()
        match_set = set(video.match_indices)

        for i in range(timeline_width):
            # Which frame range does this character represent?
            frame_start = int((i / timeline_width) * video.total_frames)
            frame_end = int(((i + 1) / timeline_width) * video.total_frames)

            # Check if any frames in this range are matches
            has_match = any(f in match_set for f in range(frame_start, frame_end + 1))

            if has_match:
                timeline.append("█", style="green")
            else:
                timeline.append("░", style="dim")

        return timeline

    def print_summary(self) -> None:
        """Print final summary statistics."""
        self.console.print()

        # Check if we actually processed any frames
        if self.stats.total_frames == 0:
            # No frames processed - likely all errors, show minimal output
            self.console.print(
                Panel(
                    f"[bold yellow]No frames processed[/bold yellow]\n"
                    f"Query: [cyan]{self.stats.query}[/cyan]",
                    border_style="yellow",
                )
            )
            return

        # Summary header
        self.console.print(
            Panel(
                f"[bold green]Extraction Complete[/bold green]\n"
                f"Query: [cyan]{self.stats.query}[/cyan]",
                border_style="green",
            )
        )

        # Results table with timeline - only show videos that were actually processed
        processed_videos = [v for v in self.stats.videos if v.frames_processed > 0]
        if processed_videos:
            # Calculate timeline width for consistent display
            timeline_width = self._get_timeline_width()

            # expand=True makes the table use full terminal width
            table = Table(title="Results by Video", show_header=True, expand=True)
            table.add_column("Video", style="cyan", ratio=2)
            table.add_column("Timeline", no_wrap=True, ratio=3)
            table.add_column("Duration", justify="right")
            table.add_column("Frames", justify="right")
            table.add_column("Matches", justify="right", style="green")
            table.add_column("Rate", justify="right")
            table.add_column("Avg Conf", justify="right")
            table.add_column("Time", justify="right")

            for video in processed_videos:
                duration_str = f"{video.duration:.1f}s"
                match_rate = f"{video.match_rate:.1f}%"
                avg_conf = f"{video.avg_confidence:.3f}" if video.confidences else "-"
                time_str = f"{video.processing_time:.1f}s"
                timeline = self._make_timeline(video, width=timeline_width)

                table.add_row(
                    video.path.name,
                    timeline,
                    duration_str,
                    str(video.frames_processed),
                    str(video.matches_found),
                    match_rate,
                    avg_conf,
                    time_str,
                )

            self.console.print(table)

        # Overall stats
        stats_table = Table.grid(padding=(0, 3))
        stats_table.add_column(style="bold")
        stats_table.add_column()

        stats_table.add_row("Total videos:", str(len(self.stats.videos)))
        stats_table.add_row("Total frames processed:", str(self.stats.total_frames))
        stats_table.add_row("Total matches:", f"[green bold]{self.stats.total_matches}[/green bold]")
        stats_table.add_row("Total time:", f"{self.stats.elapsed_time:.1f}s")

        if self.stats.elapsed_time > 0:
            fps = self.stats.frames_per_second
            stats_table.add_row("Average speed:", f"{fps:.1f} frames/s")

        self.console.print()
        self.console.print(Panel(stats_table, title="Summary", border_style="blue"))

"""
TUI (Text User Interface) components using Rich.

Provides progress bars, live statistics, and summary tables for video extraction.
Includes QuickProgress for quick mode and ExhaustiveProgress for exhaustive mode.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    Progress,
    TextColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    ProgressColumn,
    Task,
)
from rich.table import Table
from rich.text import Text


# ---------------------------------------------------------------------------
# Data classes (shared between Quick and Exhaustive)
# ---------------------------------------------------------------------------


@dataclass
class VideoStats:
    """Statistics for a single video."""

    path: Path
    duration: float = 0.0
    analyzed_start_time: float | None = None
    analyzed_end_time: float | None = None
    frames_processed: int = 0
    total_frames: int = 0
    matches_found: int = 0
    processing_time: float = 0.0
    # Confidence for every sampled frame (processed order).
    sample_confidences: list[float] = field(default_factory=list)
    # Confidence values for matches only (used by Avg Conf summary).
    confidences: list[float] = field(default_factory=list)
    # Track which frame indices were matches for the timeline
    match_indices: list[int] = field(default_factory=list)
    # Exhaustive-mode extras
    segments_found: int = 0
    frames_extracted: int = 0

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

    @property
    def analyzed_range(self) -> tuple[float, float]:
        """Analyzed [start, end] time bounds in seconds."""
        start = self.analyzed_start_time if self.analyzed_start_time is not None else 0.0
        end = self.analyzed_end_time if self.analyzed_end_time is not None else self.duration
        if end < start:
            end = start
        return start, end

    @property
    def analyzed_duration(self) -> float:
        """Duration of analyzed time window in seconds."""
        start, end = self.analyzed_range
        return max(0.0, end - start)


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
    def total_segments(self) -> int:
        return sum(v.segments_found for v in self.videos)

    @property
    def total_frames_extracted(self) -> int:
        return sum(v.frames_extracted for v in self.videos)

    @property
    def frames_per_second(self) -> float:
        if self.elapsed_time == 0:
            return 0.0
        return self.total_frames / self.elapsed_time


# ---------------------------------------------------------------------------
# Custom Rich columns
# ---------------------------------------------------------------------------


class PhaseSpinnerColumn(ProgressColumn):
    """Spinner that only renders for active (started, not finished) tasks."""

    def __init__(self, spinner_name: str = "dots", finished_text: str = "✓"):
        super().__init__()
        self._spinner_name = spinner_name
        self._finished_text = finished_text
        self._spinners: dict[int, object] = {}

    def render(self, task: Task) -> RenderableType:
        if task.finished:
            return Text(self._finished_text, style="green bold")
        if not task.started:
            return Text("  ")  # blank placeholder, same width as spinner
        # Active task — render spinner
        if task.id not in self._spinners:
            from rich.spinner import Spinner

            self._spinners[task.id] = Spinner(self._spinner_name)
        return self._spinners[task.id]


class MatchAwareBarColumn(ProgressColumn):
    """
    A progress bar that shows matches as green segments and non-matches as default color.
    Expands to fill available space.
    """

    def __init__(self, console: Console | None = None, bar_width: int | None = None):
        super().__init__()
        self._console = console
        self._fixed_width = bar_width

    def render(self, task: Task) -> RenderableType:
        """Render the progress bar with match highlighting."""
        total = task.total or 1
        completed = task.completed

        # Get match data and optional override style from task fields
        match_set: set = task.fields.get("match_set", set())
        total_frames: int = task.fields.get("total_frames", int(total))
        bar_style: str | None = task.fields.get("bar_style")

        # Calculate bar width based on terminal width
        if self._fixed_width is not None:
            bar_width = self._fixed_width
        else:
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
            frame_end = (
                int(((i + 1) / bar_width) * total_frames) if total_frames > 0 else 0
            )

            # Check if any frames in this range are matches
            has_match = any(f in match_set for f in range(frame_start, frame_end + 1))

            if i < filled_chars:
                # This part of the bar is filled (processed)
                if has_match:
                    bar.append("━", style="green bold")
                elif bar_style:
                    bar.append("━", style=bar_style)
                else:
                    bar.append("━", style="bar.complete")
            else:
                # This part is not yet processed
                bar.append("━", style="bar.back")

        return bar


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _truncate_name(name: str, max_len: int = 50) -> str:
    """Truncate a name to fit in the display."""
    if len(name) > max_len:
        return name[: max_len - 3] + "..."
    return name


def _get_timeline_width(console: Console) -> int:
    """Calculate timeline width based on terminal width."""
    terminal_width = console.width
    # Card layout uses almost full width; keep a small safety margin.
    return max(30, terminal_width - 16)


def _make_timeline(
    video: VideoStats,
    console: Console,
    width: int | None = None,
    height: int = 4,
) -> RenderableType:
    """
    Create a vertically stretched confidence timeline.

    - Each row represents an equal confidence band (top row is highest confidence).
    - Character height within a row encodes how much of that band is filled.
    - Color encodes match status: green if bucket has at least one match, cyan otherwise.
    """
    if not video.sample_confidences:
        return Text("-")

    spark_chars = [" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    confidences = video.sample_confidences
    n = len(confidences)
    timeline_width = width if width is not None else _get_timeline_width(console)
    row_count = max(1, height)
    rows = [Text() for _ in range(row_count)]
    match_set = set(video.match_indices)

    for i in range(timeline_width):
        start = int((i / timeline_width) * n)
        end = int(((i + 1) / timeline_width) * n)
        if end <= start:
            end = min(n, start + 1)

        bucket = confidences[start:end]
        bucket_conf = max(bucket) if bucket else 0.0

        has_match = any(idx in match_set for idx in range(start, end))
        style = "green bold" if has_match else "cyan"

        for row_index, row in enumerate(rows):
            band_top = 1.0 - (row_index / row_count)
            band_bottom = 1.0 - ((row_index + 1) / row_count)
            if bucket_conf <= band_bottom:
                fill = 0.0
            elif bucket_conf >= band_top:
                fill = 1.0
            else:
                fill = (bucket_conf - band_bottom) / (band_top - band_bottom)

            level = max(0, min(8, int(round(fill * 8))))
            row.append(spark_chars[level], style=style if level > 0 else "dim")

    return Group(*rows)


def _print_summary(console: Console, stats: ExtractionStats) -> None:
    """Print final summary statistics."""
    console.print()

    if stats.total_frames == 0:
        console.print(
            Panel(
                f"[bold yellow]No frames processed[/bold yellow]\n"
                f"Query: [cyan]{stats.query}[/cyan]",
                border_style="yellow",
            )
        )
        return

    console.print(
        Panel(
            f"[bold green]Extraction Complete[/bold green]\n"
            f"Query: [cyan]{stats.query}[/cyan]",
            border_style="green",
        )
    )

    processed_videos = [v for v in stats.videos if v.frames_processed > 0]
    if processed_videos:
        console.print(Text("Results by Video", style="bold italic", justify="center"))
        console.print()

        timeline_width = _get_timeline_width(console)
        total = len(processed_videos)

        for i, video in enumerate(processed_videos, start=1):
            timeline = _make_timeline(video, console, width=timeline_width, height=4)

            if video.sample_confidences:
                conf_min = min(video.sample_confidences)
                conf_avg = sum(video.sample_confidences) / len(video.sample_confidences)
                conf_max = max(video.sample_confidences)
                conf_stats = f"{conf_min:.3f}/{conf_avg:.3f}/{conf_max:.3f}"
            else:
                conf_stats = "-"

            start, end = video.analyzed_range

            metrics = Text()
            metrics.append("Range: ", style="dim")
            metrics.append(f"{start:.1f}s-{end:.1f}s", style="white")
            metrics.append("  │  ", style="dim")
            metrics.append("Analyzed: ", style="dim")
            metrics.append(f"{video.analyzed_duration:.1f}s", style="white")
            metrics.append("  │  ", style="dim")

            if stats.mode == "exhaustive":
                metrics.append("Segments: ", style="dim")
                metrics.append(str(video.segments_found), style="green bold")
                metrics.append("  │  ", style="dim")
                metrics.append("Frames extracted: ", style="dim")
                metrics.append(str(video.frames_extracted), style="white")
            else:
                metrics.append("Samples: ", style="dim")
                metrics.append(str(video.frames_processed), style="white")
                metrics.append("  │  ", style="dim")
                metrics.append("Matches: ", style="dim")
                metrics.append(str(video.matches_found), style="green bold")
                metrics.append(f" ({video.match_rate:.1f}%)", style="white")

            metrics.append("  │  ", style="dim")
            metrics.append("Time: ", style="dim")
            metrics.append(f"{video.processing_time:.1f}s", style="white")

            conf_line = Text()
            conf_line.append("Conf (min/avg/max): ", style="dim")
            conf_line.append(conf_stats, style="white")

            title = Text()
            title.append(_truncate_name(video.path.name, max_len=80), style="cyan bold")
            title.append(f" ({i}/{total})", style="dim")

            body = Group(
                timeline,
                conf_line,
                metrics,
            )
            console.print(
                Panel(
                    body,
                    title=title,
                    title_align="left",
                    border_style="blue",
                    expand=True,
                    padding=(0, 1),
                )
            )

    overall = Table.grid(padding=(0, 3))
    overall.add_column(style="bold")
    overall.add_column()

    overall.add_row("Total videos:", str(len(stats.videos)))

    if stats.mode == "exhaustive":
        overall.add_row("Total segments:", f"[green bold]{stats.total_segments}[/green bold]")
        overall.add_row("Total frames extracted:", str(stats.total_frames_extracted))
    else:
        overall.add_row("Total frames sampled:", str(stats.total_frames))
        overall.add_row("Total matches:", f"[green bold]{stats.total_matches}[/green bold]")

    overall.add_row("Total time:", f"{stats.elapsed_time:.1f}s")

    if stats.elapsed_time > 0:
        fps = stats.frames_per_second
        overall.add_row("Average speed:", f"{fps:.1f} frames/s")

    console.print()
    console.print(Panel(overall, title="Summary", border_style="blue", expand=False))


# ---------------------------------------------------------------------------
# QuickProgress — single progress bar per video (quick mode)
# ---------------------------------------------------------------------------


class ExtractionProgress:
    """
    Rich-based progress display for quick-mode video frame extraction.

    Usage:
        with ExtractionProgress(...) as progress:
            progress.start_video(...)
            progress.update(...)
            progress.finish_video()
    """

    def __init__(
        self,
        video_files: list[Path],
        query: str,
        mode: str,
        threshold: float,
        matcher_type: str,
        interval: float,
        batch_size: int,
        start_time_str: str | None = None,
        end_time_str: str | None = None,
    ):
        self.console = Console()
        self.stats = ExtractionStats(
            query=query,
            mode=mode,
            threshold=threshold,
            matcher_type=matcher_type,
        )
        self.video_files = video_files
        self._interval = interval
        self._batch_size = batch_size
        self._start_time_str = start_time_str
        self._end_time_str = end_time_str
        self._current_video_stats: VideoStats | None = None
        self._video_start_time: float = 0.0

        # Track match positions for the current video
        self._current_match_set: set[int] = set()

        # Progress bars - using custom match-aware bar for video progress
        self._progress = Progress(
            PhaseSpinnerColumn(),
            TextColumn("[progress.description]{task.description}", justify="left"),
            MatchAwareBarColumn(console=self.console),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )

        # Task IDs
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

        # Track overall frame progress
        self._overall_frames_processed: int = 0
        self._overall_frames_total: int = 0

    def _make_header(self) -> RenderableType:
        """Create a settings summary header above the progress table."""
        sep = Text("  │  ", style="dim")

        line1 = Text()
        line1.append("Query: ", style="dim")
        line1.append(f'"{self.stats.query}"', style="cyan")
        line1.append_text(sep)
        line1.append("Mode: ", style="dim")
        line1.append(self.stats.mode, style="white")
        line1.append_text(sep)
        line1.append("Matcher: ", style="dim")
        line1.append(self.stats.matcher_type, style="white")

        line2 = Text()
        line2.append("Threshold: ", style="dim")
        line2.append(f"{self.stats.threshold:.2f}", style="white")
        line2.append_text(sep)
        line2.append("Interval: ", style="dim")
        line2.append(f"{self._interval}s", style="white")
        line2.append_text(sep)
        line2.append("Batch: ", style="dim")
        line2.append(str(self._batch_size), style="white")
        line2.append_text(sep)
        line2.append("Range: ", style="dim")
        if self._start_time_str or self._end_time_str:
            start = self._start_time_str or "0s"
            end = self._end_time_str or "end"
            line2.append(f"{start} – {end}", style="white")
        else:
            line2.append("entire video", style="white")

        return Group(line1, line2)

    def _get_current_totals(self) -> tuple[int, int]:
        """Get total frames and matches including current video in progress."""
        total_frames = self.stats.total_frames
        total_matches = self.stats.total_matches

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

        n = len(self.video_files)
        done = self._current_video_index + 1 if self._is_running else len(self.stats.videos)
        pct = (
            int((self._overall_frames_processed / self._overall_frames_total) * 100)
            if self._overall_frames_total > 0
            else 0
        )
        elapsed = self.stats.elapsed_time
        elapsed_str = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
        stats_table.add_row(
            "Overall:",
            Text(f"{pct}%", style="bold blue"),
            "Videos:",
            f"{done}/{n}  {elapsed_str}",
        )

        total_frames, total_matches = self._get_current_totals()

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
            return Group(
                self._make_header(),
                Text(""),
                self._progress,
                Text(""),
                self._make_stats_panel(),
            )
        else:
            return self._progress

    def __enter__(self):
        """Start the live display."""
        for i, video_path in enumerate(self.video_files):
            video_name = _truncate_name(video_path.name)
            task_id = self._progress.add_task(
                f"[dim]{video_name}[/dim]",
                total=0,
                match_set=set(),
                total_frames=0,
                start=False,
            )
            self._video_tasks[i] = task_id

        self._live = Live(
            self._make_display(),
            console=self.console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the live display and show summary."""
        self._is_running = False
        if self._live:
            self._live.__exit__(exc_type, exc_val, exc_tb)
        _print_summary(self.console, self.stats)

    def start_video(
        self,
        video_path: Path,
        duration: float,
        estimated_frames: int,
        analyzed_start_time: float | None = None,
        analyzed_end_time: float | None = None,
    ) -> None:
        """Start processing a new video."""
        self._current_video_index += 1
        self._current_video_stats = VideoStats(
            path=video_path,
            duration=duration,
            analyzed_start_time=analyzed_start_time,
            analyzed_end_time=analyzed_end_time,
            total_frames=estimated_frames,
        )
        self._video_start_time = time.time()
        self._current_match_set = set()

        self._overall_frames_total += estimated_frames

        video_name = _truncate_name(video_path.name)
        self._video_task = self._video_tasks.get(self._current_video_index)

        if self._video_task is not None:
            self._progress.update(
                self._video_task,
                description=f"[cyan]{video_name}[/cyan]",
                total=estimated_frames,
                total_frames=estimated_frames,
                match_set=self._current_match_set,
            )
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
        self._current_video_stats.sample_confidences.append(confidence)

        if is_match:
            self._current_video_stats.matches_found += 1
            self._current_video_stats.confidences.append(confidence)
            self._current_video_stats.match_indices.append(frame_index)
            self._current_match_set.add(frame_index)

        self._overall_frames_processed += 1

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

        if self._video_task is not None:
            video_name = _truncate_name(self._current_video_stats.path.name)
            self._progress.update(
                self._video_task,
                description=f"[green]{video_name}[/green]",
            )
            self._video_task = None

        self._current_video_stats = None
        self._current_match_set = set()

        if self._live:
            self._live.update(self._make_display())

    def skip_video(self, video_path: Path, reason: str | None = None) -> None:
        """Mark a video as skipped/failed before processing starts."""
        self._current_video_index += 1
        self.stats.videos.append(VideoStats(path=video_path))

        task_id = self._video_tasks.get(self._current_video_index)
        if task_id is not None:
            video_name = _truncate_name(video_path.name)
            status = "error" if reason else "skipped"
            self._progress.update(
                task_id,
                description=f"[red]{video_name} ({status})[/red]",
            )

        if self._live:
            self._live.update(self._make_display())

    def log_saved_frame(self, output_path: Path, confidence: float) -> None:
        """Log a saved frame (shown below progress)."""
        pass


# Alias for backward compatibility
QuickProgress = ExtractionProgress


# ---------------------------------------------------------------------------
# ExhaustiveProgress — three-phase per video (exhaustive mode)
# ---------------------------------------------------------------------------


class ExhaustiveProgress:
    """
    Rich-based progress display for exhaustive-mode extraction.

    Uses a single Progress widget for overall + phase tasks so that columns
    align perfectly and spinners indicate the active task.

    Layout while running:
        Query: "A dark blue car"  │  Mode: exhaustive  │  Matcher: transformers_embedding
        Threshold: 0.70  │  Interval: 1.0s  │  Range: entire video
          ✓ video1.mp4  2 segments, 120 frames extracted, 8.3s
            video2.mp4
        ✓   Scanning     [━━green━━━grey━━━━━━━] 100%  Done (3 segments)  0:03:12
        ⠋   Refining     [━━━━green━━━━░░░░░░░░]  40%  2/3 segments      0:01:48
            Extracting   [░░░░░░░░░░░░░░░░░░░░░]   0%  waiting           -:--:--
            video3.mp4
        ┌─ Statistics ──────────────────────────────────┐
        │  Overall: 55%          Videos: 1/3  5:00      │
        │  Phase: Refining       Segments: 3            │
        │  Matches: 42           Frames: 1200           │
        │  Last conf: 0.823      Speed: 12.5 frames/s   │
        └───────────────────────────────────────────────┘
    """

    _PHASE_LABELS = {
        "scanning": "Scanning",
        "refining": "Refining",
        "extracting": "Extracting",
    }

    # Per-video overall-progress weights (must sum to 100)
    _W_SCAN = 50
    _W_REFINE = 30
    _W_EXTRACT = 20

    def __init__(
        self,
        video_files: list[Path],
        query: str,
        threshold: float,
        matcher_type: str,
        interval: float = 1.0,
        start_time_str: str | None = None,
        end_time_str: str | None = None,
    ):
        self.console = Console()
        self.stats = ExtractionStats(
            query=query,
            mode="exhaustive",
            threshold=threshold,
            matcher_type=matcher_type,
        )
        self.video_files = video_files
        self._interval = interval
        self._start_time_str = start_time_str
        self._end_time_str = end_time_str
        self._current_video_stats: VideoStats | None = None
        self._video_start_time: float = 0.0
        self._current_match_set: set[int] = set()
        self._current_phase: str = "scanning"

        self._live: Live | None = None
        self._is_running: bool = True

        self._last_confidence: float = 0.0
        self._last_was_match: bool = False

        # Single Progress widget — all tasks share columns → perfect alignment
        self._progress = Progress(
            PhaseSpinnerColumn(),
            TextColumn("{task.description}", justify="left"),
            MatchAwareBarColumn(console=self.console, bar_width=40),
            TaskProgressColumn(),
            TextColumn("{task.fields[status_text]}", justify="left"),
            TimeElapsedColumn(),
            console=self.console,
            expand=True,
        )

        self._scan_task = None
        self._refine_task = None
        self._extract_task = None

        # Completed-video one-line summaries
        self._completed_texts: list[Text] = []
        self._current_video_index: int = -1

        # For weighted overall progress (tracked in instance vars, not a Progress task)
        self._scan_total: int = 0
        self._refine_total: int = 0
        self._extract_total: int = 0
        self._overall_completed: int = 0
        self._overall_total: int = 0

    # ----- overall progress helpers -----

    def _video_base(self) -> int:
        """Base progress for the current video in overall units."""
        return self._current_video_index * 100

    def _update_overall(
        self, scan_done: int = 0, refine_done: int = 0, extract_done: int = 0
    ) -> None:
        """Recompute overall progress from phase completion fractions."""
        scan_frac = (scan_done / self._scan_total) if self._scan_total > 0 else 0
        refine_frac = (
            (refine_done / self._refine_total) if self._refine_total > 0 else 0
        )
        extract_frac = (
            (extract_done / self._extract_total) if self._extract_total > 0 else 0
        )

        weighted = (
            scan_frac * self._W_SCAN
            + refine_frac * self._W_REFINE
            + extract_frac * self._W_EXTRACT
        )
        self._overall_completed = self._video_base() + int(weighted)

    # ----- display composition -----

    def _make_header(self) -> RenderableType:
        """Create a settings summary header above the video list."""
        sep = Text("  │  ", style="dim")

        line1 = Text()
        line1.append("Query: ", style="dim")
        line1.append(f'"{self.stats.query}"', style="cyan")
        line1.append_text(sep)
        line1.append("Mode: ", style="dim")
        line1.append("exhaustive", style="white")
        line1.append_text(sep)
        line1.append("Matcher: ", style="dim")
        line1.append(self.stats.matcher_type, style="white")

        line2 = Text()
        line2.append("Threshold: ", style="dim")
        line2.append(f"{self.stats.threshold:.2f}", style="white")
        line2.append_text(sep)
        line2.append("Interval: ", style="dim")
        line2.append(f"{self._interval}s", style="white")
        line2.append_text(sep)
        line2.append("Range: ", style="dim")
        if self._start_time_str or self._end_time_str:
            start = self._start_time_str or "0s"
            end = self._end_time_str or "end"
            line2.append(f"{start} – {end}", style="white")
        else:
            line2.append("entire video", style="white")

        return Group(line1, line2)

    def _make_stats_panel(self) -> Panel:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="cyan", justify="right")
        grid.add_column(style="white")
        grid.add_column(style="cyan", justify="right")
        grid.add_column(style="white")

        # Overall progress row
        n = len(self.video_files)
        done = self._current_video_index + 1
        pct = int(self._overall_completed / self._overall_total * 100) if self._overall_total > 0 else 0
        elapsed = self.stats.elapsed_time
        elapsed_str = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
        grid.add_row(
            "Overall:",
            Text(f"{pct}%", style="bold blue"),
            "Videos:",
            f"{done}/{n}  {elapsed_str}",
        )

        total_segments, total_extracted = self._get_current_totals()

        grid.add_row(
            "Phase:",
            Text(
                self._PHASE_LABELS.get(self._current_phase, self._current_phase),
                style="bold",
            ),
            "Segments:",
            Text(str(total_segments), style="green bold"),
        )
        grid.add_row(
            "Frames extracted:",
            str(total_extracted),
            "Scanned:",
            str(self._current_video_stats.frames_processed)
            if self._current_video_stats
            else "0",
        )
        conf_text = f"{self._last_confidence:.3f}"
        match_style = "green bold" if self._last_was_match else "white"
        fps = total_extracted / elapsed if elapsed > 0 and total_extracted > 0 else 0.0
        grid.add_row(
            "Last conf:",
            Text(conf_text, style=match_style),
            "Speed:",
            f"{fps:.1f} frames/s",
        )

        return Panel(
            grid, title="[bold]Statistics[/bold]", border_style="blue", padding=(0, 1)
        )

    def _make_display(self) -> RenderableType:
        parts: list[RenderableType] = []

        # Settings header
        if self._is_running:
            parts.append(self._make_header())
            parts.append(Text(""))

        # Completed-video summaries
        for txt in self._completed_texts:
            parts.append(txt)

        # Active video name (separate line above progress)
        if self._scan_task is not None and self._current_video_stats:
            name = _truncate_name(self._current_video_stats.path.name)
            parts.append(Text(f"    {name}", style="cyan bold"))

        # Single progress widget: Scanning + Refining + Extracting
        parts.append(self._progress)

        # Pending videos
        for i in range(self._current_video_index + 1, len(self.video_files)):
            name = _truncate_name(self.video_files[i].name)
            parts.append(Text(f"    {name}", style="dim"))

        if self._is_running:
            parts.append(Text(""))
            parts.append(self._make_stats_panel())

        return Group(*parts)

    def _get_current_totals(self) -> tuple[int, int]:
        """Return (total_segments, total_frames_extracted) including in-progress video."""
        total_segments = self.stats.total_segments
        total_extracted = self.stats.total_frames_extracted
        if self._current_video_stats:
            total_segments += self._current_video_stats.segments_found
            total_extracted += self._current_video_stats.frames_extracted
        return total_segments, total_extracted

    # ----- context manager -----

    def __enter__(self):
        n = len(self.video_files)
        self._overall_total = n * 100
        self._overall_completed = 0

        self._live = Live(
            self._make_display(),
            console=self.console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._is_running = False
        if self._live:
            self._live.__exit__(exc_type, exc_val, exc_tb)
        _print_summary(self.console, self.stats)

    # ----- lifecycle -----

    def start_video(
        self,
        video_path: Path,
        duration: float,
        estimated_frames: int,
        analyzed_start_time: float | None = None,
        analyzed_end_time: float | None = None,
    ) -> None:
        self._current_video_index += 1
        self._current_video_stats = VideoStats(
            path=video_path,
            duration=duration,
            analyzed_start_time=analyzed_start_time,
            analyzed_end_time=analyzed_end_time,
            total_frames=estimated_frames,
        )
        self._video_start_time = time.time()
        self._current_match_set = set()
        self._current_phase = "scanning"
        self._scan_total = estimated_frames
        self._refine_total = 0
        self._extract_total = 0

        # Remove prior phase tasks (if any from previous video)
        for task_id in (self._scan_task, self._refine_task, self._extract_task):
            if task_id is not None:
                self._progress.remove_task(task_id)

        # Create three phase tasks in the shared Progress widget
        self._scan_task = self._progress.add_task(
            "[cyan]  Scanning[/cyan]",
            total=estimated_frames,
            match_set=self._current_match_set,
            total_frames=estimated_frames,
            status_text="",
        )
        self._refine_task = self._progress.add_task(
            "[dim]  Refining[/dim]",
            total=1,  # placeholder until segments known
            match_set=set(),
            total_frames=1,
            status_text="[dim]waiting[/dim]",
            bar_style="green bold",
            start=False,
        )
        self._extract_task = self._progress.add_task(
            "[dim]  Extracting[/dim]",
            total=1,  # placeholder until frame count known
            match_set=set(),
            total_frames=1,
            status_text="[dim]waiting[/dim]",
            bar_style="green bold",
            start=False,
        )

        if self._live:
            self._live.update(self._make_display())

    def update(self, confidence: float, is_match: bool) -> None:
        """Update scanning progress (phase 1)."""
        if self._current_video_stats is None:
            return

        frame_index = self._current_video_stats.frames_processed
        self._current_video_stats.frames_processed += 1
        self._last_confidence = confidence
        self._last_was_match = is_match
        self._current_video_stats.sample_confidences.append(confidence)

        if is_match:
            self._current_video_stats.matches_found += 1
            self._current_video_stats.confidences.append(confidence)
            self._current_video_stats.match_indices.append(frame_index)
            self._current_match_set.add(frame_index)

        if self._scan_task is not None:
            self._progress.update(
                self._scan_task,
                advance=1,
                match_set=self._current_match_set,
            )

        self._update_overall(
            scan_done=self._current_video_stats.frames_processed,
        )

        if self._live:
            self._live.update(self._make_display())

    def update_phase(self, phase: str, current: int, total: int) -> None:
        """Update phase progress (called by phase_callback from exhaustive_extract)."""

        if phase == "scan_complete":
            self._current_phase = "refining"
            segments = current
            if self._current_video_stats:
                self._current_video_stats.segments_found = segments

            # Mark scanning as done
            if self._scan_task is not None:
                scan_total = self._scan_total
                seg_text = f"[green]Done[/green] ({segments} segment{'s' if segments != 1 else ''})"
                self._progress.update(
                    self._scan_task,
                    completed=scan_total,
                    description="[green]  Scanning[/green]",
                    status_text=seg_text,
                )

            # Activate refining bar
            self._refine_total = segments
            if self._refine_task is not None:
                if segments > 0:
                    self._progress.update(
                        self._refine_task,
                        description="[cyan]  Refining[/cyan]",
                        total=segments,
                        total_frames=segments,
                        status_text=f"0/{segments} segments",
                    )
                    self._progress.start_task(self._refine_task)
                else:
                    self._progress.update(
                        self._refine_task,
                        description="[dim]  Refining[/dim]",
                        status_text="[dim]no segments[/dim]",
                    )

            self._update_overall(
                scan_done=self._scan_total,
            )

        elif phase == "refining":
            self._current_phase = "refining"
            if self._refine_task is not None:
                self._progress.update(
                    self._refine_task,
                    completed=current,
                    status_text=f"{current}/{total} segments",
                )

            self._update_overall(
                scan_done=self._scan_total,
                refine_done=current,
            )

        elif phase == "refine_complete":
            total_extract_frames = current

            # Mark refining as done
            if self._refine_task is not None:
                ref_total = self._refine_total
                self._progress.update(
                    self._refine_task,
                    completed=ref_total,
                    description="[green]  Refining[/green]",
                    status_text="[green]Done[/green]",
                )

            # Activate extracting bar
            self._current_phase = "extracting"
            self._extract_total = total_extract_frames
            if self._extract_task is not None:
                if total_extract_frames > 0:
                    self._progress.update(
                        self._extract_task,
                        description="[cyan]  Extracting[/cyan]",
                        total=total_extract_frames,
                        total_frames=total_extract_frames,
                        status_text=f"0/{total_extract_frames} frames",
                    )
                    self._progress.start_task(self._extract_task)
                else:
                    self._progress.update(
                        self._extract_task,
                        description="[dim]  Extracting[/dim]",
                        status_text="[dim]0 frames[/dim]",
                    )

            self._update_overall(
                scan_done=self._scan_total,
                refine_done=self._refine_total,
            )

        elif phase == "extracting":
            self._current_phase = "extracting"
            if self._current_video_stats:
                self._current_video_stats.frames_extracted = current

            if self._extract_task is not None:
                self._progress.update(
                    self._extract_task,
                    completed=current,
                    status_text=f"{current}/{total} frames",
                )

            self._update_overall(
                scan_done=self._scan_total,
                refine_done=self._refine_total,
                extract_done=current,
            )

        if self._live:
            self._live.update(self._make_display())

    def finish_video(self) -> None:
        if self._current_video_stats is None:
            return

        self._current_video_stats.processing_time = time.time() - self._video_start_time
        self.stats.videos.append(self._current_video_stats)

        # Mark extracting as done
        if self._extract_task is not None:
            ext_total = self._extract_total
            if ext_total > 0:
                self._progress.update(
                    self._extract_task,
                    completed=ext_total,
                    description="[green]  Extracting[/green]",
                    status_text="[green]Done[/green]",
                )

        # Snap overall to exact video boundary
        self._overall_completed = (self._current_video_index + 1) * 100

        # Remove phase tasks from widget
        for task_id in (self._scan_task, self._refine_task, self._extract_task):
            if task_id is not None:
                self._progress.remove_task(task_id)

        # Build completed-video summary line
        vs = self._current_video_stats
        name = _truncate_name(vs.path.name)
        segs = vs.segments_found
        extracted = vs.frames_extracted
        t = f"{vs.processing_time:.1f}s"
        summary = Text()
        summary.append("  ✓ ", style="green bold")
        summary.append(name, style="green")
        summary.append(
            f"  {segs} segment{'s' if segs != 1 else ''}, {extracted} frames extracted, {t}",
            style="dim",
        )
        self._completed_texts.append(summary)

        self._scan_task = None
        self._refine_task = None
        self._extract_task = None
        self._current_video_stats = None
        self._current_match_set = set()

        if self._live:
            self._live.update(self._make_display())

    def skip_video(self, video_path: Path, reason: str | None = None) -> None:
        """Mark a video as skipped/failed before processing starts."""
        self._current_video_index += 1
        self._overall_completed = min(
            self._overall_total, (self._current_video_index + 1) * 100
        )
        self.stats.videos.append(VideoStats(path=video_path))

        name = _truncate_name(video_path.name)
        summary = Text()
        summary.append("  ✗ ", style="red bold")
        summary.append(name, style="red")
        if reason:
            summary.append(f"  error: {reason}", style="dim")
        else:
            summary.append("  skipped", style="dim")
        self._completed_texts.append(summary)

        if self._live:
            self._live.update(self._make_display())

    def log_saved_frame(self, output_path: Path, confidence: float) -> None:
        pass

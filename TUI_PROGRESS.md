# TUI Progress Feature - Development Notes

## Branch: `feature/tui-progress`

## Current Status

**Quick mode** TUI is complete and working well.

**Exhaustive mode** TUI is implemented and functional — all three phases display
progress — but needs polish based on user testing feedback (see
[Open Issues](#open-issues-from-testing) below).

## What's Done

### Quick Mode (complete)

- Rich-based progress bars with green segments showing where matches occurred
- Overall progress bar + per-video progress bars
- All videos shown upfront before processing starts (dimmed until active)
- Live statistics panel showing:
  - Matches count, frames processed
  - Last confidence score (highlighted green on match)
  - Processing speed (frames/s), threshold, mode
- Full terminal width utilization (dynamic bar width)
- Summary table with timeline visualization, duration, frames, matches, etc.
- Error handling (minimal "No frames processed" output on failure)
- `--no-tui` flag to fall back to plain text output

### Exhaustive Mode (implemented, needs polish)

The `exhaustive_extract()` algorithm was restructured into three sequential phases
with a new `phase_callback` parameter. A dedicated `ExhaustiveProgress` class
renders all three phases in a single Rich `Progress` widget.

#### Architecture changes

- **`modes.py`**: `exhaustive_extract()` now accepts `phase_callback(phase, current, total)`.
  A helper `_identify_segments()` extracts runs of consecutive matches from scan
  results. The three phases fire callbacks:
  - `"scan_complete"` — after scanning, reports number of segments found
  - `"refining"` — per-segment progress during binary-search refinement
  - `"refine_complete"` — reports total frames to extract
  - `"extracting"` — per-frame progress (throttled to every 10 frames)

- **`tui.py`**: Shared helpers extracted to module level (`_truncate_name`,
  `_get_timeline_width`, `_make_timeline`, `_print_summary`) so both progress
  classes reuse them. `ExhaustiveProgress` class added with:
  - Single `Progress` widget containing Overall + Scanning + Refining + Extracting tasks
  - Weighted overall progress (Scan 50%, Refine 30%, Extract 20% per video × 100 units)
  - Phase tasks created on `start_video()`, removed on `finish_video()`
  - Completed videos shown as green summary lines above the active progress
  - Statistics panel with phase name, segments, matches, speed, etc.

- **`main.py`**: Chooses `ExhaustiveProgress` or `ExtractionProgress` based on
  `--mode`. Creates `phase_cb` lambda that calls `progress.update_phase()`.

#### Current layout (single video)

```text
  video-with-a-long-file-name.mp4
⠋   Overall          [━━━━━━━━━━━━━━━━━━░░░░░░]  49%  0/1 videos  0:05:49
⠋   Scanning         [━━green━━━red━━━━━━━━━━━]  99%              0:05:49
⠋   Refining         [░░░░░░░░░░░░░░░░░░░░░░░░]   0%  waiting    -:--:--
⠋   Extracting       [░░░░░░░░░░░░░░░░░░░░░░░░]   0%  waiting    -:--:--
┌─ Statistics ─────────────────────────────────────────────────┐
│  Phase: Scanning   Segments: 0   ...                         │
└──────────────────────────────────────────────────────────────┘
```

#### Current layout (multiple videos, 2nd video active)

```text
✓ video1.mp4  9 segments, 5899 frames extracted, 1177.1s
  video2.mp4
⠋   Overall          [━━━━━━━━━━━━━━━━━━░░░░░░]  60%  1/2 videos  0:20:30
⠋   Scanning         [━━━━━━━━━━━━━━░░░░░░░░░░]  42%              0:00:53
⠋   Refining         [░░░░░░░░░░░░░░░░░░░░░░░░]   0%  waiting    -:--:--
⠋   Extracting       [░░░░░░░░░░░░░░░░░░░░░░░░]   0%  waiting    -:--:--
  video3.mp4
┌─ Statistics ─────────────────────────────────────────────────┐
│  Phase: Scanning   Segments: 0   ...                         │
└──────────────────────────────────────────────────────────────┘
```

### Files Changed

- `extractor/tui.py` — TUI module: data classes, custom columns, QuickProgress, ExhaustiveProgress
- `extractor/modes.py` — `exhaustive_extract()` restructured, `_identify_segments()` added
- `main.py` — Integrated both progress classes, wires `phase_callback`
- `pyproject.toml` — Added `rich` dependency
- `uv.lock` — Updated lockfile

### Things We Learned

- Utilize available screen real estate — use full terminal width for progress bars,
  but don't truncate long file names when avoidable.
- Using separate `Progress` widgets for different task groups causes column
  misalignment. A **single Progress widget** with all tasks keeps columns lined up.
- `MatchAwareBarColumn` renders bar fill based on `task.fields["total_frames"]`,
  not `task.total`. Any task using it must set that field to the correct denominator
  or the bar stays empty.
- Weighted overall progress (scan 50 / refine 30 / extract 20) gives continuous
  feedback even for single-video runs, rather than jumping from 0% to 100%.

---

## Open Issues from Testing

Issues observed during test runs that still need to be fixed:

### 1. IndexError when finishing a video (crash)

The `finish_video()` and `update_phase()` methods access tasks by stored task ID
using `self._progress.tasks[self._scan_task]`. After `remove_task()` is called,
these integer indices become stale and cause `IndexError: list index out of range`.

**Root cause**: Rich's `Progress.remove_task()` removes the task from the internal
list, invalidating any saved `TaskID` that was really a list index.

**Fix**: Don't use `self._progress.tasks[task_id]` to look up tasks. Use
`self._progress._tasks[task_id]` (the dict keyed by TaskID) or, better, track
the needed totals in instance variables (`self._scan_total`, etc.) instead of
reading them back from the Progress widget.

### 2. Spinners shown on waiting/not-started phases

The `SpinnerColumn` renders a spinner on every visible task, including Refining
and Extracting while they're still in "waiting" state. The spinner should only
appear on the currently active phase.

**Fix**: Either use a custom column that checks `task.started` / `task.finished`,
or replace `SpinnerColumn` with a custom column that renders a spinner only when
the task is in progress and blank otherwise.

### 3. "Overall" row appears nested under the current video

When multiple videos are listed, the indented "Overall" line looks like it belongs
to the current video rather than being a top-level summary. This is confusing.

**Proposed fix**: Move the overall progress information into the Statistics panel
instead of having it as a Progress task row. The Statistics panel already shows
phase, matches, and speed — adding overall % and video count there would keep the
progress rows focused on the three phases.

### 4. Checkmark / filename column alignment

Completed videos show as `✓ video1.mp4` while pending videos show as
`video3.mp4`. The checkmark pushes the filename to the right, so the first
characters of completed vs. pending filenames don't line up.

**Fix**: Reserve space for the status indicator on all video lines. Use something
like `✓ name` / `name` (with a blank placeholder) so filenames always start
in the same column.

---

## Implementation Plan — Next Iteration

Priority order for the remaining fixes:

1. **Fix the crash** (IndexError) — track totals in instance variables, never
   index into `self._progress.tasks[]` with a stale TaskID.
2. **Smart spinners** — custom column or conditional rendering so only the active
   phase shows a spinner.
3. **Move Overall to Statistics panel** — remove the Overall task row, show
   overall % / video count / elapsed time inside the Statistics panel.
4. **Align filenames** — fixed-width prefix for completed (✓) / active / pending
   video lines.

After these fixes, re-test with:

- Single video: `./test1.sh` or `./test4.sh`
- Multiple videos: `./test5.sh`
- Quick mode regression: `./test3.sh`

### Phase Characteristics (reference)

| Phase | Deterministic? | Progress Type |
| ----- | -------------- | ------------- |
| Scanning | Yes | Percentage (current/total frames) |
| Refining | Semi | Percentage (segments refined/total segments) |
| Extracting | Yes (once boundaries known) | Percentage (frames extracted/total) |

---

## Usage

```bash
# Quick mode (fully working)
uv run main.py "video.mp4" "A person waving" --mode quick

# Exhaustive mode (three-phase progress, some polish needed)
uv run main.py "video.mp4" "A person waving" --mode exhaustive

# Disable TUI
uv run main.py "video.mp4" "A person waving" --no-tui
```

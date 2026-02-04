# TUI Progress Feature - Development Notes

## Branch: `feature/tui-progress`

## Current Status

The TUI works well for **Quick mode**. Exhaustive mode needs additional work.

## What's Done

### Features (working for Quick mode)

- Rich-based progress bars with green segments showing where matches occurred
- Overall progress bar + per-video progress bars
- All videos shown upfront before processing starts (dimmed until active)
- Live statistics panel showing:
  - Matches count
  - Frames processed
  - Last confidence score (highlighted green on match)
  - Processing speed (frames/s)
  - Threshold and mode
- Full terminal width utilization (dynamic bar width)
- Summary table with:
  - Timeline visualization per video
  - Duration, frames, matches, match rate, avg confidence, processing time
- Error handling (minimal "No frames processed" output on failure)
- `--no-tui` flag to fall back to plain text output

### Files Changed

- `extractor/tui.py` - New TUI module (main implementation)
- `main.py` - Integrated TUI, added `--no-tui` flag
- `pyproject.toml` - Added `rich` dependency
- `uv.lock` - Updated lockfile

## TODO: Exhaustive Mode Progress

### The Problem

Exhaustive mode has 3 phases, but currently only the first phase reports progress:

1. **Scanning** (has callbacks) - Sample frames at intervals, check for matches
2. **Refining** (NO callbacks) - Binary search to find exact segment boundaries
3. **Extracting** (NO callbacks) - Extract all frames within matched segments

After scanning completes, the TUI stops updating until the video finishes.

### Proposed Solution

Create a different UI layout for exhaustive mode with per-phase progress:

```text
video-with-a-long-file-name.mp4
    Scanning:   [━━━━━━━━━━━━━━━━━━━━] 100%  Done (found 3 segments)
    Refining:   [━━━━━━━━━━━━━━━━━━━━] 100%  2/3 segments (8 checks)
    Extracting: [━━━━━━━━━━░░░░░░░░░░]  65%  156/240 frames
```

### Implementation Steps

1. **Add `phase_callback` to `exhaustive_extract()` in `modes.py`**:

   ```python
   def exhaustive_extract(
       ...
       callback: callable = None,           # Existing: per-frame during scanning
       phase_callback: callable = None,     # New: phase transitions and progress
   )
   ```

   The `phase_callback` would be called with:
   - `phase_callback("scanning", current_frame, total_frames)`
   - `phase_callback("refining", segment_index, total_segments, checks_so_far)`
   - `phase_callback("extracting", frames_extracted, total_frames_to_extract)`

2. **Create `ExhaustiveProgress` class in `tui.py`**:
   - Different layout with video name as heading
   - Three sub-progress bars per video (Scanning, Refining, Extracting)
   - Handle the non-deterministic nature of Refining (show counter instead of percentage)

3. **Update `main.py`**:
   - Use different progress class based on mode
   - Pass the phase_callback to exhaustive_extract

### Phase Characteristics

| Phase | Deterministic? | Progress Type |
| ----- | -------------- | ------------- |
| Scanning | Yes | Percentage (current/total frames) |
| Refining | No | Counter (segments done + checks performed) |
| Extracting | Yes (once boundaries known) | Percentage (frames extracted/total) |

### Visual Continuity Considerations

- Keep the same general layout structure
- All three phases visible from the start (Refining and Extracting start dimmed/empty)
- Smooth transition between phases without jarring UI changes
- Statistics panel can show current phase name

## Usage

```bash
# Quick mode (TUI works fully)
uv run main.py "video.mp4" "A person waving" --mode quick

# Exhaustive mode (TUI only updates during scanning phase currently)
uv run main.py "video.mp4" "A person waving" --mode exhaustive

# Disable TUI
uv run main.py "video.mp4" "A person waving" --no-tui
```

# The Terminal as a Point-Cloud UI

> **Outcome.** The TUI is not an unrelated demo renderer. It consumes the exact `pp_detections` used by JSON and evaluation, projects points and boxes into a metric view, composes bounded layers into a 2×4 Braille framebuffer, and restores terminal state across normal exit, interrupts, suspend requests, and resize.

![Terminal renderer layer composition](assets/tui-layers.svg)

*A priority raster compresses metric layers into Unicode Braille while keeping class identity in ANSI color.*

## Why Braille

One Unicode Braille character encodes eight dots arranged as two columns by four rows. A terminal of `cols × rows` therefore presents a logical pixel grid of roughly:

```text
width  = 2 × cols
height = 4 × drawable_rows
```

[`pp_tui_render`](../src/tui.c) allocates one byte per logical pixel, plots world geometry, then converts each 2×4 block to a Braille code point. This provides eight times the binary spatial resolution of one-character-per-point ASCII without curses or a graphics server.

The renderer queries `TIOCGWINSZ` on every frame. Coordinates are projected after center translation, yaw rotation, and zoom. Resize signals request a redraw even while playback is paused.

## Layer priority

The byte framebuffer stores the highest-priority mark at each pixel:

| Value | Layer |
|---:|---|
| 1 | 10 m metric grid |
| 2 | ordinary point |
| 3 | elevated point |
| 10–19 | class-colored trail, velocity, or box |
| 30 | ego marker or selected target |

This is a tiny z-buffer. Grid lines cannot erase points, and points cannot erase boxes. During Braille encoding, the highest value in a cell selects ANSI color while all occupied subpixels contribute to the glyph mask.

## Geometry layers

- Points use XY position; Z chooses ordinary/elevated color.
- Rotated boxes use the same `x, y, dx, dy, yaw` decoded for evaluation.
- Velocity draws from box center to `(x + vx, y + vy)`.
- The metric grid is drawn every 10 m in world coordinates, then transformed with the view.
- Ego is a high-priority cross at world origin.

Class colors are stable across ten nuScenes classes. Keys `0`–`9` toggle class bits. `,` and `.` adjust score threshold. `[` and `]` select among currently visible detections, and the status panel shows pose, dimensions, yaw, velocity, score, local track ID, and history age.

## Bounded tracking, not a hidden tracker product

`pp_tui_update_tracks` maintains at most 64 tracks with 12 XY samples each. Each new box chooses the nearest unused track of the same class within 5 m. Tracks tolerate three missed frames. Reverse navigation or a non-consecutive frame resets all history.

This is visualization state, not benchmark output or nuScenes tracking. The code deliberately avoids pretending nearest-neighbor association is a production tracker. Its bounds make memory and worst-case matching work explicit.

## Interaction map

| Key | Action |
|---|---|
| Space | play/pause |
| Left/`p`, Right/`n` | previous/next frame |
| `WASD` | pan |
| `z` / `e` | rotate |
| `+` / `-` | zoom |
| `0`–`9`, `c` | toggle classes / restore all |
| `,` / `.` | score threshold |
| `[` / `]` | selected target |
| `b`, `v`, `g`, `t` | boxes, velocity, grid, trails |
| `r`, `q` | reset view, quit |

Paused redraws reuse the current points and detections; they do not rerun inference. Frame navigation reloads and evaluates the requested frame, then updates tracking once.

## Terminal ownership and recovery

`pp_tui_begin` requires both stdin and stdout to be TTYs, saves termios, disables canonical input and echo, enters the alternate screen, hides the cursor, and installs signal handlers. `pp_tui_end` restores termios, style, cursor, and primary screen.

Signal handlers do not call complex terminal APIs. They set `sig_atomic_t` flags; the poll loop returns a quit or redraw action, and normal control flow performs restoration. SIGINT, SIGTERM, SIGHUP, SIGQUIT, and SIGTSTP request clean exit. SIGWINCH requests redraw.

PTY tests compare termios before and after signal exit and check the final ANSI restore sequence. [`tests/test_tui.c`](../tests/test_tui.c) separately validates tracking growth, reset, and expiry.

## What to remember

- Visualization should consume the canonical result representation, or it will eventually disagree with evaluation.
- A terminal can support meaningful metric interaction when pixel packing and layer priority are explicit.
- Terminal state is an owned resource. Recovery paths deserve tests just like memory mappings and CUDA buffers.

## What remains

The TUI intentionally stays top-down and dependency-free. A future 3D view would need depth ordering and camera controls; at that point a real graphics backend may be more honest than forcing perspective into Braille.

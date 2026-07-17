# The Terminal as a Point-Cloud UI

> **Outcome.** The TUI is not an unrelated demo renderer. It consumes the exact `pp_detections` used by JSON and evaluation, projects points and boxes into an aspect-correct metric view, composes bounded layers into a 2×4 Braille framebuffer, and writes each complete responsive frame in one operation. The checked-in MP4 is recorded from this real ANSI output rather than a parallel mock visualization.

[![Live PointPillars terminal viewer](../docs/pointpillars-tui.png)](../docs/pointpillars-tui.mp4)

*The wide layout reserves a focused scene inspector without taking the BEV out of metric scale. Click to play the real 8-second capture.*

![Terminal renderer layer composition](assets/tui-layers.svg)

*A priority raster compresses metric layers into Unicode Braille while keeping class identity in ANSI color.*

## Why Braille

One Unicode Braille character encodes eight dots arranged as two columns by four rows. A terminal of `cols × rows` therefore presents a logical pixel grid of roughly:

```text
width  = 2 × cols
height = 4 × drawable_rows
```

[`pp_tui_compose`](../src/tui.c) allocates one byte per logical pixel, plots world geometry, then converts each 2×4 block to a Braille code point. This provides eight times the binary spatial resolution of one-character-per-point ASCII without curses or a graphics server. Its explicit `columns × rows` contract makes a frame deterministic and testable without owning a TTY. [`pp_tui_render`](../src/tui.c) only queries the live terminal size, composes, and performs a bounded `write` loop.

The renderer queries `TIOCGWINSZ` on every frame. Coordinates are projected after center translation, yaw rotation, and zoom. One shared metres-to-logical-pixels scale is chosen from the smaller canvas axis, so a 10 m circle stays circular instead of stretching to fill a rectangular terminal. Resize signals request a redraw even while playback is paused.

At 104 columns and 30 rows or larger, the layout becomes a dashboard: a framed BEV remains on the left while a 30-column panel shows per-class counts, active filters, the selected object, its local track, and camera state. Smaller terminals remove the panel and give the entire width to the BEV. Both layouts retain the same two-line header and two-line command/status footer; very small terminals show an explicit resize message instead of wrapping into corrupt output.

## Layer priority

The byte framebuffer stores the highest-priority mark at each pixel:

| Value | Layer |
|---:|---|
| 1 | 10 m range rings |
| 2 | forward/lateral axes |
| 3–5 | ground, low, and elevated points |
| 10–19 | dim class-colored trails |
| 20–29 | class-colored velocity vectors |
| 30–39 | bright class-colored boxes and heading marks |
| 60–61 | selected target and ego vehicle |

This is a tiny z-buffer. Grid lines cannot erase points, and points cannot erase boxes. During Braille encoding, the highest value in a cell selects ANSI color while all occupied subpixels contribute to the glyph mask.

## Geometry layers

- Points use XY position; Z chooses ordinary/elevated color.
- Rotated boxes use the same `x, y, dx, dy, yaw` decoded for evaluation; a center-to-front stroke makes heading legible even for nearly square boxes.
- Velocity draws from box center to `(x + vx, y + vy)`.
- Concentric range rings are drawn every 10 m with forward/lateral axes, then transformed with the view.
- Ego is a high-priority forward-facing triangular marker at world origin.

Class colors are stable across ten nuScenes classes. Keys `0`–`9` toggle class bits. `,` and `.` adjust the display score threshold from its uncluttered `0.20` default down to the actual `0.10` decode floor. `[` and `]` select among currently visible detections, and the status panel shows pose, dimensions, yaw, velocity, score, local track ID, and history age.

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
| `l`, `b`, `v`, `g`, `t` | points, boxes, velocity, rings, trails |
| `h` / `?` | contextual help panel |
| `r`, `q` | reset view, quit |

Paused redraws reuse the current points and detections; they do not rerun inference. Frame navigation reloads and evaluates the requested frame, then updates tracking once.

## Terminal ownership and recovery

`pp_tui_begin` requires both stdin and stdout to be TTYs, saves termios, disables canonical input and echo, enters the alternate screen, hides the cursor, and installs signal handlers. `pp_tui_end` restores termios, style, cursor, and primary screen.

Signal handlers do not call complex terminal APIs. They set `sig_atomic_t` flags; the poll loop returns a quit or redraw action, and normal control flow performs restoration. SIGINT, SIGTERM, SIGHUP, SIGQUIT, and SIGTSTP request clean exit. SIGWINCH requests redraw.

PTY tests compare termios before and after signal exit and check the final ANSI restore sequence. [`tests/test_tui.c`](../tests/test_tui.c) validates state defaults, tracking growth/reset/expiry, wide and compact layout bounds, required HUD content, Braille emission, and byte-for-byte deterministic composition.

## One frame, one write

The old renderer issued formatting calls while traversing every terminal cell. The new path builds the complete ANSI frame in a growable buffer, changes SGR state only when the winning pixel kind changes, and writes the result with an `EINTR`-safe bounded loop. This removes partial-frame tearing and makes the captured output identical to what an interactive terminal receives. Pixel and text buffers remain proportional to terminal area and are freed after the frame; visualization state itself stays fixed-size.

## Reproducible MP4

[`tools/record_tui.py`](../tools/record_tui.py) opens a 120×40 pseudo-terminal and runs the real `tui-cuda` mode over prepared nuScenes frames. It scripts selection, zoom, rotation, pause, filters, help, and layer controls; parses ANSI 256-color cells; rasterizes the Braille glyphs with a font that actually covers U+2800–U+28FF; and streams raw RGB frames to H.264 through `ffmpeg`.

```sh
make tui-video TUI_DATA=/data/nuscenes/pointpillars_10sweep
```

The checked artifact is 1200×720, 12 FPS, 8 seconds, YUV420P/H.264, and about 3.36 MiB. Its poster and MP4 live under `docs/` because they are README-facing documentation artifacts; transient recording state never enters the repository.

## What to remember

- Visualization should consume the canonical result representation, or it will eventually disagree with evaluation.
- A terminal can support meaningful metric interaction when pixel packing, physical aspect ratio, layer priority, and responsive information hierarchy are explicit.
- Terminal state is an owned resource. Recovery paths deserve tests just like memory mappings and CUDA buffers.

## What remains

The runtime TUI intentionally stays top-down and dependency-free. Pillow and `ffmpeg` are documentation-time dependencies only. A future 3D view would need depth ordering and camera controls; at that point a real graphics backend may be more honest than forcing perspective into Braille.

# Terminal Viewer

The TUI is an event-driven ANSI/Braille viewer over the same point tensor and
`pp_detections` returned by inference. It does not invent intermediate frames
or continuously animate completed data.

[![Terminal viewer recording](../docs/pointpillars-tui.gif)](../docs/pointpillars-tui.mp4)

The GIF is the GitHub-friendly preview; the linked MP4 preserves the original
recording quality.

## Rendering

One Braille character represents a 2x4 logical-pixel cell. Each composed frame
uses two terminal-sized byte buffers: one priority value and one saturated
point-density count per logical pixel. All in-view points contribute. Density
grades preserve collisions without random thinning.

Layer priority keeps boxes and the ego marker legible over points and range
rings. Point color encodes sweep age, intensity, and height. Perspective 3D
and metric BEV share pan, yaw, zoom, filters, selection, boxes, velocity, and
bounded local trails.

The foreground redraws only when:

- a completed inference result arrives;
- input changes view or filters;
- the terminal is resized.

Inference runs in one background worker. New navigation requests carry a
generation number, so a stale completed result cannot replace a newer request.
The input queue retains multiple keys returned by one `read()` and tolerates a
CSI arrow sequence split across reads.

## Controls

| Key | Action |
|---|---|
| Space | play/pause |
| Left/`p`, Right/`n` | previous/next frame |
| `WASD` | pan |
| `z` / `e` | rotate |
| `+` / `-` | zoom |
| `m` | perspective 3D / metric BEV |
| `i` | full canvas / inspector |
| `0`-`9`, `c` | class filters / restore all |
| `,` / `.` | score threshold |
| `[` / `]` | selected target |
| `l`, `b`, `v`, `g`, `t` | points, boxes, velocity, rings, trails |
| `h` / `?` | help |
| `r`, `q` | reset view, quit |

## Terminal ownership

`pp_tui_begin` requires interactive stdin/stdout, saves termios, enters the
alternate screen, and hides the cursor. Signal handlers only set atomic flags;
normal control flow restores termios, style, cursor, and the primary screen.

`tests/test_tui.c` checks PTY restoration, queued input, responsive layout,
deterministic composition, density, perspective height, BEV invariance,
selection, and track bounds.

#!/usr/bin/env python3
"""Record the real ANSI/Braille TUI from a PTY into a compact MP4."""

import argparse
import dataclasses
import fcntl
import os
import pathlib
import pty
import re
import select
import shutil
import struct
import subprocess
import sys
import termios
import time
import unicodedata

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as error:
    raise SystemExit("Pillow is required: python3 -m pip install Pillow") from error


HOME = b"\x1b[H"
CSI = re.compile(r"\x1b\[([0-9;?]*)([@-~])")
BASE_BACKGROUND = (18, 18, 18)


@dataclasses.dataclass(frozen=True)
class Cell:
    glyph: str = " "
    foreground: int = 252
    background: int = 233
    bold: bool = False
    dim: bool = False


def ansi_color(index: int) -> tuple[int, int, int]:
    basic = [
        (0, 0, 0), (205, 49, 49), (13, 188, 121), (229, 229, 16),
        (36, 114, 200), (188, 63, 188), (17, 168, 205), (229, 229, 229),
        (102, 102, 102), (241, 76, 76), (35, 209, 139), (245, 245, 67),
        (59, 142, 234), (214, 112, 214), (41, 184, 219), (255, 255, 255),
    ]
    if index < 16:
        return basic[max(index, 0)]
    if index < 232:
        value = index - 16
        levels = (0, 95, 135, 175, 215, 255)
        return levels[value // 36], levels[(value // 6) % 6], levels[value % 6]
    gray = 8 + 10 * min(index - 232, 23)
    return gray, gray, gray


def apply_sgr(parameters: list[int], style: dict[str, object]) -> None:
    if not parameters:
        parameters = [0]
    index = 0
    while index < len(parameters):
        value = parameters[index]
        if value == 0:
            style.update(fg=252, bg=233, bold=False, dim=False)
        elif value == 1:
            style["bold"] = True
            style["dim"] = False
        elif value == 2:
            style["dim"] = True
            style["bold"] = False
        elif value == 22:
            style["bold"] = False
            style["dim"] = False
        elif 30 <= value <= 37:
            style["fg"] = value - 30
        elif 90 <= value <= 97:
            style["fg"] = value - 90 + 8
        elif 40 <= value <= 47:
            style["bg"] = value - 40
        elif value in (38, 48) and index + 2 < len(parameters) and parameters[index + 1] == 5:
            style["fg" if value == 38 else "bg"] = parameters[index + 2]
            index += 2
        index += 1


def parse_screen(payload: bytes, columns: int, rows: int) -> list[list[Cell]]:
    screen = [[Cell() for _ in range(columns)] for _ in range(rows)]
    text = payload.decode("utf-8", "replace")
    style: dict[str, object] = {"fg": 252, "bg": 233, "bold": False, "dim": False}
    x = 0
    y = 0
    cursor = 0
    while cursor < len(text):
        if text[cursor] == "\x1b":
            match = CSI.match(text, cursor)
            if match:
                raw, command = match.groups()
                if command == "m":
                    values = [int(value) if value else 0 for value in raw.split(";")] if raw else [0]
                    apply_sgr(values, style)
                elif command in ("H", "f"):
                    values = [int(value) if value else 1 for value in raw.split(";")] if raw else [1, 1]
                    y = max(0, min(rows - 1, values[0] - 1))
                    x = max(0, min(columns - 1, (values[1] if len(values) > 1 else 1) - 1))
                elif command == "J" and raw in ("2", "3"):
                    screen = [[Cell(background=int(style["bg"])) for _ in range(columns)]
                              for _ in range(rows)]
                elif command == "K":
                    for column in range(x, columns):
                        screen[y][column] = Cell(background=int(style["bg"]))
                cursor = match.end()
                continue
            cursor += 1
            continue
        character = text[cursor]
        if character == "\n":
            y = min(rows - 1, y + 1)
            x = 0
        elif character == "\r":
            x = 0
        elif character >= " " and x < columns and y < rows:
            screen[y][x] = Cell(character, int(style["fg"]), int(style["bg"]),
                                bool(style["bold"]), bool(style["dim"]))
            width = 2 if unicodedata.east_asian_width(character) in "WF" else 1
            x += width
        cursor += 1
    return screen


def load_fonts(size: int) -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont,
                                   ImageFont.FreeTypeFont]:
    regular_candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    ]
    bold_candidates = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf",
    ]
    regular_path = next((path for path in regular_candidates if pathlib.Path(path).exists()), None)
    bold_path = next((path for path in bold_candidates if pathlib.Path(path).exists()), regular_path)
    if not regular_path:
        raise RuntimeError("no supported monospace font found")
    braille_candidates = [
        "/System/Library/Fonts/Apple Braille.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        regular_path,
    ]
    braille_path = next(path for path in braille_candidates if pathlib.Path(path).exists())
    return (ImageFont.truetype(regular_path, size),
            ImageFont.truetype(bold_path, size),
            ImageFont.truetype(braille_path, size))


def blend(foreground: tuple[int, int, int], background: tuple[int, int, int],
          amount: float) -> tuple[int, int, int]:
    return tuple(round(foreground[i] * amount + background[i] * (1.0 - amount))
                 for i in range(3))


def draw_screen(screen: list[list[Cell]], regular: ImageFont.FreeTypeFont,
                bold: ImageFont.FreeTypeFont, braille: ImageFont.FreeTypeFont,
                cell_width: int,
                cell_height: int) -> Image.Image:
    rows = len(screen)
    columns = len(screen[0])
    image = Image.new("RGB", (columns * cell_width, rows * cell_height), BASE_BACKGROUND)
    draw = ImageDraw.Draw(image)
    for row, cells in enumerate(screen):
        for column, cell in enumerate(cells):
            background = ansi_color(cell.background)
            x = column * cell_width
            y = row * cell_height
            if background != BASE_BACKGROUND:
                draw.rectangle((x, y, x + cell_width, y + cell_height), fill=background)
            if cell.glyph == " ":
                continue
            foreground = ansi_color(cell.foreground)
            if cell.dim:
                foreground = blend(foreground, background, 0.55)
            font = braille if "\u2800" <= cell.glyph <= "\u28ff" else bold if cell.bold else regular
            bounds = font.getbbox(cell.glyph)
            glyph_width = bounds[2] - bounds[0]
            glyph_x = x + max(0, (cell_width - glyph_width) // 2 - bounds[0])
            glyph_y = y + max(0, (cell_height - (bounds[3] - bounds[1])) // 2 - bounds[1])
            draw.text((glyph_x, glyph_y), cell.glyph, font=font, fill=foreground,
                      stroke_width=0)
    return image


def drain_until_exit(process: subprocess.Popen, master: int,
                     timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return True
        readable, _, _ = select.select([master], [], [], 0.05)
        if readable:
            try:
                if not os.read(master, 1 << 20):
                    break
            except (BlockingIOError, OSError):
                pass
    return process.poll() is not None


def capture_frames(command: list[str], columns: int, rows: int,
                   frame_count: int, timeout: float) -> tuple[list[bytes], float]:
    capture_begin = time.monotonic()
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios_tiocswinsz(), struct.pack("HHHH", rows, columns, 0, 0))
    environment = os.environ.copy()
    environment.update(TERM="xterm-256color", COLORTERM="")
    process = subprocess.Popen(command, stdin=slave, stdout=slave, stderr=slave,
                               env=environment, close_fds=True, start_new_session=True)
    os.close(slave)
    os.set_blocking(master, False)
    frames: list[bytes] = []
    buffer = b""
    deadline = time.monotonic() + timeout
    actions = {
        10: b"]", 20: b"+", 30: b"e", 38: b"e",
        48: b" ", 49: b"h", 50: b"]", 51: b"]", 52: b".",
        53: b".", 54: b"l", 55: b"l", 56: b"h", 57: b"+",
        58: b"e", 59: b" ", 64: b"m", 70: b"m",
        72: b"t", 78: b"t", 82: b"f", 88: b"f",
    }
    capture_seconds = 0.0
    try:
        while len(frames) < frame_count and time.monotonic() < deadline:
            readable, _, _ = select.select([master], [], [], 0.25)
            if readable:
                try:
                    chunk = os.read(master, 1 << 20)
                except BlockingIOError:
                    chunk = b""
                if chunk:
                    buffer += chunk
            first = buffer.find(HOME)
            if first >= 0:
                buffer = buffer[first:]
                second = buffer.find(HOME, len(HOME))
                while second >= 0 and len(frames) < frame_count:
                    frames.append(buffer[:second])
                    action = actions.get(len(frames))
                    if action:
                        os.write(master, action)
                    buffer = buffer[second:]
                    second = buffer.find(HOME, len(HOME))
            if process.poll() is not None and buffer.find(HOME, len(HOME)) < 0:
                break
        if len(frames) < frame_count:
            tail = buffer[-2000:].decode("utf-8", "replace")
            raise RuntimeError(f"captured {len(frames)}/{frame_count} frames\n{tail}")
        capture_seconds = time.monotonic() - capture_begin
    finally:
        if process.poll() is None:
            try:
                os.write(master, b"q")
                if not drain_until_exit(process, master, 0.5):
                    os.write(master, b"q")
                if not drain_until_exit(process, master, 2.5):
                    raise subprocess.TimeoutExpired(command, 3.0)
            except (OSError, subprocess.TimeoutExpired):
                process.terminate()
                if not drain_until_exit(process, master, 3.0):
                    process.kill()
                    process.wait()
        os.close(master)
    if process.returncode:
        raise RuntimeError(f"TUI exited with status {process.returncode}")
    return frames, capture_seconds


def termios_tiocswinsz() -> int:
    # Python exposes the native value on current macOS and Linux builds. Keep
    # the Linux fallback for older minimal Python distributions.
    return getattr(termios, "TIOCSWINSZ", 0x5414)


def encode_video(images: list[Image.Image], output: pathlib.Path, fps: int) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to encode the MP4")
    width, height = images[0].size
    command = [
        "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "slow",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert encoder.stdin is not None
    try:
        for index, image in enumerate(images):
            fade = min(1.0, (index + 1) / 6.0, (len(images) - index) / 6.0)
            if fade < 1.0:
                base = Image.new("RGB", image.size, BASE_BACKGROUND)
                image = Image.blend(base, image, fade)
            encoder.stdin.write(image.tobytes())
        encoder.stdin.close()
        result = encoder.wait()
    except Exception:
        encoder.kill()
        encoder.wait()
        raise
    if result:
        raise RuntimeError(f"ffmpeg exited with status {result}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=pathlib.Path)
    parser.add_argument("model", type=pathlib.Path)
    parser.add_argument("points", type=pathlib.Path,
                        help="directory containing prepared .bin frames")
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--poster", type=pathlib.Path)
    parser.add_argument("--mode", default="tui-cuda", choices=("tui", "tui-cuda"))
    parser.add_argument("--columns", type=int, default=120)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--font-size", type=int, default=17)
    parser.add_argument("--cell-width", type=int, default=10)
    parser.add_argument("--cell-height", type=int, default=18)
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    if args.columns < 32 or args.rows < 10 or args.frames < 2 or args.fps < 1:
        parser.error("terminal must be at least 32x10; frames >= 2 and fps >= 1")
    if (args.columns * args.cell_width) % 2 or (args.rows * args.cell_height) % 2:
        parser.error("H.264 YUV420P output width and height must both be even")
    for path in (args.binary, args.model, args.points):
        if not path.exists():
            parser.error(f"not found: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.poster:
        args.poster.parent.mkdir(parents=True, exist_ok=True)

    command = [str(args.binary.resolve()), args.mode, str(args.model.resolve()),
               str(args.points.resolve())]
    print(f"recording {args.columns}x{args.rows} PTY from {' '.join(command)}", file=sys.stderr)
    payloads, capture_seconds = capture_frames(command, args.columns, args.rows,
                                               args.frames, args.timeout)
    print(f"captured {len(payloads)} ANSI frames in {capture_seconds:.2f}s "
          f"({len(payloads) / capture_seconds:.1f} fps)", file=sys.stderr)
    regular, bold, braille = load_fonts(args.font_size)
    images = [draw_screen(parse_screen(payload, args.columns, args.rows), regular, bold, braille,
                          args.cell_width, args.cell_height) for payload in payloads]
    encode_video(images, args.output, args.fps)
    if args.poster:
        poster_index = min(38, len(images) - 1)
        images[poster_index].save(args.poster, optimize=True)
    duration = len(images) / args.fps
    size_mib = args.output.stat().st_size / 1048576.0
    print(f"wrote {args.output}: {len(images)} frames, {duration:.1f}s, {size_mib:.2f} MiB",
          file=sys.stderr)
    if args.poster:
        print(f"wrote {args.poster}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

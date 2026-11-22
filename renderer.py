"""Half-block ANSI presenter, terminal handling and PNG snapshots."""
from __future__ import annotations

import os
import shutil
import struct
import sys
import zlib

import numpy as np

HALF = "\u2580"  # upper half block: fg = top pixel, bg = bottom pixel


class Presenter:
    """Turns an indexed framebuffer + 256-color LUT into one ANSI string.

    Each terminal cell shows two vertically stacked pixels. Escape codes are
    only emitted when a cell's fg/bg index differs from its left neighbour.
    ``shift`` drops low index bits (256/128/64 levels) to get longer runs.
    """

    LEVELS = (0, 1, 2)

    def __init__(self, shift: int = 1):
        self.shift = shift

    def compose(self, idx: np.ndarray, lut: np.ndarray, row0: int = 1) -> str:
        if self.shift:
            idx = (idx >> self.shift) << self.shift
        h = idx.shape[0] // 2 * 2
        top, bot = idx[0:h:2], idx[1:h:2]
        cols = lut.tolist()
        fg = np.array([f"\x1b[38;2;{r};{g};{b}m" for r, g, b in cols], dtype=object)
        bg = np.array([f"\x1b[48;2;{r};{g};{b}m" for r, g, b in cols], dtype=object)
        fch = np.ones(top.shape, bool)
        bch = np.ones(top.shape, bool)
        fch[:, 1:] = top[:, 1:] != top[:, :-1]
        bch[:, 1:] = bot[:, 1:] != bot[:, :-1]
        cells = np.where(fch, fg[top], "") + np.where(bch, bg[bot], "") + HALF
        out = []
        for r in range(cells.shape[0]):
            out.append(f"\x1b[{r + row0};1H")
            out.append("".join(cells[r].tolist()))
        return "".join(out)


def to_rgb(idx: np.ndarray, lut: np.ndarray) -> np.ndarray:
    return lut[idx]


def save_png(path: str, rgb: np.ndarray, scale: int = 4, exclusive: bool = False) -> str:
    rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1).astype(np.uint8)
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    payload = (b"\x89PNG\r\n\x1a\n"
               + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(raw, 6))
               + chunk(b"IEND", b""))
    if exclusive:
        base, ext = os.path.splitext(path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        for attempt in range(100):
            candidate = path if attempt == 0 else f"{base}-{attempt}{ext}"
            try:
                fd = os.open(candidate, flags, 0o644)
            except FileExistsError:
                continue
            with os.fdopen(fd, "wb") as f:
                f.write(payload)
            return candidate
        raise RuntimeError("could not allocate a unique screenshot filename")
    with open(path, "wb") as f:
        f.write(payload)
    return path


class Terminal:
    """Alt-screen, hidden cursor, VT mode on Windows, non-blocking key reads."""

    def __enter__(self):
        self.win = os.name == "nt"
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, OSError, ValueError):
                pass
        if self.win:
            import ctypes
            k = ctypes.windll.kernel32
            h = k.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if k.GetConsoleMode(h, ctypes.byref(mode)):
                k.SetConsoleMode(h, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        else:
            import termios
            import tty
            self.fd = sys.stdin.fileno()
            self.old = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        self.write("\x1b[?1049h\x1b[?25l\x1b[2J")
        return self

    def __exit__(self, *exc):
        self.write("\x1b[0m\x1b[2J\x1b[?25h\x1b[?1049l")
        if not self.win:
            import termios
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    @staticmethod
    def write(s: str):
        sys.stdout.write(s)
        sys.stdout.flush()

    @staticmethod
    def size():
        s = shutil.get_terminal_size((100, 30))
        return max(20, s.columns), max(8, s.lines)

    def keys(self):
        out = []
        if self.win:
            import msvcrt
            while msvcrt.kbhit():
                ch = msvcrt.getwch()
                if ch in ("\x00", "\xe0"):
                    code = msvcrt.getwch()
                    out.append({"K": "LEFT", "M": "RIGHT", "H": "UP", "P": "DOWN"}.get(code, ""))
                else:
                    out.append(ch)
        else:
            import select
            data = ""
            while select.select([sys.stdin], [], [], 0)[0]:
                data += os.read(self.fd, 64).decode(errors="ignore")
            i = 0
            while i < len(data):
                if data.startswith("\x1b[", i) and i + 2 < len(data):
                    out.append({"D": "LEFT", "C": "RIGHT", "A": "UP", "B": "DOWN"}.get(data[i + 2], ""))
                    i += 3
                else:
                    out.append(data[i])
                    i += 1
        return out

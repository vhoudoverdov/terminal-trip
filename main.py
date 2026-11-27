"""terminal-trip: a 90s/2000s style music visualizer for the terminal.

    python main.py                 # system loopback audio (falls back to demo synth)
    python main.py --file song.mp3 # play + visualize a file
    python main.py --synth         # built-in procedural techno demo
    python main.py --bench 300 --snapshot shots   # headless benchmark + PNGs
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np

from audio import Analyzer, FileSource, LoopbackSource, SynthSource, sanitize_label
from effects import PRESETS
from palettes import PaletteManager
from renderer import Presenter, Terminal, save_png
from updater import __version__, check_for_update, check_for_update_cached, perform_update

HELP = [
    "TERMINAL-TRIP  hotkeys",
    "",
    "1-8 / \u2190 \u2192   switch visual preset",
    "Space       auto-cycle presets",
    "P           next color palette",
    "A           next audio source",
    "+ / -       audio sensitivity",
    "V           toggle 2x2 supersampling (hi-res AA)",
    "D           color detail (256/128/64)",
    "H           toggle bottom HUD",
    "S           save PNG screenshot",
    "?           toggle this help",
    "Q / Esc     quit",
]
TRANSITION = 0.8
AUTO_CYCLE = 20.0


class App:
    def __init__(self, args, source=None):
        self.args = args
        self.source_kinds = ["loopback", "synth"] + (["file"] if args.file else [])
        self.toast, self.toast_until = "", 0.0
        self.src = None
        if source is not None:
            self._use(source)
        else:
            first = "file" if args.file else "synth" if args.synth else "loopback"
            if not self.switch_source(first):
                self.switch_source("synth")
        self.effects = [cls() for cls in PRESETS]
        self.cur = (args.preset - 1) % len(self.effects)
        self.prev, self.trans_t = None, 0.0
        self.pal = PaletteManager(args.palette - 1)
        self.presenter = Presenter(shift=args.detail)
        self.supersample = not getattr(args, "no_ssaa", False)
        self.auto = args.auto
        self.last_switch = 0.0
        self.hud = not args.no_hud
        self.help = False
        self.fps = 0.0
        self.feats = None
        self.t = 0.0
        self.last_idx = None
        self.last_lut = None

    # ---------------------------------------------------------------- audio
    def _use(self, src):
        if self.src is not None:
            self.src.stop()
        self.src = src
        self.analyzer = Analyzer(src.samplerate)
        if hasattr(self, "gain"):
            self.analyzer.gain = self.gain

    def switch_source(self, kind):
        try:
            if kind == "loopback":
                src = LoopbackSource()
            elif kind == "file":
                src = FileSource(self.args.file)
            else:
                src = SynthSource()
            src.start()
        except Exception as e:
            self.say(f"{kind}: {e}", 6)
            return False
        self._use(src)
        self.say(f"audio: {src.label}")
        return True

    def next_source(self):
        i = self.source_kinds.index(self.src.name)
        for j in range(1, len(self.source_kinds) + 1):
            if self.switch_source(self.source_kinds[(i + j) % len(self.source_kinds)]):
                return

    def say(self, msg, secs=3.0):
        self.toast, self.toast_until = sanitize_label(msg, 80), time.perf_counter() + secs

    # -------------------------------------------------------------- presets
    def select(self, i, t):
        i %= len(self.effects)
        if i != self.cur:
            self.prev, self.trans_t, self.cur = self.cur, t, i
            self.last_switch = t
            self.say(self.effects[i].name)

    def frame(self, w, h, t, dt):
        self.t = t
        f = self.analyzer.process(self.src.read(self.analyzer.n), dt)
        self.feats = f
        scale = 2 if self.supersample else 1
        sw, sh = w * scale, h * scale

        eff = self.effects[self.cur]
        eff.ensure(sw, sh)
        out = eff.render(f, t, dt)
        if self.prev is not None:
            a = (t - self.trans_t) / TRANSITION
            if a >= 1:
                self.prev = None
            else:
                pe = self.effects[self.prev]
                pe.ensure(sw, sh)
                out = pe.render(f, t, dt) * (1 - a) + out * a
        if scale == 2:
            out = out.reshape(h, 2, w, 2).mean(axis=(1, 3))

        if self.auto and t - self.last_switch > AUTO_CYCLE and (f.beat or t - self.last_switch > AUTO_CYCLE + 4):
            self.select(self.cur + 1, t)
            self.pal.next()
        self.pal.update(dt, f.bass, f.beat)
        self.last_idx = np.clip(out, 0, 255).astype(np.uint8)
        self.last_lut = self.pal.lut()
        return self.last_idx, self.last_lut

    # ------------------------------------------------------------------ keys
    def key(self, k, t):
        if k in ("q", "Q", "\x1b", "\x03"):
            return False
        if k in "123456789" and len(k) == 1:
            idx = int(k) - 1
            if idx < len(self.effects):
                self.select(idx, t)
        elif k == "RIGHT":
            self.select(self.cur + 1, t)
        elif k == "LEFT":
            self.select(self.cur - 1, t)
        elif k == " ":
            self.auto = not self.auto
            self.last_switch = t
            self.say("auto-cycle " + ("on" if self.auto else "off"))
        elif k in ("p", "P"):
            self.pal.next()
            self.say("palette: " + self.pal.name)
        elif k in ("a", "A"):
            self.next_source()
        elif k in ("+", "="):
            self.gain = self.analyzer.gain = min(4.0, self.analyzer.gain * 1.25)
            self.say(f"sensitivity {self.analyzer.gain:.2f}")
        elif k in ("-", "_"):
            self.gain = self.analyzer.gain = max(0.25, self.analyzer.gain / 1.25)
            self.say(f"sensitivity {self.analyzer.gain:.2f}")
        elif k in ("v", "V"):
            self.supersample = not self.supersample
            self.say(f"2x2 supersampling AA: {'on' if self.supersample else 'off'}")
        elif k in ("d", "D"):
            self.presenter.shift = (self.presenter.shift + 1) % 3
            self.say(f"color detail: {256 >> self.presenter.shift} levels")
        elif k in ("h", "H"):
            self.hud = not self.hud
        elif k == "?":
            self.help = not self.help
        elif k in ("s", "S") and self.last_idx is not None:
            name = time.strftime("terminal-trip-%Y%m%d-%H%M%S.png")
            saved = save_png(name, self.last_lut[self.last_idx], exclusive=True)
            self.say("saved " + saved)
        return True

    # ------------------------------------------------------------------ HUD
    def overlay(self, cols, rows):
        s = []
        if self.hud:
            f = self.feats

            def vu(v):
                n = int(round(min(1.0, v) * 5))
                return "\u25ae" * n + "\u00b7" * (5 - n)

            bar = " \u2502 "
            heart = " \u2665" if f.pulse > 0.5 else "  "
            auto = bar + "AUTO" if self.auto else ""
            ssaa = " [2xAA]" if self.supersample else ""
            if time.perf_counter() < self.toast_until:
                extra = self.toast
            else:
                extra = f"B{vu(f.bass)} M{vu(f.mid)} T{vu(f.treble)}{heart}"
            safe_label = sanitize_label(self.src.label, 32)
            text = (f" TERMINAL-TRIP{bar}[{self.cur + 1}] {self.effects[self.cur].name}{bar}"
                    f"{self.pal.name}{ssaa}{bar}{safe_label}{bar}{self.fps:4.1f} fps"
                    f"{auto}{bar}{extra}{bar}? help")
            s.append(f"\x1b[{rows};1H\x1b[0m\x1b[38;2;235;235;235;48;2;10;10;24m"
                     + text[:cols].ljust(cols))
        if self.help:
            bw = max(len(x) for x in HELP) + 4
            x0 = max(1, (cols - bw) // 2 + 1)
            y0 = max(1, (rows - len(HELP) - 2) // 2)
            st = "\x1b[0m\x1b[38;2;120;255;220;48;2;8;8;30m"
            s.append(f"\x1b[{y0};{x0}H{st}\u250c" + "\u2500" * (bw - 2) + "\u2510")
            for i, line in enumerate(HELP):
                s.append(f"\x1b[{y0 + 1 + i};{x0}H{st}\u2502 " + line.ljust(bw - 4) + " \u2502")
            s.append(f"\x1b[{y0 + len(HELP) + 1};{x0}H{st}\u2514" + "\u2500" * (bw - 2) + "\u2518")
        return "".join(s)

    # ------------------------------------------------------------- main loop
    def _start_bg_update_check(self):
        if getattr(self.args, "no_update_check", False) or os.environ.get("TERMINAL_TRIP_NO_UPDATE_CHECK"):
            return

        def worker():
            info = check_for_update_cached()
            if info and info.get("update_available"):
                self.say(f"update v{info['latest']} available (run --update)", 6.0)

        threading.Thread(target=worker, daemon=True).start()

    def run(self, term):
        target = 1.0 / self.args.fps
        start = last = time.perf_counter()
        cols, rows = term.size()
        frame_no = 0
        self.say("press ? for help", 5)
        self._start_bg_update_check()
        while True:
            now = time.perf_counter()
            dt, last = min(now - last, 0.1), now
            t = now - start
            for k in term.keys():
                if not self.key(k, t):
                    return
            if frame_no % 10 == 0:
                size = term.size()
                if size != (cols, rows):
                    cols, rows = size
                    term.write("\x1b[0m\x1b[2J")
            idx, lut = self.frame(cols, rows * 2, t, dt)
            term.write(self.presenter.compose(idx, lut) + self.overlay(cols, rows))
            frame_no += 1
            if dt > 0:
                self.fps += (1.0 / dt - self.fps) * 0.05
            spare = target - (time.perf_counter() - now)
            if spare > 0:
                time.sleep(spare)
            if self.args.duration and t > self.args.duration:
                return


def bench(args):
    cols, rows = args.size
    w, h = cols, rows * 2
    sim = [0.0]
    src = SynthSource(clock=lambda: sim[0])
    src.start()
    app = App(args, source=src)
    if args.snapshot:
        os.makedirs(args.snapshot, exist_ok=True)
    dt = 1.0 / 60
    ssaa_str = "2x2 SSAA" if app.supersample else "1x1 native"
    print(f"canvas {w}x{h} px ({cols}x{rows} cells, {ssaa_str}), {args.bench} frames per preset")
    for i, eff in enumerate(app.effects):
        app.cur, app.prev = i, None
        tr = tc = 0.0
        nbytes = 0
        for _ in range(args.bench):
            sim[0] += dt
            t0 = time.perf_counter()
            idx, lut = app.frame(w, h, sim[0], dt)
            t1 = time.perf_counter()
            s = app.presenter.compose(idx, lut)
            tc += time.perf_counter() - t1
            tr += t1 - t0
            nbytes += len(s.encode("utf-8"))
        n = args.bench
        print(f"  [{i + 1}] {eff.name:<14} render {tr / n * 1e3:6.2f} ms  ansi {tc / n * 1e3:6.2f} ms"
              f"  ~{1 / ((tr + tc) / n):5.0f} fps max  {nbytes / n / 1024:6.1f} KiB/frame")
        if args.snapshot:
            save_png(os.path.join(args.snapshot, f"preset{i + 1}.png"), lut[idx])


def _parse_fps(val: str) -> float:
    fps = float(val)
    if not (1.0 <= fps <= 240.0):
        raise argparse.ArgumentTypeError("--fps must be between 1 and 240")
    return fps


def _parse_size(val: str) -> tuple[int, int]:
    parts = val.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--size must be formatted as COLSxROWS (e.g. 120x40)")
    cols, rows = int(parts[0]), int(parts[1])
    if not (20 <= cols <= 1000 and 8 <= rows <= 500):
        raise argparse.ArgumentTypeError("--size must be between 20x8 and 1000x500")
    return cols, rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="terminal-trip", description="Retro terminal music visualizer")
    p.add_argument("--version", action="version", version=f"terminal-trip {__version__}")
    p.add_argument("--check-update", action="store_true", help="check GitHub for a newer release and exit")
    p.add_argument("--update", action="store_true", help="self-update terminal-trip to the latest release and exit")
    p.add_argument("--no-update-check", action="store_true", help="disable background startup update check")
    p.add_argument("--file", help="audio file to play and visualize (wav; mp3/flac/ogg need miniaudio)")
    p.add_argument("--synth", action="store_true", help="start with the built-in demo synth")
    p.add_argument("--preset", type=int, default=1, help="starting preset 1-8")
    p.add_argument("--palette", type=int, default=1, help="starting palette 1-6")
    p.add_argument("--fps", type=_parse_fps, default=60.0)
    p.add_argument("--detail", type=int, default=0, choices=(0, 1, 2),
                   help="drop N low bits of color index (0=256 levels, 1=128, 2=64)")
    p.add_argument("--no-ssaa", action="store_true",
                   help="start with 2x2 supersampling anti-aliasing disabled (1:1 pixel grid)")
    p.add_argument("--auto", action="store_true", help="start with auto-cycling presets")
    p.add_argument("--no-hud", action="store_true")
    p.add_argument("--duration", type=float, default=0, help="exit after N seconds")
    p.add_argument("--bench", type=int, default=0, help="headless: render N frames per preset and report timings")
    p.add_argument("--size", type=_parse_size, default=(120, 40),
                   help="bench canvas size in terminal cells, e.g. 120x40")
    p.add_argument("--snapshot", help="bench: directory to write one PNG per preset")
    return p.parse_args(argv)


def main(argv=None, fetcher=None, runner=None):
    args = parse_args(argv)
    if args.check_update:
        try:
            info = check_for_update(fetcher=fetcher)
        except Exception as e:
            print(f"Update check failed: {e}", file=sys.stderr)
            return 1
        if info["update_available"]:
            print(
                f"Update available: v{info['current']} -> v{info['latest']} "
                f"({info['source']}). Run 'terminal-trip --update' to install."
            )
        else:
            print(f"terminal-trip is up to date (v{info['current']}, {info['source']}).")
        return 0
    if args.update:
        kwargs = {"fetcher": fetcher}
        if runner is not None:
            kwargs["runner"] = runner
        ok, msg = perform_update(**kwargs)
        stream = sys.stdout if ok else sys.stderr
        print(msg, file=stream)
        return 0 if ok else 1
    if args.bench:
        bench(args)
        return 0
    app = App(args)
    try:
        with Terminal() as term:
            app.run(term)
    except KeyboardInterrupt:
        pass
    finally:
        app.src.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Preset 3: Windows Media Player "Bars and Waves" tribute.

Log-spaced spectrum bars with gravity peak-hold caps, a glowing oscilloscope,
and a rising/decaying heat trail behind everything.
"""
from __future__ import annotations

import numpy as np

from .base import Effect, resample


class BarsAndWaves(Effect):
    name = "Bars & Waves"

    def setup(self):
        w, h = self.w, self.h
        self.fb = np.zeros((h, w), np.float32)
        self.nb = int(np.clip(w // 3, 8, 48))
        self.bw = max(1, w // self.nb)
        off = (w - self.bw * self.nb) // 2
        cols = np.arange(w) - off
        self.col_bar = np.clip(cols // self.bw, 0, self.nb - 1)
        self.col_ok = (cols >= 0) & (cols < self.bw * self.nb)
        if self.bw >= 3:
            self.col_ok &= (cols % self.bw) != self.bw - 1   # 1px gap between bars
        self.rows = np.arange(h, dtype=np.float32)[:, None]
        self.grad = (60 + 170 * (1 - np.arange(h) / h)).astype(np.float32)[:, None]
        self.peaks = np.zeros(self.nb)
        self.hold = np.zeros(self.nb)
        self.vel = np.zeros(self.nb)

    def render(self, f, t, dt):
        w, h, k = self.w, self.h, dt * 60
        fb = np.zeros_like(self.fb)
        fb[:-1] = self.fb[1:] * (0.80 ** k)                   # heat rises and fades

        # --- spectrum bars
        hp = resample(f.bars, self.nb) * 0.62 * h
        heights = hp[self.col_bar]
        mask = (self.rows >= h - heights[None, :]) & self.col_ok[None, :]
        fb = np.maximum(fb, np.where(mask, self.grad, 0))

        # --- glowing oscilloscope
        y = np.clip(0.30 * h + resample(f.wave, w) * 0.22 * h, 0, h - 1)
        yp = np.concatenate(([y[0]], y[:-1]))
        lo, hi = np.minimum(y, yp), np.maximum(y, yp)
        core = (self.rows >= np.floor(lo)) & (self.rows <= np.ceil(hi))
        glow = (self.rows >= lo - 2.5) & (self.rows <= hi + 2.5)
        fb = np.maximum(fb, np.where(core, 230 + 25 * f.pulse, np.where(glow, 110, 0)))
        self.fb = fb                    # trail excludes peak caps (no dashed smears)

        # --- gravity peak-hold caps
        up = hp > self.peaks
        self.peaks = np.where(up, hp, self.peaks)
        self.hold = np.where(up, 0.4, self.hold - dt)
        self.vel = np.where(up | (self.hold > 0), 0.0, self.vel + h * 2.5 * dt)
        self.peaks = np.maximum(self.peaks - self.vel * dt, 0)
        pr = np.clip(h - 1 - self.peaks[self.col_bar], 0, h - 1).astype(np.int32)
        pm = self.col_ok & (self.peaks[self.col_bar] > 1)
        out = fb.copy()
        out[pr[pm], np.nonzero(pm)[0]] = 255.0
        return out

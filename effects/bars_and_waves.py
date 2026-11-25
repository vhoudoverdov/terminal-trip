"""Preset 3: Windows Media Player "Bars and Waves" tribute.

Log-spaced spectrum bars with sub-pixel anti-aliased tops, membrane-relief
specular shading, gravity peak-hold caps, a continuous CRT phosphor SDF
oscilloscope beam, and a rising diffused heat trail.
"""
from __future__ import annotations

import numpy as np

from .base import Effect, blur3x3, resample


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
            self.col_ok &= (cols % self.bw) != self.bw - 1  # gap between bars
            local_x = (cols % self.bw) / max(1.0, float(self.bw - 2))
            self.bar_bevel = (0.82 + 0.18 * np.sin(np.clip(local_x, 0.0, 1.0) * np.pi)).astype(np.float32)
        else:
            self.bar_bevel = np.ones(w, dtype=np.float32)
        self.rows = np.arange(h, dtype=np.float32)[:, None]
        self.grad = (55.0 + 170.0 * (1.0 - np.arange(h, dtype=np.float32) / h))[:, None]
        self.peaks = np.zeros(self.nb, dtype=np.float32)
        self.hold = np.zeros(self.nb, dtype=np.float32)
        self.vel = np.zeros(self.nb, dtype=np.float32)

    def render(self, f, t, dt):
        w, h, k = self.w, self.h, dt * 60.0
        scale = max(1.0, h / 80.0)
        shift = max(1, int(round(scale)))
        fb = np.zeros_like(self.fb)
        fb[:-shift] = blur3x3(self.fb[shift:], mix=0.18) * (0.82 ** k)  # diffused heat rises

        # --- spectrum bars with sub-pixel fractional top coverage & specular tip
        hp = resample(f.bars, self.nb) * 0.62 * h
        heights = hp[self.col_bar][None, :]
        top_y = (h - 1) - heights
        dist_from_top = self.rows - top_y
        # Smooth sub-pixel coverage at bar top edge
        cov = np.clip(dist_from_top + 1.0, 0.0, 1.0) * self.col_ok[None, :]
        # Incandescent rim highlight right at the bar crest
        rim = np.exp(-((np.maximum(dist_from_top, 0.0) / (1.8 * scale)) ** 2)) * 45.0
        bar_val = cov * (self.grad * self.bar_bevel[None, :] + rim)
        fb = np.maximum(fb, np.clip(bar_val, 0.0, 245.0))

        # --- continuous CRT phosphor Signed Distance Field (SDF) oscilloscope
        y = np.clip(0.30 * h + resample(f.wave, w) * 0.22 * h, 0.0, h - 1.0)
        yp = np.concatenate(([y[0]], y[:-1]))
        lo = np.minimum(y, yp)[None, :]
        hi = np.maximum(y, yp)[None, :]
        # Vertical distance to the line segment connecting (x-1, yp) to (x, y)
        d = np.maximum(0.0, np.maximum(lo - self.rows, self.rows - hi)) / scale
        core = np.exp(-(d ** 2) / (2.0 * (0.55 ** 2))) * (235.0 + 20.0 * f.pulse)
        halo = np.exp(-d / 1.85) * (115.0 + 30.0 * f.treble)
        fb = np.maximum(fb, np.minimum(core + halo, 255.0))
        self.fb = fb  # trail excludes peak caps (no dashed vertical smears)

        # --- gravity peak-hold caps with sub-pixel anti-aliasing
        up = hp > self.peaks
        self.peaks = np.where(up, hp, self.peaks)
        self.hold = np.where(up, 0.4, self.hold - dt)
        self.vel = np.where(up | (self.hold > 0), 0.0, self.vel + h * 2.5 * dt)
        self.peaks = np.maximum(self.peaks - self.vel * dt, 0.0)

        peak_y = np.clip((h - 1.0) - self.peaks[self.col_bar], 0.0, h - 1.0)[None, :]
        peak_dist = np.abs(self.rows - peak_y) / scale
        peak_mask = (self.col_ok & (self.peaks[self.col_bar] > 1.0))[None, :]
        cap_glow = np.exp(-(peak_dist ** 2) / (2.0 * (0.6 ** 2))) * 255.0 * peak_mask
        return np.maximum(fb, cap_glow)

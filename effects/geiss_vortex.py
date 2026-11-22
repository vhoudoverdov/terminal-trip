"""Preset 1: Geiss / MilkDrop style feedback vortex.

Each frame the previous frame is re-sampled through a polar warp (zoom +
rotation + petal wobble), dimmed a few percent, and a fresh waveform plus
beat sparks are drawn on top. The feedback turns a simple line into a vortex.
"""
from __future__ import annotations

import numpy as np

from .base import Effect, bilinear, grid, plot, resample


class GeissVortex(Effect):
    name = "Geiss Vortex"

    def setup(self):
        X, Y, self.aspect = grid(self.w, self.h)
        self.r = np.hypot(X, Y)
        self.th = np.arctan2(Y, X)
        self.fb = np.zeros((self.h, self.w), np.float32)
        self.petals = 4
        self.zoom_dir = 1.0
        self.rot_dir = 1.0
        self.shape = 0
        self.last_change = -99.0

    def _mutate(self, t):
        rng = self.rng
        self.petals = int(rng.choice([2, 3, 4, 5, 6]))
        self.rot_dir = float(rng.choice([-1.0, 1.0]))
        self.zoom_dir = float(rng.choice([1.0, 1.0, 1.0, -0.6]))
        self.shape = (self.shape + 1) % 3
        self.last_change = t

    def render(self, f, t, dt):
        k = dt * 60.0
        if f.beat and t - self.last_change > 7.0:
            self._mutate(t)
        r, th, a = self.r, self.th, self.aspect

        # --- warp field (where each pixel samples the previous frame from)
        zoom = (0.010 + 0.05 * f.bass) * self.zoom_dir * k
        s = 1.0 - zoom + 0.012 * np.sin(self.petals * th + t * 1.3) * (0.4 + f.treble)
        rot = (0.008 + 0.03 * f.mid) * self.rot_dir * k + 0.02 * np.sin(3 * r - t * 2) * (0.3 + f.mid)
        r2 = r * s
        th2 = th + rot
        u = (r2 * np.cos(th2) / a + 1) * 0.5 * (self.w - 1)
        v = (r2 * np.sin(th2) + 1) * 0.5 * (self.h - 1)
        fb = bilinear(self.fb, u, v) * (0.955 ** k)

        # --- fresh geometry on top
        n = 3 * (self.w + self.h)
        wave = resample(f.wave, n)
        intensity = 170 + 85 * f.pulse
        R = 0.35 + 0.25 * f.bass
        if self.shape == 0:      # breathing ring
            ang = np.linspace(0, 2 * np.pi, n) + t * 0.5
            rad = R + 0.22 * wave
            x, y = rad * np.cos(ang), rad * np.sin(ang)
        elif self.shape == 1:    # horizontal oscilloscope ribbon
            x = np.linspace(-a * 0.9, a * 0.9, n)
            y = 0.45 * wave * np.cos(t * 0.4)
        else:                    # lissajous of the wave against itself
            x = 0.6 * wave * min(a, 1.6)
            y = 0.6 * np.roll(wave, n // 5)
        plot(fb, x, y, intensity, a, glow=0.55)

        if f.beat:  # sparks along the ring that get sucked into the vortex
            m = 48
            ang = self.rng.uniform(0, 2 * np.pi, m)
            rad = R + self.rng.normal(0, 0.05, m)
            plot(fb, rad * np.cos(ang), rad * np.sin(ang), 255.0, a, glow=0.8)
        nt = int(f.treble * 12)
        if nt:
            plot(fb, self.rng.uniform(-a, a, nt), self.rng.uniform(-1, 1, nt), 200.0, a)

        self.fb = np.minimum(fb, 255.0)
        return self.fb

"""Preset 4: demoscene sine-interference plasma with a spectrum-fed fire buffer."""
from __future__ import annotations

import numpy as np

from .base import Effect, grid


class PlasmaFire(Effect):
    name = "Plasma & Fire"

    def setup(self):
        self.X, self.Y, self.aspect = grid(self.w, self.h)
        self.R = np.hypot(self.X, self.Y)
        self.fire = np.zeros((self.h + 2, self.w), np.float32)
        self.phase = 0.0

    def render(self, f, t, dt):
        w, h = self.w, self.h
        X, Y = self.X, self.Y
        tt = t * 0.8
        self.phase += dt * (1.0 + 2.0 * f.bass)
        c, s = np.cos(tt * 0.3), np.sin(tt * 0.3)
        p = (np.sin(X * 3 + tt) + np.sin(Y * 4 - 1.3 * tt)
             + np.sin(3 * (X * c + Y * s) + tt) + np.sin(5 * self.R - 2 * self.phase))
        band = 0.5 + 0.5 * np.sin(p * np.pi * (1.0 + 0.5 * f.treble))     # classic contour bands
        plasma = 6 + 84 * band * (0.45 + 0.55 * f.mid)

        # seed the bottom rows with random hot spots, denser where the spectrum is loud
        nb = len(f.bars)
        spec = np.interp(np.linspace(0, nb - 1, w), np.arange(nb), f.bars) ** 0.5
        hot = self.rng.random(w) < (0.2 + 0.6 * spec) * (0.6 + 0.6 * f.bass)
        seed = np.where(hot, 255.0, 40.0 * spec)
        if f.beat:
            seed[self.rng.random(w) < 0.4] = 255
        F = self.fire
        F[-1] = seed
        F[-2] = np.maximum(F[-2] * 0.5, seed)

        cool = 255.0 / (2.0 * h) * (1.3 - 0.8 * f.bass)
        for _ in range(max(2, round(h / 40))):
            left, right = np.roll(F, 1, axis=1), np.roll(F, -1, axis=1)
            new = F.copy()
            new[:-2] = (left[1:-1] + F[1:-1] + right[1:-1] + F[2:]) / 4 - cool
            F = np.maximum(new, 0)
        self.fire = F
        heat = F[:h]
        return np.minimum(plasma * (1 - np.clip(heat / 140, 0, 0.85)) + heat * 1.1, 255)

"""Preset 2: classic demoscene lookup-table wormhole with kaleidoscope folding.

Depth (1/r) and angle tables are precomputed once for a grid twice the screen
size; each frame we just slice a moving window out of them (the old trick for
a wandering tunnel centre) and use them as texture coordinates.
"""
from __future__ import annotations

import numpy as np

from .base import Effect, grid


class Wormhole(Effect):
    name = "LUT Wormhole"

    def setup(self):
        X, Y, self.aspect = grid(2 * self.w, 2 * self.h, scale=2.0)
        r = np.hypot(X, Y)
        self.depth = (0.5 / np.maximum(r, 0.03)).astype(np.float32)
        self.angle = (np.arctan2(Y, X) / np.pi).astype(np.float32)        # -1..1
        self.shade = (np.clip(r * 1.6, 0, 1) ** 1.2).astype(np.float32)   # dark core
        self.z = 0.0
        self.spin = 0.0
        self.spin_dir = 1.0
        self.folds = 6
        self.tex_mode = 0
        self.last_change = -99.0

    def render(self, f, t, dt):
        if f.beat and t - self.last_change > 8.0:
            self.folds = int(self.rng.choice([4, 6, 8, 10]))
            self.tex_mode ^= 1
            self.spin_dir = float(self.rng.choice([-1.0, 1.0]))
            self.last_change = t
        self.z += dt * (0.6 + 2.5 * f.bass + 1.5 * f.pulse)
        self.spin += dt * (0.15 + 0.6 * f.mid) * self.spin_dir

        w, h = self.w, self.h
        dx, dy = 0.45 * np.sin(t * 0.37), 0.4 * np.sin(t * 0.53)
        ox = int(w / 2 - dx * w / 2)
        oy = int(h / 2 - dy * h / 2)
        D = self.depth[oy:oy + h, ox:ox + w]
        A = self.angle[oy:oy + h, ox:ox + w]
        S = self.shade[oy:oy + h, ox:ox + w]

        u = D * (1.0 + 0.25 * f.bass) + self.z
        a = A * (self.folds / 2) + self.spin + 0.15 * np.sin(u * 1.5 + t)
        v = np.abs((a % 2.0) - 1.0)                                       # mirrored 0..1

        if self.tex_mode == 0:   # XOR texture, the demoscene classic
            tex = (((u * 32).astype(np.int32) ^ (v * 64).astype(np.int32)) & 63) * 3.4 + 30
        else:                    # flowing plasma walls
            tex = (127 + 64 * np.sin(u * 9.4 + 4 * np.sin(v * 6.28 + t))
                   + 64 * np.sin(v * 12.6 + u * 2 - t * 1.7))

        nb = len(f.bars)
        spec = f.bars[np.minimum((v * nb).astype(np.int32), nb - 1)]
        rings = np.sin(u * 12.6 - t * 4) > 0.7
        tex = tex * (0.55 + 0.45 * spec) + 110 * spec * rings
        return np.clip(tex, 0, 255) * S * (0.8 + 0.2 * f.pulse)

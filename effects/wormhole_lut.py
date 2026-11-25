"""Preset 2: Classic demoscene wormhole with sub-pixel camera glide and kaleidoscope folding.

Evaluates polar depth (1/r) and angle around a continuously drifting sub-pixel
Lissajous camera center (eliminating integer LUT stepping), applies smooth
folds and distance-attenuated procedural textures, and pulses anti-aliased
spectrum rings toward the viewer.
"""
from __future__ import annotations

import numpy as np

from .base import Effect, grid


class Wormhole(Effect):
    name = "LUT Wormhole"

    def setup(self):
        self.X, self.Y, self.aspect = grid(self.w, self.h, scale=1.0)
        self.z = 0.0
        self.spin = 0.0
        self.spin_dir = 1.0
        self.folds = 6.0
        self.folds_target = 6.0
        self.tex_mix = 0.0
        self.tex_target = 0.0
        self.last_change = -99.0

    def render(self, f, t, dt):
        if f.beat and t - self.last_change > 8.0:
            self.folds_target = float(self.rng.choice([4, 6, 8, 10]))
            self.tex_target = 1.0 - self.tex_target
            self.spin_dir = float(self.rng.choice([-1.0, 1.0]))
            self.last_change = t

        self.z += dt * (0.65 + 2.5 * f.bass + 1.5 * f.pulse)
        self.spin += dt * (0.16 + 0.6 * f.mid) * self.spin_dir
        lerp = float(1.0 - 0.08 ** dt)
        self.folds += (self.folds_target - self.folds) * lerp
        self.tex_mix += (self.tex_target - self.tex_mix) * lerp

        # Continuous sub-pixel camera center drift (zero integer stepping)
        dx = 0.42 * np.sin(t * 0.37)
        dy = 0.38 * np.sin(t * 0.53)
        Xc = self.X - dx
        Yc = self.Y - dy

        r = np.hypot(Xc, Yc)
        D = 0.5 / np.maximum(r, 0.025)
        A = np.arctan2(Yc, Xc) / np.pi  # -1..1
        S = np.clip(r * 1.65, 0.0, 1.0) ** 1.15  # smooth dark core vignette
        detail_fade = np.clip(r * 3.0, 0.0, 1.0)  # distance fog suppresses core moire

        u = D * (1.0 + 0.25 * f.bass) + self.z
        folds_even = np.rint(self.folds / 2.0) * 2.0
        a = A * (folds_even * 0.5) + self.spin + 0.15 * np.sin(u * 1.5 + t)
        v = np.abs((a % 2.0) - 1.0)  # mirrored 0..1 kaleidoscope coordinate

        # Anti-aliased XOR texture + flowing sine plasma walls with crossfade
        tex_xor = (((u * 32).astype(np.int32) ^ (v * 64).astype(np.int32)) & 63) * 3.4 + 30.0
        tex_xor = 128.0 + (tex_xor - 128.0) * detail_fade
        tex_plasma = (
            127.0
            + 64.0 * np.sin(u * 9.4 + 4.0 * np.sin(v * (2.0 * np.pi) + t))
            + 64.0 * np.sin(v * 12.6 + u * 2.0 - t * 1.7)
        )
        tex = tex_xor * (1.0 - self.tex_mix) + tex_plasma * self.tex_mix

        # Continuous linear interpolation across spectrum bars
        nb = len(f.bars)
        v_idx = np.clip(v * (nb - 1), 0.0, nb - 1.001)
        i0 = v_idx.astype(np.int32)
        frac = v_idx - i0
        spec = f.bars[i0] * (1.0 - frac) + f.bars[i0 + 1] * frac

        # Smooth bell-curve depth rings instead of hard boolean threshold
        ring_wave = 0.5 + 0.5 * np.sin(u * 12.6 - t * 4.0)
        rings = (ring_wave ** 4.5) * detail_fade

        tex = tex * (0.55 + 0.45 * spec) + 125.0 * spec * rings
        return np.clip(tex * S * (0.82 + 0.18 * f.pulse), 0.0, 255.0).astype(np.float32)

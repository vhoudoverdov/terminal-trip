"""Preset 8: Electric Julia Fractal Kaleidoscope ("Julia Fractal").

Real-time audio-reactive Julia fractal z_{n+1} = z_n^2 + c(t) where c(t) orbits
the Mandelbrot main cardioid boundary in sync with bass/mid/treble energy,
rendered with continuous orbital-trap filaments and smooth boundary glow.
"""
from __future__ import annotations

import numpy as np

from .base import Effect, grid


class JuliaFractal(Effect):
    name = "Julia Fractal"

    def setup(self):
        self.X, self.Y, self.aspect = grid(self.w, self.h)
        self.r = np.hypot(self.X, self.Y)
        self.th = np.arctan2(self.Y, self.X)
        self.phase = 0.0
        self.spin = 0.0
        self.folds = 2
        self.last_change = -99.0

    def render(self, f, t, dt):
        if f.beat and t - self.last_change > 7.5:
            self.folds = int(self.rng.choice([1, 2, 3, 4]))
            self.last_change = t

        self.phase += dt * (0.32 + 0.80 * f.bass + 0.45 * f.mid)
        self.spin += dt * (0.14 + 0.38 * f.mid)

        # Zoom and N-fold kaleidoscope symmetry in complex plane
        scale = (1.28 - 0.20 * f.bass - 0.08 * f.pulse) / max(1.0, min(self.aspect, 1.35))
        if self.folds > 1:
            folded_th = np.abs(((self.th * self.folds / np.pi) % 2.0) - 1.0) * np.pi + self.spin
        else:
            folded_th = self.th + self.spin
        zr = (self.r * scale * np.cos(folded_th)).astype(np.float32)
        zi = (self.r * scale * np.sin(folded_th)).astype(np.float32)

        # Orbit c(t) along the electric dendritic boundary near c = -0.77 + 0.15i
        cr = np.float32(-0.775 + 0.065 * np.cos(self.phase) + 0.025 * f.bass)
        ci = np.float32(0.148 + 0.065 * np.sin(self.phase * 0.83) + 0.025 * f.treble)

        # Vectorized Julia iteration with smooth escape potential & Pickover stalk/ring trap
        n_iter = 16
        smooth_iter = np.zeros_like(zr)
        min_stalk = np.full_like(zr, 10.0)
        min_ring = np.full_like(zr, 10.0)
        alive = np.ones(zr.shape, dtype=bool)

        for _ in range(n_iter):
            zr2 = zr * zr
            zi2 = zi * zi
            mag2 = zr2 + zi2
            still = mag2 < 4.0
            smooth_iter += still.astype(np.float32)

            zri = zr * zi
            zr = np.where(still, zr2 - zi2 + cr, zr)
            zi = np.where(still, 2.0 * zri + ci, zi)

            # Pickover cross stalk trap (min(|Re z|, |Im z|)) and circle trap (||z|^2 - 0.35|)
            stalk = np.minimum(np.abs(zr), np.abs(zi))
            ring = np.abs(mag2 - 0.38)
            min_stalk = np.where(still, np.minimum(min_stalk, stalk), min_stalk)
            min_ring = np.where(still, np.minimum(min_ring, ring), min_ring)
            alive = still

        # Smooth fractional escape potential (0 at edge, peaking at fractal boundary, 0 deep inside)
        mag2_final = np.maximum(zr * zr + zi * zi, 1.001)
        frac = np.clip(1.0 - np.log2(np.maximum(np.log2(mag2_final), 1e-3)) * 0.5, 0.0, 1.0)
        ext_pot = np.where(~alive, (smooth_iter + frac) / n_iter, 0.0)

        # Exterior boundary glow + electric wave bands, plus dark interior chambers lit by Pickover traps
        boundary_glow = (ext_pot ** 0.7) * (0.55 + 0.45 * np.sin(ext_pot * np.pi * (5.0 + 2.0 * f.mid) - t * 2.0))
        stalk_glow = np.exp(-min_stalk * (14.0 - 4.0 * f.treble))
        ring_glow = np.exp(-min_ring * (8.5 - 2.5 * f.bass))

        val = (
            165.0 * boundary_glow
            + 120.0 * stalk_glow * (0.6 + 0.4 * f.treble)
            + 95.0 * ring_glow * (0.5 + 0.5 * f.pulse)
        )
        vignette = np.clip(1.38 - 0.35 * self.r, 0.25, 1.0)
        return np.clip(val * vignette, 0.0, 255.0).astype(np.float32)

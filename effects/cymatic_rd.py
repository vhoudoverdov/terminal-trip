"""Preset 5: Audio-Reactive Cymatic Reaction-Diffusion ("Cymatic Lab").

A crossover with terminal-soup: models morphogens U and V reacting and diffusing
via a 9-point isotropic Patra-Karttunen Laplacian while driven by Chladni
standing-wave acoustic nodal plates and membrane-relief shading.
"""
from __future__ import annotations

import numpy as np

from .base import Effect, grid, laplacian_9pt


class CymaticRD(Effect):
    name = "Cymatic Lab"

    def setup(self):
        self.X, self.Y, self.aspect = grid(self.w, self.h)
        self.R = np.hypot(self.X, self.Y)
        self.U = np.ones((self.h, self.w), dtype=np.float32)
        self.V = np.zeros((self.h, self.w), dtype=np.float32)
        self.mode_n = 3.0
        self.mode_m = 5.0
        self.target_n = 3.0
        self.target_m = 5.0
        self.last_beat_seed = -99.0

        # Seed initial morphogen colonies
        count = max(5, (self.w * self.h) // 2200)
        for _ in range(count):
            rx = int(self.rng.integers(self.w // 5, 4 * self.w // 5))
            ry = int(self.rng.integers(self.h // 5, 4 * self.h // 5))
            self._seed_disc(rx, ry, radius=max(4, min(self.w, self.h) // 14))

    def _seed_disc(self, x: int, y: int, radius: int = 6):
        x0, x1 = max(0, x - radius), min(self.w, x + radius + 1)
        y0, y1 = max(0, y - radius), min(self.h, y + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
        n_pts = int(np.sum(mask))
        if n_pts > 0:
            self.U[y0:y1, x0:x1][mask] = 0.50 + self.rng.uniform(-0.05, 0.05, n_pts).astype(np.float32)
            self.V[y0:y1, x0:x1][mask] = 0.28 + self.rng.uniform(-0.05, 0.05, n_pts).astype(np.float32)

    def render(self, f, t, dt):
        if f.beat and t - self.last_beat_seed > 2.5:
            pairs = [(2, 5), (3, 5), (3, 7), (4, 7), (2, 7), (5, 8)]
            n_new, m_new = pairs[int(self.rng.integers(0, len(pairs)))]
            self.target_n, self.target_m = float(n_new), float(m_new)
            self.last_beat_seed = t

        lerp = float(1.0 - 0.15 ** dt)
        self.mode_n += (self.target_n - self.mode_n) * lerp
        self.mode_m += (self.target_m - self.mode_m) * lerp

        # Chladni resonant plate equation: cos(n*pi*x)*cos(m*pi*y) - cos(m*pi*x)*cos(n*pi*y)
        nx = self.X / max(self.aspect, 1e-3)
        ny = self.Y
        n_freq = self.mode_n + 0.6 * f.mid
        m_freq = self.mode_m + 0.6 * f.treble
        chladni = (
            np.cos(n_freq * np.pi * nx) * np.cos(m_freq * np.pi * ny)
            - np.cos(m_freq * np.pi * nx) * np.cos(n_freq * np.pi * ny)
        )
        nodal_line = np.exp(-((chladni * 2.8) ** 2)).astype(np.float32)

        # On beats, nucleate fresh chemical activity along the Chladni nodal lines
        U, V = self.U, self.V
        if f.beat:
            inject = (nodal_line > 0.78) & (self.rng.random((self.h, self.w)) < 0.18)
            V = np.where(inject, np.maximum(V, 0.32), V)
            U = np.where(inject, np.minimum(U, 0.48), U)

        # Spatially modulated Feed (F) and Kill (k) parameters driven by audio + Chladni field
        F = 0.042 + 0.016 * f.bass + 0.008 * nodal_line
        k = 0.061 + 0.003 * f.mid
        Du, Dv = 0.16, 0.08

        substeps = 4
        for _ in range(substeps):
            lapU = laplacian_9pt(U)
            lapV = laplacian_9pt(V)
            UV2 = U * V * V
            U = np.clip(U + Du * lapU - UV2 + F * (1.0 - U), 0.0, 1.0)
            V = np.clip(V + Dv * lapV + UV2 - (F + k) * V, 0.0, 1.0)

        if float(V.max()) < 0.02:
            self._seed_disc(self.w // 2, self.h // 2, radius=6)

        self.U, self.V = U, V

        # Smooth membrane-relief shading (from terminal-soup) + glowing cymatic nodal filaments
        ratio = V / (U + V + 1e-5)
        core = np.clip(ratio * 1.65, 0.0, 1.0) ** 0.85
        gx = np.roll(V, -1, axis=1) - np.roll(V, 1, axis=1)
        gy = np.roll(V, -1, axis=0) - np.roll(V, 1, axis=0)
        rim = np.hypot(gx, gy) * 1.85
        cymatic_glow = nodal_line * (0.14 + 0.26 * f.pulse + 0.15 * f.treble)
        shaded = np.clip(core * 0.78 + rim * 0.36 + cymatic_glow, 0.0, 1.0)
        return (shaded * 255.0).astype(np.float32)

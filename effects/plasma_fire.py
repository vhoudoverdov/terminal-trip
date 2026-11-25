"""Preset 4: Demoscene sine-interference plasma with an isotropic turbulent fire buffer.

Combines a 4-wave contour-banded interference plasma with a spectrum-fed
cellular fire simulation featuring 9-point convection, horizontal wind swirl,
and membrane-relief flame-edge shading.
"""
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
        self.x_norm = np.linspace(0.0, 1.0, self.w, dtype=np.float32)

    def render(self, f, t, dt):
        w, h = self.w, self.h
        X, Y = self.X, self.Y
        tt = t * 0.8
        self.phase += dt * (1.0 + 2.0 * f.bass)
        c, s = np.cos(tt * 0.3), np.sin(tt * 0.3)
        p = (
            np.sin(X * 3.0 + tt)
            + np.sin(Y * 4.0 - 1.3 * tt)
            + np.sin(3.0 * (X * c + Y * s) + tt)
            + np.sin(5.0 * self.R - 2.0 * self.phase)
        )
        # Smooth contour bands with specular crest boost
        wave = 0.5 + 0.5 * np.sin(p * np.pi * (1.0 + 0.5 * f.treble))
        band = wave ** 1.15 + 0.25 * (wave ** 6.0)
        plasma = 6.0 + 78.0 * band * (0.45 + 0.55 * f.mid)

        # Seed bottom rows with clustered spectrum hotspots
        nb = len(f.bars)
        spec = np.interp(np.linspace(0, nb - 1, w), np.arange(nb), f.bars) ** 0.5
        hot = self.rng.random(w) < (0.22 + 0.60 * spec) * (0.6 + 0.6 * f.bass)
        seed = np.where(hot, 255.0, 45.0 * spec).astype(np.float32)
        if f.beat:
            seed[self.rng.random(w) < 0.45] = 255.0
        F = self.fire
        F[-1] = seed
        F[-2] = np.maximum(F[-2] * 0.55, seed)

        # Turbulent upward convection with subtle horizontal wind curl
        steps = max(2, round(h / 38))
        cool = (255.0 / (2.1 * h)) * (1.28 - 0.78 * f.bass)
        wind = float(np.sin(t * 1.4) * 0.08)
        w_left = 0.22 + wind
        w_right = 0.22 - wind
        for _ in range(steps):
            row1 = F[1:-1]
            row2 = F[2:]
            left1 = np.roll(row1, 1, axis=1)
            right1 = np.roll(row1, -1, axis=1)
            left2 = np.roll(row2, 1, axis=1)
            right2 = np.roll(row2, -1, axis=1)
            conv = (
                0.32 * row1
                + w_left * left1
                + w_right * right1
                + 0.14 * row2
                + 0.05 * left2
                + 0.05 * right2
            ) - cool
            F[:-2] = np.maximum(conv, 0.0)
        self.fire = F
        heat = F[:h]

        # Flame-edge gradient relief (|grad F|) gives licking flame tongues incandescent rims
        gx = np.roll(heat, -1, axis=1) - np.roll(heat, 1, axis=1)
        gy = np.roll(heat, -1, axis=0) - np.roll(heat, 1, axis=0)
        rim = np.hypot(gx, gy) * 0.32
        flame = np.clip(heat * 1.05 + rim, 0.0, 255.0)

        return np.minimum(plasma * (1.0 - np.clip(heat / 135.0, 0.0, 0.88)) + flame, 255.0)

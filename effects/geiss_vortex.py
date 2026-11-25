"""Preset 1: Geiss / MilkDrop style feedback vortex.

Each frame the previous frame is re-sampled through a smooth audio-reactive
polar warp (zoom + swirl + petal modulation + radial ripple), diffused with an
isotropic 9-point bloom kernel, and layered with sub-pixel anti-aliased
morphing waveform geometry and beat sparks.
"""
from __future__ import annotations

import numpy as np

from .base import Effect, bilinear, blur3x3, grid, plot, resample


class GeissVortex(Effect):
    name = "Geiss Vortex"

    def setup(self):
        X, Y, self.aspect = grid(self.w, self.h)
        self.r = np.hypot(X, Y)
        self.th = np.arctan2(Y, X)
        self.fb = np.zeros((self.h, self.w), np.float32)
        self.petals_target = 4.0
        self.petals = 4.0
        self.zoom_target = 1.0
        self.zoom_dir = 1.0
        self.rot_target = 1.0
        self.rot_dir = 1.0
        self.shape = 0
        self.prev_shape = 0
        self.shape_blend = 1.0
        self.last_change = -99.0

    def _mutate(self, t):
        rng = self.rng
        self.petals_target = float(rng.choice([2, 3, 4, 5, 6]))
        self.rot_target = float(rng.choice([-1.0, 1.0]))
        self.zoom_target = float(rng.choice([1.0, 1.0, 1.0, -0.6]))
        self.prev_shape = self.shape
        self.shape = (self.shape + 1) % 3
        self.shape_blend = 0.0
        self.last_change = t

    def _shape_coords(self, kind: int, wave: np.ndarray, R: float, t: float, a: float):
        n = len(wave)
        if kind == 0:  # breathing harmonic ring
            ang = np.linspace(0, 2 * np.pi, n, dtype=np.float32) + t * 0.5
            rad = R + 0.22 * wave
            return rad * np.cos(ang), rad * np.sin(ang)
        if kind == 1:  # twisted oscilloscope ribbon
            x = np.linspace(-a * 0.9, a * 0.9, n, dtype=np.float32)
            y = 0.45 * wave * np.cos(t * 0.4) + 0.12 * np.sin(x * 2.5 + t * 1.5)
            return x, y
        # lissajous phase scope
        x = 0.62 * wave * min(a, 1.6)
        y = 0.62 * np.roll(wave, n // 5)
        return x, y

    def render(self, f, t, dt):
        k = dt * 60.0
        if f.beat and t - self.last_change > 7.0:
            self._mutate(t)

        # Smooth exponential interpolation of warp parameters (no sudden snaps)
        lerp = float(1.0 - 0.12 ** dt)
        self.petals += (self.petals_target - self.petals) * lerp
        self.rot_dir += (self.rot_target - self.rot_dir) * lerp
        self.zoom_dir += (self.zoom_target - self.zoom_dir) * lerp
        self.shape_blend = min(1.0, self.shape_blend + dt / 0.65)

        r, th, a = self.r, self.th, self.aspect

        # --- warp field (where each pixel samples the previous frame from)
        zoom = (0.010 + 0.048 * f.bass) * self.zoom_dir * k
        petal_freq = np.rint(self.petals)
        s = 1.0 - zoom + 0.013 * np.sin(petal_freq * th + t * 1.3) * (0.4 + f.treble)
        rot = (0.008 + 0.03 * f.mid) * self.rot_dir * k + 0.022 * np.sin(3.2 * r - t * 2.0) * (0.3 + f.mid)
        r2 = r * s
        th2 = th + rot
        u = (r2 * np.cos(th2) / a + 1.0) * 0.5 * (self.w - 1)
        v = (r2 * np.sin(th2) + 1.0) * 0.5 * (self.h - 1)
        warped = bilinear(self.fb, u, v)
        # Light isotropic 9-point diffusion keeps feedback trails liquid and alias-free
        fb = blur3x3(warped, mix=0.14) * (0.958 ** k)

        # --- fresh morphing geometry on top
        n = 4 * (self.w + self.h)
        wave = resample(f.wave, n)
        intensity = 175.0 + 80.0 * f.pulse
        R = 0.35 + 0.25 * f.bass

        x, y = self._shape_coords(self.shape, wave, R, t, a)
        if self.shape_blend < 1.0:
            px, py = self._shape_coords(self.prev_shape, wave, R, t, a)
            b = self.shape_blend
            x = px * (1.0 - b) + x * b
            y = py * (1.0 - b) + y * b

        plot(fb, x, y, intensity, a, glow=0.65)

        if f.beat:  # sparks along the ring that get sucked into the vortex
            m = 56
            ang = self.rng.uniform(0, 2 * np.pi, m)
            rad = R + self.rng.normal(0, 0.05, m)
            plot(fb, rad * np.cos(ang), rad * np.sin(ang), 255.0, a, glow=0.85)
        nt = int(f.treble * 14)
        if nt:
            plot(fb, self.rng.uniform(-a, a, nt), self.rng.uniform(-1, 1, nt), 205.0, a, glow=0.35)

        self.fb = np.minimum(fb, 255.0)
        return self.fb

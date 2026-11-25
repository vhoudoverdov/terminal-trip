"""Preset 7: 3D Harmonic Attractor & Torus Knot Phase Scope ("3D Attractor").

Projects a rotating 3D volumetric harmonic torus knot & strange-attractor
ribbon modulated by the live waveform and 3-band energy onto a perspective
canvas with depth-cued intensity, bilinear splatting, and expanding trails.
"""
from __future__ import annotations

import numpy as np

from .base import Effect, bilinear, blur3x3, grid, plot, resample


class AttractorScope(Effect):
    name = "3D Attractor"

    def setup(self):
        X, Y, self.aspect = grid(self.w, self.h)
        self.r = np.hypot(X, Y)
        self.th = np.arctan2(Y, X)
        self.fb = np.zeros((self.h, self.w), dtype=np.float32)
        self.ax = 0.0
        self.ay = 0.0
        self.az = 0.0
        self.p_knot = 2.0
        self.q_knot = 3.0
        self.target_p = 2.0
        self.target_q = 3.0
        self.last_change = -99.0

    def render(self, f, t, dt):
        k = dt * 60.0
        if f.beat and t - self.last_change > 6.5:
            knots = [(2, 3), (3, 4), (2, 5), (3, 5), (4, 5), (3, 7)]
            p_new, q_new = knots[int(self.rng.integers(0, len(knots)))]
            self.target_p, self.target_q = float(p_new), float(q_new)
            self.last_change = t

        lerp = float(1.0 - 0.12 ** dt)
        self.p_knot += (self.target_p - self.p_knot) * lerp
        self.q_knot += (self.target_q - self.q_knot) * lerp

        # Advance 3D Euler angles with audio bands
        self.ax += dt * (0.50 + 1.15 * f.bass)
        self.ay += dt * (0.65 + 1.35 * f.mid)
        self.az += dt * (0.35 + 0.95 * f.treble)

        # Outward radial expansion + twist on feedback buffer fills the canvas with 3D wakes
        a = self.aspect
        s = 1.0 - (0.014 + 0.028 * f.bass + 0.015 * f.pulse) * k
        rot = (0.008 + 0.018 * f.mid) * k
        r2 = self.r * s
        th2 = self.th + rot
        u = (r2 * np.cos(th2) / a + 1.0) * 0.5 * (self.w - 1)
        v = (r2 * np.sin(th2) + 1.0) * 0.5 * (self.h - 1)
        fb = blur3x3(bilinear(self.fb, u, v), mix=0.15) * (0.935 ** k)

        # Generate 3D parametric torus-knot attractor ribbon modulated by live waveform
        n = 5 * (self.w + self.h)
        phi = np.linspace(0, 2.0 * np.pi, n, dtype=np.float32)
        wave = resample(f.wave, n)

        p = self.p_knot
        q = self.q_knot
        tube_r = 0.36 + 0.20 * f.bass + 0.18 * wave
        R_torus = 0.72 + 0.10 * np.sin(q * phi + t * 2.0) * (0.4 + f.mid)

        x3 = (R_torus + tube_r * np.cos(q * phi)) * np.cos(p * phi)
        y3 = (R_torus + tube_r * np.cos(q * phi)) * np.sin(p * phi)
        z3 = tube_r * np.sin(q * phi) + 0.18 * np.sin(2.0 * phi - t * 1.8) * f.treble

        # 3D Rotation matrix R_y(ay) * R_x(ax) * R_z(az)
        cx, sx = np.cos(self.ax), np.sin(self.ax)
        cy, sy = np.cos(self.ay), np.sin(self.ay)
        cz, sz = np.cos(self.az), np.sin(self.az)

        xz = x3 * cz - y3 * sz
        yz = x3 * sz + y3 * cz
        zz = z3

        xx = xz
        yx = yz * cx - zz * sx
        zx = yz * sx + zz * cx

        xy = xx * cy + zx * sy
        yy = yx
        zy = -xx * sy + zx * cy

        # Wide perspective projection with depth-cue intensity
        cam_d = 2.05 - 0.22 * f.pulse
        inv_z = 1.65 / np.maximum(cam_d - zy, 0.45)
        xp = xy * inv_z
        yp = yy * inv_z

        depth_shade = np.clip((zy + 1.1) / 2.2, 0.0, 1.0)
        val = 135.0 + 115.0 * depth_shade + 35.0 * f.pulse
        plot(fb, xp, yp, val, a, glow=0.75)

        self.fb = np.minimum(fb, 255.0)
        return self.fb

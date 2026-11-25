"""Preset 6: Synthwave 3D Horizon & Audio Canyon ("SynthHorizon").

Perspective-projected 3D terrain grid scrolling through an audio-reactive
valley with a symmetric spectrum mountain range, segmented scanline outrun sun,
and twinkling treble starfield.
"""
from __future__ import annotations

import numpy as np

from .base import Effect, grid, resample


class SynthHorizon(Effect):
    name = "SynthHorizon"

    def setup(self):
        self.X, self.Y, self.aspect = grid(self.w, self.h)
        self.scroll = 0.0
        # Pre-generate deterministic star positions in the upper sky
        n_stars = max(28, (self.w * self.h) // 160)
        rng = np.random.default_rng(42)
        self.star_x = rng.integers(0, self.w, size=n_stars)
        self.star_y = rng.integers(0, max(1, int(self.h * 0.48)), size=n_stars)
        self.star_phase = rng.uniform(0, 2 * np.pi, size=n_stars).astype(np.float32)

    def render(self, f, t, dt):
        w, h = self.w, self.h
        X, Y, a = self.X, self.Y, self.aspect
        self.scroll += dt * (2.2 + 5.5 * f.bass + 2.5 * f.pulse)

        horizon = -0.06  # slightly above vertical center (Y in -1..1, top is -1)
        fb = np.zeros((h, w), dtype=np.float32)

        # --- 1. Retro segmented Synthwave Sun above the horizon
        sun_cy = horizon - 0.28
        sun_r = 0.35 + 0.05 * f.bass + 0.04 * f.pulse
        dist_sun = np.hypot(X, Y - sun_cy)
        # Atmospheric horizon glow + sun halo
        sky_glow = np.exp(-np.abs(Y - horizon) * 3.8) * (45.0 + 35.0 * f.bass)
        sun_halo = np.exp(-np.maximum(dist_sun - sun_r * 0.8, 0.0) * 4.8) * (78.0 + 42.0 * f.pulse)
        fb += sky_glow + sun_halo

        # Sun disc with classic horizontal scanline cutouts on lower half
        in_sun = dist_sun <= sun_r
        sun_edge = np.clip((sun_r - dist_sun) / 0.025, 0.0, 1.0)
        scan_freq = 36.0
        scan_cut = np.where(
            Y > sun_cy - 0.05,
            np.clip(np.sin((Y - sun_cy) * scan_freq - t * 2.5) + 0.35 + (sun_cy - Y) * 1.8, 0.0, 1.0),
            1.0,
        )
        sun_val = in_sun * sun_edge * scan_cut * (200.0 + 55.0 * f.pulse)
        fb = np.maximum(fb, sun_val)

        # --- 2. Twinkling treble stars in upper sky
        twinkle = (115.0 + 135.0 * f.treble) * (0.5 + 0.5 * np.sin(t * 4.5 + self.star_phase))
        sky_mask = Y[self.star_y, self.star_x] < (horizon - 0.15)
        np.maximum.at(fb, (self.star_y[sky_mask], self.star_x[sky_mask]), twinkle[sky_mask])

        # --- 3. Symmetric Canyon Spectrum Mountain Range framing the Sun
        x_norm = np.abs(X[0]) / max(a, 1e-3)  # 0 at center canyon, 1 at outer peaks
        nb = len(f.bars)
        # Map center-to-edge (x_norm 0..1) from high frequencies in the valley to bass peaks on the flanks
        bar_pos = np.clip((1.0 - x_norm) * (nb - 1), 0.0, nb - 1.001)
        b0 = bar_pos.astype(np.int32)
        bf = bar_pos - b0
        spec = f.bars[b0] * (1.0 - bf) + f.bars[b0 + 1] * bf

        valley_profile = 0.18 + 0.82 * (x_norm ** 1.35)
        mtn_height = (0.04 + 0.30 * spec * valley_profile)[None, :]
        mtn_top = horizon - mtn_height
        in_mtn = (Y >= mtn_top) & (Y <= horizon + 0.02)
        mtn_rim = np.exp(-((np.abs(Y - mtn_top) / 0.018) ** 2)) * (195.0 + 60.0 * f.mid) * (Y <= horizon + 0.03)
        fb = np.where(in_mtn, 20.0 + 65.0 * (horizon - Y) / np.maximum(mtn_height, 0.01), fb)
        fb = np.maximum(fb, mtn_rim)

        # --- 4. 3D Perspective-Projected Canyon Grid below the horizon
        dy = np.maximum(Y - horizon, 1e-3)
        ground_mask = Y > horizon
        Z = 0.85 / np.maximum(dy, 0.045)
        fog = np.clip((Y - horizon) * 2.2, 0.0, 1.0) ** 0.85

        wave_col = resample(f.wave, w)[None, :]
        sway = 0.20 * np.sin(Z * 0.45 - self.scroll * 0.35) * (0.4 + 0.6 * f.mid)
        U = (X + sway) * Z + 0.18 * wave_col * fog
        V = Z * 2.2 - self.scroll

        u_dist = np.abs((U % 1.0) - 0.5)
        v_dist = np.abs((V % 1.0) - 0.5)
        line_u = np.exp(-((u_dist / (0.068 + 0.022 * dy)) ** 2))
        line_v = np.exp(-((v_dist / (0.078 + 0.022 * dy)) ** 2))
        grid_intensity = np.maximum(line_u, line_v)

        wall_boost = 1.0 + 0.90 * spec[None, :] * (x_norm[None, :] ** 1.15)
        pulse_ring = np.exp(-(((V % 4.0) - 2.0) ** 2) / 0.22) * (75.0 * f.pulse)
        ground_val = (grid_intensity * (160.0 + 65.0 * f.bass) * wall_boost + pulse_ring) * fog

        fb = np.where(ground_mask, np.maximum(fb * 0.3, ground_val), fb)
        return np.clip(fb, 0.0, 255.0).astype(np.float32)

"""256-entry indexed color palettes with classic palette cycling.

Every effect writes an *intensity* index (0..255) per pixel. The final color of
index ``i`` is::

    hue_cycle[(i + offset) % 256] * brightness_ramp[i]  (+ a white-hot top end)

so the low indices always stay dark (index 0 is pure black) while the hue of
everything on screen rotates as ``offset`` advances over time and jumps on beats.
"""
from __future__ import annotations

import numpy as np

_I = np.arange(256)


def _cyclic_stops(colors):
    """Interpolate a closed loop of RGB stops into a seamless 256-entry table."""
    n = len(colors)
    xs = np.arange(n + 1) / n
    cols = np.array(list(colors) + [colors[0]], dtype=np.float64)
    t = _I / 256.0
    return np.stack([np.interp(t, xs, cols[:, c]) for c in range(3)], axis=1)


def _cosine(a, b, c, d):
    """Inigo Quilez style cosine palette (periodic, so it cycles seamlessly)."""
    t = (_I / 256.0)[:, None]
    a, b, c, d = (np.array(v, dtype=np.float64) for v in (a, b, c, d))
    return 255.0 * np.clip(a + b * np.cos(2 * np.pi * (c * t + d)), 0.0, 1.0)


PALETTES = [
    ("Geiss Fire", _cyclic_stops([(70, 0, 160), (230, 0, 170), (255, 80, 0),
                                  (255, 210, 40), (255, 80, 0), (230, 0, 170)])),
    ("WMP Cobalt", _cyclic_stops([(0, 40, 220), (0, 210, 255), (150, 70, 255),
                                  (255, 40, 190), (150, 70, 255), (0, 210, 255)])),
    ("Acid Matrix", _cyclic_stops([(0, 140, 40), (120, 255, 0), (0, 255, 190),
                                   (200, 255, 120), (0, 255, 190), (120, 255, 0)])),
    ("Neon Sunset", _cyclic_stops([(255, 0, 110), (255, 120, 0), (255, 230, 80),
                                   (255, 120, 0), (255, 0, 110), (130, 0, 255)])),
    ("Thermal Cosine", _cosine((.5, .5, .5), (.5, .5, .5), (1, 1, 1), (0, .33, .67))),
    ("Infrared", _cyclic_stops([(160, 0, 0), (255, 60, 0), (255, 170, 0),
                                (255, 255, 160), (255, 170, 0), (255, 60, 0)])),
]

# Brightness ramp: index 0 is black, low indices lift smoothly for deep decay trails, top end goes white-hot.
_RAMP = (np.clip(_I / 64.0, 0.0, 1.0) ** 1.2)[:, None]
_WHITE = (np.clip((_I - 195) / 60.0, 0.0, 1.0) ** 1.8 * 0.55)[:, None]


class PaletteManager:
    def __init__(self, index: int = 0, cycle_speed: float = 14.0):
        self.index = index % len(PALETTES)
        self.prev_index = self.index
        self.blend = 1.0          # 0 -> showing prev palette, 1 -> current
        self.offset = 0.0
        self.cycle_speed = cycle_speed

    @property
    def name(self) -> str:
        return PALETTES[self.index][0]

    def next(self, step: int = 1):
        self.prev_index = self.index
        self.index = (self.index + step) % len(PALETTES)
        self.blend = 0.0

    def update(self, dt: float, bass: float = 0.0, beat: bool = False):
        self.offset += dt * self.cycle_speed * (1.0 + 2.0 * bass)
        if beat:
            self.offset += 24.0
        self.blend = min(1.0, self.blend + dt / 0.8)

    def lut(self) -> np.ndarray:
        idx = (_I + int(self.offset)) % 256
        hue = PALETTES[self.index][1][idx]
        if self.blend < 1.0:
            hue = hue * self.blend + PALETTES[self.prev_index][1][idx] * (1.0 - self.blend)
        col = hue * _RAMP
        col = col + (255.0 - col) * _WHITE
        col[0] = 0.0
        return np.clip(col, 0, 255).astype(np.uint8)

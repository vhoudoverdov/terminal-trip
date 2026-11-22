"""Shared helpers for effects: coordinate grids, bilinear feedback sampling, plotting."""
from __future__ import annotations

import numpy as np


class Effect:
    name = "Effect"

    def __init__(self):
        self.w = self.h = 0
        self.rng = np.random.default_rng()

    def ensure(self, w: int, h: int):
        if (w, h) != (self.w, self.h):
            self.w, self.h = w, h
            self.setup()

    def setup(self):
        pass

    def render(self, f, t: float, dt: float) -> np.ndarray:
        """Return a float32 (h, w) intensity buffer in 0..255."""
        raise NotImplementedError


def grid(w: int, h: int, scale: float = 1.0):
    """Square-pixel coordinates: y in [-1, 1], x in [-aspect, aspect]."""
    aspect = w / h
    xs = np.linspace(-aspect, aspect, w, dtype=np.float32) * scale
    ys = np.linspace(-1, 1, h, dtype=np.float32) * scale
    X, Y = np.meshgrid(xs, ys)
    return X, Y, aspect


def bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample ``img`` at fractional (col=u, row=v); outside the image is black."""
    h, w = img.shape
    valid = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    x0 = u.astype(np.int32)
    y0 = v.astype(np.int32)
    fx = u - x0
    fy = v - y0
    flat = img.ravel()
    i = y0 * w + x0
    top = flat[i] * (1 - fx) + flat[i + 1] * fx
    bot = flat[i + w] * (1 - fx) + flat[i + w + 1] * fx
    return (top * (1 - fy) + bot * fy) * valid


def resample(a: np.ndarray, n: int) -> np.ndarray:
    return np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)


def plot(fb: np.ndarray, x, y, val, aspect: float, glow: float = 0.0):
    """Max-blend points given in square-pixel coordinates into ``fb``."""
    h, w = fb.shape
    c = np.rint((np.asarray(x) / aspect + 1) * 0.5 * (w - 1)).astype(np.int32)
    r = np.rint((np.asarray(y) + 1) * 0.5 * (h - 1)).astype(np.int32)
    val = np.broadcast_to(np.asarray(val, np.float32), c.shape)

    def put(rr, cc, vv):
        m = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        np.maximum.at(fb, (rr[m], cc[m]), vv[m])

    put(r, c, val)
    if glow > 0:
        g = val * glow
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            put(r + dr, c + dc, g)

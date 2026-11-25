"""Shared helpers for effects: coordinate grids, bilinear feedback sampling,
sub-pixel anti-aliased plotting, and isotropic spatial stencils."""
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


def blur3x3(X: np.ndarray, mix: float = 0.25) -> np.ndarray:
    """Isotropic 9-point spatial smoothing kernel (cardinal 1/6, diagonal 1/12)."""
    n = np.roll(X, -1, axis=0)
    s = np.roll(X, 1, axis=0)
    e = np.roll(X, -1, axis=1)
    w = np.roll(X, 1, axis=1)
    ne = np.roll(n, -1, axis=1)
    nw = np.roll(n, 1, axis=1)
    se = np.roll(s, -1, axis=1)
    sw = np.roll(s, 1, axis=1)
    neigh = (n + s + e + w) * (1.0 / 6.0) + (ne + nw + se + sw) * (1.0 / 12.0)
    return (1.0 - mix) * X + mix * neigh


def laplacian_9pt(X: np.ndarray) -> np.ndarray:
    """9-point rotationally isotropic discrete Patra-Karttunen Laplacian."""
    n = np.roll(X, -1, axis=0)
    s = np.roll(X, 1, axis=0)
    e = np.roll(X, -1, axis=1)
    w = np.roll(X, 1, axis=1)
    ne = np.roll(n, -1, axis=1)
    nw = np.roll(n, 1, axis=1)
    se = np.roll(s, -1, axis=1)
    sw = np.roll(s, 1, axis=1)
    return (2.0 / 3.0) * (n + s + e + w) + (1.0 / 6.0) * (ne + nw + se + sw) - (10.0 / 3.0) * X


def plot(fb: np.ndarray, x, y, val, aspect: float, glow: float = 0.0):
    """Sub-pixel bilinear splatting with isotropic 9-point Gaussian glow."""
    h, w = fb.shape
    cf = (np.asarray(x, dtype=np.float32) / aspect + 1.0) * 0.5 * (w - 1)
    rf = (np.asarray(y, dtype=np.float32) + 1.0) * 0.5 * (h - 1)
    val = np.broadcast_to(np.asarray(val, np.float32), cf.shape)

    c0 = np.floor(cf).astype(np.int32)
    r0 = np.floor(rf).astype(np.int32)
    fc = cf - c0
    fr = rf - r0

    def put(rr, cc, vv):
        m = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        if np.any(m):
            np.maximum.at(fb, (rr[m], cc[m]), vv[m])

    # Bilinear sub-pixel core splat (boosted slightly so fractional centers stay crisp)
    w00 = (1.0 - fr) * (1.0 - fc)
    w01 = (1.0 - fr) * fc
    w10 = fr * (1.0 - fc)
    w11 = fr * fc
    put(r0, c0, val * np.sqrt(w00))
    put(r0, c0 + 1, val * np.sqrt(w01))
    put(r0 + 1, c0, val * np.sqrt(w10))
    put(r0 + 1, c0 + 1, val * np.sqrt(w11))

    if glow > 0:
        rc = np.rint(rf).astype(np.int32)
        cc = np.rint(cf).astype(np.int32)
        g_card = val * (glow * 0.85)
        g_diag = val * (glow * 0.55)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            put(rc + dr, cc + dc, g_card)
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            put(rc + dr, cc + dc, g_diag)

"""Comprehensive unit test suite for terminal-trip."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
import numpy as np

from audio import (
    Analyzer,
    Features,
    FileSource,
    RingBuffer,
    SynthSource,
    _validate_pcm,
    sanitize_label,
)
from effects import PRESETS
from effects.base import bilinear, blur3x3, grid, laplacian_9pt, plot
from main import App, main, parse_args
from palettes import PALETTES, PaletteManager
from renderer import Presenter, save_png, to_rgb
from updater import (
    RAW_PYPROJECT_URL,
    RELEASES_API_URL,
    TAGS_API_URL,
    __version__,
    check_for_update,
    check_for_update_cached,
    fetch_latest_version,
    format_version,
    parse_version,
    perform_update,
)


def make_features(
    bass: float = 0.5,
    mid: float = 0.4,
    treble: float = 0.3,
    beat: bool = False,
    pulse: float = 0.5,
    nbars: int = 48,
    nwave: int = 1024,
) -> Features:
    t = np.linspace(0, 2 * np.pi, nwave, dtype=np.float32)
    w = np.sin(3 * t) * 0.6 + np.cos(7 * t) * 0.3
    bars = np.linspace(0.8, 0.2, nbars, dtype=np.float32)
    return Features(
        wave=w,
        bars=bars,
        bass=bass,
        mid=mid,
        treble=treble,
        energy=float(bars.mean()),
        beat=beat,
        pulse=pulse,
        level=0.25,
    )


class TestAudioPipeline(unittest.TestCase):
    def test_ring_buffer_wrap_and_overwrite(self):
        rb = RingBuffer(16)
        rb.write(np.arange(10, dtype=np.float32))
        np.testing.assert_array_equal(rb.read(5), np.arange(5, 10, dtype=np.float32))

        # Write past wrap boundary
        rb.write(np.arange(10, 22, dtype=np.float32))
        np.testing.assert_array_equal(rb.read(16), np.arange(6, 22, dtype=np.float32))

        # Oversized write larger than buffer
        rb.write(np.arange(100, 140, dtype=np.float32))
        np.testing.assert_array_equal(rb.read(16), np.arange(124, 140, dtype=np.float32))

    def test_synth_source_and_analyzer(self):
        clock = [0.0]
        src = SynthSource(samplerate=44100, clock=lambda: clock[0])
        src.start()
        analyzer = Analyzer(src.samplerate)

        beats_seen = 0
        for _ in range(45):
            clock[0] += 1.0 / 60.0
            samples = src.read(analyzer.n)
            self.assertEqual(samples.shape, (analyzer.n,))
            self.assertEqual(samples.dtype, np.float32)
            self.assertTrue(np.all(np.isfinite(samples)))

            feats = analyzer.process(samples, 1.0 / 60.0)
            self.assertEqual(feats.bars.shape, (48,))
            self.assertEqual(feats.wave.shape, (1024,))
            self.assertTrue(0.0 <= feats.bass <= 1.0)
            self.assertTrue(0.0 <= feats.mid <= 1.0)
            self.assertTrue(0.0 <= feats.treble <= 1.0)
            self.assertTrue(0.0 <= feats.pulse <= 1.0)
            if feats.beat:
                beats_seen += 1

        self.assertGreaterEqual(beats_seen, 1)

    def test_sanitize_label_and_validate_pcm(self):
        dirty = "Song\x1b[31mName\x00\x7f\x85WithControls" + ("A" * 50)
        clean = sanitize_label(dirty, max_len=24)
        self.assertNotIn("\x1b", clean)
        self.assertNotIn("\x00", clean)
        self.assertLessEqual(len(clean), 24)

        with self.assertRaises(ValueError):
            _validate_pcm(np.zeros((10, 2), dtype=np.float32), 0)
        with self.assertRaises(ValueError):
            _validate_pcm(np.zeros((0, 2), dtype=np.float32), 44100)

    def test_wav_file_source_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = os.path.join(tmp, "test.wav")
            sr = 22050
            samples = (np.sin(np.linspace(0, 50 * np.pi, sr)) * 16000).astype("<i2")
            with wave.open(wav_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes(samples.tobytes())

            clock = [0.5]
            fs = FileSource(wav_path, clock=lambda: clock[0])
            chunk = fs.read(1024)
            self.assertEqual(chunk.shape, (1024,))
            self.assertTrue(np.all(np.isfinite(chunk)))


class TestPalettesAndRenderer(unittest.TestCase):
    def test_palette_manager_invariants(self):
        pm = PaletteManager(0)
        for _ in range(len(PALETTES) + 1):
            lut = pm.lut()
            self.assertEqual(lut.shape, (256, 3))
            self.assertEqual(lut.dtype, np.uint8)
            # Index 0 must always remain pure black regardless of offset
            np.testing.assert_array_equal(lut[0], np.array([0, 0, 0], dtype=np.uint8))
            pm.update(0.1, bass=0.8, beat=True)
            lut_shifted = pm.lut()
            np.testing.assert_array_equal(lut_shifted[0], np.array([0, 0, 0], dtype=np.uint8))
            pm.next()

    def test_presenter_compose_and_deduplication(self):
        lut = PaletteManager(0).lut()
        # Uniform row of index 120 should only emit 1 fg and 1 bg escape sequence per row
        idx = np.full((6, 10), 120, dtype=np.uint8)
        pres = Presenter(shift=0)
        ansi = pres.compose(idx, lut)
        self.assertIn("\u2580" * 10, ansi)
        self.assertEqual(ansi.count("\x1b[38;2;"), 3)  # 3 cell rows (6 pixel rows)
        self.assertEqual(ansi.count("\x1b[48;2;"), 3)

    def test_save_png(self):
        lut = PaletteManager(1).lut()
        idx = np.arange(64, dtype=np.uint8).reshape(8, 8)
        rgb = to_rgb(idx, lut)
        with tempfile.TemporaryDirectory() as tmp:
            p1 = os.path.join(tmp, "shot.png")
            out1 = save_png(p1, rgb, scale=2, exclusive=True)
            out2 = save_png(p1, rgb, scale=2, exclusive=True)
            self.assertNotEqual(out1, out2)
            with open(out1, "rb") as f:
                data = f.read()
            self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))


class TestEffectsAndApp(unittest.TestCase):
    def test_base_primitives_and_stencils(self):
        X, Y, aspect = grid(40, 20)
        self.assertEqual(X.shape, (20, 40))
        self.assertEqual(Y.shape, (20, 40))
        self.assertAlmostEqual(aspect, 2.0)

        img = np.ones((10, 10), dtype=np.float32) * 100.0
        u = np.full((10, 10), 4.5, dtype=np.float32)
        v = np.full((10, 10), 4.5, dtype=np.float32)
        sampled = bilinear(img, u, v)
        np.testing.assert_allclose(sampled, 100.0, rtol=1e-4)

        # Constant field has 0 Laplacian and unchanged blur3x3
        np.testing.assert_allclose(blur3x3(img, mix=0.25), 100.0, rtol=1e-4)
        np.testing.assert_allclose(laplacian_9pt(img), 0.0, atol=1e-4)

        # Sub-pixel bilinear splatting + isotropic 9-point glow
        fb = np.zeros((20, 40), dtype=np.float32)
        plot(fb, np.array([0.0]), np.array([0.0]), 200.0, aspect, glow=0.6)
        self.assertGreater(float(fb.max()), 100.0)
        self.assertLessEqual(float(fb.max()), 255.0)
        self.assertGreater(np.count_nonzero(fb > 0), 4)

    def test_all_eight_presets_render_and_resize(self):
        self.assertEqual(len(PRESETS), 8)
        f_normal = make_features(beat=False)
        f_beat = make_features(bass=0.95, mid=0.8, treble=0.85, beat=True, pulse=1.0)

        for cls in PRESETS:
            with self.subTest(preset=cls.name):
                eff = cls()
                eff.ensure(48, 32)
                out1 = eff.render(f_normal, 0.1, 1.0 / 60.0)
                self.assertEqual(out1.shape, (32, 48))
                self.assertTrue(np.all(np.isfinite(out1)))
                self.assertGreaterEqual(float(out1.min()), 0.0)
                self.assertLessEqual(float(out1.max()), 255.0)

                out2 = eff.render(f_beat, 9.5, 1.0 / 60.0)
                self.assertEqual(out2.shape, (32, 48))
                self.assertTrue(np.all(np.isfinite(out2)))
                self.assertGreater(float(out2.max()), 5.0)

                # Verify dynamic resize
                eff.ensure(64, 40)
                out3 = eff.render(f_normal, 9.6, 1.0 / 60.0)
                self.assertEqual(out3.shape, (40, 64))
                self.assertTrue(np.all(np.isfinite(out3)))

    def test_app_controller_ssaa_and_transitions(self):
        clock = [0.0]
        src = SynthSource(clock=lambda: clock[0])
        src.start()
        args = parse_args(["--synth", "--preset", "1", "--palette", "1", "--no-update-check"])
        app = App(args, source=src)
        self.assertTrue(app.supersample)
        self.assertEqual(app.presenter.shift, 0)

        # Render with 2x2 SSAA enabled
        idx_ssaa, lut = app.frame(40, 24, 0.1, 1.0 / 60.0)
        self.assertEqual(idx_ssaa.shape, (24, 40))
        self.assertEqual(idx_ssaa.dtype, np.uint8)
        self.assertEqual(lut.shape, (256, 3))

        # Toggle 2x2 SSAA off via 'v' key and verify native 1x1 frame
        self.assertTrue(app.key("v", 0.15))
        self.assertFalse(app.supersample)
        idx_native, _ = app.frame(40, 24, 0.2, 1.0 / 60.0)
        self.assertEqual(idx_native.shape, (24, 40))

        # Test direct preset keys 1..8 and crossfade
        for k_num in "12345678":
            self.assertTrue(app.key(k_num, 0.3))
        idx_trans, _ = app.frame(40, 24, 0.4, 1.0 / 60.0)
        self.assertEqual(idx_trans.shape, (24, 40))

        # Test remaining hotkeys and HUD overlay
        for key in ("RIGHT", "LEFT", "p", "+", "-", "d", "h", "?", " "):
            self.assertTrue(app.key(key, 0.5))
        hud_str = app.overlay(80, 24)
        self.assertIsInstance(hud_str, str)
        self.assertFalse(app.key("q", 0.6))


class TestClientUpdater(unittest.TestCase):
    def test_parse_and_format_version(self):
        self.assertEqual(parse_version("0.2.1"), (0, 2, 1))
        self.assertEqual(parse_version("v1.10.3"), (1, 10, 3))
        self.assertEqual(format_version("v0.2.1"), "0.2.1")
        self.assertGreater(parse_version("0.2.1"), parse_version("0.2.0"))
        self.assertGreater(parse_version("1.0.0"), parse_version("0.9.9"))

        for bad in ("", "1.2", "1.2.3.4", "v1.2.x", "0.2.1; rm -rf /", "../0.2.1"):
            with self.assertRaises(ValueError):
                parse_version(bad)

    def test_fetch_latest_version_tiers(self):
        # Tier 1: GitHub Releases API
        def fetch_release(url, timeout):
            if url == RELEASES_API_URL:
                return json.dumps({"tag_name": "v0.3.0"})
            raise RuntimeError("unexpected URL")

        self.assertEqual(fetch_latest_version(fetcher=fetch_release), ("0.3.0", "github-release"))

        # Tier 2: Fallback to GitHub Tags API when no release exists
        def fetch_tags(url, timeout):
            if url == RELEASES_API_URL:
                raise RuntimeError("404 Not Found")
            if url == TAGS_API_URL:
                return json.dumps([{"name": "v0.1.0"}, {"name": "v0.2.5"}, {"name": "v0.2.1"}])
            raise RuntimeError("unexpected URL")

        self.assertEqual(fetch_latest_version(fetcher=fetch_tags), ("0.2.5", "github-tag"))

        # Tier 3: Fallback to raw pyproject.toml on main
        def fetch_pyproject(url, timeout):
            if url in (RELEASES_API_URL, TAGS_API_URL):
                raise RuntimeError("404")
            if url == RAW_PYPROJECT_URL:
                return '[project]\nname = "terminal-trip"\nversion = "0.4.2"\n'
            raise RuntimeError("unexpected URL")

        self.assertEqual(fetch_latest_version(fetcher=fetch_pyproject), ("0.4.2", "pyproject-main"))

    def test_check_for_update_and_caching(self):
        calls = [0]

        def fake_fetch(url, timeout):
            calls[0] += 1
            return json.dumps({"tag_name": "v0.5.0"})

        info = check_for_update(current_version="0.2.1", fetcher=fake_fetch)
        self.assertTrue(info["update_available"])
        self.assertEqual(info["current"], "0.2.1")
        self.assertEqual(info["latest"], "0.5.0")

        with tempfile.TemporaryDirectory() as tmp:
            cpath = Path(tmp) / "update_check.json"
            calls[0] = 0
            r1 = check_for_update_cached(
                current_version="0.2.1",
                cache_path=cpath,
                fetcher=fake_fetch,
                now_fn=lambda: 1000.0,
            )
            self.assertFalse(r1["cached"])
            self.assertEqual(calls[0], 1)

            # Second call within 24h TTL uses cache without network call
            r2 = check_for_update_cached(
                current_version="0.2.1",
                cache_path=cpath,
                fetcher=fake_fetch,
                now_fn=lambda: 2000.0,
            )
            self.assertTrue(r2["cached"])
            self.assertTrue(r2["update_available"])
            self.assertEqual(calls[0], 1)

    def test_perform_update_git_and_pip_modes(self):
        def fetch_newer(url, timeout):
            return json.dumps({"tag_name": "v0.3.0"})

        def fetch_same(url, timeout):
            return json.dumps({"tag_name": f"v{__version__}"})

        # 1. Already up to date
        ok, msg = perform_update(current_version=__version__, fetcher=fetch_same)
        self.assertTrue(ok)
        self.assertIn("already up to date", msg)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".git").mkdir()

            # 2. Git mode refuses to overwrite dirty working tree
            def dirty_runner(cmd, **kw):
                return subprocess.CompletedProcess(cmd, 0, stdout=" M main.py\n", stderr="")

            ok_dirty, msg_dirty = perform_update(
                current_version="0.2.1",
                repo_dir=root,
                fetcher=fetch_newer,
                runner=dirty_runner,
            )
            self.assertFalse(ok_dirty)
            self.assertIn("uncommitted changes", msg_dirty)

            # 3. Git mode clean working tree performs fetch + pull --ff-only
            git_cmds = []

            def clean_git_runner(cmd, **kw):
                git_cmds.append(cmd)
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

            ok_git, msg_git = perform_update(
                current_version="0.2.1",
                repo_dir=root,
                fetcher=fetch_newer,
                runner=clean_git_runner,
            )
            self.assertTrue(ok_git)
            self.assertIn("via git", msg_git)
            self.assertEqual(len(git_cmds), 3)  # status, fetch --tags, pull --ff-only

        # 4. Pip package mode (no .git directory)
        with tempfile.TemporaryDirectory() as tmp_pkg:
            pip_cmds = []

            def pip_runner(cmd, **kw):
                pip_cmds.append(cmd)
                return subprocess.CompletedProcess(cmd, 0, stdout="Successfully installed", stderr="")

            ok_pip, msg_pip = perform_update(
                current_version="0.2.1",
                repo_dir=Path(tmp_pkg),
                fetcher=fetch_newer,
                runner=pip_runner,
            )
            self.assertTrue(ok_pip)
            self.assertIn("via pip", msg_pip)
            self.assertEqual(len(pip_cmds), 1)
            self.assertIn("pip", pip_cmds[0])

    def test_cli_check_update_and_update_flags(self):
        def fetch_newer(url, timeout):
            return json.dumps({"tag_name": "v0.3.0"})

        def fake_runner(cmd, **kw):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        rc_check = main(["--check-update"], fetcher=fetch_newer)
        self.assertEqual(rc_check, 0)
        rc_upd = main(["--update"], fetcher=fetch_newer, runner=fake_runner)
        self.assertEqual(rc_upd, 0)


if __name__ == "__main__":
    unittest.main()

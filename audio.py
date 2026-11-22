"""Audio sources (system loopback, file playback, procedural synth) and the
FFT / band / beat feature extractor that drives the visuals."""
from __future__ import annotations

import threading
import time
import wave
from collections import deque
from dataclasses import dataclass

import numpy as np

try:  # optional: decodes wav/mp3/flac/ogg and plays audio to the speakers
    import miniaudio
except Exception:  # pragma: no cover - optional dependency
    miniaudio = None


# --------------------------------------------------------------------------
# Ring buffer shared between capture threads and the render loop
# --------------------------------------------------------------------------
class RingBuffer:
    def __init__(self, size: int):
        self.size = size
        self.buf = np.zeros(size, np.float32)
        self.pos = 0
        self.lock = threading.Lock()
        self.last_write = 0.0

    def write(self, x: np.ndarray):
        x = np.asarray(x, np.float32)
        n = len(x)
        with self.lock:
            if n >= self.size:
                self.buf[:] = x[-self.size:]
                self.pos = 0
            else:
                end = self.pos + n
                if end <= self.size:
                    self.buf[self.pos:end] = x
                else:
                    k = self.size - self.pos
                    self.buf[self.pos:] = x[:k]
                    self.buf[:n - k] = x[k:]
                self.pos = end % self.size
            self.last_write = time.perf_counter()

    def read(self, n: int) -> np.ndarray:
        with self.lock:
            idx = (self.pos - n + np.arange(n)) % self.size
            return self.buf[idx]


class AudioSource:
    name = "base"
    label = "base"
    samplerate = 44100

    def start(self):
        pass

    def stop(self):
        pass

    def read(self, n: int) -> np.ndarray:  # newest ``n`` mono samples
        raise NotImplementedError


# --------------------------------------------------------------------------
# Procedural 125 BPM acid-techno synth (pure NumPy, no dependencies)
# --------------------------------------------------------------------------
BPM = 125.0
BASS_PATTERN = np.array([36, 36, 48, 36, 39, 36, 43, 36, 36, 48, 36, 46, 39, 51, 36, 43])
CHORDS = np.array([[57, 60, 64, 69], [53, 57, 60, 65], [55, 59, 62, 67], [52, 55, 59, 64]])
TAU = 2.0 * np.pi


def _hash_noise(s: np.ndarray) -> np.ndarray:
    x = (s * 1103515245 + 12345) & 0x7FFFFFFF
    x = ((x ^ (x >> 13)) * 1274126177) & 0x7FFFFFFF
    return x / float(2 ** 30) - 1.0


def _mtof(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


def synth_block(s: np.ndarray, sr: int) -> np.ndarray:
    """Render samples at absolute sample indices ``s`` (vectorized)."""
    t = s / sr
    beat = 60.0 / BPM
    six = beat / 4
    tb = t % beat
    step = (t // six).astype(np.int64)
    ts = t % six
    bar = (t // (beat * 4)).astype(np.int64)
    beat_idx = (t // beat).astype(np.int64)
    section = bar % 16

    kick_phase = TAU * (45 * tb + (110 / 30) * (1 - np.exp(-30 * tb)))
    kick = np.sin(kick_phase) * np.exp(-tb * 7) * 0.9 * (section < 12)
    hat = _hash_noise(s) * np.exp(-ts * 90) * np.where(step % 4 == 2, 0.35, 0.12)
    snare = _hash_noise(s * 7 + 3) * np.where(beat_idx % 2 == 1, np.exp(-tb * 25), 0) * 0.35

    note = BASS_PATTERN[step % 16] + np.where((bar // 4) % 2 == 1, 5, 0)
    saw = 2 * ((t * _mtof(note)) % 1.0) - 1
    bass = saw * np.exp(-ts * 10) * 0.35 * (0.6 + 0.4 * np.sin(TAU * t * 0.125))

    fa = _mtof(CHORDS[(bar // 2) % 4, step % 4] + 12)
    arp = (np.sin(TAU * fa * t) + 0.3 * np.sin(2 * TAU * fa * t)) * np.exp(-ts * 14) * 0.18

    return np.tanh((kick + hat + snare + bass + arp) * 1.2).astype(np.float32)


class SynthSource(AudioSource):
    name = "synth"
    label = "demo synth"

    def __init__(self, samplerate: int = 44100, clock=time.perf_counter):
        self.samplerate = samplerate
        self.clock = clock
        self.t0 = 0.0

    def start(self):
        self.t0 = self.clock()

    def read(self, n):
        end = int((self.clock() - self.t0) * self.samplerate) + 10 * self.samplerate
        return synth_block(np.arange(end - n, end, dtype=np.int64), self.samplerate)


# --------------------------------------------------------------------------
# File playback (miniaudio for decode+playback, stdlib wave as fallback)
# --------------------------------------------------------------------------
MAX_AUDIO_FILE_BYTES = 256 * 1024 * 1024  # 256 MiB guard against memory exhaustion
MAX_AUDIO_FRAMES = 48000 * 60 * 30        # 30 minutes @ 48 kHz


def sanitize_label(text: str, max_len: int = 32) -> str:
    """Strip ASCII/Unicode control characters to prevent terminal escape injection."""
    cleaned = "".join(ch if (ord(ch) >= 0x20 and ord(ch) != 0x7F and not (0x80 <= ord(ch) <= 0x9F)) else " "
                      for ch in str(text))
    return cleaned.strip()[:max_len]


def _validate_pcm(a: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    if sr <= 0 or sr > 384000:
        raise ValueError(f"invalid sample rate {sr}")
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError("audio file contains no valid audio samples")
    return a, int(sr)


def _load_wav(path):
    with wave.open(path, "rb") as wf:
        ch, sw, sr, n = wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()
        if ch <= 0 or ch > 16:
            raise ValueError(f"invalid WAV channel count {ch}")
        if n <= 0 or n > MAX_AUDIO_FRAMES:
            raise ValueError(f"invalid or excessive WAV frame count {n}")
        raw = wf.readframes(n)
    if sw == 1:
        a = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128) / 128
    elif sw == 2:
        a = np.frombuffer(raw, "<i2").astype(np.float32) / 32768
    elif sw == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        a = (np.where(v >= 2 ** 23, v - 2 ** 24, v) / 2 ** 23).astype(np.float32)
    elif sw == 4:
        a = np.frombuffer(raw, "<i4").astype(np.float32) / 2 ** 31
    else:
        raise RuntimeError(f"unsupported WAV sample width {sw}")
    frames = len(a) // ch
    if frames == 0:
        raise ValueError("WAV file contains no complete sample frames")
    return _validate_pcm(a[:frames * ch].reshape(frames, ch), sr)


def load_audio(path):
    import os
    size = os.path.getsize(path)
    if size <= 0:
        raise ValueError("audio file is empty")
    if size > MAX_AUDIO_FILE_BYTES:
        raise ValueError(f"audio file exceeds {MAX_AUDIO_FILE_BYTES // (1024 * 1024)} MiB safety limit")
    if miniaudio is not None:
        d = miniaudio.decode_file(path, output_format=miniaudio.SampleFormat.FLOAT32)
        if d.nchannels <= 0 or len(d.samples) == 0:
            raise ValueError("decoded audio file contains no samples")
        frames = len(d.samples) // d.nchannels
        if frames > MAX_AUDIO_FRAMES:
            raise ValueError("decoded audio exceeds maximum frame limit")
        a = np.frombuffer(d.samples, dtype=np.float32)[:frames * d.nchannels].reshape(frames, d.nchannels).copy()
        return _validate_pcm(a, d.sample_rate)
    if path.lower().endswith(".wav"):
        return _load_wav(path)
    raise RuntimeError("install 'miniaudio' (pip install miniaudio) to play mp3/flac/ogg")


class FileSource(AudioSource):
    name = "file"

    def __init__(self, path: str, clock=time.perf_counter):
        import os
        data, sr = load_audio(path)
        if data.shape[1] > 2:
            data = data[:, :2]
        self.data = np.ascontiguousarray(data, np.float32)
        self.mono = self.data.mean(axis=1)
        self.samplerate = sr
        self.label = sanitize_label(os.path.basename(path))
        self.clock = clock
        self.device = None
        self.pos = 0
        self.latency = 0
        self.t0 = 0.0

    def _producer(self):
        frames = yield b""
        n = len(self.data)
        while True:
            idx = (self.pos + np.arange(frames)) % n
            self.pos += frames
            frames = yield self.data[idx].tobytes()

    def start(self):
        self.t0 = self.clock()
        if miniaudio is None:
            return
        try:
            dev = miniaudio.PlaybackDevice(output_format=miniaudio.SampleFormat.FLOAT32,
                                           nchannels=self.data.shape[1],
                                           sample_rate=self.samplerate,
                                           buffersize_msec=60)
            gen = self._producer()
            next(gen)
            dev.start(gen)
            self.device = dev
            self.latency = int(0.06 * self.samplerate)
        except Exception:
            self.device = None

    def stop(self):
        if self.device is not None:
            self.device.close()
            self.device = None

    def read(self, n):
        if self.device is not None:
            end = self.pos - self.latency
        else:  # no playback backend: advance silently on the wall clock
            end = int((self.clock() - self.t0) * self.samplerate)
        idx = (end - n + np.arange(n)) % len(self.mono)
        return self.mono[idx]


# --------------------------------------------------------------------------
# System loopback (WASAPI via pyaudiowpatch, or the soundcard package)
# --------------------------------------------------------------------------
class LoopbackSource(AudioSource):
    name = "loopback"
    label = "system loopback"

    def __init__(self):
        self.ring = RingBuffer(48000)
        self._stop = None

    def start(self):
        errors = []
        for starter in (self._start_pyaudiowpatch, self._start_soundcard):
            try:
                starter()
                return
            except Exception as e:  # try the next backend
                errors.append(f"{starter.__name__[7:]}: {e}")
        raise RuntimeError("system loopback unavailable (" + "; ".join(errors) + ")")

    def _start_pyaudiowpatch(self):
        import pyaudiowpatch as pyaudio
        p = pyaudio.PyAudio()
        try:
            wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            spk = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
            if not spk.get("isLoopbackDevice"):
                for lb in p.get_loopback_device_info_generator():
                    if spk["name"] in lb["name"]:
                        spk = lb
                        break
                else:
                    raise RuntimeError("no loopback device for default speakers")
            ch = max(1, int(spk["maxInputChannels"]))
            sr = int(spk["defaultSampleRate"])
            self.samplerate = sr
            self.ring = RingBuffer(sr)

            def cb(in_data, frame_count, time_info, status):
                a = np.frombuffer(in_data, np.float32).reshape(-1, ch).mean(axis=1)
                self.ring.write(a)
                return (None, pyaudio.paContinue)

            stream = p.open(format=pyaudio.paFloat32, channels=ch, rate=sr, input=True,
                            input_device_index=spk["index"], frames_per_buffer=512,
                            stream_callback=cb)
            stream.start_stream()
        except Exception:
            p.terminate()
            raise
        self.label = "loopback: " + sanitize_label(spk["name"].replace(" [Loopback]", ""), 28)

        def stop():
            stream.stop_stream()
            stream.close()
            p.terminate()
        self._stop = stop

    def _start_soundcard(self):
        import soundcard as sc
        spk = sc.default_speaker()
        mic = sc.get_microphone(id=str(spk.name), include_loopback=True)
        self.samplerate = 48000
        self.ring = RingBuffer(48000)
        running = threading.Event()
        running.set()

        def loop():
            with mic.recorder(samplerate=48000, blocksize=512) as rec:
                while running.is_set():
                    self.ring.write(rec.record(numframes=512).mean(axis=1))

        threading.Thread(target=loop, daemon=True).start()
        self.label = "loopback: " + sanitize_label(spk.name, 28)
        self._stop = running.clear

    def stop(self):
        if self._stop:
            self._stop()
            self._stop = None

    def read(self, n):
        if time.perf_counter() - self.ring.last_write > 0.25:  # WASAPI sends nothing on silence
            return np.zeros(n, np.float32)
        return self.ring.read(n)


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
@dataclass
class Features:
    wave: np.ndarray      # recent waveform, roughly -1..1 (auto-gained)
    bars: np.ndarray      # log-spaced spectrum bars, 0..1
    bass: float
    mid: float
    treble: float
    energy: float
    beat: bool            # True on the frame a beat is detected
    pulse: float          # 1.0 at a beat, decays to 0
    level: float          # raw RMS (for "is anything playing")


class Analyzer:
    BANDS = ((20, 160), (160, 2000), (2000, 12000))

    def __init__(self, samplerate: int, n: int = 2048, nbars: int = 48):
        self.sr, self.n, self.nbars = samplerate, n, nbars
        self.window = np.hanning(n).astype(np.float32)
        freqs = np.fft.rfftfreq(n, 1.0 / samplerate)
        edges = np.geomspace(40, min(16000, samplerate * 0.45), nbars + 1)
        self.lo = np.searchsorted(freqs, edges[:-1])
        self.hi = np.maximum(np.searchsorted(freqs, edges[1:]), self.lo + 1)
        self.tilt = np.linspace(1.0, 4.0, nbars)  # treble carries less energy
        self.band_bins = [(np.searchsorted(freqs, a), max(np.searchsorted(freqs, b),
                                                         np.searchsorted(freqs, a) + 1))
                          for a, b in self.BANDS]
        self.bar_ref = 0.05
        self.band_ref = np.full(3, 0.02)
        self.wave_ref = 0.05
        self.bars_s = np.zeros(nbars)
        self.bands_s = np.zeros(3)
        self.hist = deque(maxlen=45)
        self.clock = 0.0
        self.last_beat = -1.0
        self.pulse = 0.0
        self.gain = 1.0

    def process(self, x: np.ndarray, dt: float) -> Features:
        dt = max(dt, 1e-3)
        self.clock += dt
        x = np.asarray(x[-self.n:], np.float32)
        spec = np.abs(np.fft.rfft(x * self.window)) / (self.n / 4)
        cs = np.concatenate(([0.0], np.cumsum(spec)))
        decay = 0.5 ** (dt / 4.0)  # auto-gain references halve every 4 s

        raw_bars = (cs[self.hi] - cs[self.lo]) / (self.hi - self.lo) * self.tilt
        self.bar_ref = max(self.bar_ref * decay, float(raw_bars.max()), 0.02)
        bars = np.log1p(20 * raw_bars / self.bar_ref) / np.log1p(20)

        raw_bands = np.array([(cs[b] - cs[a]) / (b - a) for a, b in self.band_bins])
        self.band_ref = np.maximum(np.maximum(self.band_ref * decay, raw_bands), 0.01)
        bands = (raw_bands / self.band_ref) ** 1.5

        att = 1 - 0.35 ** (dt * 60)
        rel = 1 - 0.88 ** (dt * 60)
        self.bars_s += (bars - self.bars_s) * np.where(bars > self.bars_s, att, rel)
        self.bands_s += (bands - self.bands_s) * np.where(bands > self.bands_s, att, rel)

        # beat: bass energy jumps well above its ~0.75 s running mean
        e = raw_bands[0]
        avg = (sum(self.hist) / len(self.hist)) if self.hist else e
        self.hist.append(e)
        beat = (e > 1.4 * avg and e > 0.3 * self.band_ref[0] and e > 0.01
                and self.clock - self.last_beat > 0.18)
        if beat:
            self.last_beat = self.clock
            self.pulse = 1.0
        else:
            self.pulse *= 0.5 ** (dt / 0.12)

        w = x[-1024:]
        self.wave_ref = max(self.wave_ref * decay, float(np.abs(w).max()), 0.02)
        g = self.gain
        b3 = np.clip(self.bands_s * g, 0, 1)
        bars_out = np.clip(self.bars_s * g, 0, 1)
        return Features(wave=w / self.wave_ref, bars=bars_out,
                        bass=float(b3[0]), mid=float(b3[1]), treble=float(b3[2]),
                        energy=float(bars_out.mean()), beat=bool(beat), pulse=float(self.pulse),
                        level=float(np.sqrt(np.mean(x * x))))

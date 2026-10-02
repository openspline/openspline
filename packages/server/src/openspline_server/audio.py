"""Stateful audio conversion. Preserve partial samples and resampler delay."""

from fractions import Fraction

import av
import numpy as np


class AudioConverter:
    def __init__(self, sample_rate=24000, channels=1, encoding="pcm_s16le"):
        if sample_rate not in {8000, 16000, 22050, 24000, 32000, 44100, 48000}:
            raise ValueError("Unsupported sample rate")
        if channels not in (1, 2):
            raise ValueError("channels must be 1 or 2")
        if encoding not in {"pcm_s16le", "pcm_f32le"}:
            raise ValueError("Unsupported PCM encoding")
        self.rate, self.channels, self.encoding = sample_rate, channels, encoding
        self.dtype = np.dtype("<i2" if encoding == "pcm_s16le" else "<f4")
        self.resampler = av.AudioResampler(format="s16", layout="mono", rate=48000)
        self.pending = b""
        self.position = 0
        self.emitted = 0

    def push(self, raw: bytes) -> np.ndarray:
        self.pending += raw
        size = self.dtype.itemsize * self.channels
        n = len(self.pending) // size * size
        if not n:
            return np.empty(0, dtype=np.int16)
        samples = np.frombuffer(self.pending[:n], dtype=self.dtype)
        self.pending = self.pending[n:]
        if self.encoding == "pcm_f32le":
            samples = (np.clip(np.nan_to_num(samples), -1, 1) * 32767).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(
            samples.reshape(1, -1), format="s16", layout="mono" if self.channels == 1 else "stereo"
        )
        frame.sample_rate = self.rate
        frame.pts = self.position
        frame.time_base = Fraction(1, self.rate)
        self.position += len(samples) // self.channels
        out = self._collect(self.resampler.resample(frame))
        self.emitted += len(out)
        return out

    def flush(self):
        if self.pending:
            raise ValueError("Audio ends with an incomplete sample")
        # libswresample can discard inputs shorter than its filter delay. Pad
        # the converter, then trim to the exact input duration.
        needed = round(self.position * 48000 / self.rate) - self.emitted
        padding = self.push(bytes(64 * self.channels * self.dtype.itemsize))
        tail = self._collect(self.resampler.resample(None))
        return np.concatenate([padding, tail])[:needed]

    @staticmethod
    def _collect(frames):
        return (
            np.concatenate([f.to_ndarray().reshape(-1) for f in frames])
            if frames
            else np.empty(0, dtype=np.int16)
        )


class EncodedDecoder:
    def __init__(self, encoding):
        if encoding not in {"mp3"}:
            raise ValueError("Use the file endpoint for WAV containers")
        self.codec = av.CodecContext.create(encoding, "r")
        self.header = bytearray()
        self.header_done = False
        self.skip = 0
        self.resampler = av.AudioResampler(format="s16", layout="mono", rate=48000)

    def push(self, data):
        if not self.header_done:
            self.header.extend(data)
            if len(self.header) < 10:
                return np.empty(0, dtype=np.int16)
            if self.header[:3] == b"ID3":
                size = sum(
                    (b & 127) << shift for b, shift in zip(self.header[6:10], (21, 14, 7, 0))
                )
                if size > 1024 * 1024:
                    raise ValueError("MP3 metadata exceeds 1 MiB")
                self.skip = 10 + size + (10 if self.header[5] & 16 else 0)
            data = bytes(self.header)
            self.header.clear()
            self.header_done = True
        if self.skip:
            count = min(self.skip, len(data))
            self.skip -= count
            data = data[count:]
        frames = [f for p in self.codec.parse(data) for f in self.codec.decode(p)]
        return AudioConverter._collect([r for f in frames for r in self.resampler.resample(f)])

    def flush(self):
        out = self.push(b"")
        tail = [r for f in self.codec.decode(None) for r in self.resampler.resample(f)]
        tail += self.resampler.resample(None)
        return np.concatenate([out, AudioConverter._collect(tail)])

"""WAV file I/O for AudioBuffer.

Uses only the stdlib ``wave`` module -- zero external dependencies.
Supports reading 8/16/24/32-bit PCM and writing 16/24-bit PCM.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from cydsp.buffer import AudioBuffer


def read_wav(path: str | Path) -> AudioBuffer:
    """Read a WAV file and return an AudioBuffer.

    Supports 8-bit unsigned, 16-bit signed, 24-bit signed, and 32-bit signed PCM.
    Output is float32 normalized to [-1, 1].
    """
    path = Path(path)
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    total_samples = n_frames * n_channels

    if sampwidth == 1:
        # 8-bit unsigned
        samples = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128.0) / 128.0
    elif sampwidth == 2:
        # 16-bit signed
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        samples = samples / 32768.0
    elif sampwidth == 3:
        # 24-bit signed -- vectorized approach
        raw = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(-1, 3)
        padded = np.zeros((len(raw), 4), dtype=np.uint8)
        padded[:, 0:3] = raw
        # Sign extend: if high bit of third byte is set, fill fourth byte
        padded[:, 3] = np.where(raw[:, 2] & 0x80, 0xFF, 0x00)
        samples = padded.view(np.int32).flatten().astype(np.float32)
        samples = samples / 8388608.0
    elif sampwidth == 4:
        # 32-bit signed
        samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32)
        samples = samples / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

    if len(samples) != total_samples:
        raise ValueError(f"Expected {total_samples} samples, got {len(samples)}")

    # Deinterleave to planar [channels, frames]
    if n_channels == 1:
        data = samples.reshape(1, -1)
    else:
        # Interleaved: [L0, R0, L1, R1, ...] -> [[L0, L1, ...], [R0, R1, ...]]
        data = samples.reshape(-1, n_channels).T

    data = np.ascontiguousarray(data, dtype=np.float32)
    return AudioBuffer(data, sample_rate=float(sample_rate))


def write_wav(
    path: str | Path,
    buf: AudioBuffer,
    bit_depth: int = 16,
) -> None:
    """Write an AudioBuffer to a WAV file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    buf : AudioBuffer
        Audio data to write.
    bit_depth : int
        Output bit depth: 16 or 24.
    """
    if bit_depth not in (16, 24):
        raise ValueError(f"Unsupported bit_depth: {bit_depth} (use 16 or 24)")

    path = Path(path)
    n_channels = buf.channels
    sample_rate = int(buf.sample_rate)
    sampwidth = bit_depth // 8

    # Interleave: [channels, frames] -> [frames, channels] -> flat
    data = buf.data.copy()
    # Clip to [-1, 1]
    np.clip(data, -1.0, 1.0, out=data)

    if bit_depth == 16:
        interleaved = data.T.flatten()
        scaled = (interleaved * 32767.0).astype(np.int16)
        raw_bytes = scaled.tobytes()
    elif bit_depth == 24:
        interleaved = data.T.flatten()
        scaled = np.clip(interleaved * 8388607.0, -8388608.0, 8388607.0).astype(
            np.int32
        )
        # int32 -> view as uint8 -> take lower 3 bytes (little-endian)
        bytes_4 = scaled.view(np.uint8).reshape(-1, 4)
        raw_bytes = bytes_4[:, :3].tobytes()

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(raw_bytes)

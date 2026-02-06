"""High-level DSP functions operating on AudioBuffer.

All functions accept AudioBuffer inputs and return AudioBuffer outputs.
Frequencies are specified in Hz and automatically converted to the
normalized [0, 0.5) range expected by the C++ bindings.
"""

from __future__ import annotations

import numpy as np

from cydsp.buffer import AudioBuffer
from cydsp._core import filters, fft, delay as _delay, envelopes, rates, mix


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hz_to_normalized(freq_hz: float, sample_rate: float) -> float:
    """Convert Hz to normalized frequency [0, 0.5).

    Raises ValueError if freq_hz is negative or >= Nyquist.
    """
    if freq_hz < 0:
        raise ValueError(f"Frequency must be non-negative, got {freq_hz}")
    nyquist = sample_rate / 2.0
    if freq_hz >= nyquist:
        raise ValueError(f"Frequency {freq_hz} Hz >= Nyquist ({nyquist} Hz)")
    return freq_hz / sample_rate


def _process_per_channel(buf: AudioBuffer, process_fn) -> AudioBuffer:
    """Apply process_fn(1d_array) -> 1d_array per channel, return new AudioBuffer."""
    out = np.zeros_like(buf.data)
    for ch in range(buf.channels):
        out[ch] = process_fn(buf.ensure_1d(ch))
    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# Filter functions
# ---------------------------------------------------------------------------


def lowpass(
    buf: AudioBuffer,
    cutoff_hz: float,
    octaves: float | None = None,
    design=filters.BiquadDesign.bilinear,
) -> AudioBuffer:
    freq = _hz_to_normalized(cutoff_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        if octaves is not None:
            bq.lowpass(freq, octaves, design)
        else:
            bq.lowpass(freq, design=design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def highpass(
    buf: AudioBuffer,
    cutoff_hz: float,
    octaves: float | None = None,
    design=filters.BiquadDesign.bilinear,
) -> AudioBuffer:
    freq = _hz_to_normalized(cutoff_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        if octaves is not None:
            bq.highpass(freq, octaves, design)
        else:
            bq.highpass(freq, design=design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def bandpass(
    buf: AudioBuffer,
    center_hz: float,
    octaves: float | None = None,
    design=filters.BiquadDesign.one_sided,
) -> AudioBuffer:
    freq = _hz_to_normalized(center_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        if octaves is not None:
            bq.bandpass(freq, octaves, design)
        else:
            bq.bandpass(freq, design=design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def notch(
    buf: AudioBuffer,
    center_hz: float,
    octaves: float | None = None,
    design=filters.BiquadDesign.one_sided,
) -> AudioBuffer:
    freq = _hz_to_normalized(center_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        if octaves is not None:
            bq.notch(freq, octaves, design)
        else:
            bq.notch(freq, design=design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def peak(
    buf: AudioBuffer,
    center_hz: float,
    gain: float,
    octaves: float = 1.0,
    design=filters.BiquadDesign.one_sided,
) -> AudioBuffer:
    freq = _hz_to_normalized(center_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        bq.peak(freq, gain, octaves, design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def peak_db(
    buf: AudioBuffer,
    center_hz: float,
    db: float,
    octaves: float = 1.0,
    design=filters.BiquadDesign.one_sided,
) -> AudioBuffer:
    freq = _hz_to_normalized(center_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        bq.peak_db(freq, db, octaves, design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def high_shelf(
    buf: AudioBuffer,
    cutoff_hz: float,
    gain: float,
    octaves: float | None = None,
    design=filters.BiquadDesign.one_sided,
) -> AudioBuffer:
    freq = _hz_to_normalized(cutoff_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        if octaves is not None:
            bq.high_shelf(freq, gain, octaves, design)
        else:
            bq.high_shelf(freq, gain, design=design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def high_shelf_db(
    buf: AudioBuffer,
    cutoff_hz: float,
    db: float,
    octaves: float | None = None,
    design=filters.BiquadDesign.one_sided,
) -> AudioBuffer:
    freq = _hz_to_normalized(cutoff_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        if octaves is not None:
            bq.high_shelf_db(freq, db, octaves, design)
        else:
            bq.high_shelf_db(freq, db, design=design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def low_shelf(
    buf: AudioBuffer,
    cutoff_hz: float,
    gain: float,
    octaves: float = 2.0,
    design=filters.BiquadDesign.one_sided,
) -> AudioBuffer:
    freq = _hz_to_normalized(cutoff_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        bq.low_shelf(freq, gain, octaves, design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def low_shelf_db(
    buf: AudioBuffer,
    cutoff_hz: float,
    db: float,
    octaves: float = 2.0,
    design=filters.BiquadDesign.one_sided,
) -> AudioBuffer:
    freq = _hz_to_normalized(cutoff_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        bq.low_shelf_db(freq, db, octaves, design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def allpass(
    buf: AudioBuffer,
    freq_hz: float,
    octaves: float = 1.0,
    design=filters.BiquadDesign.one_sided,
) -> AudioBuffer:
    freq = _hz_to_normalized(freq_hz, buf.sample_rate)

    def _process(x):
        bq = filters.Biquad()
        bq.allpass(freq, octaves, design)
        return bq.process(x)

    return _process_per_channel(buf, _process)


def biquad_process(buf: AudioBuffer, biquad) -> AudioBuffer:
    """Process buffer through a pre-configured Biquad, resetting between channels."""

    def _process(x):
        biquad.reset()
        return biquad.process(x)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# Delay functions
# ---------------------------------------------------------------------------


def delay(
    buf: AudioBuffer,
    delay_samples: float,
    capacity: int | None = None,
    interpolation: str = "linear",
) -> AudioBuffer:
    """Apply a fixed delay (in samples) per channel."""
    cap = capacity if capacity is not None else int(delay_samples) + 64

    def _process(x):
        if interpolation == "cubic":
            d = _delay.DelayCubic(cap)
        else:
            d = _delay.Delay(cap)
        return d.process(x, delay_samples)

    return _process_per_channel(buf, _process)


def delay_varying(
    buf: AudioBuffer,
    delays,
    interpolation: str = "linear",
) -> AudioBuffer:
    """Apply time-varying delay per channel.

    Parameters
    ----------
    delays : ndarray
        1D (broadcast to all channels) or 2D [channels, frames].
    """
    delays = np.asarray(delays, dtype=np.float32)
    if delays.ndim == 1:
        delays_2d = np.tile(delays, (buf.channels, 1))
    elif delays.ndim == 2:
        if delays.shape[0] != buf.channels:
            raise ValueError(
                f"delays has {delays.shape[0]} channels, buffer has {buf.channels}"
            )
        delays_2d = delays
    else:
        raise ValueError(f"delays must be 1D or 2D, got {delays.ndim}D")

    if delays_2d.shape[1] != buf.frames:
        raise ValueError(
            f"delays has {delays_2d.shape[1]} frames, buffer has {buf.frames}"
        )

    max_delay = int(np.max(delays_2d)) + 64
    out = np.zeros_like(buf.data)
    for ch in range(buf.channels):
        if interpolation == "cubic":
            d = _delay.DelayCubic(max_delay)
        else:
            d = _delay.Delay(max_delay)
        ch_delays = np.ascontiguousarray(delays_2d[ch])
        out[ch] = d.process_varying(buf.ensure_1d(ch), ch_delays)

    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# Envelope functions
# ---------------------------------------------------------------------------


def box_filter(buf: AudioBuffer, length: int) -> AudioBuffer:
    """Apply a BoxFilter (moving average) per channel."""

    def _process(x):
        bf = envelopes.BoxFilter(length)
        bf.set(length)
        return bf.process(x)

    return _process_per_channel(buf, _process)


def box_stack_filter(buf: AudioBuffer, size: int, layers: int = 4) -> AudioBuffer:
    """Apply a BoxStackFilter (stacked moving average) per channel."""

    def _process(x):
        bs = envelopes.BoxStackFilter(size, layers)
        bs.set(size)
        return bs.process(x)

    return _process_per_channel(buf, _process)


def peak_hold(buf: AudioBuffer, length: int) -> AudioBuffer:
    """Apply PeakHold per channel."""

    def _process(x):
        ph = envelopes.PeakHold(length)
        ph.set(length)
        return ph.process(x)

    return _process_per_channel(buf, _process)


def peak_decay(buf: AudioBuffer, length: int) -> AudioBuffer:
    """Apply PeakDecayLinear per channel."""

    def _process(x):
        pd = envelopes.PeakDecayLinear(length)
        pd.set(length)
        return pd.process(x)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# FFT functions
# ---------------------------------------------------------------------------


def rfft(buf: AudioBuffer) -> list[np.ndarray]:
    """Forward real FFT per channel.

    Returns a list of complex64 arrays (one per channel, N/2 bins each).
    Uses RealFFT.fast_size_above for efficient FFT size, zero-pads if needed.
    """
    fft_size = fft.RealFFT.fast_size_above(buf.frames)
    rfft_obj = fft.RealFFT(fft_size)
    result = []
    for ch in range(buf.channels):
        x = buf.ensure_1d(ch)
        if len(x) < fft_size:
            x = np.pad(x, (0, fft_size - len(x)), mode="constant")
        result.append(rfft_obj.fft(x))
    return result


def irfft(
    spectra: list[np.ndarray],
    size: int,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Inverse real FFT from list of spectra to AudioBuffer.

    Returns unscaled output (matches C++ convention). Divide by N if needed.
    """
    channels = len(spectra)
    bins = spectra[0].shape[0]
    fft_size = bins * 2
    rfft_obj = fft.RealFFT(fft_size)

    out = np.zeros((channels, size), dtype=np.float32)
    for ch in range(channels):
        full = rfft_obj.ifft(spectra[ch])
        out[ch] = full[:size]

    return AudioBuffer(out, sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# Rates functions
# ---------------------------------------------------------------------------


def upsample_2x(
    buf: AudioBuffer,
    max_block: int | None = None,
    half_latency: int = 16,
    pass_freq: float = 0.43,
) -> AudioBuffer:
    """Upsample by 2x. Returns AudioBuffer with 2x frames and 2x sample rate."""
    block = max_block if max_block is not None else buf.frames
    os = rates.Oversampler2x(buf.channels, block, half_latency, pass_freq)
    upsampled = os.up(buf.data)
    return AudioBuffer(
        upsampled,
        sample_rate=buf.sample_rate * 2,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


def oversample_roundtrip(
    buf: AudioBuffer,
    max_block: int | None = None,
    half_latency: int = 16,
    pass_freq: float = 0.43,
) -> AudioBuffer:
    """Upsample then downsample (roundtrip). Same shape and sample rate as input."""
    block = max_block if max_block is not None else buf.frames
    os = rates.Oversampler2x(buf.channels, block, half_latency, pass_freq)
    processed = os.process(buf.data)
    return AudioBuffer(
        processed,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


# ---------------------------------------------------------------------------
# Mix functions
# ---------------------------------------------------------------------------


def hadamard(buf: AudioBuffer) -> AudioBuffer:
    """Apply Hadamard mixing across channels at each frame.

    Requires power-of-2 channel count.
    """
    ch = buf.channels
    if ch == 0 or (ch & (ch - 1)) != 0:
        raise ValueError(f"Hadamard requires power-of-2 channel count, got {ch}")
    h = mix.Hadamard(ch)
    out = np.zeros_like(buf.data)
    for i in range(buf.frames):
        frame = np.ascontiguousarray(buf.data[:, i].copy())
        out[:, i] = h.in_place(frame)
    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


def householder(buf: AudioBuffer) -> AudioBuffer:
    """Apply Householder reflection across channels at each frame."""
    ch = buf.channels
    h = mix.Householder(ch)
    out = np.zeros_like(buf.data)
    for i in range(buf.frames):
        frame = np.ascontiguousarray(buf.data[:, i].copy())
        out[:, i] = h.in_place(frame)
    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


def crossfade(buf_a: AudioBuffer, buf_b: AudioBuffer, x: float) -> AudioBuffer:
    """Crossfade between two buffers using cheap_energy_crossfade coefficients.

    x=0 returns buf_a, x=1 returns buf_b.
    """
    if buf_a.sample_rate != buf_b.sample_rate:
        raise ValueError(
            f"Sample rate mismatch: {buf_a.sample_rate} vs {buf_b.sample_rate}"
        )
    if buf_a.channels != buf_b.channels:
        raise ValueError(
            f"Channel count mismatch: {buf_a.channels} vs {buf_b.channels}"
        )
    if buf_a.frames != buf_b.frames:
        raise ValueError(f"Frame count mismatch: {buf_a.frames} vs {buf_b.frames}")
    to_c, from_c = mix.cheap_energy_crossfade(x)
    out = buf_a.data * from_c + buf_b.data * to_c
    return AudioBuffer(
        out.astype(np.float32),
        sample_rate=buf_a.sample_rate,
        channel_layout=buf_a.channel_layout,
        label=buf_a.label,
    )


# ---------------------------------------------------------------------------
# STFT functions
# ---------------------------------------------------------------------------


class Spectrogram:
    """Lightweight container for STFT output, consumed by ``istft``."""

    __slots__ = (
        "data",
        "window_size",
        "hop_size",
        "fft_size",
        "sample_rate",
        "original_frames",
    )

    def __init__(
        self,
        data: np.ndarray,
        window_size: int,
        hop_size: int,
        fft_size: int,
        sample_rate: float,
        original_frames: int,
    ):
        self.data = data  # [channels, num_stft_frames, bins] complex64
        self.window_size = window_size
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.sample_rate = sample_rate
        self.original_frames = original_frames

    @property
    def channels(self) -> int:
        return self.data.shape[0]

    @property
    def num_frames(self) -> int:
        return self.data.shape[1]

    @property
    def bins(self) -> int:
        return self.data.shape[2]


def stft(
    buf: AudioBuffer,
    window_size: int = 2048,
    hop_size: int | None = None,
) -> Spectrogram:
    """Short-time Fourier transform using windowed RealFFT + overlap.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    window_size : int
        Analysis window length in samples.
    hop_size : int or None
        Hop between successive windows.  Defaults to ``window_size // 4``.

    Returns
    -------
    Spectrogram
        Complex64 data shaped ``[channels, num_stft_frames, fft_size // 2]``.
    """
    if hop_size is None:
        hop_size = window_size // 4

    fft_size = fft.RealFFT.fast_size_above(window_size)
    rfft_obj = fft.RealFFT(fft_size)
    bins = fft_size // 2

    window = np.hanning(window_size).astype(np.float32)

    n_frames = buf.frames
    num_stft_frames = max(0, (n_frames - window_size) // hop_size + 1)

    out = np.zeros((buf.channels, num_stft_frames, bins), dtype=np.complex64)

    for ch in range(buf.channels):
        channel_data = buf.ensure_1d(ch)
        for t in range(num_stft_frames):
            start = t * hop_size
            segment = channel_data[start : start + window_size] * window
            if window_size < fft_size:
                padded = np.zeros(fft_size, dtype=np.float32)
                padded[:window_size] = segment
                segment = padded
            out[ch, t, :] = rfft_obj.fft(segment)

    return Spectrogram(
        data=out,
        window_size=window_size,
        hop_size=hop_size,
        fft_size=fft_size,
        sample_rate=buf.sample_rate,
        original_frames=n_frames,
    )


def istft(spec: Spectrogram) -> AudioBuffer:
    """Inverse STFT via overlap-add with COLA normalization.

    Parameters
    ----------
    spec : Spectrogram
        Output from :func:`stft`.

    Returns
    -------
    AudioBuffer
        Reconstructed audio, trimmed to the original frame count.
    """
    window_size = spec.window_size
    hop_size = spec.hop_size
    fft_size = spec.fft_size

    rfft_obj = fft.RealFFT(fft_size)
    window = np.hanning(window_size).astype(np.float32)

    out_len = (spec.num_frames - 1) * hop_size + window_size
    out = np.zeros((spec.channels, out_len), dtype=np.float32)
    win_sum = np.zeros(out_len, dtype=np.float32)

    for ch in range(spec.channels):
        for t in range(spec.num_frames):
            full = rfft_obj.ifft(np.ascontiguousarray(spec.data[ch, t, :]))
            # ifft is unscaled -- divide by fft_size
            frame = (
                np.asarray(full[:window_size], dtype=np.float32) / fft_size
            ) * window
            start = t * hop_size
            out[ch, start : start + window_size] += frame

    # Window normalization (sum of squared windows at each position)
    for t in range(spec.num_frames):
        start = t * hop_size
        win_sum[start : start + window_size] += window**2

    # Avoid division by zero at edges
    win_sum = np.maximum(win_sum, 1e-8)
    out /= win_sum[np.newaxis, :]

    # Trim to original length
    trim = min(spec.original_frames, out.shape[1])
    out = out[:, :trim]

    return AudioBuffer(out, sample_rate=spec.sample_rate)


# ---------------------------------------------------------------------------
# LFO function
# ---------------------------------------------------------------------------


def lfo(
    frames: int,
    low: float,
    high: float,
    rate: float,
    sample_rate: float = 48000.0,
    rate_variation: float = 0.0,
    depth_variation: float = 0.0,
    seed: int | None = None,
) -> AudioBuffer:
    """Generate an LFO signal using CubicLfo.

    Parameters
    ----------
    frames : int
        Number of output samples.
    low, high : float
        Output value range.
    rate : float
        Base rate (cycles per sample).
    sample_rate : float
        Sample rate for the returned AudioBuffer metadata.
    rate_variation, depth_variation : float
        Randomization parameters (0 = deterministic).
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    AudioBuffer
        Mono buffer containing the LFO waveform.
    """
    if seed is not None:
        osc = envelopes.CubicLfo(seed)
    else:
        osc = envelopes.CubicLfo()
    osc.set(low, high, rate, rate_variation, depth_variation)
    data = osc.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)

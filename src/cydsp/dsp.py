"""High-level DSP functions operating on AudioBuffer.

All functions accept AudioBuffer inputs and return AudioBuffer outputs.
Frequencies are specified in Hz and automatically converted to the
normalized [0, 0.5) range expected by the C++ bindings.
"""

from __future__ import annotations

import numpy as np

from cydsp.buffer import AudioBuffer
from cydsp._core import filters, fft, delay as _delay, envelopes, rates, mix
from cydsp._core import daisysp as _daisysp


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
# DaisySP submodule aliases
# ---------------------------------------------------------------------------

_dsy_osc = _daisysp.oscillators
_dsy_filt = _daisysp.filters
_dsy_fx = _daisysp.effects
_dsy_dyn = _daisysp.dynamics
_dsy_noise = _daisysp.noise
_dsy_drums = _daisysp.drums
_dsy_pm = _daisysp.physical_modeling
_dsy_util = _daisysp.utility

_WAVEFORM_MAP: dict[str, int] = {
    "sine": _dsy_osc.WAVE_SIN,
    "sin": _dsy_osc.WAVE_SIN,
    "tri": _dsy_osc.WAVE_TRI,
    "triangle": _dsy_osc.WAVE_TRI,
    "saw": _dsy_osc.WAVE_SAW,
    "ramp": _dsy_osc.WAVE_RAMP,
    "square": _dsy_osc.WAVE_SQUARE,
    "polyblep_tri": _dsy_osc.WAVE_POLYBLEP_TRI,
    "polyblep_saw": _dsy_osc.WAVE_POLYBLEP_SAW,
    "polyblep_square": _dsy_osc.WAVE_POLYBLEP_SQUARE,
}

_BLOSC_WAVEFORM_MAP: dict[str, int] = {
    "triangle": _dsy_osc.BLOSC_WAVE_TRIANGLE,
    "tri": _dsy_osc.BLOSC_WAVE_TRIANGLE,
    "saw": _dsy_osc.BLOSC_WAVE_SAW,
    "square": _dsy_osc.BLOSC_WAVE_SQUARE,
    "off": _dsy_osc.BLOSC_WAVE_OFF,
}

_LADDER_MODE_MAP: dict[str, int] = {
    "lp24": _dsy_filt.LadderFilterMode.LP24,
    "lp12": _dsy_filt.LadderFilterMode.LP12,
    "bp24": _dsy_filt.LadderFilterMode.BP24,
    "bp12": _dsy_filt.LadderFilterMode.BP12,
    "hp24": _dsy_filt.LadderFilterMode.HP24,
    "hp12": _dsy_filt.LadderFilterMode.HP12,
}


def _resolve_waveform(waveform: int | str, mapping: dict[str, int]) -> int:
    """Resolve a waveform name or int constant to an int."""
    if isinstance(waveform, int):
        return waveform
    key = waveform.lower()
    if key not in mapping:
        raise ValueError(
            f"Unknown waveform {waveform!r}, valid names: {list(mapping.keys())}"
        )
    return mapping[key]


# ---------------------------------------------------------------------------
# Filter functions (signalsmith)
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
# Convolution
# ---------------------------------------------------------------------------


def convolve(
    buf: AudioBuffer,
    ir: AudioBuffer,
    normalize: bool = False,
    trim: bool = True,
) -> AudioBuffer:
    """FFT-based overlap-add convolution.

    Parameters
    ----------
    buf : AudioBuffer
        Input signal.
    ir : AudioBuffer
        Impulse response.
    normalize : bool
        If True, scale IR to unit energy before convolving.
    trim : bool
        If True (default), output has the same length as *buf*.
        If False, output is the full convolution (buf.frames + ir.frames - 1).

    Raises
    ------
    ValueError
        If sample rates differ or channel counts are incompatible.
    """
    if buf.sample_rate != ir.sample_rate:
        raise ValueError(
            f"Sample rate mismatch: buf={buf.sample_rate}, ir={ir.sample_rate}"
        )

    # Channel matching
    if ir.channels == 1 and buf.channels > 1:
        ir_data = np.tile(ir.data, (buf.channels, 1))
    elif ir.channels == buf.channels:
        ir_data = ir.data
    else:
        raise ValueError(
            f"Channel mismatch: buf has {buf.channels}, ir has {ir.channels}. "
            "IR must be mono (broadcasts) or match buf channel count."
        )

    if normalize:
        for ch in range(ir_data.shape[0]):
            energy = np.sqrt(np.sum(ir_data[ch] ** 2))
            if energy > 0:
                ir_data = ir_data.copy()
                ir_data[ch] /= energy

    sig_len = buf.frames
    ir_len = ir.frames
    full_len = sig_len + ir_len - 1
    block_size = ir_len
    fft_size = fft.RealFFT.fast_size_above(2 * block_size)

    n_blocks = (sig_len + block_size - 1) // block_size
    out = np.zeros((buf.channels, full_len), dtype=np.float32)

    for ch in range(buf.channels):
        # FFT the IR once
        ir_padded = np.zeros(fft_size, dtype=np.float32)
        ir_padded[:ir_len] = ir_data[ch]
        IR_freq = np.fft.rfft(ir_padded)

        # Pre-slice all signal blocks into [n_blocks, fft_size]
        blocks = np.zeros((n_blocks, fft_size), dtype=np.float32)
        for b in range(n_blocks):
            start = b * block_size
            end = min(start + block_size, sig_len)
            blocks[b, : end - start] = buf.data[ch, start:end]

        # Batch FFT, multiply, IFFT
        block_freqs = np.fft.rfft(blocks, n=fft_size, axis=1)
        block_freqs *= IR_freq[np.newaxis, :]
        block_results = np.fft.irfft(block_freqs, n=fft_size, axis=1).astype(np.float32)

        # Overlap-add
        for b in range(n_blocks):
            pos = b * block_size
            out_end = min(pos + fft_size, full_len)
            out[ch, pos:out_end] += block_results[b, : out_end - pos]

    if trim:
        out = out[:, :sig_len]

    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


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
# Spectral utility functions
# ---------------------------------------------------------------------------


def magnitude(spec: Spectrogram) -> np.ndarray:
    """Return magnitude of spectral data.

    Parameters
    ----------
    spec : Spectrogram
        Input spectrogram.

    Returns
    -------
    np.ndarray
        float32 array shaped ``[channels, num_frames, bins]``.
    """
    return np.abs(spec.data).astype(np.float32)


def phase(spec: Spectrogram) -> np.ndarray:
    """Return phase angle of spectral data.

    Parameters
    ----------
    spec : Spectrogram
        Input spectrogram.

    Returns
    -------
    np.ndarray
        float32 array shaped ``[channels, num_frames, bins]`` in radians.
    """
    return np.angle(spec.data).astype(np.float32)


def from_polar(mag: np.ndarray, ph: np.ndarray, spec: Spectrogram) -> Spectrogram:
    """Reconstruct a Spectrogram from magnitude and phase arrays.

    Parameters
    ----------
    mag : np.ndarray
        Magnitude array, broadcastable to ``spec.data.shape``.
    ph : np.ndarray
        Phase array in radians, broadcastable to ``spec.data.shape``.
    spec : Spectrogram
        Reference spectrogram whose metadata is copied.

    Returns
    -------
    Spectrogram
        New spectrogram with ``mag * exp(j * ph)`` as data.
    """
    data = (mag * np.exp(1j * ph)).astype(np.complex64)
    return Spectrogram(
        data=data,
        window_size=spec.window_size,
        hop_size=spec.hop_size,
        fft_size=spec.fft_size,
        sample_rate=spec.sample_rate,
        original_frames=spec.original_frames,
    )


def apply_mask(spec: Spectrogram, mask: np.ndarray) -> Spectrogram:
    """Multiply spectral data by a real-valued mask.

    Parameters
    ----------
    spec : Spectrogram
        Input spectrogram.
    mask : np.ndarray
        Real-valued mask broadcastable to ``[channels, num_frames, bins]``.

    Returns
    -------
    Spectrogram
        New spectrogram with masked data.

    Raises
    ------
    ValueError
        If *mask* cannot be broadcast to the spectrogram shape.
    """
    try:
        result = spec.data * mask
    except ValueError:
        raise ValueError(
            f"Mask shape {mask.shape} is not broadcastable to "
            f"spectrogram shape {spec.data.shape}"
        )
    return Spectrogram(
        data=result.astype(np.complex64),
        window_size=spec.window_size,
        hop_size=spec.hop_size,
        fft_size=spec.fft_size,
        sample_rate=spec.sample_rate,
        original_frames=spec.original_frames,
    )


def spectral_gate(
    spec: Spectrogram,
    threshold_db: float = -40.0,
    noise_floor_db: float = -80.0,
) -> Spectrogram:
    """Gate spectral bins below a dB threshold.

    Bins whose magnitude falls below *threshold_db* are attenuated to
    *noise_floor_db* rather than zeroed, reducing musical noise artifacts.

    Parameters
    ----------
    spec : Spectrogram
        Input spectrogram.
    threshold_db : float
        Magnitude threshold in dB.  Bins at or above this pass through.
    noise_floor_db : float
        Attenuation applied to bins below the threshold, in dB relative to
        the threshold.

    Returns
    -------
    Spectrogram
        Gated spectrogram.
    """
    eps = 1e-10
    mag = np.abs(spec.data)
    mag_db = 20.0 * np.log10(mag + eps)
    attenuation = 10.0 ** ((noise_floor_db - threshold_db) / 20.0)
    mask = np.where(mag_db >= threshold_db, 1.0, attenuation).astype(np.float32)
    return apply_mask(spec, mask)


def spectral_emphasis(
    spec: Spectrogram,
    low_db: float = 0.0,
    high_db: float = 0.0,
) -> Spectrogram:
    """Apply a linear dB tilt across frequency bins.

    Gain varies linearly from *low_db* at DC to *high_db* at Nyquist.

    Parameters
    ----------
    spec : Spectrogram
        Input spectrogram.
    low_db : float
        Gain at DC in dB.
    high_db : float
        Gain at Nyquist in dB.

    Returns
    -------
    Spectrogram
        Emphasized spectrogram.
    """
    n_bins = spec.bins
    db_ramp = np.linspace(low_db, high_db, n_bins, dtype=np.float32)
    mask = 10.0 ** (db_ramp / 20.0)
    return apply_mask(spec, mask)


def bin_freq(spec: Spectrogram, bin_index: int) -> float:
    """Return the center frequency in Hz of a given FFT bin.

    Parameters
    ----------
    spec : Spectrogram
        Reference spectrogram.
    bin_index : int
        Bin index (0 = DC).

    Returns
    -------
    float
        Frequency in Hz.
    """
    return float(bin_index * spec.sample_rate / spec.fft_size)


def freq_to_bin(spec: Spectrogram, freq_hz: float) -> int:
    """Return the nearest FFT bin for a given frequency.

    Parameters
    ----------
    spec : Spectrogram
        Reference spectrogram.
    freq_hz : float
        Frequency in Hz.

    Returns
    -------
    int
        Nearest bin index, clamped to ``[0, bins - 1]``.

    Raises
    ------
    ValueError
        If *freq_hz* is negative or >= Nyquist.
    """
    nyquist = spec.sample_rate / 2.0
    if freq_hz < 0:
        raise ValueError(f"Frequency must be non-negative, got {freq_hz}")
    if freq_hz >= nyquist:
        raise ValueError(f"Frequency {freq_hz} Hz >= Nyquist ({nyquist} Hz)")
    exact = freq_hz * spec.fft_size / spec.sample_rate
    return int(np.clip(round(exact), 0, spec.bins - 1))


def time_stretch(spec: Spectrogram, rate: float) -> Spectrogram:
    """Phase-vocoder time stretch.

    Parameters
    ----------
    spec : Spectrogram
        Input spectrogram.
    rate : float
        Stretch rate.  ``rate > 1`` makes audio shorter (faster),
        ``rate < 1`` makes audio longer (slower).

    Returns
    -------
    Spectrogram
        Time-stretched spectrogram with updated ``original_frames``.

    Raises
    ------
    ValueError
        If *rate* <= 0.
    """
    if rate <= 0:
        raise ValueError(f"Rate must be > 0, got {rate}")

    n_ch, n_frames, n_bins = spec.data.shape
    new_frames = max(1, round(n_frames / rate))
    hop = spec.hop_size

    # Expected phase advance per hop for each bin
    omega = 2.0 * np.pi * np.arange(n_bins) * hop / spec.fft_size

    out = np.zeros((n_ch, new_frames, n_bins), dtype=np.complex64)

    input_mag = np.abs(spec.data)
    input_phase = np.angle(spec.data)

    for ch in range(n_ch):
        # Initialize phase accumulator from first frame
        phase_acc = input_phase[ch, 0].copy()

        for t_out in range(new_frames):
            t_in = t_out * rate
            t_floor = int(t_in)
            frac = t_in - t_floor

            # Clamp to valid input range
            t0 = min(t_floor, n_frames - 1)
            t1 = min(t_floor + 1, n_frames - 1)

            # Interpolate magnitude
            mag = (1.0 - frac) * input_mag[ch, t0] + frac * input_mag[ch, t1]

            if t_out == 0:
                phase_acc = input_phase[ch, t0].copy()
            else:
                # Instantaneous frequency from input phase difference
                if t0 < n_frames - 1:
                    dphi = input_phase[ch, t0 + 1] - input_phase[ch, t0]
                else:
                    dphi = np.zeros(n_bins)
                # Deviation from expected phase advance
                deviation = dphi - omega
                # Wrap to [-pi, pi]
                deviation = deviation - 2.0 * np.pi * np.round(
                    deviation / (2.0 * np.pi)
                )
                # True instantaneous frequency
                inst_freq = omega + deviation
                # Accumulate phase at output hop rate
                phase_acc += inst_freq

            out[ch, t_out] = mag * np.exp(1j * phase_acc)

    new_original = max(1, round(spec.original_frames / rate))
    return Spectrogram(
        data=out,
        window_size=spec.window_size,
        hop_size=spec.hop_size,
        fft_size=spec.fft_size,
        sample_rate=spec.sample_rate,
        original_frames=new_original,
    )


def phase_lock(spec: Spectrogram) -> Spectrogram:
    """Identity phase-locking (Laroche & Dolson 1999).

    Finds spectral peaks in each frame and propagates their phase to
    neighboring bins, reducing phasiness.

    Parameters
    ----------
    spec : Spectrogram
        Input spectrogram.

    Returns
    -------
    Spectrogram
        Phase-locked spectrogram with identical magnitudes.
    """
    n_ch, n_frames, n_bins = spec.data.shape
    hop = spec.hop_size
    fft_size = spec.fft_size

    input_mag = np.abs(spec.data)
    input_phase = np.angle(spec.data)
    out_phase = input_phase.copy()

    all_bins = np.arange(n_bins)
    phase_scale = 2.0 * np.pi * hop / fft_size

    for ch in range(n_ch):
        for t in range(n_frames):
            mag = input_mag[ch, t]

            # Vectorized peak detection: mag >= left neighbor AND mag >= right neighbor
            left = np.empty(n_bins, dtype=np.float32)
            left[0] = -1.0
            left[1:] = mag[:-1]
            right = np.empty(n_bins, dtype=np.float32)
            right[-1] = -1.0
            right[:-1] = mag[1:]
            peak_mask = (mag >= left) & (mag >= right)
            peaks = np.nonzero(peak_mask)[0]

            if len(peaks) == 0:
                continue

            # Nearest-peak assignment via searchsorted + left/right comparison
            idx = np.searchsorted(peaks, all_bins, side="left")
            idx = np.clip(idx, 0, len(peaks) - 1)
            # Compare candidate on the right with candidate on the left
            nearest = peaks[idx]
            left_idx = np.clip(idx - 1, 0, len(peaks) - 1)
            left_candidate = peaks[left_idx]
            use_left = np.abs(all_bins - left_candidate) < np.abs(all_bins - nearest)
            nearest = np.where(use_left, left_candidate, nearest)

            # Vectorized phase propagation
            offset = all_bins - nearest
            out_phase[ch, t, :] = input_phase[ch, t, nearest] + offset * (
                phase_scale * nearest
            )

    out_data = (input_mag * np.exp(1j * out_phase)).astype(np.complex64)
    return Spectrogram(
        data=out_data,
        window_size=spec.window_size,
        hop_size=spec.hop_size,
        fft_size=spec.fft_size,
        sample_rate=spec.sample_rate,
        original_frames=spec.original_frames,
    )


def spectral_freeze(
    spec: Spectrogram,
    frame_index: int = 0,
    num_frames: int | None = None,
) -> Spectrogram:
    """Repeat a single STFT frame to produce a static ("frozen") texture.

    Parameters
    ----------
    spec : Spectrogram
        Input spectrogram.
    frame_index : int
        Index of the frame to freeze.  Negative indices are supported.
    num_frames : int or None
        Number of output STFT frames.  Defaults to ``spec.num_frames``.

    Returns
    -------
    Spectrogram
        Spectrogram with the chosen frame repeated *num_frames* times.

    Raises
    ------
    IndexError
        If *frame_index* is out of range.
    """
    if num_frames is None:
        num_frames = spec.num_frames
    if frame_index < -spec.num_frames or frame_index >= spec.num_frames:
        raise IndexError(
            f"frame_index {frame_index} out of range for {spec.num_frames} frames"
        )
    frame = spec.data[:, frame_index, :]  # [n_ch, n_bins]
    data = np.tile(frame[:, None, :], (1, num_frames, 1))
    original_frames = (num_frames - 1) * spec.hop_size + spec.window_size
    return Spectrogram(
        data=data.astype(np.complex64),
        window_size=spec.window_size,
        hop_size=spec.hop_size,
        fft_size=spec.fft_size,
        sample_rate=spec.sample_rate,
        original_frames=original_frames,
    )


def spectral_morph(
    spec_a: Spectrogram,
    spec_b: Spectrogram,
    mix: float | np.ndarray = 0.5,
) -> Spectrogram:
    """Interpolate between two spectrograms in the polar domain.

    Magnitudes are interpolated linearly; phases use shortest-arc circular
    interpolation, avoiding the cancellation artefacts of complex-valued
    lerp.

    Parameters
    ----------
    spec_a, spec_b : Spectrogram
        Input spectrograms.  Must share ``fft_size``, ``window_size``,
        ``hop_size``, and channel count.  If frame counts differ the
        shorter length is used.
    mix : float or np.ndarray
        Blend factor.  ``0.0`` returns *spec_a*, ``1.0`` returns *spec_b*.
        May be a scalar or an array broadcastable to
        ``[channels, num_frames, bins]`` for time-varying morphing.

    Returns
    -------
    Spectrogram

    Raises
    ------
    ValueError
        If the two spectrograms have incompatible parameters.
    """
    if spec_a.fft_size != spec_b.fft_size:
        raise ValueError(f"fft_size mismatch: {spec_a.fft_size} vs {spec_b.fft_size}")
    if spec_a.window_size != spec_b.window_size:
        raise ValueError(
            f"window_size mismatch: {spec_a.window_size} vs {spec_b.window_size}"
        )
    if spec_a.hop_size != spec_b.hop_size:
        raise ValueError(f"hop_size mismatch: {spec_a.hop_size} vs {spec_b.hop_size}")
    if spec_a.channels != spec_b.channels:
        raise ValueError(
            f"channel count mismatch: {spec_a.channels} vs {spec_b.channels}"
        )

    n_frames = min(spec_a.num_frames, spec_b.num_frames)
    data_a = spec_a.data[:, :n_frames, :]
    data_b = spec_b.data[:, :n_frames, :]

    mag_a = np.abs(data_a)
    mag_b = np.abs(data_b)
    phase_a = np.angle(data_a)
    phase_b = np.angle(data_b)

    mix = np.asarray(mix, dtype=np.float32)

    # Magnitude: linear interpolation
    mag = (1.0 - mix) * mag_a + mix * mag_b

    # Phase: shortest-arc circular interpolation
    phase_diff = phase_b - phase_a
    phase_diff = phase_diff - 2.0 * np.pi * np.round(phase_diff / (2.0 * np.pi))
    ph = phase_a + mix * phase_diff

    data = (mag * np.exp(1j * ph)).astype(np.complex64)
    original_frames = min(spec_a.original_frames, spec_b.original_frames)
    return Spectrogram(
        data=data,
        window_size=spec_a.window_size,
        hop_size=spec_a.hop_size,
        fft_size=spec_a.fft_size,
        sample_rate=spec_a.sample_rate,
        original_frames=original_frames,
    )


def pitch_shift_spectral(
    buf: AudioBuffer,
    semitones: float,
    window_size: int = 2048,
    hop_size: int | None = None,
) -> AudioBuffer:
    """Pitch-shift audio via phase vocoder + resampling.

    Combines :func:`time_stretch` with linear resampling so that pitch
    changes without altering duration.

    Parameters
    ----------
    buf : AudioBuffer
        Input audio.
    semitones : float
        Pitch shift in semitones.  Positive = higher, negative = lower.
    window_size : int
        STFT analysis window size.
    hop_size : int or None
        STFT hop size.  Defaults to ``window_size // 4``.

    Returns
    -------
    AudioBuffer
        Pitch-shifted audio with the same duration and sample rate.
    """
    if semitones == 0.0:
        return AudioBuffer(
            buf.data.copy(),
            sample_rate=buf.sample_rate,
            channel_layout=buf.channel_layout,
            label=buf.label,
        )

    alpha = 2.0 ** (semitones / 12.0)
    # Time-stretch to compensate for the resampling that follows
    stretch_rate = 1.0 / alpha

    spec = stft(buf, window_size=window_size, hop_size=hop_size)
    stretched = time_stretch(spec, stretch_rate)
    audio = istft(stretched)

    # Resample to original length using linear interpolation
    target_frames = buf.frames
    if audio.frames == target_frames:
        resampled = audio.data
    else:
        old_x = np.linspace(0.0, 1.0, audio.frames, dtype=np.float64)
        new_x = np.linspace(0.0, 1.0, target_frames, dtype=np.float64)
        resampled = np.empty((audio.channels, target_frames), dtype=np.float32)
        for ch in range(audio.channels):
            resampled[ch] = np.interp(new_x, old_x, audio.data[ch]).astype(np.float32)

    return AudioBuffer(
        resampled,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


def spectral_denoise(
    spec: Spectrogram,
    noise_frames: int = 10,
    reduction_db: float = -20.0,
    smoothing: int = 0,
) -> Spectrogram:
    """Spectral noise reduction using a profile estimated from leading frames.

    Computes the mean magnitude of the first *noise_frames* STFT frames
    per bin, then attenuates bins whose magnitude falls at or below that
    noise floor.  The leading frames should ideally contain only noise.

    Parameters
    ----------
    spec : Spectrogram
        Input spectrogram.
    noise_frames : int
        Number of leading STFT frames used to build the noise profile.
    reduction_db : float
        Attenuation in dB applied to bins at or below the noise floor.
        More negative = more aggressive reduction.
    smoothing : int
        If > 0, apply a moving-average of this width (in bins) to the
        noise profile, reducing musical-noise artefacts.

    Returns
    -------
    Spectrogram
        Denoised spectrogram.

    Raises
    ------
    ValueError
        If *noise_frames* < 1 or exceeds the number of available frames.
    """
    if noise_frames < 1:
        raise ValueError(f"noise_frames must be >= 1, got {noise_frames}")
    if noise_frames > spec.num_frames:
        raise ValueError(
            f"noise_frames ({noise_frames}) exceeds available frames "
            f"({spec.num_frames})"
        )

    # Mean magnitude across noise frames, per channel per bin: [ch, bins]
    noise_mag = np.mean(np.abs(spec.data[:, :noise_frames, :]), axis=1)

    if smoothing > 0:
        kernel = np.ones(smoothing, dtype=np.float32) / smoothing
        smoothed = np.empty_like(noise_mag)
        for ch in range(spec.channels):
            smoothed[ch] = np.convolve(noise_mag[ch], kernel, mode="same")
        noise_mag = smoothed

    # Gate: pass bins above noise floor, attenuate the rest
    sig_mag = np.abs(spec.data)
    noise_threshold = noise_mag[:, None, :]  # broadcast [ch, 1, bins]
    attenuation = 10.0 ** (reduction_db / 20.0)
    mask = np.where(sig_mag > noise_threshold, 1.0, attenuation).astype(np.float32)
    return apply_mask(spec, mask)


# ---------------------------------------------------------------------------
# EQ matching
# ---------------------------------------------------------------------------


def eq_match(
    buf: AudioBuffer,
    target: AudioBuffer,
    window_size: int = 4096,
    smoothing: int = 0,
) -> AudioBuffer:
    """Match the spectral envelope of *buf* to *target*.

    Parameters
    ----------
    buf : AudioBuffer
        Source audio to be adjusted.
    target : AudioBuffer
        Reference audio whose spectral envelope is matched.
    window_size : int
        STFT window size.
    smoothing : int
        If > 0, apply a moving-average of this width (in bins) to the
        correction curve.

    Raises
    ------
    ValueError
        If sample rates or channel counts differ.
    """
    if buf.sample_rate != target.sample_rate:
        raise ValueError(
            f"Sample rate mismatch: buf={buf.sample_rate}, target={target.sample_rate}"
        )
    if buf.channels != target.channels:
        raise ValueError(
            f"Channel count mismatch: buf has {buf.channels}, target has "
            f"{target.channels}. Convert to matching layout first "
            f"(e.g. to_mono())."
        )

    src_spec = stft(buf, window_size=window_size)
    tgt_spec = stft(target, window_size=window_size)

    # Mean magnitude across all channels and frames -> [bins]
    src_avg = np.mean(np.abs(src_spec.data), axis=(0, 1))
    tgt_avg = np.mean(np.abs(tgt_spec.data), axis=(0, 1))

    eps = 1e-10
    correction = tgt_avg / (src_avg + eps)
    correction = np.clip(correction, 0.0, 100.0).astype(np.float32)

    if smoothing > 0:
        kernel = np.ones(smoothing, dtype=np.float32) / smoothing
        correction = np.convolve(correction, kernel, mode="same").astype(np.float32)

    # Apply correction as a 1D mask [bins] -- broadcasts across channels/frames
    corrected = apply_mask(src_spec, correction)
    result = istft(corrected)

    # Trim to original length
    if result.frames > buf.frames:
        result = AudioBuffer(
            result.data[:, : buf.frames],
            sample_rate=buf.sample_rate,
            channel_layout=buf.channel_layout,
            label=buf.label,
        )

    return AudioBuffer(
        result.data,
        sample_rate=buf.sample_rate,
        channel_layout=buf.channel_layout,
        label=buf.label,
    )


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


# ---------------------------------------------------------------------------
# DaisySP Effects
# ---------------------------------------------------------------------------


def autowah(
    buf: AudioBuffer,
    wah: float = 0.5,
    dry_wet: float = 1.0,
    level: float = 0.5,
) -> AudioBuffer:
    """Apply auto-wah effect per channel."""

    def _process(x):
        aw = _dsy_fx.Autowah()
        aw.init(buf.sample_rate)
        aw.set_wah(wah)
        aw.set_dry_wet(dry_wet)
        aw.set_level(level)
        return aw.process(x)

    return _process_per_channel(buf, _process)


def chorus(
    buf: AudioBuffer,
    lfo_freq: float = 0.3,
    lfo_depth: float = 0.5,
    delay_ms: float = 5.0,
    feedback: float = 0.2,
) -> AudioBuffer:
    """Apply chorus effect.

    Mono input produces stereo output via process_stereo.
    Multi-channel input is processed per-channel (mono chorus).
    """
    if buf.channels == 1:
        ch = _dsy_fx.Chorus()
        ch.init(buf.sample_rate)
        ch.set_lfo_freq(lfo_freq)
        ch.set_lfo_depth(lfo_depth)
        ch.set_delay_ms(delay_ms)
        ch.set_feedback(feedback)
        stereo = ch.process_stereo(buf.ensure_1d(0))
        return AudioBuffer(
            stereo,
            sample_rate=buf.sample_rate,
            channel_layout="stereo",
            label=buf.label,
        )

    def _process(x):
        ch = _dsy_fx.Chorus()
        ch.init(buf.sample_rate)
        ch.set_lfo_freq(lfo_freq)
        ch.set_lfo_depth(lfo_depth)
        ch.set_delay_ms(delay_ms)
        ch.set_feedback(feedback)
        return ch.process(x)

    return _process_per_channel(buf, _process)


def decimator(
    buf: AudioBuffer,
    downsample_factor: float = 0.5,
    bitcrush_factor: float = 0.5,
    bits_to_crush: int = 8,
    smooth: bool = False,
) -> AudioBuffer:
    """Apply decimator (bitcrushing / downsampling) per channel."""

    def _process(x):
        d = _dsy_fx.Decimator()
        d.init()
        d.set_downsample_factor(downsample_factor)
        d.set_bitcrush_factor(bitcrush_factor)
        d.set_bits_to_crush(bits_to_crush)
        d.set_smooth_crushing(smooth)
        return d.process(x)

    return _process_per_channel(buf, _process)


def flanger(
    buf: AudioBuffer,
    lfo_freq: float = 0.2,
    lfo_depth: float = 0.5,
    feedback: float = 0.3,
    delay_ms: float = 1.0,
) -> AudioBuffer:
    """Apply flanger effect per channel."""

    def _process(x):
        f = _dsy_fx.Flanger()
        f.init(buf.sample_rate)
        f.set_lfo_freq(lfo_freq)
        f.set_lfo_depth(lfo_depth)
        f.set_feedback(feedback)
        f.set_delay_ms(delay_ms)
        return f.process(x)

    return _process_per_channel(buf, _process)


def overdrive(buf: AudioBuffer, drive: float = 0.5) -> AudioBuffer:
    """Apply overdrive distortion per channel."""

    def _process(x):
        od = _dsy_fx.Overdrive()
        od.init()
        od.set_drive(drive)
        return od.process(x)

    return _process_per_channel(buf, _process)


def phaser(
    buf: AudioBuffer,
    lfo_freq: float = 0.3,
    lfo_depth: float = 0.5,
    freq: float = 1000.0,
    feedback: float = 0.5,
    poles: int = 4,
) -> AudioBuffer:
    """Apply phaser effect per channel."""

    def _process(x):
        p = _dsy_fx.Phaser()
        p.init(buf.sample_rate)
        p.set_lfo_freq(lfo_freq)
        p.set_lfo_depth(lfo_depth)
        p.set_freq(freq)
        p.set_feedback(feedback)
        p.set_poles(poles)
        return p.process(x)

    return _process_per_channel(buf, _process)


def pitch_shift(
    buf: AudioBuffer,
    semitones: float = 0.0,
    del_size: int = 256,
    fun: float = 0.0,
) -> AudioBuffer:
    """Apply pitch shifting per channel."""

    def _process(x):
        ps = _dsy_fx.PitchShifter()
        ps.init(buf.sample_rate)
        ps.set_transposition(semitones)
        ps.set_del_size(del_size)
        ps.set_fun(fun)
        return ps.process(x)

    return _process_per_channel(buf, _process)


def sample_rate_reduce(buf: AudioBuffer, freq: float = 0.5) -> AudioBuffer:
    """Apply sample-rate reduction per channel.

    Parameters
    ----------
    freq : float
        Normalized frequency 0-1 controlling the reduction amount.
    """

    def _process(x):
        srr = _dsy_fx.SampleRateReducer()
        srr.init()
        srr.set_freq(freq)
        return srr.process(x)

    return _process_per_channel(buf, _process)


def tremolo(
    buf: AudioBuffer,
    freq: float = 5.0,
    depth: float = 0.5,
    waveform: int = 0,
) -> AudioBuffer:
    """Apply tremolo effect per channel."""

    def _process(x):
        t = _dsy_fx.Tremolo()
        t.init(buf.sample_rate)
        t.set_freq(freq)
        t.set_depth(depth)
        t.set_waveform(waveform)
        return t.process(x)

    return _process_per_channel(buf, _process)


def wavefold(
    buf: AudioBuffer,
    gain: float = 1.0,
    offset: float = 0.0,
) -> AudioBuffer:
    """Apply wavefolding per channel."""

    def _process(x):
        wf = _dsy_fx.Wavefolder()
        wf.init()
        wf.set_gain(gain)
        wf.set_offset(offset)
        return wf.process(x)

    return _process_per_channel(buf, _process)


def bitcrush(
    buf: AudioBuffer,
    bit_depth: int = 8,
    crush_rate: float | None = None,
) -> AudioBuffer:
    """Apply bitcrushing per channel.

    Parameters
    ----------
    crush_rate : float or None
        Sample-and-hold rate. Defaults to sample_rate / 4 if None.
    """
    rate = crush_rate if crush_rate is not None else buf.sample_rate / 4.0

    def _process(x):
        bc = _dsy_fx.Bitcrush()
        bc.init(buf.sample_rate)
        bc.set_bit_depth(bit_depth)
        bc.set_crush_rate(rate)
        return bc.process(x)

    return _process_per_channel(buf, _process)


def fold(buf: AudioBuffer, increment: float = 1.0) -> AudioBuffer:
    """Apply fold distortion per channel."""

    def _process(x):
        f = _dsy_fx.Fold()
        f.init()
        f.set_increment(increment)
        return f.process(x)

    return _process_per_channel(buf, _process)


def reverb_sc(
    buf: AudioBuffer,
    feedback: float = 0.7,
    lp_freq: float = 10000.0,
) -> AudioBuffer:
    """Apply ReverbSc stereo reverb.

    Mono input is duplicated to stereo. Stereo input is passed through.
    3+ channels raises ValueError.
    """
    if buf.channels > 2:
        raise ValueError(
            f"reverb_sc requires mono or stereo input, got {buf.channels} channels"
        )
    rv = _dsy_fx.ReverbSc()
    rv.init(buf.sample_rate)
    rv.set_feedback(feedback)
    rv.set_lp_freq(lp_freq)
    if buf.channels == 1:
        stereo_in = np.vstack([buf.data[0], buf.data[0]])
    else:
        stereo_in = buf.data
    out = rv.process(stereo_in)
    return AudioBuffer(
        out,
        sample_rate=buf.sample_rate,
        channel_layout="stereo",
        label=buf.label,
    )


def dc_block(buf: AudioBuffer) -> AudioBuffer:
    """Remove DC offset per channel using DaisySP DcBlock."""

    def _process(x):
        dc = _dsy_util.DcBlock()
        dc.init(buf.sample_rate)
        return dc.process(x)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# DaisySP Filters
# ---------------------------------------------------------------------------


def _make_svf(buf, freq_hz, resonance, drive, process_method):
    """Internal helper for SVF filter variants."""

    def _process(x):
        svf = _dsy_filt.Svf()
        svf.init(buf.sample_rate)
        svf.set_freq(freq_hz)
        svf.set_res(resonance)
        svf.set_drive(drive)
        return getattr(svf, process_method)(x)

    return _process_per_channel(buf, _process)


def svf_lowpass(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    resonance: float = 0.0,
    drive: float = 0.0,
) -> AudioBuffer:
    """State-variable filter lowpass."""
    return _make_svf(buf, freq_hz, resonance, drive, "process_low")


def svf_highpass(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    resonance: float = 0.0,
    drive: float = 0.0,
) -> AudioBuffer:
    """State-variable filter highpass."""
    return _make_svf(buf, freq_hz, resonance, drive, "process_high")


def svf_bandpass(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    resonance: float = 0.0,
    drive: float = 0.0,
) -> AudioBuffer:
    """State-variable filter bandpass."""
    return _make_svf(buf, freq_hz, resonance, drive, "process_band")


def svf_notch(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    resonance: float = 0.0,
    drive: float = 0.0,
) -> AudioBuffer:
    """State-variable filter notch."""
    return _make_svf(buf, freq_hz, resonance, drive, "process_notch")


def svf_peak(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    resonance: float = 0.0,
    drive: float = 0.0,
) -> AudioBuffer:
    """State-variable filter peak."""
    return _make_svf(buf, freq_hz, resonance, drive, "process_peak")


def ladder_filter(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    resonance: float = 0.0,
    mode: str = "lp24",
    drive: float = 0.0,
) -> AudioBuffer:
    """Ladder filter with selectable mode.

    Parameters
    ----------
    mode : str
        One of "lp24", "lp12", "bp24", "bp12", "hp24", "hp12".
    """
    mode_key = mode.lower()
    if mode_key not in _LADDER_MODE_MAP:
        raise ValueError(
            f"Unknown ladder mode {mode!r}, valid: {list(_LADDER_MODE_MAP.keys())}"
        )
    mode_val = _LADDER_MODE_MAP[mode_key]

    def _process(x):
        lf = _dsy_filt.LadderFilter()
        lf.init(buf.sample_rate)
        lf.set_freq(freq_hz)
        lf.set_res(resonance)
        lf.set_filter_mode(mode_val)
        lf.set_input_drive(drive)
        return lf.process(x)

    return _process_per_channel(buf, _process)


def moog_ladder(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    resonance: float = 0.0,
) -> AudioBuffer:
    """Moog-style ladder lowpass filter."""

    def _process(x):
        ml = _dsy_filt.MoogLadder()
        ml.init(buf.sample_rate)
        ml.set_freq(freq_hz)
        ml.set_res(resonance)
        return ml.process(x)

    return _process_per_channel(buf, _process)


def tone_lowpass(buf: AudioBuffer, freq_hz: float = 1000.0) -> AudioBuffer:
    """One-pole lowpass filter (Tone)."""

    def _process(x):
        t = _dsy_filt.Tone()
        t.init(buf.sample_rate)
        t.set_freq(freq_hz)
        return t.process(x)

    return _process_per_channel(buf, _process)


def tone_highpass(buf: AudioBuffer, freq_hz: float = 1000.0) -> AudioBuffer:
    """One-pole highpass filter (ATone)."""

    def _process(x):
        at = _dsy_filt.ATone()
        at.init(buf.sample_rate)
        at.set_freq(freq_hz)
        return at.process(x)

    return _process_per_channel(buf, _process)


def modal_bandpass(
    buf: AudioBuffer,
    freq_hz: float = 1000.0,
    q: float = 500.0,
) -> AudioBuffer:
    """Modal resonator bandpass filter."""

    def _process(x):
        m = _dsy_filt.Mode()
        m.init(buf.sample_rate)
        m.set_freq(freq_hz)
        m.set_q(q)
        return m.process(x)

    return _process_per_channel(buf, _process)


def comb_filter(
    buf: AudioBuffer,
    freq_hz: float = 500.0,
    rev_time: float = 0.5,
    max_size: int = 4096,
) -> AudioBuffer:
    """Comb filter."""

    def _process(x):
        c = _dsy_filt.Comb(buf.sample_rate, max_size)
        c.set_freq(freq_hz)
        c.set_rev_time(rev_time)
        return c.process(x)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# DaisySP Dynamics
# ---------------------------------------------------------------------------


def compress(
    buf: AudioBuffer,
    ratio: float = 4.0,
    threshold: float = -20.0,
    attack: float = 0.01,
    release: float = 0.1,
    makeup: float = 0.0,
    auto_makeup: bool = False,
) -> AudioBuffer:
    """Apply compression per channel."""

    def _process(x):
        c = _dsy_dyn.Compressor()
        c.init(buf.sample_rate)
        c.set_ratio(ratio)
        c.set_threshold(threshold)
        c.set_attack(attack)
        c.set_release(release)
        c.set_makeup(makeup)
        c.auto_makeup(auto_makeup)
        return c.process(x)

    return _process_per_channel(buf, _process)


def limit(buf: AudioBuffer, pre_gain: float = 1.0) -> AudioBuffer:
    """Apply limiter per channel."""

    def _process(x):
        lm = _dsy_dyn.Limiter()
        lm.init()
        return lm.process(x, pre_gain)

    return _process_per_channel(buf, _process)


# ---------------------------------------------------------------------------
# Loudness metering (ITU-R BS.1770-4)
# ---------------------------------------------------------------------------


def _k_weight(x: np.ndarray, sample_rate: float) -> np.ndarray:
    """Apply two-stage K-weighting to a 1D signal via C++ Biquad.

    Stage 1: high shelf ~+4 dB at 1681 Hz (head/ear acoustic model).
    Stage 2: highpass at 38 Hz (revised low-frequency B-weighting).
    """
    freq_pre = 1681.0 / sample_rate
    bq_pre = filters.Biquad()
    bq_pre.high_shelf_db(freq_pre, 4.0)
    stage1 = bq_pre.process(x)
    freq_hp = 38.0 / sample_rate
    bq_hp = filters.Biquad()
    bq_hp.highpass(freq_hp)
    return bq_hp.process(stage1)


def loudness_lufs(buf: AudioBuffer) -> float:
    """Measure integrated loudness per ITU-R BS.1770-4.

    Returns
    -------
    float
        Integrated loudness in LUFS. Returns ``-inf`` for silence or
        signals shorter than 400 ms.
    """
    sr = buf.sample_rate
    block_samples = int(sr * 0.4)  # 400 ms
    hop_samples = int(sr * 0.1)  # 100 ms (75% overlap)

    if buf.frames < block_samples:
        return float("-inf")

    # K-weight each channel
    weighted = []
    for ch in range(buf.channels):
        weighted.append(_k_weight(buf.ensure_1d(ch), sr))

    # Channel weights per ITU-R BS.1770-4
    # 5.1 (6 ch): L=1.0, R=1.0, C=1.0, LFE=0.0, Ls=1.41, Rs=1.41
    # All other layouts: uniform 1.0
    if buf.channels == 6:
        ch_weights = np.array([1.0, 1.0, 1.0, 0.0, 1.41, 1.41], dtype=np.float64)
    else:
        ch_weights = np.ones(buf.channels, dtype=np.float64)

    # Compute per-block loudness
    n_blocks = (buf.frames - block_samples) // hop_samples + 1
    block_power = np.zeros(n_blocks, dtype=np.float64)

    for i in range(n_blocks):
        start = i * hop_samples
        end = start + block_samples
        power = 0.0
        for ch in range(buf.channels):
            segment = weighted[ch][start:end].astype(np.float64)
            power += ch_weights[ch] * np.mean(segment**2)
        block_power[i] = power

    # Convert to LUFS
    eps = 1e-20
    block_lufs = -0.691 + 10.0 * np.log10(block_power + eps)

    # Absolute gate: -70 LUFS
    abs_gate_mask = block_lufs >= -70.0
    if not np.any(abs_gate_mask):
        return float("-inf")

    # Relative gate: mean of surviving blocks - 10 dB
    mean_power_abs = np.mean(block_power[abs_gate_mask])
    rel_gate_lufs = -0.691 + 10.0 * np.log10(mean_power_abs + eps) - 10.0
    rel_gate_mask = abs_gate_mask & (block_lufs >= rel_gate_lufs)

    if not np.any(rel_gate_mask):
        return float("-inf")

    # Integrated loudness
    mean_power = np.mean(block_power[rel_gate_mask])
    return float(-0.691 + 10.0 * np.log10(mean_power + eps))


def normalize_lufs(
    buf: AudioBuffer,
    target_lufs: float = -14.0,
) -> AudioBuffer:
    """Normalize loudness to *target_lufs*.

    Parameters
    ----------
    target_lufs : float
        Target integrated loudness in LUFS.

    Raises
    ------
    ValueError
        If the input is silent or too short to measure.
    """
    current = loudness_lufs(buf)
    if np.isinf(current):
        raise ValueError("Cannot normalize: input is silent or too short to measure")
    delta = target_lufs - current
    return buf.gain_db(delta)


# ---------------------------------------------------------------------------
# DaisySP Oscillators
# ---------------------------------------------------------------------------


def oscillator(
    frames: int,
    freq: float = 440.0,
    amp: float = 1.0,
    waveform: int | str = "sine",
    pw: float = 0.5,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a waveform using DaisySP Oscillator.

    Parameters
    ----------
    waveform : int or str
        Waveform constant or name: "sine", "tri", "saw", "ramp", "square",
        "polyblep_tri", "polyblep_saw", "polyblep_square".
    """
    wf = _resolve_waveform(waveform, _WAVEFORM_MAP)
    osc = _dsy_osc.Oscillator()
    osc.init(sample_rate)
    osc.set_freq(freq)
    osc.set_amp(amp)
    osc.set_waveform(wf)
    osc.set_pw(pw)
    data = osc.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def fm2(
    frames: int,
    freq: float = 440.0,
    ratio: float = 2.0,
    index: float = 1.0,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate 2-operator FM synthesis."""
    fm = _dsy_osc.Fm2()
    fm.init(sample_rate)
    fm.set_frequency(freq)
    fm.set_ratio(ratio)
    fm.set_index(index)
    data = fm.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def formant_oscillator(
    frames: int,
    carrier_freq: float = 440.0,
    formant_freq: float = 1000.0,
    phase_shift: float = 0.0,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate formant oscillator signal."""
    fo = _dsy_osc.FormantOscillator()
    fo.init(sample_rate)
    fo.set_carrier_freq(carrier_freq)
    fo.set_formant_freq(formant_freq)
    fo.set_phase_shift(phase_shift)
    data = fo.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def bl_oscillator(
    frames: int,
    freq: float = 440.0,
    amp: float = 1.0,
    waveform: int | str = "saw",
    pw: float = 0.5,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a band-limited waveform using DaisySP BlOsc.

    Parameters
    ----------
    waveform : int or str
        Waveform constant or name: "triangle"/"tri", "saw", "square", "off".
    """
    wf = _resolve_waveform(waveform, _BLOSC_WAVEFORM_MAP)
    osc = _dsy_osc.BlOsc()
    osc.init(sample_rate)
    osc.set_freq(freq)
    osc.set_amp(amp)
    osc.set_waveform(wf)
    osc.set_pw(pw)
    data = osc.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# DaisySP Noise
# ---------------------------------------------------------------------------


def white_noise(
    frames: int,
    amp: float = 1.0,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate white noise."""
    wn = _dsy_noise.WhiteNoise()
    wn.init()
    wn.set_amp(amp)
    data = wn.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def clocked_noise(
    frames: int,
    freq: float = 1000.0,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate clocked (sample-and-hold) noise."""
    cn = _dsy_noise.ClockedNoise()
    cn.init(sample_rate)
    cn.set_freq(freq)
    data = cn.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def dust(
    frames: int,
    density: float = 1.0,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate Dust (random impulses at given density)."""
    d = _dsy_noise.Dust()
    d.init()
    d.set_density(density)
    data = d.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# DaisySP Drums
# ---------------------------------------------------------------------------


def analog_bass_drum(
    frames: int,
    freq: float = 60.0,
    tone: float = 0.5,
    decay: float = 0.5,
    accent: float = 0.5,
    sustain: bool = False,
    attack_fm: float = 0.5,
    self_fm: float = 0.5,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate an analog bass drum hit (triggered at sample 0)."""
    d = _dsy_drums.AnalogBassDrum()
    d.init(sample_rate)
    d.set_freq(freq)
    d.set_tone(tone)
    d.set_decay(decay)
    d.set_accent(accent)
    d.set_sustain(sustain)
    d.set_attack_fm_amount(attack_fm)
    d.set_self_fm_amount(self_fm)
    d.trig()
    data = d.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def analog_snare_drum(
    frames: int,
    freq: float = 200.0,
    tone: float = 0.5,
    decay: float = 0.5,
    snappy: float = 0.5,
    accent: float = 0.5,
    sustain: bool = False,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate an analog snare drum hit (triggered at sample 0)."""
    d = _dsy_drums.AnalogSnareDrum()
    d.init(sample_rate)
    d.set_freq(freq)
    d.set_tone(tone)
    d.set_decay(decay)
    d.set_snappy(snappy)
    d.set_accent(accent)
    d.set_sustain(sustain)
    d.trig()
    data = d.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def hihat(
    frames: int,
    freq: float = 3000.0,
    tone: float = 0.5,
    decay: float = 0.3,
    noisiness: float = 0.8,
    accent: float = 0.5,
    sustain: bool = False,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a hi-hat hit (triggered at sample 0)."""
    d = _dsy_drums.HiHat()
    d.init(sample_rate)
    d.set_freq(freq)
    d.set_tone(tone)
    d.set_decay(decay)
    d.set_noisiness(noisiness)
    d.set_accent(accent)
    d.set_sustain(sustain)
    d.trig()
    data = d.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def synthetic_bass_drum(
    frames: int,
    freq: float = 60.0,
    tone: float = 0.5,
    decay: float = 0.5,
    dirtiness: float = 0.3,
    fm_env_amount: float = 0.5,
    fm_env_decay: float = 0.3,
    accent: float = 0.5,
    sustain: bool = False,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a synthetic bass drum hit (triggered at sample 0)."""
    d = _dsy_drums.SyntheticBassDrum()
    d.init(sample_rate)
    d.set_freq(freq)
    d.set_tone(tone)
    d.set_decay(decay)
    d.set_dirtiness(dirtiness)
    d.set_fm_envelope_amount(fm_env_amount)
    d.set_fm_envelope_decay(fm_env_decay)
    d.set_accent(accent)
    d.set_sustain(sustain)
    d.trig()
    data = d.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def synthetic_snare_drum(
    frames: int,
    freq: float = 200.0,
    decay: float = 0.5,
    snappy: float = 0.5,
    fm_amount: float = 0.3,
    accent: float = 0.5,
    sustain: bool = False,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a synthetic snare drum hit (triggered at sample 0)."""
    d = _dsy_drums.SyntheticSnareDrum()
    d.init(sample_rate)
    d.set_freq(freq)
    d.set_decay(decay)
    d.set_snappy(snappy)
    d.set_fm_amount(fm_amount)
    d.set_accent(accent)
    d.set_sustain(sustain)
    d.trig()
    data = d.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


# ---------------------------------------------------------------------------
# DaisySP Physical Modeling
# ---------------------------------------------------------------------------


def karplus_strong(
    buf: AudioBuffer,
    freq_hz: float = 440.0,
    brightness: float = 0.5,
    damping: float = 0.5,
    non_linearity: float = 0.0,
) -> AudioBuffer:
    """Karplus-Strong string model (excitation input, per channel)."""

    def _process(x):
        s = _dsy_pm.String()
        s.init(buf.sample_rate)
        s.set_freq(freq_hz)
        s.set_brightness(brightness)
        s.set_damping(damping)
        s.set_non_linearity(non_linearity)
        return s.process(x)

    return _process_per_channel(buf, _process)


def modal_voice(
    frames: int,
    freq: float = 440.0,
    accent: float = 0.5,
    structure: float = 0.5,
    brightness: float = 0.5,
    damping: float = 0.5,
    sustain: bool = False,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a modal voice hit (triggered at sample 0)."""
    mv = _dsy_pm.ModalVoice()
    mv.init(sample_rate)
    mv.set_freq(freq)
    mv.set_accent(accent)
    mv.set_structure(structure)
    mv.set_brightness(brightness)
    mv.set_damping(damping)
    mv.set_sustain(sustain)
    mv.trig()
    data = mv.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def string_voice(
    frames: int,
    freq: float = 440.0,
    accent: float = 0.5,
    structure: float = 0.5,
    brightness: float = 0.5,
    damping: float = 0.5,
    sustain: bool = False,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a string voice hit (triggered at sample 0)."""
    sv = _dsy_pm.StringVoice()
    sv.init(sample_rate)
    sv.set_freq(freq)
    sv.set_accent(accent)
    sv.set_structure(structure)
    sv.set_brightness(brightness)
    sv.set_damping(damping)
    sv.set_sustain(sustain)
    sv.trig()
    data = sv.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def pluck(
    frames: int,
    freq: float = 440.0,
    amp: float = 0.8,
    decay: float = 0.95,
    damp: float = 0.9,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a plucked string sound (triggered at sample 0)."""
    npt = max(256, int(sample_rate / freq) + 1)
    p = _dsy_pm.Pluck(sample_rate, npt)
    p.set_freq(freq)
    p.set_amp(amp)
    p.set_decay(decay)
    p.set_damp(damp)
    data = p.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)


def drip(
    frames: int,
    dettack: float = 0.01,
    sample_rate: float = 48000.0,
) -> AudioBuffer:
    """Generate a water-drip sound (triggered at sample 0)."""
    d = _dsy_pm.Drip()
    d.init(sample_rate, dettack)
    data = d.process(frames)
    return AudioBuffer(np.asarray(data).reshape(1, -1), sample_rate=sample_rate)

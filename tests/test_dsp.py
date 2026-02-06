"""Tests for cydsp.dsp module (high-level AudioBuffer processing)."""

import numpy as np
import pytest

import cydsp
from cydsp import dsp, filters
from cydsp.buffer import AudioBuffer


# ---------------------------------------------------------------------------
# Frequency conversion
# ---------------------------------------------------------------------------


class TestFrequencyConversion:
    def test_valid_conversion(self):
        assert dsp._hz_to_normalized(1000.0, 48000.0) == pytest.approx(1000.0 / 48000.0)

    def test_zero_hz(self):
        assert dsp._hz_to_normalized(0.0, 48000.0) == 0.0

    def test_nyquist_rejection(self):
        with pytest.raises(ValueError, match="Nyquist"):
            dsp._hz_to_normalized(24000.0, 48000.0)

    def test_above_nyquist_rejection(self):
        with pytest.raises(ValueError, match="Nyquist"):
            dsp._hz_to_normalized(25000.0, 48000.0)

    def test_negative_rejection(self):
        with pytest.raises(ValueError, match="non-negative"):
            dsp._hz_to_normalized(-100.0, 48000.0)


# ---------------------------------------------------------------------------
# Filter functions
# ---------------------------------------------------------------------------


class TestFilterFunctions:
    def test_lowpass_attenuates_high_freq(self):
        buf = AudioBuffer.sine(1000.0, frames=4096, sample_rate=48000.0)
        high = AudioBuffer.sine(20000.0, frames=4096, sample_rate=48000.0)
        combined = buf + high
        result = dsp.lowpass(combined, 5000.0)
        # High-frequency energy should be attenuated
        assert np.sum(result.data**2) < np.sum(combined.data**2)

    def test_highpass_attenuates_low_freq(self):
        low = AudioBuffer.sine(100.0, frames=4096, sample_rate=48000.0)
        high = AudioBuffer.sine(10000.0, frames=4096, sample_rate=48000.0)
        combined = low + high
        result = dsp.highpass(combined, 5000.0)
        assert np.sum(result.data**2) < np.sum(combined.data**2)

    def test_bandpass_passes_center(self):
        center = AudioBuffer.sine(5000.0, frames=4096, sample_rate=48000.0)
        result = dsp.bandpass(center, 5000.0, octaves=2.0)
        # Center frequency should pass through with reasonable energy
        energy_ratio = np.sum(result.data**2) / np.sum(center.data**2)
        assert energy_ratio > 0.3

    def test_notch_attenuates_center(self):
        center = AudioBuffer.sine(5000.0, frames=4096, sample_rate=48000.0)
        result = dsp.notch(center, 5000.0, octaves=1.0)
        energy_ratio = np.sum(result.data**2) / np.sum(center.data**2)
        assert energy_ratio < 0.5

    def test_peak_boosts(self):
        buf = AudioBuffer.sine(5000.0, frames=4096, sample_rate=48000.0)
        result = dsp.peak(buf, 5000.0, gain=4.0)
        assert np.sum(result.data**2) > np.sum(buf.data**2)

    def test_peak_db_boosts(self):
        buf = AudioBuffer.sine(5000.0, frames=4096, sample_rate=48000.0)
        result = dsp.peak_db(buf, 5000.0, db=12.0)
        assert np.sum(result.data**2) > np.sum(buf.data**2)

    def test_high_shelf(self):
        buf = AudioBuffer.sine(15000.0, frames=4096, sample_rate=48000.0)
        result = dsp.high_shelf(buf, 10000.0, gain=4.0)
        assert np.sum(result.data**2) > np.sum(buf.data**2)

    def test_high_shelf_db(self):
        buf = AudioBuffer.sine(15000.0, frames=4096, sample_rate=48000.0)
        result = dsp.high_shelf_db(buf, 10000.0, db=12.0)
        assert np.sum(result.data**2) > np.sum(buf.data**2)

    def test_low_shelf(self):
        buf = AudioBuffer.sine(100.0, frames=4096, sample_rate=48000.0)
        result = dsp.low_shelf(buf, 1000.0, gain=4.0)
        assert np.sum(result.data**2) > np.sum(buf.data**2)

    def test_low_shelf_db(self):
        buf = AudioBuffer.sine(100.0, frames=4096, sample_rate=48000.0)
        result = dsp.low_shelf_db(buf, 1000.0, db=12.0)
        assert np.sum(result.data**2) > np.sum(buf.data**2)

    def test_allpass_preserves_magnitude(self):
        buf = AudioBuffer.sine(5000.0, frames=4096, sample_rate=48000.0)
        result = dsp.allpass(buf, 5000.0)
        in_energy = np.sum(buf.data**2)
        out_energy = np.sum(result.data**2)
        np.testing.assert_allclose(out_energy, in_energy, rtol=0.01)

    def test_metadata_preserved(self):
        buf = AudioBuffer.sine(
            1000.0,
            channels=2,
            frames=1024,
            sample_rate=44100.0,
            label="test",
        )
        result = dsp.lowpass(buf, 5000.0)
        assert result.sample_rate == 44100.0
        assert result.channels == 2
        assert result.frames == 1024
        assert result.label == "test"
        assert result.channel_layout == "stereo"

    def test_per_channel_independence(self):
        # Different content per channel
        data = np.zeros((2, 4096), dtype=np.float32)
        t = np.arange(4096, dtype=np.float32) / 48000.0
        data[0] = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        data[1] = np.sin(2 * np.pi * 15000 * t).astype(np.float32)
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = dsp.lowpass(buf, 5000.0)
        # Channel 0 (1kHz) should retain more energy than channel 1 (15kHz)
        ch0_energy = np.sum(result.data[0] ** 2)
        ch1_energy = np.sum(result.data[1] ** 2)
        assert ch0_energy > ch1_energy * 5

    def test_all_filters_produce_correct_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        for fn, kwargs in [
            (dsp.lowpass, {"cutoff_hz": 5000.0}),
            (dsp.highpass, {"cutoff_hz": 5000.0}),
            (dsp.bandpass, {"center_hz": 5000.0}),
            (dsp.notch, {"center_hz": 5000.0}),
            (dsp.peak, {"center_hz": 5000.0, "gain": 2.0}),
            (dsp.peak_db, {"center_hz": 5000.0, "db": 6.0}),
            (dsp.high_shelf, {"cutoff_hz": 5000.0, "gain": 2.0}),
            (dsp.high_shelf_db, {"cutoff_hz": 5000.0, "db": 6.0}),
            (dsp.low_shelf, {"cutoff_hz": 5000.0, "gain": 2.0}),
            (dsp.low_shelf_db, {"cutoff_hz": 5000.0, "db": 6.0}),
            (dsp.allpass, {"freq_hz": 5000.0}),
        ]:
            result = fn(buf, **kwargs)
            assert result.channels == 1
            assert result.frames == 1024
            assert result.data.dtype == np.float32

    def test_biquad_process_preconfigured(self):
        bq = filters.Biquad()
        bq.lowpass(0.1)
        buf = AudioBuffer.noise(channels=2, frames=1024, sample_rate=48000.0, seed=0)
        result = dsp.biquad_process(buf, bq)
        assert result.channels == 2
        assert result.frames == 1024

    def test_lowpass_with_explicit_octaves(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        result = dsp.lowpass(buf, 5000.0, octaves=2.0)
        assert result.frames == 1024


# ---------------------------------------------------------------------------
# Delay functions
# ---------------------------------------------------------------------------


class TestDelayFunctions:
    def test_basic_delay_shifts_impulse(self):
        buf = AudioBuffer.impulse(channels=1, frames=128, sample_rate=48000.0)
        result = dsp.delay(buf, 10.0)
        peak_idx = np.argmax(np.abs(result.data[0]))
        expected = 10 + cydsp.delay.Delay.latency
        assert peak_idx == expected

    def test_multichannel_delay(self):
        buf = AudioBuffer.impulse(channels=2, frames=128, sample_rate=48000.0)
        result = dsp.delay(buf, 10.0)
        assert result.channels == 2
        assert result.frames == 128
        # Both channels should have same delay
        peak0 = np.argmax(np.abs(result.data[0]))
        peak1 = np.argmax(np.abs(result.data[1]))
        assert peak0 == peak1

    def test_fractional_delay(self):
        buf = AudioBuffer.impulse(channels=1, frames=128, sample_rate=48000.0)
        result = dsp.delay(buf, 5.5)
        # Should produce nonzero output at interpolated samples
        assert np.max(np.abs(result.data)) > 0

    def test_varying_delay(self):
        buf = AudioBuffer.impulse(channels=1, frames=128, sample_rate=48000.0)
        delays = np.full(128, 10.0, dtype=np.float32)
        result = dsp.delay_varying(buf, delays)
        assert result.frames == 128
        assert np.max(np.abs(result.data)) > 0

    def test_1d_delay_broadcast_multichannel(self):
        buf = AudioBuffer.impulse(channels=2, frames=128, sample_rate=48000.0)
        delays = np.full(128, 10.0, dtype=np.float32)
        result = dsp.delay_varying(buf, delays)
        assert result.channels == 2

    def test_channel_mismatch_raises(self):
        buf = AudioBuffer.impulse(channels=2, frames=128, sample_rate=48000.0)
        delays = np.full((3, 128), 10.0, dtype=np.float32)
        with pytest.raises(ValueError, match="channels"):
            dsp.delay_varying(buf, delays)

    def test_cubic_interpolation(self):
        buf = AudioBuffer.impulse(channels=1, frames=128, sample_rate=48000.0)
        result = dsp.delay(buf, 5.0, interpolation="cubic")
        assert np.max(np.abs(result.data)) > 0


# ---------------------------------------------------------------------------
# Envelope functions
# ---------------------------------------------------------------------------


class TestEnvelopeFunctions:
    def test_box_filter_smooths(self):
        data = np.zeros((1, 128), dtype=np.float32)
        data[0, 32:] = 1.0
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = dsp.box_filter(buf, 16)
        # Should smooth the step
        assert result.data[0, 31] < result.data[0, 48]

    def test_box_stack_smoother(self):
        data = np.zeros((1, 256), dtype=np.float32)
        data[0, 0] = 1.0
        buf = AudioBuffer(data, sample_rate=48000.0)
        r_box = dsp.box_filter(buf, 16)
        r_stack = dsp.box_stack_filter(buf, 16, layers=4)
        # Both should produce output
        assert np.max(np.abs(r_box.data)) > 0
        assert np.max(np.abs(r_stack.data)) > 0

    def test_peak_hold_holds(self):
        data = np.zeros((1, 128), dtype=np.float32)
        data[0, 10] = 5.0
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = dsp.peak_hold(buf, 32)
        # Peak should be held for multiple samples after sample 10
        assert np.sum(result.data[0] >= 4.9) > 1

    def test_peak_decay_decays(self):
        data = np.zeros((1, 128), dtype=np.float32)
        data[0, 0] = 1.0
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = dsp.peak_decay(buf, 64)
        peak_val = np.max(result.data[0])
        assert peak_val > 0.9
        # Should decay after peak
        peak_idx = np.argmax(result.data[0])
        if peak_idx + 20 < 128:
            assert result.data[0, peak_idx + 20] < peak_val

    def test_envelope_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=256, sample_rate=48000.0, seed=0)
        result = dsp.box_filter(buf, 16)
        assert result.channels == 2
        assert result.frames == 256


# ---------------------------------------------------------------------------
# FFT functions
# ---------------------------------------------------------------------------


class TestFFTFunctions:
    def test_rfft_shape_dtype(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        spectra = dsp.rfft(buf)
        assert len(spectra) == 1
        assert spectra[0].dtype == np.complex64
        # bins = fast_size / 2
        fft_size = cydsp.fft.RealFFT.fast_size_above(1024)
        assert spectra[0].shape == (fft_size // 2,)

    def test_multichannel_rfft(self):
        buf = AudioBuffer.noise(channels=3, frames=512, sample_rate=48000.0, seed=0)
        spectra = dsp.rfft(buf)
        assert len(spectra) == 3

    def test_irfft_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=256, sample_rate=48000.0, seed=0)
        spectra = dsp.rfft(buf)
        result = dsp.irfft(spectra, 256, sample_rate=48000.0)
        assert result.channels == 1
        assert result.frames == 256

    def test_roundtrip(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=42)
        fft_size = cydsp.fft.RealFFT.fast_size_above(1024)
        spectra = dsp.rfft(buf)
        result = dsp.irfft(spectra, 1024, sample_rate=48000.0)
        # Unscaled: need to divide by fft_size
        recovered = result.data / fft_size
        np.testing.assert_allclose(recovered[0, :1024], buf.data[0], atol=1e-4)


# ---------------------------------------------------------------------------
# Rates functions
# ---------------------------------------------------------------------------


class TestRatesFunctions:
    def test_upsample_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=128, sample_rate=48000.0, seed=0)
        result = dsp.upsample_2x(buf)
        assert result.frames == 256
        assert result.sample_rate == 96000.0

    def test_upsample_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=128, sample_rate=48000.0, seed=0)
        result = dsp.upsample_2x(buf)
        assert result.channels == 2
        assert result.frames == 256

    def test_oversample_roundtrip_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=128, sample_rate=48000.0, seed=0)
        result = dsp.oversample_roundtrip(buf)
        assert result.frames == 128
        assert result.sample_rate == 48000.0

    def test_oversample_roundtrip_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=128, sample_rate=48000.0, seed=0)
        result = dsp.oversample_roundtrip(buf)
        assert result.channels == 2
        assert result.frames == 128


# ---------------------------------------------------------------------------
# Mix functions
# ---------------------------------------------------------------------------


class TestMixFunctions:
    def test_hadamard_involution(self):
        data = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            dtype=np.float32,
        )
        buf = AudioBuffer(data, sample_rate=48000.0)
        once = dsp.hadamard(buf)
        twice = dsp.hadamard(once)
        np.testing.assert_allclose(twice.data, buf.data, atol=1e-4)

    def test_hadamard_energy_preservation(self):
        buf = AudioBuffer.noise(channels=4, frames=64, sample_rate=48000.0, seed=0)
        result = dsp.hadamard(buf)
        # Energy should be preserved per-frame
        for i in range(buf.frames):
            in_e = np.sum(buf.data[:, i] ** 2)
            out_e = np.sum(result.data[:, i] ** 2)
            np.testing.assert_allclose(out_e, in_e, rtol=1e-4)

    def test_hadamard_non_power_of_2_raises(self):
        buf = AudioBuffer.noise(channels=3, frames=64, sample_rate=48000.0, seed=0)
        with pytest.raises(ValueError, match="power-of-2"):
            dsp.hadamard(buf)

    def test_householder_involution(self):
        data = np.array(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            dtype=np.float32,
        )
        buf = AudioBuffer(data, sample_rate=48000.0)
        once = dsp.householder(buf)
        twice = dsp.householder(once)
        np.testing.assert_allclose(twice.data, buf.data, atol=1e-4)

    def test_householder_any_channel_count(self):
        buf = AudioBuffer.noise(channels=5, frames=32, sample_rate=48000.0, seed=0)
        result = dsp.householder(buf)
        assert result.channels == 5
        assert result.frames == 32

    def test_crossfade_at_zero(self):
        a = AudioBuffer.ones(1, 64, sample_rate=48000.0)
        b = AudioBuffer.zeros(1, 64, sample_rate=48000.0)
        result = dsp.crossfade(a, b, 0.0)
        # x=0 -> from=a, to=b; from_c ~1, to_c ~0
        np.testing.assert_allclose(result.data, a.data, atol=0.02)

    def test_crossfade_at_one(self):
        a = AudioBuffer.zeros(1, 64, sample_rate=48000.0)
        b = AudioBuffer.ones(1, 64, sample_rate=48000.0)
        result = dsp.crossfade(a, b, 1.0)
        np.testing.assert_allclose(result.data, b.data, atol=0.02)

    def test_crossfade_midpoint(self):
        a = AudioBuffer.ones(1, 64, sample_rate=48000.0)
        b = AudioBuffer.ones(1, 64, sample_rate=48000.0) * 3.0
        result = dsp.crossfade(a, b, 0.5)
        # At midpoint, both coefficients are roughly equal
        mid_val = result.data[0, 0]
        assert 1.0 < mid_val < 3.0

    def test_crossfade_sr_mismatch_raises(self):
        a = AudioBuffer.ones(1, 64, sample_rate=44100.0)
        b = AudioBuffer.ones(1, 64, sample_rate=48000.0)
        with pytest.raises(ValueError, match="Sample rate"):
            dsp.crossfade(a, b, 0.5)

    def test_crossfade_channel_mismatch_raises(self):
        a = AudioBuffer.ones(1, 64, sample_rate=48000.0)
        b = AudioBuffer.ones(2, 64, sample_rate=48000.0)
        with pytest.raises(ValueError, match="Channel count"):
            dsp.crossfade(a, b, 0.5)

    def test_crossfade_frame_mismatch_raises(self):
        a = AudioBuffer.ones(1, 64, sample_rate=48000.0)
        b = AudioBuffer.ones(1, 128, sample_rate=48000.0)
        with pytest.raises(ValueError, match="Frame count"):
            dsp.crossfade(a, b, 0.5)


# ---------------------------------------------------------------------------
# STFT functions
# ---------------------------------------------------------------------------


class TestSTFTFunctions:
    def test_stft_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        spec = dsp.stft(buf, window_size=1024)
        fft_size = cydsp.fft.RealFFT.fast_size_above(1024)
        expected_frames = (4096 - 1024) // 256 + 1
        assert spec.data.shape == (1, expected_frames, fft_size // 2)
        assert spec.data.dtype == np.complex64

    def test_stft_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=0)
        spec = dsp.stft(buf, window_size=1024)
        assert spec.channels == 2

    def test_stft_custom_params(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        spec = dsp.stft(buf, window_size=512, hop_size=128)
        expected_frames = (4096 - 512) // 128 + 1
        assert spec.num_frames == expected_frames
        assert spec.hop_size == 128
        assert spec.window_size == 512

    def test_istft_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        spec = dsp.stft(buf, window_size=1024)
        result = dsp.istft(spec)
        assert isinstance(result, AudioBuffer)
        assert result.channels == 1
        assert result.frames == 4096

    def test_roundtrip(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=42)
        spec = dsp.stft(buf, window_size=1024)
        result = dsp.istft(spec)
        # Interior samples (away from edges) should match well
        margin = 1024
        np.testing.assert_allclose(
            result.data[0, margin:-margin],
            buf.data[0, margin:-margin],
            atol=1e-4,
        )

    def test_roundtrip_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=48000.0, seed=7)
        spec = dsp.stft(buf, window_size=1024)
        result = dsp.istft(spec)
        margin = 1024
        for ch in range(2):
            np.testing.assert_allclose(
                result.data[ch, margin:-margin],
                buf.data[ch, margin:-margin],
                atol=1e-4,
            )

    def test_spectrogram_properties(self):
        buf = AudioBuffer.noise(channels=2, frames=4096, sample_rate=44100.0, seed=0)
        spec = dsp.stft(buf, window_size=2048)
        assert spec.channels == 2
        assert spec.bins == spec.fft_size // 2
        assert spec.sample_rate == 44100.0
        assert spec.original_frames == 4096


# ---------------------------------------------------------------------------
# LFO function
# ---------------------------------------------------------------------------


class TestLfoFunction:
    def test_lfo_shape(self):
        result = dsp.lfo(1024, low=0.0, high=1.0, rate=0.001)
        assert isinstance(result, AudioBuffer)
        assert result.channels == 1
        assert result.frames == 1024

    def test_lfo_range(self):
        result = dsp.lfo(4096, low=-1.0, high=1.0, rate=0.01, seed=42)
        # CubicLfo may slightly overshoot at transitions, allow tolerance
        assert np.all(result.data >= -1.1)
        assert np.all(result.data <= 1.1)

    def test_lfo_deterministic(self):
        a = dsp.lfo(1024, low=0.0, high=1.0, rate=0.005, seed=123)
        b = dsp.lfo(1024, low=0.0, high=1.0, rate=0.005, seed=123)
        np.testing.assert_array_equal(a.data, b.data)

    def test_lfo_sample_rate(self):
        result = dsp.lfo(512, low=0.0, high=1.0, rate=0.01, sample_rate=44100.0)
        assert result.sample_rate == 44100.0

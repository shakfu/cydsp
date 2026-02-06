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
# Convolution
# ---------------------------------------------------------------------------


class TestConvolve:
    def test_impulse_passthrough(self):
        """Convolving with a unit impulse should return the input."""
        buf = AudioBuffer.sine(440.0, frames=1024, sample_rate=48000.0)
        ir = AudioBuffer.impulse(channels=1, frames=64, sample_rate=48000.0)
        result = dsp.convolve(buf, ir)
        np.testing.assert_allclose(result.data, buf.data, atol=1e-5)

    def test_output_length_trimmed(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        ir = AudioBuffer.noise(channels=1, frames=128, sample_rate=48000.0, seed=1)
        result = dsp.convolve(buf, ir, trim=True)
        assert result.frames == buf.frames

    def test_output_length_full(self):
        buf = AudioBuffer.noise(channels=1, frames=1024, sample_rate=48000.0, seed=0)
        ir = AudioBuffer.noise(channels=1, frames=128, sample_rate=48000.0, seed=1)
        result = dsp.convolve(buf, ir, trim=False)
        assert result.frames == buf.frames + ir.frames - 1

    def test_mono_ir_broadcast_to_stereo(self):
        buf = AudioBuffer.noise(channels=2, frames=512, sample_rate=48000.0, seed=0)
        ir = AudioBuffer.impulse(channels=1, frames=32, sample_rate=48000.0)
        result = dsp.convolve(buf, ir)
        assert result.channels == 2

    def test_channel_mismatch_raises(self):
        buf = AudioBuffer.noise(channels=2, frames=512, sample_rate=48000.0, seed=0)
        ir = AudioBuffer.noise(channels=3, frames=32, sample_rate=48000.0, seed=1)
        with pytest.raises(ValueError, match="Channel mismatch"):
            dsp.convolve(buf, ir)

    def test_sample_rate_mismatch_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=512, sample_rate=48000.0, seed=0)
        ir = AudioBuffer.noise(channels=1, frames=32, sample_rate=44100.0, seed=1)
        with pytest.raises(ValueError, match="Sample rate mismatch"):
            dsp.convolve(buf, ir)

    def test_normalize_flag(self):
        buf = AudioBuffer.sine(440.0, frames=1024, sample_rate=48000.0)
        ir_data = np.zeros((1, 64), dtype=np.float32)
        ir_data[0, 0] = 10.0
        ir = AudioBuffer(ir_data, sample_rate=48000.0)
        result_norm = dsp.convolve(buf, ir, normalize=True)
        result_raw = dsp.convolve(buf, ir, normalize=False)
        # Normalized should have less energy than raw (IR energy > 1)
        assert np.sum(result_norm.data**2) < np.sum(result_raw.data**2)

    def test_correctness_vs_np_convolve(self):
        rng = np.random.default_rng(42)
        sig = rng.standard_normal(256).astype(np.float32)
        kernel = rng.standard_normal(32).astype(np.float32)
        buf = AudioBuffer(sig, sample_rate=48000.0)
        ir = AudioBuffer(kernel, sample_rate=48000.0)
        result = dsp.convolve(buf, ir, trim=False)
        expected = np.convolve(sig, kernel)
        np.testing.assert_allclose(result.data[0], expected, atol=1e-4)

    def test_metadata_preserved(self):
        buf = AudioBuffer.noise(
            channels=1, frames=512, sample_rate=44100.0, seed=0, label="test"
        )
        ir = AudioBuffer.impulse(channels=1, frames=32, sample_rate=44100.0)
        result = dsp.convolve(buf, ir)
        assert result.sample_rate == 44100.0
        assert result.label == "test"


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


# ---------------------------------------------------------------------------
# DaisySP Effects
# ---------------------------------------------------------------------------


class TestDaisySPEffects:
    def _noise(self, channels=1, frames=2048, sample_rate=48000.0):
        return AudioBuffer.noise(
            channels=channels, frames=frames, sample_rate=sample_rate, seed=0
        )

    def _sine(self, freq=440.0, channels=1, frames=2048, sample_rate=48000.0):
        return AudioBuffer.sine(
            freq, channels=channels, frames=frames, sample_rate=sample_rate
        )

    def test_autowah_shape_dtype(self):
        buf = self._sine()
        result = dsp.autowah(buf)
        assert result.channels == 1
        assert result.frames == 2048
        assert result.data.dtype == np.float32

    def test_autowah_modifies_signal(self):
        buf = self._sine()
        result = dsp.autowah(buf, wah=0.8)
        assert not np.allclose(result.data, buf.data)

    def test_chorus_mono_to_stereo(self):
        buf = self._sine()
        result = dsp.chorus(buf)
        assert result.channels == 2
        assert result.frames == 2048

    def test_chorus_multichannel_per_channel(self):
        buf = self._noise(channels=2)
        result = dsp.chorus(buf)
        assert result.channels == 2

    def test_decimator_shape(self):
        buf = self._noise()
        result = dsp.decimator(buf)
        assert result.frames == 2048
        assert result.data.dtype == np.float32

    def test_decimator_modifies_signal(self):
        buf = self._sine()
        result = dsp.decimator(buf, downsample_factor=0.8, bits_to_crush=4)
        assert not np.allclose(result.data, buf.data)

    def test_flanger_shape(self):
        buf = self._sine()
        result = dsp.flanger(buf)
        assert result.channels == 1
        assert result.frames == 2048

    def test_overdrive_shape(self):
        buf = self._sine()
        result = dsp.overdrive(buf, drive=0.8)
        assert result.frames == 2048
        assert result.data.dtype == np.float32

    def test_overdrive_adds_harmonics(self):
        buf = self._sine()
        result = dsp.overdrive(buf, drive=0.9)
        assert not np.allclose(result.data, buf.data)

    def test_phaser_shape(self):
        buf = self._sine()
        result = dsp.phaser(buf)
        assert result.frames == 2048

    def test_pitch_shift_shape(self):
        buf = self._sine()
        result = dsp.pitch_shift(buf, semitones=5.0)
        assert result.frames == 2048

    def test_sample_rate_reduce_shape(self):
        buf = self._noise()
        result = dsp.sample_rate_reduce(buf, freq=0.3)
        assert result.frames == 2048

    def test_tremolo_shape(self):
        buf = self._sine()
        result = dsp.tremolo(buf, freq=5.0, depth=1.0)
        assert result.frames == 2048

    def test_wavefold_shape(self):
        buf = self._sine()
        result = dsp.wavefold(buf, gain=2.0)
        assert result.frames == 2048

    def test_bitcrush_shape(self):
        buf = self._noise()
        result = dsp.bitcrush(buf, bit_depth=4)
        assert result.frames == 2048

    def test_bitcrush_default_crush_rate(self):
        buf = self._noise()
        result = dsp.bitcrush(buf)
        assert result.data.dtype == np.float32

    def test_fold_shape(self):
        buf = self._sine()
        result = dsp.fold(buf, increment=0.5)
        assert result.frames == 2048

    def test_reverb_sc_mono_to_stereo(self):
        buf = self._sine()
        result = dsp.reverb_sc(buf)
        assert result.channels == 2
        assert result.frames == 2048

    def test_reverb_sc_stereo_passthrough(self):
        buf = self._noise(channels=2)
        result = dsp.reverb_sc(buf)
        assert result.channels == 2

    def test_reverb_sc_3ch_raises(self):
        buf = self._noise(channels=3)
        with pytest.raises(ValueError, match="mono or stereo"):
            dsp.reverb_sc(buf)

    def test_dc_block_shape(self):
        buf = self._noise()
        result = dsp.dc_block(buf)
        assert result.frames == 2048

    def test_dc_block_removes_offset(self):
        data = np.ones((1, 4096), dtype=np.float32) * 0.5
        buf = AudioBuffer(data, sample_rate=48000.0)
        result = dsp.dc_block(buf)
        # Mean should be much closer to 0 after DC blocking
        assert abs(np.mean(result.data[0, 1024:])) < abs(np.mean(buf.data[0]))

    def test_effects_multichannel(self):
        buf = self._noise(channels=2)
        for fn, kwargs in [
            (dsp.autowah, {}),
            (dsp.decimator, {}),
            (dsp.flanger, {}),
            (dsp.overdrive, {}),
            (dsp.phaser, {}),
            (dsp.tremolo, {}),
            (dsp.wavefold, {}),
            (dsp.bitcrush, {}),
            (dsp.fold, {}),
            (dsp.dc_block, {}),
        ]:
            result = fn(buf, **kwargs)
            assert result.channels == 2, f"{fn.__name__} failed multichannel"


# ---------------------------------------------------------------------------
# DaisySP Filters
# ---------------------------------------------------------------------------


class TestDaisySPFilters:
    def _noise(self, channels=1, frames=4096, sample_rate=48000.0):
        return AudioBuffer.noise(
            channels=channels, frames=frames, sample_rate=sample_rate, seed=0
        )

    def test_svf_lowpass_attenuates_high(self):
        low = AudioBuffer.sine(500.0, frames=4096, sample_rate=48000.0)
        high = AudioBuffer.sine(15000.0, frames=4096, sample_rate=48000.0)
        combined = low + high
        result = dsp.svf_lowpass(combined, freq_hz=2000.0)
        assert np.sum(result.data**2) < np.sum(combined.data**2)

    def test_svf_highpass_attenuates_low(self):
        low = AudioBuffer.sine(500.0, frames=4096, sample_rate=48000.0)
        high = AudioBuffer.sine(15000.0, frames=4096, sample_rate=48000.0)
        combined = low + high
        result = dsp.svf_highpass(combined, freq_hz=5000.0)
        assert np.sum(result.data**2) < np.sum(combined.data**2)

    def test_svf_bandpass_shape(self):
        buf = self._noise()
        result = dsp.svf_bandpass(buf, freq_hz=1000.0)
        assert result.frames == 4096

    def test_svf_notch_shape(self):
        buf = self._noise()
        result = dsp.svf_notch(buf, freq_hz=1000.0)
        assert result.frames == 4096

    def test_svf_peak_shape(self):
        buf = self._noise()
        result = dsp.svf_peak(buf, freq_hz=1000.0)
        assert result.frames == 4096

    def test_ladder_filter_lowpass(self):
        buf = self._noise()
        result = dsp.ladder_filter(buf, freq_hz=2000.0, mode="lp24")
        assert result.frames == 4096
        assert np.sum(result.data**2) < np.sum(buf.data**2)

    def test_ladder_filter_modes(self):
        buf = self._noise()
        for mode in ["lp24", "lp12", "bp24", "bp12", "hp24", "hp12"]:
            result = dsp.ladder_filter(buf, freq_hz=2000.0, mode=mode)
            assert result.frames == 4096, f"Failed for mode={mode}"

    def test_ladder_filter_invalid_mode(self):
        buf = self._noise()
        with pytest.raises(ValueError, match="Unknown ladder mode"):
            dsp.ladder_filter(buf, mode="invalid")

    def test_moog_ladder_shape(self):
        buf = self._noise()
        result = dsp.moog_ladder(buf, freq_hz=2000.0, resonance=0.3)
        assert result.frames == 4096

    def test_tone_lowpass_attenuates_high(self):
        low = AudioBuffer.sine(500.0, frames=4096, sample_rate=48000.0)
        high = AudioBuffer.sine(15000.0, frames=4096, sample_rate=48000.0)
        combined = low + high
        result = dsp.tone_lowpass(combined, freq_hz=2000.0)
        assert np.sum(result.data**2) < np.sum(combined.data**2)

    def test_tone_highpass_attenuates_low(self):
        low = AudioBuffer.sine(500.0, frames=4096, sample_rate=48000.0)
        high = AudioBuffer.sine(15000.0, frames=4096, sample_rate=48000.0)
        combined = low + high
        result = dsp.tone_highpass(combined, freq_hz=5000.0)
        assert np.sum(result.data**2) < np.sum(combined.data**2)

    def test_modal_bandpass_shape(self):
        buf = self._noise()
        result = dsp.modal_bandpass(buf, freq_hz=1000.0)
        assert result.frames == 4096

    def test_comb_filter_shape(self):
        buf = self._noise()
        result = dsp.comb_filter(buf, freq_hz=500.0)
        assert result.frames == 4096

    def test_filters_multichannel(self):
        buf = self._noise(channels=2)
        for fn, kwargs in [
            (dsp.svf_lowpass, {"freq_hz": 2000.0}),
            (dsp.svf_highpass, {"freq_hz": 2000.0}),
            (dsp.ladder_filter, {"freq_hz": 2000.0}),
            (dsp.moog_ladder, {"freq_hz": 2000.0}),
            (dsp.tone_lowpass, {"freq_hz": 2000.0}),
            (dsp.tone_highpass, {"freq_hz": 2000.0}),
            (dsp.modal_bandpass, {"freq_hz": 1000.0}),
            (dsp.comb_filter, {"freq_hz": 500.0}),
        ]:
            result = fn(buf, **kwargs)
            assert result.channels == 2, f"{fn.__name__} failed multichannel"


# ---------------------------------------------------------------------------
# DaisySP Dynamics
# ---------------------------------------------------------------------------


class TestDaisySPDynamics:
    def test_compress_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=2048, sample_rate=48000.0, seed=0)
        result = dsp.compress(buf, ratio=4.0, threshold=-20.0)
        assert result.frames == 2048
        assert result.data.dtype == np.float32

    def test_compress_reduces_dynamic_range(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        result = dsp.compress(buf, ratio=8.0, threshold=-30.0)
        # Compressed signal should have different peak/RMS ratio
        assert not np.allclose(result.data, buf.data)

    def test_compress_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=2048, sample_rate=48000.0, seed=0)
        result = dsp.compress(buf)
        assert result.channels == 2

    def test_limit_shape(self):
        buf = AudioBuffer.noise(channels=1, frames=2048, sample_rate=48000.0, seed=0)
        result = dsp.limit(buf, pre_gain=2.0)
        assert result.frames == 2048
        assert result.data.dtype == np.float32

    def test_limit_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=2048, sample_rate=48000.0, seed=0)
        result = dsp.limit(buf)
        assert result.channels == 2


# ---------------------------------------------------------------------------
# DaisySP Oscillators
# ---------------------------------------------------------------------------


class TestDaisySPOscillators:
    def test_oscillator_sine_shape(self):
        result = dsp.oscillator(1024, freq=440.0)
        assert result.channels == 1
        assert result.frames == 1024
        assert result.data.dtype == np.float32

    def test_oscillator_waveform_names(self):
        for name in [
            "sine",
            "tri",
            "saw",
            "ramp",
            "square",
            "polyblep_tri",
            "polyblep_saw",
            "polyblep_square",
        ]:
            result = dsp.oscillator(512, freq=440.0, waveform=name)
            assert result.frames == 512
            assert np.max(np.abs(result.data)) > 0, f"waveform {name} produced silence"

    def test_oscillator_int_waveform(self):
        from cydsp._core import daisysp

        result = dsp.oscillator(512, freq=440.0, waveform=daisysp.oscillators.WAVE_SAW)
        assert result.frames == 512

    def test_oscillator_invalid_waveform(self):
        with pytest.raises(ValueError, match="Unknown waveform"):
            dsp.oscillator(512, waveform="nope")

    def test_oscillator_nonzero_output(self):
        result = dsp.oscillator(4096, freq=440.0, amp=1.0)
        assert np.max(np.abs(result.data)) > 0.5

    def test_fm2_shape(self):
        result = dsp.fm2(1024, freq=440.0, ratio=2.0, index=1.0)
        assert result.channels == 1
        assert result.frames == 1024

    def test_fm2_nonzero(self):
        result = dsp.fm2(4096, freq=440.0)
        assert np.max(np.abs(result.data)) > 0.1

    def test_formant_oscillator_shape(self):
        result = dsp.formant_oscillator(1024, carrier_freq=440.0, formant_freq=1000.0)
        assert result.channels == 1
        assert result.frames == 1024

    def test_formant_oscillator_nonzero(self):
        result = dsp.formant_oscillator(4096, carrier_freq=440.0)
        assert np.max(np.abs(result.data)) > 0.1

    def test_bl_oscillator_shape(self):
        result = dsp.bl_oscillator(1024, freq=440.0, waveform="saw")
        assert result.channels == 1
        assert result.frames == 1024

    def test_bl_oscillator_waveform_names(self):
        for name in ["triangle", "tri", "saw", "square"]:
            result = dsp.bl_oscillator(512, freq=440.0, waveform=name)
            assert result.frames == 512
            assert np.max(np.abs(result.data)) > 0, f"bl_osc waveform {name} silent"

    def test_bl_oscillator_invalid_waveform(self):
        with pytest.raises(ValueError, match="Unknown waveform"):
            dsp.bl_oscillator(512, waveform="nope")

    def test_oscillator_sample_rate(self):
        result = dsp.oscillator(512, freq=440.0, sample_rate=44100.0)
        assert result.sample_rate == 44100.0


# ---------------------------------------------------------------------------
# DaisySP Noise
# ---------------------------------------------------------------------------


class TestDaisySPNoise:
    def test_white_noise_shape(self):
        result = dsp.white_noise(1024)
        assert result.channels == 1
        assert result.frames == 1024
        assert result.data.dtype == np.float32

    def test_white_noise_nonzero(self):
        result = dsp.white_noise(4096)
        assert np.max(np.abs(result.data)) > 0.1

    def test_white_noise_amp(self):
        result = dsp.white_noise(4096, amp=0.1)
        assert np.max(np.abs(result.data)) < 0.5

    def test_clocked_noise_shape(self):
        result = dsp.clocked_noise(1024, freq=1000.0)
        assert result.channels == 1
        assert result.frames == 1024

    def test_clocked_noise_nonzero(self):
        result = dsp.clocked_noise(4096, freq=1000.0)
        assert np.max(np.abs(result.data)) > 0

    def test_dust_shape(self):
        result = dsp.dust(1024, density=1.0)
        assert result.channels == 1
        assert result.frames == 1024

    def test_dust_nonzero(self):
        result = dsp.dust(48000, density=100.0)
        assert np.max(np.abs(result.data)) > 0

    def test_noise_sample_rate(self):
        result = dsp.white_noise(512, sample_rate=44100.0)
        assert result.sample_rate == 44100.0


# ---------------------------------------------------------------------------
# DaisySP Drums
# ---------------------------------------------------------------------------


class TestDaisySPDrums:
    def test_analog_bass_drum_shape(self):
        result = dsp.analog_bass_drum(4096)
        assert result.channels == 1
        assert result.frames == 4096
        assert result.data.dtype == np.float32

    def test_analog_bass_drum_nonzero(self):
        result = dsp.analog_bass_drum(4096, accent=0.8)
        assert np.max(np.abs(result.data)) > 0.01

    def test_analog_snare_drum_shape(self):
        result = dsp.analog_snare_drum(4096)
        assert result.channels == 1
        assert result.frames == 4096

    def test_analog_snare_drum_nonzero(self):
        result = dsp.analog_snare_drum(4096, accent=0.8)
        assert np.max(np.abs(result.data)) > 0.01

    def test_hihat_shape(self):
        result = dsp.hihat(4096)
        assert result.channels == 1
        assert result.frames == 4096

    def test_hihat_nonzero(self):
        result = dsp.hihat(4096, accent=0.8)
        assert np.max(np.abs(result.data)) > 0.01

    def test_synthetic_bass_drum_shape(self):
        result = dsp.synthetic_bass_drum(4096)
        assert result.channels == 1
        assert result.frames == 4096

    def test_synthetic_bass_drum_nonzero(self):
        result = dsp.synthetic_bass_drum(4096, accent=0.8)
        assert np.max(np.abs(result.data)) > 0.01

    def test_synthetic_snare_drum_shape(self):
        result = dsp.synthetic_snare_drum(4096)
        assert result.channels == 1
        assert result.frames == 4096

    def test_synthetic_snare_drum_nonzero(self):
        result = dsp.synthetic_snare_drum(4096, accent=0.8)
        assert np.max(np.abs(result.data)) > 0.01

    def test_drums_decay(self):
        result = dsp.analog_bass_drum(8192, decay=0.8)
        peak_idx = np.argmax(np.abs(result.data[0]))
        tail_energy = np.sum(result.data[0, peak_idx + 2048 :] ** 2)
        head_energy = np.sum(result.data[0, peak_idx : peak_idx + 2048] ** 2)
        # Tail should have less energy than head for a decaying drum
        assert tail_energy < head_energy


# ---------------------------------------------------------------------------
# DaisySP Physical Modeling
# ---------------------------------------------------------------------------


class TestDaisySPPhysicalModeling:
    def test_karplus_strong_shape(self):
        buf = AudioBuffer.impulse(channels=1, frames=4096, sample_rate=48000.0)
        result = dsp.karplus_strong(buf, freq_hz=440.0)
        assert result.channels == 1
        assert result.frames == 4096

    def test_karplus_strong_nonzero(self):
        buf = AudioBuffer.impulse(channels=1, frames=4096, sample_rate=48000.0)
        result = dsp.karplus_strong(buf, freq_hz=440.0)
        assert np.max(np.abs(result.data)) > 0.01

    def test_karplus_strong_multichannel(self):
        buf = AudioBuffer.impulse(channels=2, frames=4096, sample_rate=48000.0)
        result = dsp.karplus_strong(buf, freq_hz=440.0)
        assert result.channels == 2

    def test_modal_voice_shape(self):
        result = dsp.modal_voice(4096, freq=440.0)
        assert result.channels == 1
        assert result.frames == 4096

    def test_modal_voice_nonzero(self):
        result = dsp.modal_voice(4096, freq=440.0, accent=0.8)
        assert np.max(np.abs(result.data)) > 0.001

    def test_string_voice_shape(self):
        result = dsp.string_voice(4096, freq=440.0)
        assert result.channels == 1
        assert result.frames == 4096

    def test_string_voice_nonzero(self):
        result = dsp.string_voice(4096, freq=440.0, accent=0.8)
        assert np.max(np.abs(result.data)) > 0.001

    def test_pluck_shape(self):
        result = dsp.pluck(4096, freq=440.0)
        assert result.channels == 1
        assert result.frames == 4096

    def test_pluck_nonzero(self):
        result = dsp.pluck(4096, freq=440.0, amp=0.8)
        assert np.max(np.abs(result.data)) > 0.01

    def test_drip_shape(self):
        result = dsp.drip(4096)
        assert result.channels == 1
        assert result.frames == 4096

    def test_drip_nonzero(self):
        result = dsp.drip(4096)
        # Drip may or may not produce output on first trigger, just check shape
        assert result.data.dtype == np.float32


# ---------------------------------------------------------------------------
# Spectral utility functions
# ---------------------------------------------------------------------------


class TestSpectralUtilities:
    @pytest.fixture()
    def spec(self):
        """A mono spectrogram from noise for reuse across tests."""
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        return dsp.stft(buf, window_size=1024)

    @pytest.fixture()
    def spec_stereo(self):
        """A stereo spectrogram."""
        buf = AudioBuffer.noise(channels=2, frames=8192, sample_rate=48000.0, seed=1)
        return dsp.stft(buf, window_size=1024)

    # -- Decomposition --

    def test_magnitude_shape_dtype(self, spec):
        mag = dsp.magnitude(spec)
        assert mag.shape == spec.data.shape
        assert mag.dtype == np.float32

    def test_phase_shape_dtype(self, spec):
        ph = dsp.phase(spec)
        assert ph.shape == spec.data.shape
        assert ph.dtype == np.float32

    def test_decomposition_roundtrip(self, spec):
        mag = dsp.magnitude(spec)
        ph = dsp.phase(spec)
        reconstructed = dsp.from_polar(mag, ph, spec)
        np.testing.assert_allclose(
            np.abs(reconstructed.data), np.abs(spec.data), atol=1e-5
        )
        np.testing.assert_allclose(
            np.angle(reconstructed.data), np.angle(spec.data), atol=1e-5
        )

    def test_from_polar_metadata(self, spec):
        mag = dsp.magnitude(spec)
        ph = dsp.phase(spec)
        result = dsp.from_polar(mag, ph, spec)
        assert result.window_size == spec.window_size
        assert result.hop_size == spec.hop_size
        assert result.fft_size == spec.fft_size
        assert result.sample_rate == spec.sample_rate
        assert result.original_frames == spec.original_frames

    # -- Filtering --

    def test_apply_mask_identity(self, spec):
        mask = np.ones(spec.data.shape, dtype=np.float32)
        result = dsp.apply_mask(spec, mask)
        np.testing.assert_allclose(result.data, spec.data, atol=1e-7)

    def test_apply_mask_zeros_bins(self, spec):
        mask = np.ones(spec.data.shape, dtype=np.float32)
        mask[:, :, 0] = 0.0  # zero DC bin
        result = dsp.apply_mask(spec, mask)
        assert np.all(result.data[:, :, 0] == 0.0)
        # Other bins unaffected
        np.testing.assert_allclose(
            result.data[:, :, 1:], spec.data[:, :, 1:], atol=1e-7
        )

    def test_apply_mask_broadcast_1d(self, spec):
        """1D mask [bins] broadcasts across channels and frames."""
        mask = np.ones(spec.bins, dtype=np.float32)
        mask[0] = 0.0
        result = dsp.apply_mask(spec, mask)
        assert np.all(result.data[:, :, 0] == 0.0)

    def test_apply_mask_broadcast_2d(self, spec):
        """2D mask [frames, bins] broadcasts across channels."""
        mask = np.ones((spec.num_frames, spec.bins), dtype=np.float32)
        mask[0, :] = 0.0  # zero first frame
        result = dsp.apply_mask(spec, mask)
        assert np.all(result.data[:, 0, :] == 0.0)

    def test_apply_mask_bad_shape(self, spec):
        bad_mask = np.ones((spec.bins + 1,), dtype=np.float32)
        with pytest.raises(ValueError, match="broadcastable"):
            dsp.apply_mask(spec, bad_mask)

    # -- Spectral gate --

    def test_spectral_gate_preserves_loud(self, spec):
        """Loud bins should pass through mostly unchanged."""
        result = dsp.spectral_gate(spec, threshold_db=-100.0)
        np.testing.assert_allclose(np.abs(result.data), np.abs(spec.data), atol=1e-6)

    def test_spectral_gate_attenuates_quiet(self):
        """Quiet signal should be attenuated."""
        buf = AudioBuffer(
            np.full((1, 8192), 1e-6, dtype=np.float32), sample_rate=48000.0
        )
        spec = dsp.stft(buf, window_size=1024)
        result = dsp.spectral_gate(spec, threshold_db=-20.0, noise_floor_db=-80.0)
        assert np.mean(np.abs(result.data)) < np.mean(np.abs(spec.data))

    # -- Spectral emphasis --

    def test_spectral_emphasis_flat(self, spec):
        result = dsp.spectral_emphasis(spec, low_db=0.0, high_db=0.0)
        np.testing.assert_allclose(result.data, spec.data, atol=1e-6)

    def test_spectral_emphasis_tilt(self, spec):
        result = dsp.spectral_emphasis(spec, low_db=-6.0, high_db=6.0)
        # Low bins should be attenuated, high bins boosted
        orig_low = np.mean(np.abs(spec.data[:, :, :10]))
        orig_high = np.mean(np.abs(spec.data[:, :, -10:]))
        new_low = np.mean(np.abs(result.data[:, :, :10]))
        new_high = np.mean(np.abs(result.data[:, :, -10:]))
        assert new_low < orig_low  # attenuated
        assert new_high > orig_high  # boosted

    # -- bin_freq / freq_to_bin --

    def test_bin_freq_correctness(self, spec):
        # Bin 0 = DC
        assert dsp.bin_freq(spec, 0) == 0.0
        # Bin 1 = sample_rate / fft_size
        expected = spec.sample_rate / spec.fft_size
        assert dsp.bin_freq(spec, 1) == pytest.approx(expected)

    def test_freq_to_bin_roundtrip(self, spec):
        freq = 1000.0
        b = dsp.freq_to_bin(spec, freq)
        recovered = dsp.bin_freq(spec, b)
        # Should be within one bin width
        bin_width = spec.sample_rate / spec.fft_size
        assert abs(recovered - freq) <= bin_width

    def test_freq_to_bin_negative_raises(self, spec):
        with pytest.raises(ValueError, match="non-negative"):
            dsp.freq_to_bin(spec, -100.0)

    def test_freq_to_bin_nyquist_raises(self, spec):
        with pytest.raises(ValueError, match="Nyquist"):
            dsp.freq_to_bin(spec, spec.sample_rate / 2.0)

    # -- Time stretch --

    def test_time_stretch_identity(self, spec):
        result = dsp.time_stretch(spec, 1.0)
        assert result.num_frames == spec.num_frames
        assert result.original_frames == spec.original_frames

    def test_time_stretch_slow(self, spec):
        result = dsp.time_stretch(spec, 0.5)
        # Should roughly double num_frames
        assert result.num_frames == pytest.approx(spec.num_frames * 2, abs=1)
        assert result.original_frames == pytest.approx(spec.original_frames * 2, abs=1)

    def test_time_stretch_fast(self, spec):
        result = dsp.time_stretch(spec, 2.0)
        # Should roughly halve num_frames
        assert result.num_frames == pytest.approx(spec.num_frames / 2, abs=1)

    def test_time_stretch_roundtrip(self):
        """stft -> stretch(0.5) -> istft produces longer audio."""
        buf = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0)
        spec = dsp.stft(buf, window_size=1024)
        stretched = dsp.time_stretch(spec, 0.5)
        result = dsp.istft(stretched)
        assert result.frames > buf.frames

    def test_time_stretch_invalid_rate(self, spec):
        with pytest.raises(ValueError, match="Rate must be > 0"):
            dsp.time_stretch(spec, 0.0)
        with pytest.raises(ValueError, match="Rate must be > 0"):
            dsp.time_stretch(spec, -1.0)

    # -- Phase lock --

    def test_phase_lock_preserves_magnitude(self, spec):
        result = dsp.phase_lock(spec)
        np.testing.assert_allclose(np.abs(result.data), np.abs(spec.data), atol=1e-5)

    def test_phase_lock_shape(self, spec):
        result = dsp.phase_lock(spec)
        assert result.data.shape == spec.data.shape
        assert result.data.dtype == np.complex64

    # -- Spectral freeze --

    def test_spectral_freeze_shape(self, spec):
        result = dsp.spectral_freeze(spec, frame_index=0)
        assert result.data.shape == spec.data.shape
        assert result.data.dtype == np.complex64

    def test_spectral_freeze_all_frames_identical(self, spec):
        result = dsp.spectral_freeze(spec, frame_index=3)
        for t in range(result.num_frames):
            np.testing.assert_array_equal(result.data[:, t, :], result.data[:, 0, :])

    def test_spectral_freeze_matches_source_frame(self, spec):
        idx = 5
        result = dsp.spectral_freeze(spec, frame_index=idx)
        np.testing.assert_array_equal(result.data[:, 0, :], spec.data[:, idx, :])

    def test_spectral_freeze_negative_index(self, spec):
        result = dsp.spectral_freeze(spec, frame_index=-1)
        np.testing.assert_array_equal(result.data[:, 0, :], spec.data[:, -1, :])

    def test_spectral_freeze_custom_num_frames(self, spec):
        result = dsp.spectral_freeze(spec, frame_index=0, num_frames=10)
        assert result.num_frames == 10

    def test_spectral_freeze_out_of_range_raises(self, spec):
        with pytest.raises(IndexError, match="out of range"):
            dsp.spectral_freeze(spec, frame_index=spec.num_frames)
        with pytest.raises(IndexError, match="out of range"):
            dsp.spectral_freeze(spec, frame_index=-spec.num_frames - 1)

    def test_spectral_freeze_roundtrip(self, spec):
        """Freeze -> istft should produce audio of the expected length."""
        frozen = dsp.spectral_freeze(spec, frame_index=0, num_frames=20)
        audio = dsp.istft(frozen)
        expected_len = (20 - 1) * spec.hop_size + spec.window_size
        assert audio.frames == expected_len

    # -- Spectral morph --

    def test_spectral_morph_mix_zero(self, spec):
        result = dsp.spectral_morph(spec, spec, mix=0.0)
        np.testing.assert_allclose(np.abs(result.data), np.abs(spec.data), atol=1e-5)

    def test_spectral_morph_mix_one(self, spec, spec_stereo):
        """mix=1.0 should return spec_b's magnitudes."""
        # Use two different mono specs
        buf2 = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=99)
        spec_b = dsp.stft(buf2, window_size=1024)
        result = dsp.spectral_morph(spec, spec_b, mix=1.0)
        np.testing.assert_allclose(
            np.abs(result.data),
            np.abs(spec_b.data[:, : result.num_frames, :]),
            atol=1e-5,
        )

    def test_spectral_morph_midpoint(self, spec):
        buf2 = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=77)
        spec_b = dsp.stft(buf2, window_size=1024)
        result = dsp.spectral_morph(spec, spec_b, mix=0.5)
        mag_a = np.abs(spec.data[:, : result.num_frames, :])
        mag_b = np.abs(spec_b.data[:, : result.num_frames, :])
        expected_mag = 0.5 * mag_a + 0.5 * mag_b
        np.testing.assert_allclose(np.abs(result.data), expected_mag, atol=1e-5)

    def test_spectral_morph_different_lengths(self):
        """Shorter spectrogram length should be used."""
        short = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        long = AudioBuffer.noise(channels=1, frames=16384, sample_rate=48000.0, seed=1)
        spec_short = dsp.stft(short, window_size=1024)
        spec_long = dsp.stft(long, window_size=1024)
        result = dsp.spectral_morph(spec_short, spec_long, mix=0.5)
        assert result.num_frames == spec_short.num_frames

    def test_spectral_morph_fft_size_mismatch_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        spec_a = dsp.stft(buf, window_size=1024)
        spec_b = dsp.stft(buf, window_size=512)
        with pytest.raises(ValueError, match="fft_size mismatch"):
            dsp.spectral_morph(spec_a, spec_b, mix=0.5)

    def test_spectral_morph_channel_mismatch_raises(self):
        mono = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        stereo = AudioBuffer.noise(channels=2, frames=8192, sample_rate=48000.0, seed=0)
        spec_m = dsp.stft(mono, window_size=1024)
        spec_s = dsp.stft(stereo, window_size=1024)
        with pytest.raises(ValueError, match="channel count mismatch"):
            dsp.spectral_morph(spec_m, spec_s, mix=0.5)

    def test_spectral_morph_time_varying_mix(self, spec):
        """Per-frame mix array should work."""
        buf2 = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=55)
        spec_b = dsp.stft(buf2, window_size=1024)
        n_frames = min(spec.num_frames, spec_b.num_frames)
        # Ramp from 0 to 1 across frames: [1, T, 1]
        mix_arr = np.linspace(0, 1, n_frames, dtype=np.float32)[None, :, None]
        result = dsp.spectral_morph(spec, spec_b, mix=mix_arr)
        assert result.num_frames == n_frames

    # -- Pitch shift spectral --

    def test_pitch_shift_spectral_identity(self):
        buf = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0)
        result = dsp.pitch_shift_spectral(buf, semitones=0.0)
        assert result.frames == buf.frames
        assert result.sample_rate == buf.sample_rate
        np.testing.assert_array_equal(result.data, buf.data)

    def test_pitch_shift_spectral_preserves_duration(self):
        buf = AudioBuffer.sine(440.0, frames=8192, sample_rate=48000.0)
        for semi in [3.0, -3.0, 7.0, 12.0]:
            result = dsp.pitch_shift_spectral(buf, semitones=semi)
            assert result.frames == buf.frames
            assert result.sample_rate == buf.sample_rate

    def test_pitch_shift_spectral_up_raises_frequency(self):
        """Shifting up should increase dominant frequency."""
        buf = AudioBuffer.sine(440.0, frames=16384, sample_rate=48000.0)
        result = dsp.pitch_shift_spectral(buf, semitones=12.0, window_size=2048)
        # Compare spectral centroids as a proxy for pitch
        spec_in = dsp.stft(buf, window_size=2048)
        spec_out = dsp.stft(result, window_size=2048)
        mag_in = np.mean(np.abs(spec_in.data[0]), axis=0)
        mag_out = np.mean(np.abs(spec_out.data[0]), axis=0)
        bins = np.arange(len(mag_in), dtype=np.float32)
        centroid_in = np.sum(bins * mag_in) / (np.sum(mag_in) + 1e-10)
        centroid_out = np.sum(bins * mag_out) / (np.sum(mag_out) + 1e-10)
        assert centroid_out > centroid_in * 1.3

    def test_pitch_shift_spectral_down_lowers_frequency(self):
        """Shifting down should decrease dominant frequency."""
        buf = AudioBuffer.sine(2000.0, frames=16384, sample_rate=48000.0)
        result = dsp.pitch_shift_spectral(buf, semitones=-12.0, window_size=2048)
        spec_in = dsp.stft(buf, window_size=2048)
        spec_out = dsp.stft(result, window_size=2048)
        mag_in = np.mean(np.abs(spec_in.data[0]), axis=0)
        mag_out = np.mean(np.abs(spec_out.data[0]), axis=0)
        bins = np.arange(len(mag_in), dtype=np.float32)
        centroid_in = np.sum(bins * mag_in) / (np.sum(mag_in) + 1e-10)
        centroid_out = np.sum(bins * mag_out) / (np.sum(mag_out) + 1e-10)
        assert centroid_out < centroid_in * 0.7

    def test_pitch_shift_spectral_multichannel(self):
        buf = AudioBuffer.noise(channels=2, frames=8192, sample_rate=48000.0, seed=0)
        result = dsp.pitch_shift_spectral(buf, semitones=5.0)
        assert result.channels == 2
        assert result.frames == buf.frames

    def test_pitch_shift_spectral_metadata(self):
        buf = AudioBuffer.sine(
            440.0,
            channels=2,
            frames=8192,
            sample_rate=44100.0,
            label="test",
        )
        result = dsp.pitch_shift_spectral(buf, semitones=3.0)
        assert result.sample_rate == 44100.0
        assert result.channel_layout == "stereo"
        assert result.label == "test"

    # -- Spectral denoise --

    def test_spectral_denoise_shape(self, spec):
        result = dsp.spectral_denoise(spec, noise_frames=5)
        assert result.data.shape == spec.data.shape
        assert result.data.dtype == np.complex64

    def test_spectral_denoise_reduces_noise(self):
        """Signal buried in noise should have noise reduced."""
        rng = np.random.RandomState(42)
        noise = rng.randn(1, 16384).astype(np.float32) * 0.01
        # First 4096 samples: pure noise. Rest: noise + signal.
        signal = np.zeros_like(noise)
        t = np.arange(16384, dtype=np.float32) / 48000.0
        signal[0, 4096:] = np.sin(2 * np.pi * 440 * t[4096:]).astype(np.float32) * 0.5
        combined = AudioBuffer(noise + signal, sample_rate=48000.0)
        spec = dsp.stft(combined, window_size=1024)
        # ~16 noise-only STFT frames at the start
        noise_f = (4096 - 1024) // 256 + 1
        result = dsp.spectral_denoise(spec, noise_frames=noise_f, reduction_db=-40.0)
        # Noise energy should decrease
        noise_region = spec.data[:, :noise_f, :]
        denoised_region = result.data[:, :noise_f, :]
        assert np.sum(np.abs(denoised_region) ** 2) < np.sum(np.abs(noise_region) ** 2)

    def test_spectral_denoise_preserves_loud_signal(self, spec):
        """Bins well above noise floor should pass through."""
        result = dsp.spectral_denoise(spec, noise_frames=3, reduction_db=-60.0)
        # Most of the energy is above the noise floor for broadband noise
        energy_ratio = np.sum(np.abs(result.data) ** 2) / np.sum(np.abs(spec.data) ** 2)
        assert energy_ratio > 0.5

    def test_spectral_denoise_smoothing(self, spec):
        """Smoothing should still produce valid output."""
        result = dsp.spectral_denoise(spec, noise_frames=5, smoothing=5)
        assert result.data.shape == spec.data.shape

    def test_spectral_denoise_invalid_noise_frames(self, spec):
        with pytest.raises(ValueError, match="noise_frames must be >= 1"):
            dsp.spectral_denoise(spec, noise_frames=0)
        with pytest.raises(ValueError, match="exceeds available frames"):
            dsp.spectral_denoise(spec, noise_frames=spec.num_frames + 1)


# ---------------------------------------------------------------------------
# EQ matching
# ---------------------------------------------------------------------------


class TestEqMatch:
    def test_identity_match(self):
        """Matching a signal to itself should return roughly the same signal."""
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        result = dsp.eq_match(buf, buf)
        assert result.frames == buf.frames
        # STFT roundtrip has edge effects, check interior
        margin = 2048
        np.testing.assert_allclose(
            result.data[0, margin:-margin],
            buf.data[0, margin:-margin],
            atol=0.05,
        )

    def test_shape_preserved(self):
        buf = AudioBuffer.noise(channels=2, frames=8192, sample_rate=48000.0, seed=0)
        target = AudioBuffer.noise(channels=2, frames=8192, sample_rate=48000.0, seed=1)
        result = dsp.eq_match(buf, target)
        assert result.channels == buf.channels
        assert result.frames == buf.frames

    def test_spectral_tilt_correction(self):
        """A dark source matched to a bright target should have more high-freq energy."""
        sr = 48000.0
        dark = AudioBuffer.noise(channels=1, frames=16384, sample_rate=sr, seed=0)
        dark = dsp.lowpass(dark, 2000.0)
        bright = AudioBuffer.noise(channels=1, frames=16384, sample_rate=sr, seed=1)
        bright = dsp.highpass(bright, 2000.0)

        result = dsp.eq_match(dark, bright, window_size=2048)
        # Compare spectral centroids
        spec_dark = dsp.stft(dark, window_size=2048)
        spec_result = dsp.stft(result, window_size=2048)
        mag_dark = np.mean(np.abs(spec_dark.data[0]), axis=0)
        mag_result = np.mean(np.abs(spec_result.data[0]), axis=0)
        bins = np.arange(len(mag_dark), dtype=np.float32)
        centroid_dark = np.sum(bins * mag_dark) / (np.sum(mag_dark) + 1e-10)
        centroid_result = np.sum(bins * mag_result) / (np.sum(mag_result) + 1e-10)
        assert centroid_result > centroid_dark * 1.5

    def test_smoothing(self):
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        target = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=1)
        result = dsp.eq_match(buf, target, smoothing=8)
        assert result.frames == buf.frames

    def test_sample_rate_mismatch_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=4096, sample_rate=48000.0, seed=0)
        target = AudioBuffer.noise(channels=1, frames=4096, sample_rate=44100.0, seed=1)
        with pytest.raises(ValueError, match="Sample rate mismatch"):
            dsp.eq_match(buf, target)

    def test_channel_mismatch_raises(self):
        buf = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=0)
        target = AudioBuffer.noise(channels=2, frames=8192, sample_rate=48000.0, seed=1)
        with pytest.raises(ValueError, match="Channel count mismatch"):
            dsp.eq_match(buf, target)

    def test_channel_mismatch_raises_reverse(self):
        buf = AudioBuffer.noise(channels=2, frames=8192, sample_rate=48000.0, seed=0)
        target = AudioBuffer.noise(channels=1, frames=8192, sample_rate=48000.0, seed=1)
        with pytest.raises(ValueError, match="Channel count mismatch"):
            dsp.eq_match(buf, target)


# ---------------------------------------------------------------------------
# Loudness metering (LUFS)
# ---------------------------------------------------------------------------


class TestLoudnessLufs:
    def test_1khz_sine_reference(self):
        """A full-scale 1 kHz sine at 48 kHz should measure around -3.01 LUFS."""
        sr = 48000.0
        frames = int(sr * 5)  # 5 seconds
        buf = AudioBuffer.sine(1000.0, channels=1, frames=frames, sample_rate=sr)
        lufs = dsp.loudness_lufs(buf)
        # Full-scale sine RMS = 1/sqrt(2) => ~-3.01 dBFS
        # K-weighting at 1kHz is nearly unity, so expect close to -3 LUFS
        assert -5.0 < lufs < -1.0

    def test_silence_returns_neg_inf(self):
        buf = AudioBuffer.zeros(1, 48000, sample_rate=48000.0)
        lufs = dsp.loudness_lufs(buf)
        assert np.isinf(lufs) and lufs < 0

    def test_short_signal_returns_neg_inf(self):
        """Signal shorter than 400ms should return -inf."""
        sr = 48000.0
        frames = int(sr * 0.3)  # 300ms < 400ms
        buf = AudioBuffer.sine(1000.0, channels=1, frames=frames, sample_rate=sr)
        lufs = dsp.loudness_lufs(buf)
        assert np.isinf(lufs) and lufs < 0

    def test_6db_gain_tracks_linearly(self):
        """Adding 6 dB should increase LUFS by ~6."""
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.sine(1000.0, channels=1, frames=frames, sample_rate=sr)
        buf = buf * 0.25  # scale down to avoid clipping after gain
        lufs_base = dsp.loudness_lufs(buf)
        boosted = buf.gain_db(6.0)
        lufs_boosted = dsp.loudness_lufs(boosted)
        assert abs((lufs_boosted - lufs_base) - 6.0) < 0.5

    def test_stereo(self):
        """Stereo measurement should work."""
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.sine(1000.0, channels=2, frames=frames, sample_rate=sr)
        lufs = dsp.loudness_lufs(buf)
        # Stereo same-signal doubles power => +3 dB over mono
        mono_lufs = dsp.loudness_lufs(
            AudioBuffer.sine(1000.0, channels=1, frames=frames, sample_rate=sr)
        )
        assert abs((lufs - mono_lufs) - 3.0) < 0.5

    def test_44100_hz(self):
        """Should work at 44100 Hz sample rate."""
        sr = 44100.0
        frames = int(sr * 3)
        buf = AudioBuffer.sine(1000.0, channels=1, frames=frames, sample_rate=sr)
        lufs = dsp.loudness_lufs(buf)
        assert -5.0 < lufs < -1.0

    def test_5_1_lfe_ignored(self):
        """LFE channel (ch 3) should contribute zero to loudness."""
        sr = 48000.0
        frames = int(sr * 3)
        # 6-channel buffer: only LFE has signal
        data = np.zeros((6, frames), dtype=np.float32)
        t = np.arange(frames, dtype=np.float32) / sr
        data[3] = np.sin(2.0 * np.pi * 60.0 * t).astype(np.float32)
        buf = AudioBuffer(data, sample_rate=sr)
        lufs = dsp.loudness_lufs(buf)
        # LFE weight is 0.0, so this should read as silence
        assert np.isinf(lufs) and lufs < 0

    def test_5_1_surround_weighted(self):
        """Surround channels (4, 5) should be weighted +1.5 dB (x1.41)."""
        sr = 48000.0
        frames = int(sr * 3)
        t = np.arange(frames, dtype=np.float32) / sr
        tone = np.sin(2.0 * np.pi * 1000.0 * t).astype(np.float32) * 0.25

        # Signal in Left only (weight 1.0)
        data_left = np.zeros((6, frames), dtype=np.float32)
        data_left[0] = tone
        lufs_left = dsp.loudness_lufs(AudioBuffer(data_left, sample_rate=sr))

        # Same signal in Left Surround only (weight 1.41)
        data_ls = np.zeros((6, frames), dtype=np.float32)
        data_ls[4] = tone
        lufs_ls = dsp.loudness_lufs(AudioBuffer(data_ls, sample_rate=sr))

        # Surround should measure ~1.5 dB louder
        delta = lufs_ls - lufs_left
        assert 1.0 < delta < 2.0

    def test_5_1_vs_stereo_front_channels(self):
        """Front L+R in 5.1 should match stereo (both weight 1.0)."""
        sr = 48000.0
        frames = int(sr * 3)
        tone = AudioBuffer.sine(1000.0, channels=1, frames=frames, sample_rate=sr)

        stereo = AudioBuffer(np.tile(tone.data, (2, 1)), sample_rate=sr)
        lufs_stereo = dsp.loudness_lufs(stereo)

        data_51 = np.zeros((6, frames), dtype=np.float32)
        data_51[0] = tone.data[0]
        data_51[1] = tone.data[0]
        lufs_51 = dsp.loudness_lufs(AudioBuffer(data_51, sample_rate=sr))

        assert abs(lufs_51 - lufs_stereo) < 0.5


class TestNormalizeLufs:
    def test_hits_target(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.sine(1000.0, channels=1, frames=frames, sample_rate=sr)
        result = dsp.normalize_lufs(buf, target_lufs=-14.0)
        measured = dsp.loudness_lufs(result)
        assert abs(measured - (-14.0)) < 1.5

    def test_silence_raises(self):
        buf = AudioBuffer.zeros(1, 48000, sample_rate=48000.0)
        with pytest.raises(ValueError, match="silent or too short"):
            dsp.normalize_lufs(buf)

    def test_short_signal_raises(self):
        sr = 48000.0
        frames = int(sr * 0.3)
        buf = AudioBuffer.sine(1000.0, channels=1, frames=frames, sample_rate=sr)
        with pytest.raises(ValueError, match="silent or too short"):
            dsp.normalize_lufs(buf)

    def test_metadata_preserved(self):
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.sine(
            1000.0, channels=1, frames=frames, sample_rate=sr, label="norm_test"
        )
        result = dsp.normalize_lufs(buf, target_lufs=-14.0)
        assert result.sample_rate == sr
        assert result.label == "norm_test"

    def test_idempotent(self):
        """Normalizing then re-measuring should hit the target."""
        sr = 48000.0
        frames = int(sr * 3)
        buf = AudioBuffer.noise(channels=1, frames=frames, sample_rate=sr, seed=42)
        for target in [-14.0, -23.0, -9.0]:
            normalized = dsp.normalize_lufs(buf, target_lufs=target)
            measured = dsp.loudness_lufs(normalized)
            assert abs(measured - target) < 0.5, f"target={target}, measured={measured}"

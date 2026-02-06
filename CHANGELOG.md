# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `cydsp.dsp` module -- high-level AudioBuffer-in/AudioBuffer-out processing functions
  - Filter functions (`lowpass`, `highpass`, `bandpass`, `notch`, `peak`, `peak_db`,
    `high_shelf`, `high_shelf_db`, `low_shelf`, `low_shelf_db`, `allpass`,
    `biquad_process`) with frequency parameters in Hz (auto-converted to normalized)
  - Delay functions (`delay`, `delay_varying`) with per-channel processing
  - Envelope functions (`box_filter`, `box_stack_filter`, `peak_hold`, `peak_decay`)
  - FFT functions (`rfft`, `irfft`) with automatic zero-padding to efficient FFT sizes
  - STFT functions (`stft`, `istft`) with Hann window and COLA overlap-add reconstruction
  - `Spectrogram` data class for STFT output (`[channels, frames, bins]` complex64)
  - `lfo` function wrapping `CubicLfo` with optional seed/variation parameters
  - Rate conversion (`upsample_2x`, `oversample_roundtrip`)
  - Mix functions (`hadamard`, `householder`, `crossfade`)
- `cydsp._core.pyi` -- complete type stubs for the C++ extension module (all 7 submodules)
- `cydsp.io` module -- WAV file I/O using stdlib `wave` (zero dependencies)
  - `read_wav`: reads 8/16/24/32-bit PCM WAV files to AudioBuffer (float32, planar)
  - `write_wav`: writes AudioBuffer to 16-bit or 24-bit PCM WAV files
  - Vectorized 24-bit encoding/decoding (no per-sample Python loops)
- Test suites for both new modules (78 new tests, 281 total)

### Changed

- Removed `disable_error_code = ["import-untyped"]` from mypy config (stubs fix this)

## [0.1.0]

- Initial project structure
- Core C++ bindings via nanobind: filters, fft, delay, envelopes, spectral, rates, mix
- `AudioBuffer` class (pure Python, 2D `[channels, frames]` float32 with metadata)
- Test suite with pytest (203 tests)
- Build system using scikit-build-core + CMake + uv

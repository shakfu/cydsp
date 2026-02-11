# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Pure NumPy DSP algorithms** -- 7 new functions added for API completeness without scipy dependency
  - `ops.xcorr(buf_a, buf_b=None)` -- FFT-based cross-correlation (or autocorrelation in single-arg form)
  - `ops.hilbert(buf)` -- amplitude envelope via analytic signal (FFT method)
  - `ops.envelope(buf)` -- alias for `hilbert`
  - `ops.median_filter(buf, kernel_size=3)` -- per-channel median filtering via stride tricks
  - `ops.lms_filter(buf, ref, filter_len=32, step_size=0.01, normalized=True)` -- NLMS adaptive filter returning `(output, error)`
  - `effects.agc(buf, target_level, max_gain_db, average_len, attack, release)` -- automatic gain control with asymmetric attack/release
  - `analysis.gcc_phat(buf, ref, sample_rate)` -- GCC-PHAT time-delay estimation returning `(delay_seconds, correlation)`

- **GrainflowLib bindings** (`cydsp._core.grainflow`) -- granular synthesis engine (header-only, MIT license)
  - `GfBuffer` -- buffer wrapper bridging numpy `[channels, frames]` arrays to GrainflowLib's internal AudioFile storage
  - `GrainCollection` -- core multi-grain granulator with block-based processing, parameter control (enum and string reflection), buffer assignment, stream management, and auto-overlap
  - `Panner` -- stereo grain panning with three modes (bipolar, unipolar, stereo) using equal-power quarter-sine interpolation
  - `Recorder` -- live recording into buffers with overdub, freeze, sync, and multi-band filter support
  - `Phasor` -- clock generator for grain triggering (continuous-phase ramp [0, 1))
  - 37 enum constants: `PARAM_*` (23 parameter names), `PTYPE_*` (5 parameter types), `STREAM_*` (4 stream modes), `BUF_*` (6 buffer types), `BUFMODE_*` (3 buffer modes), `PAN_*` (3 pan modes)
  - String-based parameter reflection (e.g. `"delayRandom"`, `"rateOffset"`, `"channelMode"`)
  - All processing methods release the GIL for thread safety
  - 49 tests (`tests/test_grainflow.py`)
  - Patched two GrainflowLib upstream bugs for `SigType=float`: `gf_utils::mod` template deduction failure, `stream` method vs member access

- **Demo scripts** (`demos/`) -- 16 runnable demo scripts showcasing the full API surface
  - `demo_filters.py` -- 13 biquad filter variants (lowpass, highpass, bandpass, notch, peak, shelving)
  - `demo_modulation.py` -- 10 modulation effects (chorus, flanger, phaser, tremolo)
  - `demo_distortion.py` -- 14 distortion/saturation effects (overdrive, wavefold, bitcrush, decimator, saturate, fold)
  - `demo_reverb.py` -- 12 reverb algorithms (FDN presets, ReverbSc, STK freeverb/jcrev/nrev/prcrev)
  - `demo_dynamics.py` -- 9 dynamics processors (compression, limiting, gating, parallel/multiband compression)
  - `demo_delay.py` -- 8 delay effects (stereo delay, ping-pong, slapback, echo)
  - `demo_pitch.py` -- 10 pitch shifters (time-domain and spectral at various intervals)
  - `demo_spectral.py` -- 12 spectral transforms (time stretch, phase lock, spectral gate, tilt EQ, freeze)
  - `demo_daisysp_filters.py` -- 21 DaisySP filter variants (SVF, ladder, moog, tone, modal, comb)
  - `demo_composed.py` -- 13 composed effects (autowah, sample rate reduce, DC block, exciter, de-esser, vocal chain, mastering, STK chorus)
  - `demo_spectral_extra.py` -- 8 additional spectral transforms (denoise, EQ match, spectral morph)
  - `demo_ops.py` -- 29 core DSP operations (delay, vibrato, convolution, envelopes, fades, panning, stereo widening, crossfade, normalization, trim, oversample)
  - `demo_resample.py` -- 6 resampling variants (madronalib and FFT methods at 22k/48k/96k)
  - `demo_synthesis.py` -- 44 synthesis sounds (oscillators, FM, formant, noise, drums, physical modeling, STK instruments, sequence) -- no input file required
  - `demo_analysis.py` -- audio analysis printout (loudness, spectral features, pitch detection, onset detection, chromagram) -- no audio output
  - `demo_grainflow.py` -- 7 granular synthesis variants (basic cloud, dense cloud, pitch shift up/down, sparse stochastic, stereo panned, recorder)
  - All file-processing scripts accept positional `infile`, optional `-o`/`--out-dir` (default `build/demo-output/`), and `-n`/`--no-normalize` to skip peak normalization
  - Peak normalization (0 dBFS) applied by default to prevent clipping on PCM output
- `make demos` target -- runs all 16 demo scripts in sequence (`DEMO_INPUT=demos/s01.wav` by default)

## [0.1.2]

### Changed

- **GIL release in C++ bindings** -- all ~160 processing functions across 6 binding files now release the Python GIL during computation via `nb::gil_scoped_release`, enabling true multi-threaded parallelism
  - `_core_signalsmith.cpp` -- Biquad, FFT, RealFFT, Delay, LFO, envelope, STFT, Oversampler processing
  - `_core_daisysp.cpp` -- 73 functions: oscillators, filters, effects, dynamics, control, noise, drums, physical modeling, utility
  - `_core_stk.cpp` -- generators, filters (via macro), reverbs (via macro), instruments (via macro), effects, Guitar, Twang
  - `_core_madronalib.cpp` -- `ml_process`/`ml_process_stereo`/`ml_process2` templates (propagates to FDN reverbs, delay, resampling, generators), projections, amp/dB conversions
  - `_core_hisstools.cpp` -- MonoConvolve, Convolver, SpectralProcessor (convolve/correlate/change_phase), KernelSmoother
  - `_core_choc.cpp` -- FLAC read/write file I/O

### Fixed

- **Cross-platform build** (Linux, macOS, Windows)
  - Linux: `CMAKE_POSITION_INDEPENDENT_CODE` for static libs linked into shared `.so`
  - Linux: Suppressed GCC `-Wmaybe-uninitialized` false positives from HISSTools `Statistics.hpp`
  - Linux: Dropped aarch64 wheels (HISSTools NEON code requires Apple Clang-specific implicit type conversions)
  - macOS: Set `MACOSX_DEPLOYMENT_TARGET=10.15` for `std::filesystem::path` and nanobind aligned deallocation
  - macOS: Architecture detection via compiler built-in defines (`__aarch64__`) instead of `CMAKE_SYSTEM_PROCESSOR` (correct under cross-compilation)
  - macOS: `cmake/hisstools_arch_compat.h` -- bridges `__aarch64__` (GCC/Linux) to `__arm64__` (Apple/HISSTools)
  - Windows: `NOMINMAX` and `_USE_MATH_DEFINES` for MSVC across all targets
  - Windows: `cmake/msvc_compat.h` -- `__attribute__` no-op and `<cmath>` includes for DaisySP
  - Python < 3.12: Guarded `AudioBuffer.__buffer__` (PEP 688) behind version check

- **CI/CD** (`.github/workflows/`)
  - `build-publish.yml` -- cibuildwheel v3.3.1 wheel builds for Linux x86_64, macOS arm64+x86_64, Windows AMD64; TestPyPI + PyPI publish via trusted publishing
  - `ci.yml` -- QA (ruff lint/format, mypy typecheck) + native build/test matrix (ubuntu/macOS/Windows, Python 3.10+3.14)
  - Cross-compile macOS x86_64 wheels from ARM64 runner (macos-latest); tests skipped for x86_64

## [0.1.1]

### Added

- **CLI** (`cydsp.__main__`, `cydsp._cli`)
  - `cydsp info <file>` -- audio file metadata (path, format, duration, sample_rate, channels, frames, peak_db, loudness_lufs), `--json` output
  - `cydsp process <inputs...> -o OUT|-O DIR` -- chainable effect pipeline with `--fx`/`-f` (repeatable) and `--preset`/`-p` (repeatable)
  - Batch mode: `cydsp process *.wav -O out/` processes multiple files to an output directory
  - Dry-run: `cydsp process in.wav -n -f lowpass:cutoff_hz=1000` shows the chain without reading or writing files
  - Global `-v`/`--verbose` flag for detailed step-by-step output, `-q`/`--quiet` to suppress non-essential output (mutually exclusive)
  - `cydsp analyze <file> <type>` -- 10 analysis subcommands (loudness, pitch, onsets, centroid, bandwidth, rolloff, flux, flatness, chromagram, info), `--json` output
  - `cydsp synth <out> <type>` -- 7 synth types (sine, noise, drum, oscillator, fm, note, sequence)
  - `cydsp convert <in> <out>` -- format conversion (WAV/FLAC), resampling (`--sample-rate`), channel conversion (`--channels`), bit depth (`-b`)
  - `cydsp pipe` -- read WAV from stdin, apply `-f`/`-p` effect chain, write WAV to stdout; supports Unix pipe chaining
  - `cydsp benchmark <function>` -- profile a DSP function with configurable iterations, warmup, buffer size; reports min/max/mean/median/std timing and realtime throughput multiplier, `--json` output
  - `cydsp preset list|info|apply` -- 30 presets across 8 categories (mastering, voice, spatial, dynamics, lofi, cleanup, creative)
  - `cydsp list [category]` -- browse all registered functions with signatures across 7 categories (filters, effects, dynamics, spectral, analysis, synthesis, ops)
  - 13 new presets: genre mastering (`master_pop`, `master_hiphop`, `master_classical`, `master_edm`, `master_podcast`), creative effects (`radio`, `underwater`, `megaphone`, `tape_warmth`, `shimmer`, `vaporwave`, `walkie_talkie`), lofi (`8bit`)
  - Function registry with auto-discovery from all modules, `inspect.signature`-based parameter display
  - Preset registry with single-function and chain-based presets, parameter overrides
  - FX token parser (`name:k=v,k=v`) with type coercion from signature defaults
  - `[project.scripts]` entry point: `cydsp` command

- **Audio I/O** (`cydsp.io`)
  - `read_wav_bytes(data)` -- parse WAV from raw bytes (for stdin/pipe workflows)
  - `write_wav_bytes(buf, bit_depth)` -- serialize AudioBuffer to WAV bytes (for stdout/pipe workflows)

- **CHOC FLAC codec** -- read/write FLAC files (16/24-bit) via header-only CHOC library
  - `cydsp._core.choc` C++ bindings for FLAC read/write
  - `io.read_flac()`, `io.write_flac()` Python wrappers
  - `io.read()`, `io.write()` auto-detect WAV vs FLAC by extension
  - Fixed CHOC upstream bug in 24-bit float-to-int scale factor

- **Streaming infrastructure** (`cydsp.stream`)
  - `RingBuffer` -- multi-channel ring buffer with independent read/write positions
  - `BlockProcessor` -- base class for block-based audio processors
  - `CallbackProcessor` -- wrap a callable as a block processor
  - `ProcessorChain` -- chain multiple processors in series
  - `process_blocks()` -- process a buffer through a function in blocks with optional overlap-add

- **DaisySP effects** (via `cydsp.effects`)
  - Effects: `autowah`, `chorus`, `decimator`, `flanger`, `overdrive`, `phaser`, `pitch_shift`, `sample_rate_reduce`, `tremolo`, `wavefold`, `bitcrush`, `fold`, `reverb_sc`, `dc_block`
  - Filters: `svf_lowpass`, `svf_highpass`, `svf_bandpass`, `svf_notch`, `svf_peak`, `ladder_filter`, `moog_ladder`, `tone_lowpass`, `tone_highpass`, `modal_bandpass`, `comb_filter`
  - Dynamics: `compress`, `limit`

- **DaisySP synthesis** (via `cydsp.synthesis`)
  - Oscillators: `oscillator`, `fm2`, `formant_oscillator`, `bl_oscillator`
  - Noise: `white_noise`, `clocked_noise`, `dust`
  - Drums: `analog_bass_drum`, `analog_snare_drum`, `hihat`, `synthetic_bass_drum`, `synthetic_snare_drum`
  - Physical modeling: `karplus_strong`, `modal_voice`, `string_voice`, `pluck`, `drip`

- **STK bindings** (`cydsp._core.stk`) -- 5 submodules, 39 classes
  - Instruments: `Clarinet`, `Flute`, `Brass`, `Bowed`, `Plucked`, `Sitar`, `StifKarp`, `Saxofony`, `Recorder`, `BlowBotl`, `BlowHole`, `Whistle`, `Guitar`, `Twang`
  - Generators: `SineWave`, `Noise`, `Blit`, `BlitSaw`, `BlitSquare`, `ADSR`, `Asymp`, `Envelope`, `Modulate`
  - Filters: `BiQuad`, `OnePole`, `OneZero`, `TwoPole`, `TwoZero`, `PoleZero`, `FormSwep`
  - Delays: `Delay`, `DelayA`, `DelayL`, `TapDelay`
  - Effects: `FreeVerb`, `JCRev`, `NRev`, `PRCRev`, `Echo`, `Chorus`, `PitShift`, `LentPitShift`
  - High-level wrappers: `stk_reverb`, `stk_chorus`, `stk_echo`, `synth_note`, `synth_sequence`

- **Madronalib bindings** (`cydsp._core.madronalib`) -- 7 submodules
  - FDN reverbs: `FDN4`, `FDN8`, `FDN16` with configurable delays, cutoffs, and feedback
  - Delays: `PitchbendableDelay`
  - Resampling: `Downsampler`, `Upsampler`
  - Generators: `OneShotGen`, `LinearGlide`, `SampleAccurateLinearGlide`, `TempoLock`
  - Projections: 18 easing functions (`smoothstep`, `bell`, `ease_in`, `ease_out`, etc.)
  - Windows: `hamming`, `blackman`, `flat_top`, `triangle`, `raised_cosine`, `rectangle`
  - Utilities: `amp_to_db`, `db_to_amp` (scalar and array overloads)

- **HISSTools bindings** (`cydsp._core.hisstools`) -- 4 submodules
  - Convolution: `MonoConvolve`, `Convolver` (multi-channel) with selectable latency modes
  - Spectral processing: `SpectralProcessor` (convolve, correlate, phase change), `KernelSmoother`
  - Analysis: 24 statistics functions (`stat_mean`, `stat_rms`, `stat_centroid`, `stat_kurtosis`, etc.), `PartialTracker`
  - Windows: 28 window functions (Hann, Blackman-Harris variants, Nuttall variants, flat-top variants, Kaiser, Tukey, etc.)

- **Spectral processing** (`cydsp.spectral`)
  - STFT/ISTFT with Hann window and COLA overlap-add reconstruction
  - Spectral utilities: `magnitude`, `phase`, `from_polar`, `apply_mask`, `spectral_gate`, `spectral_emphasis`, `bin_freq`, `freq_to_bin`
  - Spectral transforms: `time_stretch`, `phase_lock`, `spectral_freeze`, `spectral_morph`, `pitch_shift_spectral`, `spectral_denoise`
  - `eq_match` -- match spectral envelope between two buffers

- **Analysis** (`cydsp.analysis`)
  - Loudness: `loudness_lufs` (ITU-R BS.1770-4), `normalize_lufs`
  - Spectral features: `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_flux`, `spectral_flatness_curve`, `chromagram`
  - Pitch detection: `pitch_detect` (YIN algorithm)
  - Onset detection: `onset_detect` (spectral flux with peak picking)
  - Resampling: `resample` (madronalib backend), `resample_fft` (FFT-based)
  - Delay estimation: `gcc_phat` (GCC-PHAT)

- **Composed effects** (`cydsp.effects`)
  - `saturate` (soft/hard/tape modes), `exciter`, `de_esser`, `parallel_compress`
  - `noise_gate`, `stereo_delay` (with ping-pong mode), `multiband_compress`
  - `reverb` with FDN backend and presets (room, hall, plate, chamber, cathedral)
  - `master` -- mastering chain (dc_block, EQ, compress, limit, normalize_lufs)
  - `vocal_chain` -- vocal processing chain (de-esser, EQ, compress, limit, normalize)
  - `agc` -- automatic gain control with asymmetric attack/release

- **Core DSP operations** (`cydsp.ops`)
  - Delay: `delay`, `delay_varying`
  - Envelopes: `box_filter`, `box_stack_filter`, `peak_hold`, `peak_decay`
  - FFT: `rfft`, `irfft`
  - `convolve` -- FFT-based overlap-add convolution
  - Rate conversion: `upsample_2x`, `oversample_roundtrip`
  - Mixing: `hadamard`, `householder`, `crossfade`, `mix_buffers`
  - `lfo` -- cubic LFO with rate/depth variation
  - Normalization: `normalize_peak`, `trim_silence`, `fade_in`, `fade_out`
  - Stereo: `pan`, `mid_side_encode`, `mid_side_decode`, `stereo_widen`
  - Correlation: `xcorr` (FFT-based cross-/auto-correlation)
  - Analytic signal: `hilbert`, `envelope`
  - Filtering: `median_filter`, `lms_filter`

- **Biquad filter wrappers** (`cydsp.effects`)
  - `lowpass`, `highpass`, `bandpass`, `notch`, `peak`, `peak_db`
  - `high_shelf`, `high_shelf_db`, `low_shelf`, `low_shelf_db`, `allpass`
  - `biquad_process` -- process through a pre-configured Biquad
  - All accept frequency in Hz with automatic normalization

- **AudioBuffer I/O methods**
  - `AudioBuffer.from_file(path)` -- classmethod to read WAV/FLAC by extension
  - `buf.write(path, bit_depth=16)` -- instance method to write WAV/FLAC by extension
- `cydsp._core.pyi` -- complete type stubs for all 12 C++ submodules
- `Spectrogram` data class for STFT output (`[channels, frames, bins]` complex64)

### Changed

- **Module split** -- monolithic `dsp.py` replaced by focused modules:
  - `_helpers.py` -- shared private utilities
  - `ops.py` -- delay, envelopes, FFT, convolution, rates, mix, pan, normalization
  - `effects.py` -- filters, effects, dynamics, reverb, mastering chains
  - `spectral.py` -- STFT, spectral utilities, spectral transforms, eq_match
  - `synthesis.py` -- oscillators, noise, drums, physical modeling, STK synth
  - `analysis.py` -- loudness, spectral features, pitch/onset detection, resampling
- `__init__.py` stripped to `__version__` only -- no re-exports; use explicit imports
- `io.py` now supports both WAV and FLAC formats
- Test suite reorganized into per-module test files (1114 tests)
- Removed `disable_error_code = ["import-untyped"]` from mypy config (stubs fix this)

## [0.1.0]

### Added

- Initial project structure with scikit-build-core + CMake + uv
- Core C++ bindings via nanobind (`cydsp._core`):
  - `filters` -- `Biquad` with 16 filter designs, `BiquadDesign` enum
  - `fft` -- `FFT` (complex-to-complex), `RealFFT` (real-to-complex)
  - `delay` -- `Delay` (linear interpolation), `DelayCubic` (cubic interpolation)
  - `envelopes` -- `CubicLfo`, `BoxFilter`, `BoxStackFilter`, `PeakHold`, `PeakDecayLinear`
  - `spectral` -- `STFT` (multi-channel analysis/synthesis)
  - `rates` -- `Oversampler2x`
  - `mix` -- `Hadamard`, `Householder`, `cheap_energy_crossfade`
- `AudioBuffer` class (pure Python, 2D `[channels, frames]` float32 with metadata)
  - Factory methods: `zeros`, `ones`, `impulse`, `sine`, `noise`, `from_numpy`
  - Channel operations: `to_mono`, `to_channels`, `split`, `concat_channels`
  - Arithmetic operators: `+`, `-`, `*`, `/`, `gain_db`
  - Pipeline: `pipe()` for chaining DSP functions
- `io.read_wav()`, `io.write_wav()` -- WAV file I/O (8/16/24/32-bit PCM, stdlib `wave`)
- Test suite with pytest (203 tests)

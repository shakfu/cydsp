"""
cydsp - Python DSP bindings via nanobind.

Submodules:
    cydsp.filters   - Biquad IIR filters
    cydsp.fft       - FFT (complex and real)
    cydsp.delay     - Delay line utilities
    cydsp.envelopes - Envelopes, LFOs, smoothing filters
    cydsp.spectral  - STFT and spectral processing
    cydsp.rates     - Multi-rate processing (oversampling)
    cydsp.mix       - Multichannel mixing utilities
    cydsp.daisysp   - DaisySP bindings (oscillators, filters, effects, drums, etc.)
    cydsp.stk       - STK bindings (physical models, reverbs, biquad filters, etc.)
    cydsp.madronalib - Madronalib DSP bindings (FDN reverbs, projections, windows, etc.)
    cydsp.hisstools - HISSTools Library (SIMD convolution, spectral, analysis, windows)
    cydsp.dsp       - High-level AudioBuffer processing functions
    cydsp.io        - WAV file I/O
"""

from cydsp._core import add, greet
from cydsp._core import filters, fft, delay, envelopes, spectral, rates, mix, daisysp, stk, madronalib, hisstools
from cydsp.buffer import AudioBuffer
from cydsp import dsp, io

__all__ = [
    "add",
    "greet",
    "filters",
    "fft",
    "delay",
    "envelopes",
    "spectral",
    "rates",
    "mix",
    "daisysp",
    "stk",
    "madronalib",
    "hisstools",
    "AudioBuffer",
    "dsp",
    "io",
]
__version__ = "0.1.0"

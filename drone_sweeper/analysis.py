"""IQ analysis utilities for Drone Sweeper.

This module provides a rolling FFT analyzer that consumes IQ samples and
produces decoded metadata for detected peaks.  The implementation is designed
for field deployments where ``hackrf_transfer`` (or an equivalent IQ streaming
utility) feeds raw complex samples into the analyzer.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Sequence

import numpy as np
from scipy.signal import find_peaks, get_window

logger = logging.getLogger(__name__)


@dataclass
class AnalyzerConfig:
    """Configuration for :class:`RollingFFTAnalyzer`.

    Attributes
    ----------
    sample_rate:
        The complex sample rate of the captured IQ stream in Hertz.
    window_size:
        Number of samples included in each FFT window.  Larger windows improve
        frequency resolution at the cost of latency.
    overlap:
        Window overlap ratio (0.0-0.99).  The hop size is derived from the
        window size and overlap.
    window:
        Name of the windowing function supplied to :func:`scipy.signal.get_window`.
    min_peak_height_db:
        Minimum power (in dBFS) that a candidate peak must exceed to be
        considered for decoding.
    min_peak_distance_hz:
        Minimum spacing (in Hertz) between detected peaks to avoid duplicate
        detections.
    enable_am / enable_fm / enable_fsk:
        Enable or disable the corresponding demodulators.
    export_payload:
        When ``True`` the analyzer will embed compact payload summaries in the
        decoded metadata.
    """

    sample_rate: float
    window_size: int = 4096
    overlap: float = 0.5
    window: str = "hann"
    min_peak_height_db: float = -55.0
    min_peak_distance_hz: float = 12_500.0
    enable_am: bool = True
    enable_fm: bool = True
    enable_fsk: bool = True
    export_payload: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("Analyzer overlap must be within [0.0, 1.0)")
        if self.window_size <= 0:
            raise ValueError("Analyzer window_size must be positive")
        if self.sample_rate <= 0:
            raise ValueError("Analyzer sample_rate must be positive")


@dataclass
class DecodedMetadata:
    """Represents the result of a demodulation stage."""

    modulation: str
    confidence: float
    bandwidth_hz: float
    payload: str | None = None
    extra: Dict[str, float | int | str] = field(default_factory=dict)


@dataclass
class AnalyzedPeak:
    """Metadata describing a detected spectral peak."""

    frequency_hz: float
    power_db: float
    window_start_sample: int
    timestamp: datetime
    metadata: List[DecodedMetadata] = field(default_factory=list)


class RollingFFTAnalyzer:
    """Performs rolling FFT analysis and modulation-specific decoding."""

    def __init__(self, config: AnalyzerConfig) -> None:
        self.config = config
        self.window = get_window(config.window, config.window_size)
        hop = int(round(config.window_size * (1.0 - config.overlap)))
        self.hop_size = max(1, hop)
        logger.debug(
            "Initialized RollingFFTAnalyzer with window=%s hop_size=%d", config.window, self.hop_size
        )

    def process(
        self,
        iq_samples: np.ndarray,
        center_frequency_hz: float,
        sweep_ranges: Sequence[tuple[float, float]],
        *,
        timestamp: datetime | None = None,
    ) -> List[AnalyzedPeak]:
        """Analyze ``iq_samples`` and return decoded peaks.

        Parameters
        ----------
        iq_samples:
            Complex64 array of baseband samples centred on ``center_frequency_hz``.
        center_frequency_hz:
            Tuned centre frequency used while capturing the samples.
        sweep_ranges:
            Frequency ranges (absolute Hz) that should be considered during peak
            selection.  Peaks falling outside these ranges are discarded.
        timestamp:
            Reference timestamp associated with the IQ capture.  If omitted the
            current UTC time is used.
        """

        if iq_samples.ndim != 1:
            raise ValueError("IQ samples must be a one-dimensional array")
        if iq_samples.size < self.config.window_size:
            logger.debug(
                "Skipping analysis block with insufficient samples (got %d, need %d)",
                iq_samples.size,
                self.config.window_size,
            )
            return []

        ts = timestamp or datetime.now(timezone.utc)
        window_size = self.config.window_size
        hop_size = self.hop_size
        sample_rate = self.config.sample_rate

        window = self.window
        time_axis = np.arange(window_size, dtype=np.float64) / sample_rate
        results: List[AnalyzedPeak] = []

        for start in range(0, iq_samples.size - window_size + 1, hop_size):
            segment = iq_samples[start : start + window_size]
            windowed = segment * window
            spectrum = np.fft.fft(windowed)
            magnitude = 20.0 * np.log10(np.abs(spectrum) + 1e-12)
            freqs = np.fft.fftfreq(window_size, d=1.0 / sample_rate)

            spectrum = np.fft.fftshift(magnitude)
            freqs = np.fft.fftshift(freqs)
            abs_freqs = freqs + center_frequency_hz

            bins_per_hz = window_size / sample_rate
            min_distance_bins = max(1, int(self.config.min_peak_distance_hz * bins_per_hz))

            peaks, properties = find_peaks(
                spectrum,
                height=self.config.min_peak_height_db,
                distance=min_distance_bins,
            )
            logger.debug(
                "Window start=%d detected %d peak(s) (threshold %.1f dB)",
                start,
                len(peaks),
                self.config.min_peak_height_db,
            )

            for offset, peak_idx in enumerate(peaks):
                frequency_hz = float(abs_freqs[peak_idx])
                if not _within_ranges(frequency_hz, sweep_ranges):
                    continue
                power_db = float(properties["peak_heights"][offset])
                baseband = segment * np.exp(-2j * math.pi * freqs[peak_idx] * time_axis)
                metadata = self._decode(baseband)
                if not metadata:
                    continue
                results.append(
                    AnalyzedPeak(
                        frequency_hz=frequency_hz,
                        power_db=power_db,
                        window_start_sample=start,
                        timestamp=ts,
                        metadata=metadata,
                    )
                )
        return results

    def _decode(self, baseband: np.ndarray) -> List[DecodedMetadata]:
        metadata: List[DecodedMetadata] = []
        if self.config.enable_am:
            metadata.append(_demodulate_am(baseband, self.config))
        if self.config.enable_fm:
            metadata.append(_demodulate_fm(baseband, self.config))
        if self.config.enable_fsk:
            metadata.append(_demodulate_fsk(baseband, self.config))
        return [entry for entry in metadata if entry.confidence > 0.0]


def _within_ranges(frequency: float, ranges: Sequence[tuple[float, float]]) -> bool:
    for start, stop in ranges:
        low, high = sorted((start, stop))
        if low <= frequency <= high:
            return True
    return False


def _demodulate_am(baseband: np.ndarray, config: AnalyzerConfig) -> DecodedMetadata:
    envelope = np.abs(baseband)
    mean = float(np.mean(envelope))
    rms = float(np.sqrt(np.mean((envelope - mean) ** 2)))
    confidence = min(1.0, rms / (mean + 1e-12)) if mean else 0.0
    payload = None
    if config.export_payload:
        payload = _serialize_series(envelope[:256])
    return DecodedMetadata(
        modulation="AM",
        confidence=confidence,
        bandwidth_hz=config.sample_rate / 2.0,
        payload=payload,
        extra={"mean_envelope": round(mean, 6), "rms": round(rms, 6)},
    )


def _demodulate_fm(baseband: np.ndarray, config: AnalyzerConfig) -> DecodedMetadata:
    if baseband.size < 2:
        return DecodedMetadata("FM", 0.0, config.sample_rate / 2.0)
    phase = np.unwrap(np.angle(baseband))
    diff = np.diff(phase)
    deviation = float(np.std(diff))
    confidence = min(1.0, deviation / math.pi)
    payload = None
    if config.export_payload:
        payload = _serialize_series(diff[:256])
    return DecodedMetadata(
        modulation="FM",
        confidence=confidence,
        bandwidth_hz=config.sample_rate / 2.0,
        payload=payload,
        extra={"phase_deviation": round(deviation, 6)},
    )


def _demodulate_fsk(baseband: np.ndarray, config: AnalyzerConfig) -> DecodedMetadata:
    if baseband.size < 2:
        return DecodedMetadata("FSK", 0.0, config.sample_rate / 2.0)
    phase = np.unwrap(np.angle(baseband))
    freq_dev = np.diff(phase)
    threshold = float(np.median(np.abs(freq_dev)))
    bits = (freq_dev > threshold).astype(np.uint8)
    transitions = int(np.count_nonzero(np.diff(bits)))
    confidence = min(1.0, transitions / max(1, baseband.size // 16))
    payload = None
    if config.export_payload:
        payload = ''.join(map(str, bits[:64]))
    return DecodedMetadata(
        modulation="FSK",
        confidence=confidence,
        bandwidth_hz=config.sample_rate / 2.0,
        payload=payload,
        extra={"transitions": transitions},
    )


def _serialize_series(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=np.float64)
    return ",".join(f"{value:.6f}" for value in values)


__all__ = [
    "AnalyzerConfig",
    "DecodedMetadata",
    "AnalyzedPeak",
    "RollingFFTAnalyzer",
]

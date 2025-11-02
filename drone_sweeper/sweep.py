"""Helpers for running and parsing ``hackrf_sweep`` output."""

from __future__ import annotations

import json
import logging
import math
import shlex
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from .analysis import DecodedMetadata, RollingFFTAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class SweepFrame:
    """Represents a single line of ``hackrf_sweep`` CSV output."""

    timestamp: datetime
    start_frequency_hz: int
    stop_frequency_hz: int
    bin_size_hz: int
    power_db: List[float]

    @property
    def bin_count(self) -> int:
        return len(self.power_db)

    def frequency_for_bin(self, index: int) -> float:
        return self.start_frequency_hz + index * self.bin_size_hz


@dataclass
class SignalOfInterest:
    frequency_hz: float
    label: str
    tolerance_hz: float


def parse_sweep_line(line: str) -> SweepFrame:
    parts = [part.strip() for part in line.split(",") if part.strip()]
    if len(parts) < 6:
        raise ValueError(f"Unexpected hackrf_sweep output line: {line!r}")

    timestamp = datetime.now(timezone.utc)
    idx = 0
    if _looks_like_date(parts[0]) and _looks_like_time(parts[1]):
        timestamp = _parse_timestamp(parts[0], parts[1])
        idx = 2

    start_frequency_hz = int(float(parts[idx])); idx += 1
    stop_frequency_hz = int(float(parts[idx])); idx += 1
    bin_size_hz = int(float(parts[idx])); idx += 1
    bin_count = int(float(parts[idx])); idx += 1
    power = [float(value) for value in parts[idx:idx + bin_count]]
    if len(power) != bin_count:
        raise ValueError("Incomplete power bin data in hackrf_sweep output")

    return SweepFrame(
        timestamp=timestamp,
        start_frequency_hz=start_frequency_hz,
        stop_frequency_hz=stop_frequency_hz,
        bin_size_hz=bin_size_hz,
        power_db=power,
    )


def _looks_like_date(value: str) -> bool:
    return value.count("-") == 2 and len(value) >= 8


def _looks_like_time(value: str) -> bool:
    return value.count(":") >= 2


def _parse_timestamp(date_str: str, time_str: str) -> datetime:
    try:
        if "." in time_str:
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S.%f")
        else:
            dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        logger.debug("Falling back to current time for timestamp parsing of %s %s", date_str, time_str)
        return datetime.now(timezone.utc)


class HackRFSweepRunner:
    """Runs ``hackrf_sweep`` and yields parsed frames."""

    def __init__(
        self,
        frequency_ranges: Sequence[str],
        additional_args: Optional[Sequence[str]] = None,
        *,
        bin_width: Optional[int] = None,
        sample_rate: Optional[int] = None,
    ) -> None:
        self.frequency_ranges = frequency_ranges
        self.additional_args = list(additional_args or [])
        self.bin_width = bin_width
        self.sample_rate = sample_rate
        self._process: Optional[subprocess.Popen[str]] = None

    def build_command(self) -> List[str]:
        command = ["hackrf_sweep", "--format", "csv"]
        for freq_range in self.frequency_ranges:
            command.extend(["-f", freq_range])
        if self.bin_width:
            command.extend(["-w", str(self.bin_width)])
        if self.sample_rate:
            command.extend(["-r", str(self.sample_rate)])
        command.extend(self.additional_args)
        logger.debug("Running command: %s", shlex.join(command))
        return command

    def frames(self) -> Iterator[SweepFrame]:
        command = self.build_command()
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self._process.stdout is not None
        for line in self._process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                yield parse_sweep_line(line)
            except Exception:
                logger.exception("Unable to parse hackrf_sweep output line: %s", line)

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()


@dataclass
class Detection:
    signal: SignalOfInterest
    frequency_hz: float
    power_db: float
    bin_index: int
    analysis: List["DecodedMetadata"] = field(default_factory=list)


def detect_signals(frame: SweepFrame, signals: Iterable[SignalOfInterest]) -> List[Detection]:
    detections: List[Detection] = []
    for signal in signals:
        index = _bin_index_for_frequency(frame, signal.frequency_hz)
        if index is None:
            continue
        frequency = frame.frequency_for_bin(index)
        if abs(frequency - signal.frequency_hz) > signal.tolerance_hz:
            continue
        power_db = frame.power_db[index]
        detections.append(
            Detection(
                signal=signal,
                frequency_hz=frequency,
                power_db=power_db,
                bin_index=index,
            )
        )
    return detections


def _bin_index_for_frequency(frame: SweepFrame, frequency_hz: float) -> Optional[int]:
    if frequency_hz < frame.start_frequency_hz or frequency_hz > frame.stop_frequency_hz:
        return None
    offset = frequency_hz - frame.start_frequency_hz
    index = int(math.floor(offset / frame.bin_size_hz))
    if index < 0 or index >= frame.bin_count:
        return None
    return index


def write_detections_csv(path: Path, frame: SweepFrame, detections: Sequence[Detection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header_needed = not path.exists()
    with path.open("a", encoding="utf-8") as fp:
        if header_needed:
            fp.write(
                "timestamp,start_frequency_hz,stop_frequency_hz,bin_size_hz,label,center_frequency_hz,detected_frequency_hz,power_db,bin_index,analysis_modulations,analysis_payloads,analysis_metadata\n"
            )
        for detection in detections:
            modulations = ";".join(meta.modulation for meta in detection.analysis)
            payloads = ";".join(meta.payload or "" for meta in detection.analysis if meta.payload)
            metadata = [
                {
                    "modulation": meta.modulation,
                    "confidence": meta.confidence,
                    "bandwidth_hz": meta.bandwidth_hz,
                    "payload": meta.payload,
                    "extra": meta.extra,
                }
                for meta in detection.analysis
            ]
            metadata_json = json.dumps(metadata, separators=(",", ":")) if metadata else ""
            fp.write(
                f"{frame.timestamp.isoformat()},{frame.start_frequency_hz},{frame.stop_frequency_hz},{frame.bin_size_hz},{detection.signal.label},{detection.signal.frequency_hz},{detection.frequency_hz},{detection.power_db:.2f},{detection.bin_index},{modulations},{payloads},{metadata_json}\n"
            )


class IQStreamRunner:
    """Captures IQ samples around a detection and forwards them to an analyzer."""

    def __init__(
        self,
        analyzer: "RollingFFTAnalyzer",
        *,
        sample_rate: int,
        capture_seconds: float = 0.25,
        additional_args: Optional[Sequence[str]] = None,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive for IQ capture")
        if capture_seconds <= 0:
            raise ValueError("capture_seconds must be positive")
        self.analyzer = analyzer
        self.sample_rate = sample_rate
        self.capture_seconds = capture_seconds
        self.additional_args = list(additional_args or [])

    def capture_and_analyze(
        self,
        detection: Detection,
        ranges_hz: Sequence[tuple[float, float]],
        *,
        tolerance_hz: Optional[float] = None,
    ) -> List["DecodedMetadata"]:
        """Capture IQ data for ``detection`` and return decoded metadata."""

        try:
            iq_samples = self._capture_samples(detection.frequency_hz)
        except FileNotFoundError:
            logger.error(
                "hackrf_transfer is not available on the system; unable to run IQ analysis"
            )
            return []
        except Exception:  # pragma: no cover - hardware interaction
            logger.exception("Unable to capture IQ samples for analysis")
            return []

        if iq_samples.size == 0:
            return []

        peaks = self.analyzer.process(
            iq_samples,
            center_frequency_hz=detection.frequency_hz,
            sweep_ranges=ranges_hz,
        )
        if not peaks:
            return []

        tol = tolerance_hz or self.sample_rate / max(1, self.analyzer.config.window_size)
        metadata: List["DecodedMetadata"] = []
        for peak in peaks:
            if abs(peak.frequency_hz - detection.frequency_hz) <= tol:
                metadata.extend(peak.metadata)
        return metadata

    def _capture_samples(self, center_frequency_hz: float) -> "np.ndarray":
        import numpy as np

        command = [
            "hackrf_transfer",
            "-f",
            str(int(center_frequency_hz)),
            "-s",
            str(int(self.sample_rate)),
            "-r",
            "-",
            "-n",
            str(int(self.sample_rate * self.capture_seconds)),
        ]
        command.extend(self.additional_args)
        logger.debug("Running IQ capture command: %s", shlex.join(command))
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert process.stdout is not None

        bytes_needed = int(self.sample_rate * self.capture_seconds * 2)
        buffer = bytearray()
        while len(buffer) < bytes_needed:
            chunk = process.stdout.read(bytes_needed - len(buffer))
            if not chunk:
                break
            buffer.extend(chunk)

        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:  # pragma: no cover - defensive
            process.kill()

        if len(buffer) < 2:
            return np.array([], dtype=np.complex64)

        raw = np.frombuffer(buffer, dtype=np.int8).astype(np.float32)
        if raw.size % 2:
            raw = raw[:-1]
        i_samples = raw[0::2] / 127.5
        q_samples = raw[1::2] / 127.5
        return (i_samples + 1j * q_samples).astype(np.complex64)


__all__ = [
    "SweepFrame",
    "SignalOfInterest",
    "HackRFSweepRunner",
    "IQStreamRunner",
    "Detection",
    "detect_signals",
    "write_detections_csv",
    "parse_sweep_line",
]

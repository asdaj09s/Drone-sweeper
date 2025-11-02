"""Command line interface for Drone Sweeper field capture."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence

from .gps import GPSReader
from .events import DetectionEvent
from .models import ClassificationResult, classify_detection
from .sweep import (
    HackRFSweepRunner,
    IQStreamRunner,
    SignalOfInterest,
    Detection,
    SweepFrame,
    detect_signals,
    write_detections_csv,
)
from .tdoa import TDOAEvent, TDOALogger, default_sensor_id, monotonic_time_ns

from .analysis import AnalyzerConfig, RollingFFTAnalyzer, DecodedMetadata


LOGGER = logging.getLogger("drone_sweeper")


def parse_frequency(value: str) -> float:
    value = value.strip().lower()
    multipliers = {
        "hz": 1.0,
        "khz": 1e3,
        "mhz": 1e6,
        "ghz": 1e9,
    }
    for suffix, multiplier in multipliers.items():
        if value.endswith(suffix):
            return float(value[: -len(suffix)]) * multiplier
    if value.endswith("k"):
        return float(value[:-1]) * 1e3
    if value.endswith("m"):
        return float(value[:-1]) * 1e6
    if value.endswith("g"):
        return float(value[:-1]) * 1e9
    return float(value)


def parse_sweep_range(value: str) -> tuple[float, float]:
    try:
        start_str, stop_str = value.split(":", 1)
    except ValueError as exc:  # pragma: no cover - defensive
        raise argparse.ArgumentTypeError(
            "Sweep ranges must be formatted as start:end"
        ) from exc
    start = parse_frequency(start_str)
    stop = parse_frequency(stop_str)
    if start >= stop:
        raise argparse.ArgumentTypeError("Sweep range start must be less than stop")
    return (start, stop)


def parse_signal_argument(value: str, default_tolerance_hz: float) -> SignalOfInterest:
    parts = value.split(":")
    if len(parts) == 1:
        label = value
        freq_str = value
        tol_str = None
    elif len(parts) == 2:
        label, freq_str = parts
        tol_str = None
    elif len(parts) == 3:
        label, freq_str, tol_str = parts
    else:
        raise argparse.ArgumentTypeError(
            "Signals must be specified as label:frequency[:tolerance]"
        )

    frequency_hz = parse_frequency(freq_str)
    tolerance_hz = parse_frequency(tol_str) if tol_str else default_tolerance_hz
    label = label or f"{frequency_hz/1e6:.3f}MHz"
    return SignalOfInterest(
        frequency_hz=frequency_hz,
        label=label,
        tolerance_hz=tolerance_hz,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-range",
        dest="sweep_ranges",
        action="append",
        required=True,
        help="Frequency range passed to hackrf_sweep in the form start:end (Hz)",
    )
    parser.add_argument(
        "--signal",
        dest="signals",
        action="append",
        default=[],
        help="Signal of interest as label:frequency[:tolerance] (e.g. dji:915MHz:500kHz)",
    )
    parser.add_argument(
        "--default-tolerance",
        type=parse_frequency,
        default=25000.0,
        help="Fallback tolerance (Hz) when not specified per signal",
    )
    parser.add_argument(
        "--min-power",
        type=float,
        default=-40.0,
        help="Minimum power (dB) required to register a detection",
    )
    parser.add_argument(
        "--csv-log",
        type=Path,
        help="Optional CSV file to append detections to",
    )
    parser.add_argument(
        "--tdoa-log",
        type=Path,
        help="Optional JSONL file to append TDOA events to",
    )
    parser.add_argument(
        "--sensor-id",
        default=default_sensor_id(),
        help="Identifier for this sensor in the TDOA log",
    )
    parser.add_argument(
        "--gps-port",
        help="Serial port path for the GPS puck (e.g. /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--gps-baudrate",
        type=int,
        default=9600,
        help="Serial baudrate for the GPS puck",
    )
    parser.add_argument(
        "--bin-width",
        type=int,
        help="Optional bin width to pass to hackrf_sweep (-w)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        help="Optional sample rate to pass to hackrf_sweep (-r)",
    )
    parser.add_argument(
        "--hackrf-arg",
        dest="hackrf_args",
        action="append",
        default=[],
        help="Additional raw arguments to forward to hackrf_sweep",
    )
    parser.add_argument(
        "--iq-arg",
        dest="iq_args",
        action="append",
        default=[],
        help="Additional arguments to forward to hackrf_transfer",
    )
    parser.add_argument(
        "--enable-analyzer",
        action="store_true",
        help="Enable IQ streaming analysis for detected signals",
    )
    parser.add_argument(
        "--iq-sample-rate",
        type=int,
        default=10_000_000,
        help="Sample rate (Hz) to use when capturing IQ data",
    )
    parser.add_argument(
        "--iq-capture-seconds",
        type=float,
        default=0.25,
        help="Duration of each IQ capture per detection (seconds)",
    )
    parser.add_argument(
        "--analyzer-window-size",
        type=int,
        default=4096,
        help="FFT window size for the IQ analyzer",
    )
    parser.add_argument(
        "--analyzer-overlap",
        type=float,
        default=0.5,
        help="Window overlap ratio (0-1) for the IQ analyzer",
    )
    parser.add_argument(
        "--analyzer-window",
        default="hann",
        help="Windowing function name for the IQ analyzer",
    )
    parser.add_argument(
        "--analyzer-min-peak",
        type=float,
        default=-55.0,
        help="Minimum peak height (dB) required by the IQ analyzer",
    )
    parser.add_argument(
        "--analyzer-min-distance",
        type=float,
        default=12_500.0,
        help="Minimum peak separation (Hz) for the IQ analyzer",
    )
    parser.add_argument(
        "--disable-am",
        action="store_true",
        help="Disable AM demodulation in the IQ analyzer",
    )
    parser.add_argument(
        "--disable-fm",
        action="store_true",
        help="Disable FM demodulation in the IQ analyzer",
    )
    parser.add_argument(
        "--disable-fsk",
        action="store_true",
        help="Disable FSK demodulation in the IQ analyzer",
    )
    parser.add_argument(
        "--analysis-export-payloads",
        action="store_true",
        help="Serialize compact payload samples in analyzer output",
    )
    parser.add_argument(
        "--analysis-jsonl",
        type=Path,
        help="Optional JSONL file to append analyzer results to",
    )
    parser.add_argument(
        "--cot-log",
        type=Path,
        help="Optional Cursor-on-Target XML log file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Emit detections as JSON on stdout",
    )
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_signals(values: Sequence[str], default_tolerance_hz: float) -> List[SignalOfInterest]:
    if not values:
        raise SystemExit("At least one --signal is required to run detections")
    return [parse_signal_argument(value, default_tolerance_hz) for value in values]


def _build_feature_vector(frame: SweepFrame, detection: Detection) -> List[float]:
    span_hz = float(frame.stop_frequency_hz - frame.start_frequency_hz) or 1.0
    relative_frequency = (detection.frequency_hz - frame.start_frequency_hz) / span_hz
    frequency_offset = detection.frequency_hz - detection.signal.frequency_hz
    return [
        float(detection.power_db),
        float(frequency_offset),
        float(relative_frequency),
        float(frame.bin_size_hz),
    ]


def _summarise_classification(result: ClassificationResult) -> str:
    confidence_pct = result.confidence * 100.0
    return f"{result.label} ({confidence_pct:.1f}% confidence)"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    signals = ensure_signals(args.signals, args.default_tolerance)
    LOGGER.info("Loaded %d signal(s) of interest", len(signals))

    try:
        sweep_ranges_numeric = [parse_sweep_range(value) for value in args.sweep_ranges]
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc))

    gps_reader: Optional[GPSReader] = None
    if args.gps_port:
        LOGGER.info("Starting GPS reader on %s", args.gps_port)
        gps_reader = GPSReader(args.gps_port, baudrate=args.gps_baudrate)
        try:
            gps_reader.start()
        except Exception as exc:  # pragma: no cover - hardware specific
            LOGGER.error("Unable to start GPS reader: %s", exc)
            gps_reader = None

    analyzer: Optional[RollingFFTAnalyzer] = None
    iq_runner: Optional[IQStreamRunner] = None
    if args.enable_analyzer:
        try:
            analyzer_config = AnalyzerConfig(
                sample_rate=float(args.iq_sample_rate),
                window_size=args.analyzer_window_size,
                overlap=args.analyzer_overlap,
                window=args.analyzer_window,
                min_peak_height_db=args.analyzer_min_peak,
                min_peak_distance_hz=args.analyzer_min_distance,
                enable_am=not args.disable_am,
                enable_fm=not args.disable_fm,
                enable_fsk=not args.disable_fsk,
                export_payload=args.analysis_export_payloads,
            )
        except ValueError as exc:
            raise SystemExit(str(exc))
        analyzer = RollingFFTAnalyzer(analyzer_config)
        iq_runner = IQStreamRunner(
            analyzer,
            sample_rate=int(analyzer_config.sample_rate),
            capture_seconds=args.iq_capture_seconds,
            additional_args=args.iq_args,
        )
        LOGGER.info(
            "IQ analyzer enabled (window=%d overlap=%.2f sample_rate=%.0f)",
            analyzer_config.window_size,
            analyzer_config.overlap,
            analyzer_config.sample_rate,
        )

    sweep_runner = HackRFSweepRunner(
        args.sweep_ranges,
        args.hackrf_args,
        bin_width=args.bin_width,
        sample_rate=args.sample_rate,
    )

    csv_path = args.csv_log
    analysis_jsonl_path = args.analysis_jsonl
    cot_path = args.cot_log
    tdoa_logger = TDOALogger(args.tdoa_log) if args.tdoa_log else None

    def shutdown_handler(signum, frame):  # pragma: no cover - signal handler
        LOGGER.info("Received signal %s, shutting down", signum)
        sweep_runner.stop()
        if gps_reader:
            gps_reader.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        for frame in sweep_runner.frames():
            detections = [d for d in detect_signals(frame, signals) if d.power_db >= args.min_power]
            if not detections:
                continue

            if iq_runner:
                for detection in detections:
                    try:
                        detection.analysis = iq_runner.capture_and_analyze(
                            detection,
                            sweep_ranges_numeric,
                            tolerance_hz=detection.signal.tolerance_hz,
                        )
                    except Exception:  # pragma: no cover - hardware interaction
                        LOGGER.exception(
                            "Failed to analyze detection %s", detection.signal.label
                        )

            gps_fix = gps_reader.latest_fix() if gps_reader else None
            monotonic_ns = monotonic_time_ns()

            if csv_path:
                write_detections_csv(csv_path, frame, detections)

            for detection in detections:
                features = _build_feature_vector(frame, detection)
                metadata = {
                    "signal_label": detection.signal.label,
                    "frequency_hz": detection.frequency_hz,
                    "sensor_id": args.sensor_id,
                }
                try:
                    classification = classify_detection(features, metadata)
                except Exception:  # pragma: no cover - defensive logging
                    LOGGER.exception("Classifier failed, marking detection as unknown")
                    classification = ClassificationResult(
                        label="unknown",
                        confidence=0.0,
                        probabilities={},
                        model_name=None,
                        model_version=None,
                    )

                detection_event = DetectionEvent(
                    timestamp=frame.timestamp,
                    label=detection.signal.label,
                    frequency_hz=detection.frequency_hz,
                    power_db=detection.power_db,
                    sensor_id=args.sensor_id,
                    gps=gps_fix,
                    ai_label=classification.label,
                    ai_confidence=classification.confidence,
                    ai_probabilities=classification.probabilities,
                    model_name=classification.model_name,
                    model_version=classification.model_version,
                )

                if args.print_json:
                    print(detection_event.to_json(), flush=True)
                else:
                    LOGGER.info(
                        "Detection %s at %.0f Hz (%.1f dB) -> %s",
                        detection.signal.label,
                        detection.frequency_hz,
                        detection.power_db,
                        _summarise_classification(classification),
                    )

                if tdoa_logger:
                    tdoa_event = TDOAEvent(
                        detection_time_utc=frame.timestamp,
                        detection_time_monotonic_ns=monotonic_ns,
                        sensor_id=args.sensor_id,
                        frequency_hz=detection.frequency_hz,
                        power_db=detection.power_db,
                        sweep_start_frequency_hz=frame.start_frequency_hz,
                        sweep_stop_frequency_hz=frame.stop_frequency_hz,
                        bin_size_hz=frame.bin_size_hz,
                        bin_index=detection.bin_index,
                        gps=gps_fix,
                    )
                    tdoa_logger.record(tdoa_event)
    except KeyboardInterrupt:  # pragma: no cover - interactive session
        LOGGER.info("Interrupted by user")
    finally:
        sweep_runner.stop()
        if gps_reader:
            gps_reader.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Command line interface for Drone Sweeper field capture."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from .gps import GPSReader
from .sweep import (
    HackRFSweepRunner,
    SignalOfInterest,
    detect_signals,
    write_detections_csv,
)
from .tdoa import TDOAEvent, TDOALogger, default_sensor_id, monotonic_time_ns


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


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    configure_logging(args.log_level)

    signals = ensure_signals(args.signals, args.default_tolerance)
    LOGGER.info("Loaded %d signal(s) of interest", len(signals))

    gps_reader: Optional[GPSReader] = None
    if args.gps_port:
        LOGGER.info("Starting GPS reader on %s", args.gps_port)
        gps_reader = GPSReader(args.gps_port, baudrate=args.gps_baudrate)
        try:
            gps_reader.start()
        except Exception as exc:  # pragma: no cover - hardware specific
            LOGGER.error("Unable to start GPS reader: %s", exc)
            gps_reader = None

    sweep_runner = HackRFSweepRunner(
        args.sweep_ranges,
        args.hackrf_args,
        bin_width=args.bin_width,
        sample_rate=args.sample_rate,
    )

    csv_path = args.csv_log
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

            gps_fix = gps_reader.latest_fix() if gps_reader else None
            monotonic_ns = monotonic_time_ns()

            if csv_path:
                write_detections_csv(csv_path, frame, detections)

            for detection in detections:
                payload = {
                    "timestamp": frame.timestamp.isoformat(),
                    "label": detection.signal.label,
                    "frequency_hz": detection.frequency_hz,
                    "power_db": detection.power_db,
                    "sensor_id": args.sensor_id,
                }
                if gps_fix:
                    payload["gps"] = gps_fix.as_dict()
                if args.print_json:
                    print(json.dumps(payload), flush=True)
                else:
                    LOGGER.info(
                        "Detection %s at %.0f Hz (%.1f dB)",
                        detection.signal.label,
                        detection.frequency_hz,
                        detection.power_db,
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

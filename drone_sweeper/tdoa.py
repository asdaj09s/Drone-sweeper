"""TDOA sample helpers for correlating detections across sensors."""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .gps import GPSFix


@dataclass
class TDOAEvent:
    """A serialisable representation of a detection for TDOA analysis."""

    detection_time_utc: datetime
    detection_time_monotonic_ns: int
    sensor_id: str
    frequency_hz: float
    power_db: float
    sweep_start_frequency_hz: float
    sweep_stop_frequency_hz: float
    bin_size_hz: float
    bin_index: int
    gps: Optional[GPSFix]

    def to_json(self) -> str:
        payload = asdict(self)
        payload["detection_time_utc"] = self.detection_time_utc.isoformat()
        payload["gps"] = self.gps.as_dict() if self.gps else None
        return json.dumps(payload)


class TDOALogger:
    """Streams :class:`TDOAEvent` objects to a JSONL file for later fusion."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event: TDOAEvent) -> None:
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(event.to_json())
            fp.write("\n")


def default_sensor_id() -> str:
    hostname = socket.gethostname()
    return f"{hostname}-hackrf"


def monotonic_time_ns() -> int:
    return time.monotonic_ns()


__all__ = ["TDOAEvent", "TDOALogger", "default_sensor_id", "monotonic_time_ns"]

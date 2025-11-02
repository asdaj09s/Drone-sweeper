"""GPS puck integration helpers."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pynmea2
import serial

logger = logging.getLogger(__name__)


@dataclass
class GPSFix:
    """Represents a GPS fix captured from an attached GPS puck."""

    timestamp: datetime
    latitude: Optional[float]
    longitude: Optional[float]
    altitude_m: Optional[float]
    speed_mps: Optional[float]
    track_deg: Optional[float]
    hdop: Optional[float]
    valid: bool

    def as_dict(self) -> dict:
        """Convert the fix into a serialisable dictionary."""

        return {
            "timestamp": self.timestamp.isoformat(),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude_m": self.altitude_m,
            "speed_mps": self.speed_mps,
            "track_deg": self.track_deg,
            "hdop": self.hdop,
            "valid": self.valid,
        }


class GPSReader:
    """Continuously reads NMEA sentences from a serial GPS puck."""

    def __init__(
        self,
        port: str,
        baudrate: int = 9600,
        read_timeout: float = 1.0,
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._timeout = read_timeout
        self._serial: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_fix: Optional[GPSFix] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._serial = serial.Serial(self._port, self._baudrate, timeout=self._timeout)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        logger.debug("Started GPS reader thread for port %s", self._port)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._serial and self._serial.is_open:
            self._serial.close()
        logger.debug("Stopped GPS reader thread for port %s", self._port)

    def _reader_loop(self) -> None:
        assert self._serial is not None
        buffer = b""
        while not self._stop_event.is_set():
            try:
                chunk = self._serial.read(256)
                if not chunk:
                    continue
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    self._handle_line(line.decode(errors="ignore"))
            except serial.SerialException as exc:
                logger.warning("GPS serial exception: %s", exc)
                time.sleep(1)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Unexpected error while reading GPS data")
                time.sleep(1)

    def _handle_line(self, line: str) -> None:
        try:
            msg = pynmea2.parse(line)
        except pynmea2.nmea.ParseError:
            logger.debug("Skipping unparsable NMEA sentence: %s", line)
            return

        fix = self._create_fix(msg)
        if fix:
            with self._lock:
                self._latest_fix = fix

    def _create_fix(self, msg: pynmea2.nmea.Sentence) -> Optional[GPSFix]:
        timestamp = datetime.now(timezone.utc)
        latitude: Optional[float] = None
        longitude: Optional[float] = None
        altitude: Optional[float] = None
        speed_mps: Optional[float] = None
        track_deg: Optional[float] = None
        hdop: Optional[float] = None
        valid = False

        if hasattr(msg, "latitude") and hasattr(msg, "longitude"):
            try:
                latitude = msg.latitude if msg.latitude != "" else None
                if latitude is not None:
                    latitude = float(latitude)
                longitude = msg.longitude if msg.longitude != "" else None
                if longitude is not None:
                    longitude = float(longitude)
            except (TypeError, ValueError):
                latitude = longitude = None

        if hasattr(msg, "altitude"):
            try:
                altitude = float(msg.altitude)
            except (TypeError, ValueError):
                altitude = None

        if hasattr(msg, "spd_over_grnd"):
            try:
                speed_mps = float(msg.spd_over_grnd) * 0.514444
            except (TypeError, ValueError):
                speed_mps = None

        if hasattr(msg, "true_course"):
            try:
                track_deg = float(msg.true_course)
            except (TypeError, ValueError):
                track_deg = None

        if hasattr(msg, "horizontal_dil"):
            try:
                hdop = float(msg.horizontal_dil)
            except (TypeError, ValueError):
                hdop = None

        if hasattr(msg, "gps_qual"):
            try:
                valid = int(msg.gps_qual) > 0
            except (TypeError, ValueError):
                valid = False
        elif hasattr(msg, "status"):
            valid = getattr(msg, "status", "V") == "A"

        return GPSFix(
            timestamp=timestamp,
            latitude=latitude,
            longitude=longitude,
            altitude_m=altitude,
            speed_mps=speed_mps,
            track_deg=track_deg,
            hdop=hdop,
            valid=valid,
        )

    def latest_fix(self) -> Optional[GPSFix]:
        with self._lock:
            return self._latest_fix


__all__ = ["GPSFix", "GPSReader"]

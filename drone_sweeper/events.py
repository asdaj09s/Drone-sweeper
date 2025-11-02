"""Detection event data structures and serialization utilities."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Mapping, Optional
from xml.etree.ElementTree import Element, SubElement, tostring

from .gps import GPSFix


@dataclass
class DetectionEvent:
    """Enriched detection information propagated to external consumers."""

    timestamp: datetime
    label: str
    frequency_hz: float
    power_db: float
    sensor_id: str
    gps: Optional[GPSFix] = None
    ai_label: Optional[str] = None
    ai_confidence: Optional[float] = None
    ai_probabilities: Optional[Mapping[str, float]] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "timestamp": self.timestamp.isoformat(),
            "label": self.label,
            "frequency_hz": self.frequency_hz,
            "power_db": self.power_db,
            "sensor_id": self.sensor_id,
        }
        if self.gps:
            payload["gps"] = self.gps.as_dict()
        if self.ai_label is not None:
            payload["ai_label"] = self.ai_label
        if self.ai_confidence is not None:
            payload["ai_confidence"] = self.ai_confidence
        if self.ai_probabilities is not None:
            payload["ai_probabilities"] = dict(self.ai_probabilities)
        if self.model_name:
            payload["model_name"] = self.model_name
        if self.model_version:
            payload["model_version"] = self.model_version
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def to_cot_xml(self, stale_ttl_seconds: float = 60.0) -> str:
        """Serialise the event into ATAK Cursor-on-Target XML."""

        start_time = self.timestamp.astimezone(timezone.utc)
        stale_time = start_time + timedelta(seconds=stale_ttl_seconds)
        uid = self._cot_uid()
        event = Element(
            "event",
            {
                "version": "2.0",
                "type": "a-f-G-U-C",
                "uid": uid,
                "how": "m-g",
                "time": start_time.isoformat(),
                "start": start_time.isoformat(),
                "stale": stale_time.isoformat(),
            },
        )
        lat, lon, hae = self._position_components()
        SubElement(
            event,
            "point",
            {
                "lat": f"{lat:.6f}",
                "lon": f"{lon:.6f}",
                "hae": f"{hae:.1f}",
                "ce": "9999999.0",
                "le": "9999999.0",
            },
        )
        detail = SubElement(event, "detail")
        contact = SubElement(detail, "contact")
        contact.set("callsign", self.label)

        if self.ai_label:
            ai_element = SubElement(detail, "rfAi")
            ai_element.set("label", self.ai_label)
            if self.ai_confidence is not None:
                ai_element.set("confidence", f"{self.ai_confidence:.3f}")
            if self.ai_probabilities:
                for name, value in sorted(self.ai_probabilities.items()):
                    SubElement(
                        ai_element,
                        "probability",
                        {
                            "label": str(name),
                            "value": f"{float(value):.3f}",
                        },
                    )
        if self.model_name:
            model_element = SubElement(detail, "model")
            model_element.set("name", self.model_name)
            if self.model_version:
                model_element.set("version", self.model_version)

        SubElement(
            detail,
            "rf",
            {
                "frequency_hz": f"{self.frequency_hz:.0f}",
                "power_db": f"{self.power_db:.2f}",
                "sensor": self.sensor_id,
            },
        )

        return tostring(event, encoding="utf-8").decode("utf-8")

    def _position_components(self) -> tuple[float, float, float]:
        if self.gps and self.gps.latitude is not None and self.gps.longitude is not None:
            hae = self.gps.altitude_m if self.gps.altitude_m is not None else 0.0
            return self.gps.latitude, self.gps.longitude, hae
        return 0.0, 0.0, 0.0

    def _cot_uid(self) -> str:
        seed = f"{self.sensor_id}-{self.frequency_hz:.0f}-{self.timestamp.timestamp():.0f}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


__all__ = ["DetectionEvent"]


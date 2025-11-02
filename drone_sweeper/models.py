"""Runtime helpers for RF classification models."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np

LOGGER = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency during docs builds
    import onnxruntime as ort
except Exception:  # pragma: no cover - import guard for lightweight environments
    ort = None


DEFAULT_MODEL_PATH = Path(os.environ.get("DRONE_SWEEPER_MODEL", ""))
DEFAULT_LABELS = ("benign", "suspicious", "threat")


@dataclass
class ClassificationResult:
    """Represents the output of the RF classifier."""

    label: str
    confidence: float
    probabilities: Mapping[str, float]
    model_name: Optional[str]
    model_version: Optional[str]

    def as_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
            "model_name": self.model_name,
            "model_version": self.model_version,
        }


class _ModelRuntime:
    """Wrapper around ONNX Runtime with a heuristic fallback."""

    def __init__(self, model_path: Optional[Path]) -> None:
        self.model_path = model_path if model_path and model_path.exists() else None
        self._session: Optional["ort.InferenceSession"] = None
        self._input_name: Optional[str] = None
        self._labels: Sequence[str] = DEFAULT_LABELS
        self._model_name: Optional[str] = None
        self._model_version: Optional[str] = None
        if self.model_path and ort is not None:
            try:
                self._session = ort.InferenceSession(str(self.model_path))
                self._input_name = self._session.get_inputs()[0].name
                metadata = self._session.get_modelmeta()
                self._model_name = metadata.graph_name or metadata.description
                try:
                    custom_metadata = json.loads(metadata.custom_metadata_map.get("labels", "[]"))
                except json.JSONDecodeError:
                    custom_metadata = []
                if custom_metadata:
                    self._labels = tuple(str(label) for label in custom_metadata)
                version = metadata.version
                self._model_version = version if version else None
                LOGGER.info("Loaded RF classifier from %s", self.model_path)
            except Exception:
                LOGGER.exception("Unable to load ONNX model from %s", self.model_path)
                self._session = None
        elif self.model_path:
            LOGGER.warning("onnxruntime is unavailable, falling back to heuristic classifier")

    def classify(self, features: Sequence[float], metadata: Optional[Mapping[str, object]] = None) -> ClassificationResult:
        if self._session and self._input_name:
            return self._onnx_classify(features)
        return self._heuristic_classify(features, metadata)

    def _onnx_classify(self, features: Sequence[float]) -> ClassificationResult:
        assert self._session is not None
        assert self._input_name is not None
        input_array = np.asarray([features], dtype=np.float32)
        try:
            outputs = self._session.run(None, {self._input_name: input_array})
        except Exception:
            LOGGER.exception("ONNX inference failed, falling back to heuristic classifier")
            return self._heuristic_classify(features, None)
        if not outputs:
            return self._heuristic_classify(features, None)
        logits = np.asarray(outputs[0]).reshape(-1)
        if logits.size != len(self._labels):
            LOGGER.warning(
                "Unexpected classifier output size %d (expected %d)",
                logits.size,
                len(self._labels),
            )
            return self._heuristic_classify(features, None)
        probabilities = _softmax(logits)
        label_index = int(np.argmax(probabilities))
        label = self._labels[label_index]
        confidence = float(probabilities[label_index])
        prob_map: Dict[str, float] = {
            str(label_name): float(probabilities[idx])
            for idx, label_name in enumerate(self._labels)
        }
        return ClassificationResult(
            label=label,
            confidence=confidence,
            probabilities=prob_map,
            model_name=self._model_name,
            model_version=self._model_version,
        )

    def _heuristic_classify(
        self,
        features: Sequence[float],
        metadata: Optional[Mapping[str, object]],
    ) -> ClassificationResult:
        power_db = float(features[0]) if features else -100.0
        frequency_offset = abs(float(features[1])) if len(features) > 1 else 0.0
        span_position = float(features[2]) if len(features) > 2 else 0.0

        # Power driven threat estimate with additional context shaping the probabilities.
        if power_db >= -20.0:
            base = {"threat": 0.75, "suspicious": 0.2, "benign": 0.05}
        elif power_db >= -40.0:
            base = {"threat": 0.35, "suspicious": 0.45, "benign": 0.20}
        else:
            base = {"threat": 0.1, "suspicious": 0.25, "benign": 0.65}

        # A tight frequency lock reduces uncertainty, drift increases suspicion.
        drift_factor = min(frequency_offset / 1_000_000.0, 1.0)
        base["suspicious"] += drift_factor * 0.1
        base["threat"] += max(0.0, 0.3 - span_position * 0.3)

        normalised = _normalise_probabilities(base, DEFAULT_LABELS)
        label = max(normalised, key=normalised.get)
        confidence = normalised[label]
        model_name = "heuristic-rf"
        model_version = "1.0"
        return ClassificationResult(
            label=label,
            confidence=confidence,
            probabilities=normalised,
            model_name=model_name,
            model_version=model_version,
        )


def _softmax(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = arr - np.max(arr)
    exp = np.exp(arr)
    total = np.sum(exp)
    if total == 0.0:
        return np.full_like(exp, 1.0 / len(exp))
    return exp / total


def _normalise_probabilities(
    raw: MutableMapping[str, float], labels: Iterable[str]
) -> Dict[str, float]:
    values = {label: max(0.0, float(raw.get(label, 0.0))) for label in labels}
    total = sum(values.values())
    if total <= 0.0:
        uniform = 1.0 / max(len(values), 1)
        return {label: uniform for label in values}
    return {label: value / total for label, value in values.items()}


@lru_cache(maxsize=1)
def _runtime(model_path: Optional[str] = None) -> _ModelRuntime:
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if path and not path.exists():
        LOGGER.warning("Model path %s does not exist; using heuristic classifier", path)
        path = None
    return _ModelRuntime(path)


def classify_detection(
    features: Sequence[float],
    metadata: Optional[Mapping[str, object]] = None,
    *,
    model_path: Optional[Path] = None,
) -> ClassificationResult:
    """Classify a detection feature vector.

    Parameters
    ----------
    features:
        Model-ready numeric features.
    metadata:
        Optional contextual metadata (e.g. signal label, location).
    model_path:
        Override for the RF classification model path.
    """

    runtime = _runtime(str(model_path) if model_path else None)
    return runtime.classify(features, metadata)


__all__ = ["ClassificationResult", "classify_detection"]


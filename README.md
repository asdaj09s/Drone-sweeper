# Drone Sweeper — RF-Driven Drone Detection at the Edge

## Overview
**Drone Sweeper** is an RF-based detection and classification tool designed for **real-time drone activity monitoring** using SDR hardware and on-device machine learning.  
It detects control signals in known drone frequency bands, flags suspicious activity, and provides instant alerts — without requiring an internet connection.

Supports:
- **HackRF One**
- **BladeRF 2.0 A9**
- **SDRplay RSPduo**
- **Other SoapySDR-compatible SDRs**

Runs on:
- **NVIDIA Jetson Orin Nano Super**
- **Raspberry Pi 5 with Hailo AI Hat**
- **Raspberry Pi 4 (reduced features mode)**
- **Standard Linux x86_64 systems**
- **Windows and Apple M-Series devices support TBD**

---

## Features
- **Wideband Spectrum Sweeps** — Scan known drone bands (433 MHz, 915 MHz, 2.4 GHz, 5.8 GHz, etc.)
- **Real-Time Detection** — Flag signals exceeding a configurable threshold.
- **On-Device Classification** — Use AI models to identify the type/brand/model of detected drones.
- **Alert System**
  - Audio notification (TTS or tone)
  - Email alert with frequency and timestamp
  - Visual log with markers and suspected drone ID
- **GPS Integration** — Tag detections with location data from a USB GPS puck.
- **Automatic Capture** — Save IQ samples or sweep CSVs for post-analysis.
- **Extensible AI Pipeline** — Supports TensorRT, PyTorch, or ONNX models for RF spectrogram classification.
- **Optional Multi-Modal Fusion** — Combine RF detection with computer vision (YOLOv8) for confirmation.

---

## Architecture

[HackRF/BladeRF] -> [Sweep/Detect] -> (mark.jsonl) -> [Web UI]
                               \-> (auto-cap .iq) -> [On-Device Classifier (Jetson/Pi5+Hailo)]
                                                      \-> (enriched mark) -> [UI + Audio + Email]
(mark) --optional--> [OpenAI Enrichment] --------------/
[UI "AoA"] -> [RSPduo AoA] -> [ATAK CoT] -> [UI]

---

## Installation

### Prerequisites
- Python 3.9+
- SDR hardware (BladeRF, HackRF, or SDRplay RSPduo)
- Linux environment (Ubuntu 20.04+, JetPack 5+, Raspberry Pi OS 64-bit)
- `SoapySDR` library and SDR-specific drivers
- Optional: NVIDIA Jetson + TensorRT for accelerated AI

### Clone the Repository
```bash
git clone https://github.com/yourusername/drone-sweeper.git
cd drone-sweeper

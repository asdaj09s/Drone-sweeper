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

```mermaid
flowchart LR
  subgraph Radios
    H[HackRF One]:::sdr
    B[bladeRF 2.0 A9]:::sdr
    R[RSPduo]:::sdr
  end

  H & B --> S1[RF Sweep / Peak Picking<br/>bin clustering + thresholds]:::pi
  S1 --> M1{Mark Writer}:::pi
  M1 -->|JSONL mark| UI[Web UI<br/>/latest, /marks]:::ui
  M1 -->|auto-cap hit| IQ[.iq capture]:::fs

  %% Enrichment paths
  IQ --> NPU[On‑Device Classifier<br/>(Jetson Orin / Pi5+AI HAT)<br/>TensorRT / HailoRT]:::ai
  M1 --> OAI[(Optional) OpenAI Enrichment<br/>label, confidence, suggestion]:::cloud
  NPU --> ENR[Enriched Mark<br/>(*_marks_enriched.jsonl)]:::fs
  OAI --> ENR
  ENR --> UI

  %% Alerts
  ENR --> TTS[Audio Alert<br/>(browser TTS / MP3)]:::ux
  ENR --> MAIL[Email / Push]:::ux

  %% AoA path
  UI -->|user clicks "AoA"| AOA[AoA Engine (SoapySDR)<br/>RSPduo / coherent SDR]:::pi
  AOA --> COT[CoT/ATAK Event<br/>bearing fan / remarks]:::net
  COT --> UI

  classDef sdr fill:#eef,stroke:#88a,stroke-width:1px;
  classDef pi fill:#efe,stroke:#6a6,stroke-width:1px;
  classDef ui fill:#ffe,stroke:#aa6,stroke-width:1px;
  classDef fs fill:#f7f7f7,stroke:#bbb,stroke-width:1px;
  classDef ai fill:#eaf7ff,stroke:#59a,stroke-width:1px;
  classDef cloud fill:#f0eaff,stroke:#95a,stroke-width:1px;
  classDef ux fill:#fff2e6,stroke:#c96,stroke-width:1px;
  classDef net fill:#e6fff2,stroke:#6c9,stroke-width:1px;

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

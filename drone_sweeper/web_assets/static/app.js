(function () {
  'use strict';

  function formatConfidence(confidence) {
    if (confidence === undefined || confidence === null || Number.isNaN(confidence)) {
      return 'n/a';
    }
    const pct = Math.max(0, Math.min(1, Number(confidence))) * 100;
    return `${pct.toFixed(1)}%`;
  }

  function renderDetection(detection, containerSelector = '#detections') {
    const container = document.querySelector(containerSelector);
    if (!container) {
      return;
    }

    const card = document.createElement('div');
    card.className = 'detection-card';

    const header = document.createElement('div');
    header.className = 'detection-card__header';
    header.textContent = detection.label || 'Unknown signal';
    card.appendChild(header);

    const meta = document.createElement('dl');
    meta.className = 'detection-card__meta';

    const frequency = document.createElement('div');
    frequency.innerHTML = `<dt>Frequency</dt><dd>${formatFrequency(detection.frequency_hz)}</dd>`;
    meta.appendChild(frequency);

    const power = document.createElement('div');
    power.innerHTML = `<dt>Power</dt><dd>${formatPower(detection.power_db)}</dd>`;
    meta.appendChild(power);

    if (detection.sensor_id) {
      const sensor = document.createElement('div');
      sensor.innerHTML = `<dt>Sensor</dt><dd>${detection.sensor_id}</dd>`;
      meta.appendChild(sensor);
    }

    card.appendChild(meta);

    const aiSummary = document.createElement('div');
    aiSummary.className = 'detection-card__ai';
    aiSummary.textContent = detection.ai_label
      ? `AI Assessment: ${detection.ai_label} (${formatConfidence(detection.ai_confidence)})`
      : 'AI Assessment: unavailable';
    card.appendChild(aiSummary);

    if (detection.ai_probabilities) {
      const list = document.createElement('ul');
      list.className = 'detection-card__ai-probabilities';
      Object.entries(detection.ai_probabilities).forEach(([label, value]) => {
        const item = document.createElement('li');
        item.textContent = `${label}: ${formatConfidence(value)}`;
        list.appendChild(item);
      });
      card.appendChild(list);
    }

    container.prepend(card);
  }

  function formatFrequency(freq) {
    if (!freq && freq !== 0) {
      return 'unknown';
    }
    const mhz = Number(freq) / 1e6;
    return `${mhz.toFixed(3)} MHz`;
  }

  function formatPower(power) {
    if (power === undefined || power === null) {
      return 'unknown';
    }
    return `${Number(power).toFixed(1)} dB`;
  }

  window.DroneSweeperUI = {
    renderDetection,
    formatConfidence,
  };
})();

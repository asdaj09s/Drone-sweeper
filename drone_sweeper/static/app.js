const detectionList = document.getElementById("detection-list");
const mapView = document.getElementById("map-view");
const refreshButton = document.getElementById("refresh-button");
const lastUpdatedLabel = document.getElementById("last-updated");

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Request failed with status ${response.status}`);
  }
  return response.json();
}

function renderDetectionItem(detection) {
  const li = document.createElement("li");
  li.className = "detection-item";
  const title = document.createElement("strong");
  title.textContent = detection.label || detection.frequency_label || "Unknown signal";
  li.appendChild(title);

  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = detection.status || detection.type || "Detection";
  li.appendChild(badge);

  const meta = document.createElement("div");
  meta.className = "meta";

  const frequency = detection.frequency_hz || detection.frequency || detection.center_frequency_hz;
  if (frequency !== undefined) {
    const frequencyNumber = Number(frequency);
    if (!Number.isNaN(frequencyNumber)) {
      meta.appendChild(makeMeta(`Frequency: ${(frequencyNumber / 1e6).toFixed(3)} MHz`));
    }
  }

  const power = detection.power_dbm ?? detection.power;
  if (power !== undefined) {
    const powerNumber = Number(power);
    if (!Number.isNaN(powerNumber)) {
      meta.appendChild(makeMeta(`Power: ${powerNumber.toFixed(1)} dB`));
    }
  }

  const timestamp = detection.timestamp ?? detection.time;
  if (timestamp !== undefined) {
    meta.appendChild(makeMeta(`Time: ${formatTimestamp(timestamp)}`));
  }

  if (detection.location) {
    const { lat, lon } = detection.location;
    if (typeof lat === "number" && typeof lon === "number") {
      meta.appendChild(makeMeta(`Lat ${lat.toFixed(4)}, Lon ${lon.toFixed(4)}`));
    }
  }

  if (meta.children.length > 0) {
    li.appendChild(meta);
  }

  return li;
}

function makeMeta(text) {
  const span = document.createElement("span");
  span.textContent = text;
  return span;
}

function formatTimestamp(value) {
  const numeric = Number(value);
  if (!Number.isNaN(numeric)) {
    const timestampMs = numeric > 1e12 ? numeric : numeric * 1000;
    const date = new Date(timestampMs);
    if (!Number.isNaN(date.getTime())) {
      return date.toLocaleString();
    }
  }
  try {
    const date = new Date(value);
    if (!Number.isNaN(date.getTime())) {
      return date.toLocaleString();
    }
  } catch (error) {
    // fall through to string conversion
  }
  return String(value);
}

function renderDetections(detections) {
  detectionList.innerHTML = "";
  if (!detections || detections.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No detections yet";
    detectionList.appendChild(empty);
    return;
  }

  const fragment = document.createDocumentFragment();
  detections.slice(0, 20).forEach((detection) => {
    fragment.appendChild(renderDetectionItem(detection));
  });
  detectionList.appendChild(fragment);
}

function renderMap(data) {
  if (!mapView) {
    return;
  }
  mapView.textContent = "";
  if (!data || Object.keys(data).length === 0) {
    mapView.textContent = "No map data available";
    return;
  }
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(data, null, 2);
  mapView.appendChild(pre);
}

async function refresh() {
  try {
    const [detections, map] = await Promise.all([
      fetchJson("/api/detections"),
      fetchJson("/api/map"),
    ]);
    renderDetections(detections.detections ?? detections);
    renderMap(map);
    if (lastUpdatedLabel) {
      lastUpdatedLabel.textContent = `Updated ${new Date().toLocaleTimeString()}`;
    }
  } catch (error) {
    console.error("Unable to refresh data", error);
    if (lastUpdatedLabel) {
      lastUpdatedLabel.textContent = "Data refresh failed";
    }
  }
}

if (refreshButton) {
  refreshButton.addEventListener("click", () => {
    refresh();
  });
}

document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible") {
    refresh();
  }
});

refresh();
setInterval(refresh, 5000);

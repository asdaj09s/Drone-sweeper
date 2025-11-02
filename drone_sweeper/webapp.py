"""Flask application for serving the Drone Sweeper web UI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from flask import Flask, jsonify, render_template

WEB_ASSETS_DIR = Path(__file__).with_name("web_assets")
STATIC_DIR = Path(__file__).with_name("static")

DETECTIONS_FILENAME = "detections.jsonl"
MAP_FILENAME = "map.geojson"


def _load_json_lines(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def create_app(data_dir: Path, ui_mode: str = "desktop") -> Flask:
    """Create a configured Flask application.

    Parameters
    ----------
    data_dir:
        Directory containing detection and map artifacts.
    ui_mode:
        Which UI template to expose: ``desktop``, ``mobile`` or ``both``.
    """

    normalized_mode = ui_mode.lower()
    if normalized_mode not in {"desktop", "mobile", "both"}:
        raise ValueError("ui_mode must be one of: desktop, mobile, both")

    app = Flask(
        __name__,
        static_folder=str(STATIC_DIR),
        template_folder=str(WEB_ASSETS_DIR),
    )

    data_dir = Path(data_dir)

    def get_detections() -> Iterable[Dict[str, Any]]:
        return _load_json_lines(data_dir / DETECTIONS_FILENAME)

    def get_map() -> Dict[str, Any]:
        return _load_json(data_dir / MAP_FILENAME)

    @app.route("/api/detections")
    def api_detections() -> Any:
        return jsonify({"detections": list(get_detections())})

    @app.route("/api/map")
    def api_map() -> Any:
        return jsonify(get_map())

    def render(template_name: str):
        return render_template(template_name)

    if normalized_mode == "desktop":
        app.add_url_rule("/", "index", lambda: render("desktop.html"))
    elif normalized_mode == "mobile":
        app.add_url_rule("/", "index", lambda: render("mobile.html"))
    else:  # both
        app.add_url_rule("/", "index", lambda: render("desktop.html"))
        app.add_url_rule("/desktop", "desktop", lambda: render("desktop.html"))
        app.add_url_rule("/mobile", "mobile", lambda: render("mobile.html"))

    return app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drone Sweeper web UI server")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory with detections.jsonl and map.geojson artifacts",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Interface for the HTTP server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server",
    )
    parser.add_argument(
        "--ui-mode",
        choices=["desktop", "mobile", "both"],
        default="desktop",
        help="Select which UI template to expose",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    app = create_app(args.data_dir, ui_mode=args.ui_mode)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual execution
    raise SystemExit(main())

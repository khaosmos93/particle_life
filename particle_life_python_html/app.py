from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).parent
REFERENCE_DIR = ROOT.parent / "particle-life-app"
INDEX_PATH = ROOT / "static" / "index.html"


def build_reference_payload() -> dict[str, object]:
    files = sorted(str(path.relative_to(REFERENCE_DIR)) for path in REFERENCE_DIR.rglob("*") if path.is_file())
    return {
        "reference_directory": str(REFERENCE_DIR),
        "file_count": len(files),
        "files": files,
        "simulation": {
            "rules": "none",
            "parameters": [],
            "initial_state": "empty",
        },
    }


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 (http method name)
        if self.path == "/":
            body = INDEX_PATH.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/reference":
            body = json.dumps(build_reference_payload(), indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:
        return


def run(host: str = "0.0.0.0", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"Serving on http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()

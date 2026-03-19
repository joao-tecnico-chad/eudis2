"""MJPEG + JSON telemetry HTTP server using stdlib.

Endpoints:
    GET /           HTML dashboard with live video and telemetry
    GET /stream     MJPEG multipart stream
    GET /telemetry  JSON with current system state
"""

import json
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

from guardian.activation.filter import ActivationState
from guardian.config import GuardianConfig


DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Drone Guardian</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #1a1a2e; color: #eee; font-family: monospace; }
        .container { display: flex; gap: 20px; padding: 20px; max-width: 1200px; margin: auto; }
        .video { flex: 2; }
        .video img { width: 100%%; border: 2px solid #333; border-radius: 4px; }
        .panel { flex: 1; }
        h1 { padding: 20px; text-align: center; color: #0f0; font-size: 1.4em; }
        .card { background: #16213e; padding: 15px; margin-bottom: 10px; border-radius: 4px;
                border-left: 3px solid #333; }
        .card.active { border-left-color: #0f0; }
        .card.inactive { border-left-color: #f00; }
        .card h3 { margin-bottom: 8px; font-size: 0.9em; color: #888; }
        .card .value { font-size: 1.3em; }
        .armed { color: #ff0; font-weight: bold; }
        .fired { color: #f00; font-weight: bold; animation: pulse 0.5s infinite; }
        @keyframes pulse { 50%% { opacity: 0.5; } }
        .layer { display: flex; justify-content: space-between; padding: 4px 0;
                 border-bottom: 1px solid #222; }
        .layer .status { font-weight: bold; }
        .pass { color: #0f0; }
        .fail { color: #666; }
    </style>
</head>
<body>
    <h1>DRONE GUARDIAN</h1>
    <div class="container">
        <div class="video">
            <img id="stream" src="/stream" alt="Live Feed">
        </div>
        <div class="panel">
            <div class="card" id="status-card">
                <h3>SYSTEM STATUS</h3>
                <div class="value" id="system-status">INITIALIZING</div>
            </div>
            <div class="card">
                <h3>ALTITUDE</h3>
                <div class="value" id="altitude">--</div>
            </div>
            <div class="card">
                <h3>DETECTIONS</h3>
                <div class="value" id="detections">--</div>
            </div>
            <div class="card">
                <h3>FPS</h3>
                <div class="value" id="fps">--</div>
            </div>
            <div class="card">
                <h3>ACTIVATION LAYERS</h3>
                <div class="layer">
                    <span>L1 Altitude</span>
                    <span class="status" id="l1">--</span>
                </div>
                <div class="layer">
                    <span>L2 Centroid</span>
                    <span class="status" id="l2">--</span>
                </div>
                <div class="layer">
                    <span>L3 Size</span>
                    <span class="status" id="l3">--</span>
                </div>
                <div class="layer">
                    <span>L4 Persistence</span>
                    <span class="status" id="l4">--</span>
                </div>
            </div>
        </div>
    </div>
    <script>
        function update() {
            fetch('/telemetry').then(r => r.json()).then(d => {
                document.getElementById('altitude').textContent =
                    d.altitude_delta_m.toFixed(1) + 'm (ref + ' + d.altitude_margin_m.toFixed(0) + 'm)';
                document.getElementById('detections').textContent = d.detection_count;
                document.getElementById('fps').textContent = d.fps.toFixed(1);

                const a = d.activation;
                const setLayer = (id, val) => {
                    const el = document.getElementById(id);
                    el.textContent = val ? 'PASS' : 'FAIL';
                    el.className = 'status ' + (val ? 'pass' : 'fail');
                };
                setLayer('l1', a.layer1_altitude);
                setLayer('l2', a.layer2_centroid);
                setLayer('l3', a.layer3_size);
                const l4 = document.getElementById('l4');
                l4.textContent = a.layer4_count + '/5';
                l4.className = 'status ' + (a.layer4_count >= 5 ? 'pass' : 'fail');

                const statusEl = document.getElementById('system-status');
                const card = document.getElementById('status-card');
                if (a.fired) {
                    statusEl.textContent = 'NET DEPLOYED';
                    statusEl.className = 'value fired';
                    card.className = 'card active';
                } else if (a.armed) {
                    statusEl.textContent = 'ARMED';
                    statusEl.className = 'value armed';
                    card.className = 'card active';
                } else {
                    statusEl.textContent = 'MONITORING';
                    statusEl.className = 'value';
                    card.className = 'card';
                }
            }).catch(() => {});
        }
        setInterval(update, 200);
        update();
    </script>
</body>
</html>
"""


class FrameBuffer:
    """Thread-safe buffer for sharing frames between producer and consumers."""

    def __init__(self):
        self._frame_bytes: bytes = b""
        self._condition = threading.Condition()

    def push(self, jpeg_bytes: bytes) -> None:
        with self._condition:
            self._frame_bytes = jpeg_bytes
            self._condition.notify_all()

    def wait_and_get(self, timeout: float = 1.0) -> bytes:
        with self._condition:
            self._condition.wait(timeout=timeout)
            return self._frame_bytes


class TelemetryStore:
    """Thread-safe store for current telemetry data."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            "altitude_delta_m": 0.0,
            "altitude_margin_m": 5.0,
            "detection_count": 0,
            "activation": asdict(ActivationState()),
            "fps": 0.0,
            "timestamp": 0.0,
        }

    def update(self, altitude_delta_m: float, detection_count: int,
               activation: ActivationState, fps: float) -> None:
        with self._lock:
            self._data = {
                "altitude_delta_m": altitude_delta_m,
                "altitude_margin_m": self._data["altitude_margin_m"],
                "detection_count": detection_count,
                "activation": asdict(activation),
                "fps": fps,
                "timestamp": time.time(),
            }

    def get_json(self) -> str:
        with self._lock:
            return json.dumps(self._data)


class StreamServer:
    """HTTP server providing MJPEG stream and JSON telemetry."""

    def __init__(self, config: GuardianConfig):
        self._config = config
        self.frame_buffer = FrameBuffer()
        self.telemetry = TelemetryStore()
        self._server = None
        self._thread = None

    def start(self) -> None:
        frame_buffer = self.frame_buffer
        telemetry = self.telemetry

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/":
                    self._serve_dashboard()
                elif self.path == "/stream":
                    self._serve_mjpeg()
                elif self.path == "/telemetry":
                    self._serve_telemetry()
                else:
                    self.send_error(404)

            def _serve_dashboard(self):
                content = DASHBOARD_HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def _serve_mjpeg(self):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while True:
                        jpeg = frame_buffer.wait_and_get(timeout=2.0)
                        if not jpeg:
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n".encode())
                        self.wfile.write(b"\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _serve_telemetry(self):
                data = telemetry.get_json().encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format, *args):
                pass  # Suppress request logs

        self._server = ThreadingHTTPServer(
            (self._config.stream_host, self._config.stream_port), Handler
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"Stream server started at http://{self._config.stream_host}:{self._config.stream_port}")

    def push_frame(self, frame: np.ndarray) -> None:
        """Encode frame as JPEG and push to buffer."""
        _, jpeg = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, self._config.jpeg_quality])
        self.frame_buffer.push(jpeg.tobytes())

    def push_telemetry(self, altitude_delta_m: float, detection_count: int,
                       activation: ActivationState, fps: float) -> None:
        self.telemetry.update(altitude_delta_m, detection_count, activation, fps)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        print("Stream server stopped")

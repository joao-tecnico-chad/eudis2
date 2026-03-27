"""MJPEG + JSON telemetry HTTP server.

Endpoints:
    GET /           HTML dashboard with live video and JS-drawn detection boxes
    GET /stream     MJPEG multipart stream (from OAK hardware encoder or CPU fallback)
    GET /telemetry  JSON with detections, activation state, and telemetry
"""

import json
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from guardian.activation.filter import ActivationState
from guardian.config import GuardianConfig


DASHBOARD_HTML = """<!DOCTYPE html>
<html>
<head>
    <title>Drone Guardian</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #0a0a1a; color: #eee; font-family: 'Courier New', monospace;
               height: 100vh; overflow: hidden; }
        .header { display: flex; align-items: center; justify-content: space-between;
                  padding: 10px 20px; background: #0d1117; border-bottom: 1px solid #1a2332; }
        .header h1 { color: #00ff88; font-size: 1.1em; letter-spacing: 3px; }
        .header .status-badge { padding: 4px 14px; border-radius: 3px; font-size: 0.85em;
                                font-weight: bold; letter-spacing: 1px; }
        .badge-monitoring { background: #1a2332; color: #4a9eff; border: 1px solid #234; }
        .badge-armed { background: #332b00; color: #ffcc00; border: 1px solid #554400;
                       animation: pulse 1s infinite; }
        .badge-fired { background: #330000; color: #ff3333; border: 1px solid #550000;
                       animation: pulse 0.4s infinite; }
        @keyframes pulse { 50%% { opacity: 0.6; } }

        .main { display: flex; height: calc(100vh - 50px); }

        .video-panel { flex: 1; min-width: 0; display: flex; align-items: center;
                       justify-content: center; padding: 10px; background: #0a0a1a;
                       position: relative; }
        .video-container { position: relative; display: inline-block; }
        .video-container img { max-width: 100%%; max-height: 100%%; display: block;
                               border: 1px solid #1a2332; border-radius: 3px; }
        #overlay { position: absolute; top: 0; left: 0; width: 100%%; height: 100%%;
                   pointer-events: none; }

        .side-panel { width: 260px; min-width: 260px; background: #0d1117;
                      border-left: 1px solid #1a2332; padding: 12px; overflow-y: auto;
                      display: flex; flex-direction: column; gap: 8px; }

        .card { background: #161b22; padding: 10px 12px; border-radius: 4px;
                border: 1px solid #1a2332; }
        .card h3 { font-size: 0.7em; color: #555; text-transform: uppercase;
                   letter-spacing: 1px; margin-bottom: 4px; }
        .card .value { font-size: 1.15em; color: #ccc; }

        .layers-card { background: #161b22; padding: 10px 12px; border-radius: 4px;
                       border: 1px solid #1a2332; }
        .layers-card h3 { font-size: 0.7em; color: #555; text-transform: uppercase;
                          letter-spacing: 1px; margin-bottom: 8px; }
        .layer-row { display: flex; justify-content: space-between; align-items: center;
                     padding: 5px 0; border-bottom: 1px solid #1a2332; }
        .layer-row:last-child { border-bottom: none; }
        .layer-name { font-size: 0.8em; color: #888; }
        .layer-status { font-size: 0.8em; font-weight: bold; padding: 1px 8px;
                        border-radius: 2px; }
        .layer-pass { color: #00ff88; background: #0a2a1a; }
        .layer-fail { color: #555; background: #1a1a1a; }

        .stat-row { display: flex; gap: 8px; }
        .stat-row .card { flex: 1; }
    </style>
</head>
<body>
    <div class="header">
        <h1>DRONE GUARDIAN</h1>
        <span class="status-badge badge-monitoring" id="status-badge">MONITORING</span>
    </div>
    <div class="main">
        <div class="video-panel">
            <div class="video-container">
                <img id="stream" src="/stream" alt="Live Feed">
                <canvas id="overlay"></canvas>
            </div>
        </div>
        <div class="side-panel">
            <div class="stat-row">
                <div class="card">
                    <h3>FPS</h3>
                    <div class="value" id="fps">--</div>
                </div>
                <div class="card">
                    <h3>Drones</h3>
                    <div class="value" id="detections">0</div>
                </div>
            </div>
            <div class="card">
                <h3>Altitude</h3>
                <div class="value" id="altitude">--</div>
            </div>
            <div class="layers-card">
                <h3>Activation Layers</h3>
                <div class="layer-row">
                    <span class="layer-name">L1 Altitude</span>
                    <span class="layer-status layer-fail" id="l1">FAIL</span>
                </div>
                <div class="layer-row">
                    <span class="layer-name">L2 Centroid</span>
                    <span class="layer-status layer-fail" id="l2">FAIL</span>
                </div>
                <div class="layer-row">
                    <span class="layer-name">L3 Size</span>
                    <span class="layer-status layer-fail" id="l3">FAIL</span>
                </div>
                <div class="layer-row">
                    <span class="layer-name">L4 Persist</span>
                    <span class="layer-status layer-fail" id="l4">0/5</span>
                </div>
            </div>
        </div>
    </div>
    <script>
        const img = document.getElementById('stream');
        const canvas = document.getElementById('overlay');
        const ctx = canvas.getContext('2d');

        function drawBoxes(detections, nnSize) {
            const w = img.clientWidth;
            const h = img.clientHeight;
            if (!w || !h) return;

            canvas.width = w;
            canvas.height = h;

            ctx.clearRect(0, 0, w, h);
            if (!detections || !detections.length) return;

            const sx = w / nnSize[0];
            const sy = h / nnSize[1];

            for (const d of detections) {
                const x1 = d[0] * sx, y1 = d[1] * sy;
                const x2 = d[2] * sx, y2 = d[3] * sy;
                const conf = d[4];
                const armed = d[5];

                ctx.strokeStyle = armed ? '#ff0' : '#0f0';
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

                ctx.fillStyle = armed ? '#ff0' : '#0f0';
                ctx.font = '14px monospace';
                ctx.fillText('drone ' + (conf * 100).toFixed(0) + '%%', x1, y1 - 6);
            }
        }

        function update() {
            fetch('/telemetry').then(r => r.json()).then(d => {
                document.getElementById('altitude').textContent =
                    d.altitude_delta_m.toFixed(1) + 'm';
                document.getElementById('detections').textContent = d.detection_count;
                document.getElementById('fps').textContent = d.fps.toFixed(0);

                // Draw detection boxes on canvas overlay
                drawBoxes(d.detections, d.nn_size);

                const a = d.activation;
                const setLayer = (id, val) => {
                    const el = document.getElementById(id);
                    el.textContent = val ? 'PASS' : 'FAIL';
                    el.className = 'layer-status ' + (val ? 'layer-pass' : 'layer-fail');
                };
                setLayer('l1', a.layer1_altitude);
                setLayer('l2', a.layer2_centroid);
                setLayer('l3', a.layer3_size);
                const l4 = document.getElementById('l4');
                l4.textContent = a.layer4_count + '/5';
                l4.className = 'layer-status ' + (a.layer4_count >= 5 ? 'layer-pass' : 'layer-fail');

                const badge = document.getElementById('status-badge');
                if (a.cooling_down) {
                    badge.textContent = 'REARMING';
                    badge.className = 'status-badge badge-fired';
                } else if (a.armed) {
                    badge.textContent = 'ARMED';
                    badge.className = 'status-badge badge-armed';
                } else {
                    badge.textContent = 'MONITORING';
                    badge.className = 'status-badge badge-monitoring';
                }
            }).catch(() => {});
        }
        setInterval(update, 100);
        update();
    </script>
</body>
</html>
"""


class FrameBuffer:
    """Thread-safe buffer for sharing JPEG frames between producer and consumers."""

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

    def __init__(self, config: GuardianConfig):
        self._lock = threading.Lock()
        self._config = config
        self._data = {
            "altitude_delta_m": 0.0,
            "detection_count": 0,
            "detections": [],
            "nn_size": [config.img_size, config.img_size],
            "activation": asdict(ActivationState()),
            "fps": 0.0,
            "timestamp": 0.0,
        }

    def update(self, altitude_delta_m: float, detections: list,
               activation: ActivationState, fps: float) -> None:
        det_list = [
            [d.x1, d.y1, d.x2, d.y2, d.confidence, activation.armed]
            for d in detections
        ]
        with self._lock:
            self._data = {
                "altitude_delta_m": altitude_delta_m,
                "detection_count": len(detections),
                "detections": det_list,
                "nn_size": [self._config.img_size, self._config.img_size],
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
        self.telemetry = TelemetryStore(config)
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
                pass

        self._server = ThreadingHTTPServer(
            (self._config.stream_host, self._config.stream_port), Handler
        )
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"Stream: http://{self._config.stream_host}:{self._config.stream_port}")

    def push_jpeg(self, jpeg_bytes: bytes) -> None:
        """Push pre-encoded JPEG bytes (from OAK hardware encoder)."""
        self.frame_buffer.push(jpeg_bytes)

    def push_frame(self, frame: np.ndarray) -> None:
        """Encode frame as JPEG on CPU and push (fallback for desktop mode)."""
        import cv2
        _, jpeg = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, self._config.jpeg_quality])
        self.frame_buffer.push(jpeg.tobytes())

    def push_telemetry(self, altitude_delta_m: float, detections: list,
                       activation: ActivationState, fps: float) -> None:
        self.telemetry.update(altitude_delta_m, detections, activation, fps)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server = None
        print("Stream server stopped")

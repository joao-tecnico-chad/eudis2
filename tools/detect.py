"""Drone detection with live web dashboard.

Runs YOLOv6s drone detection on OAK-1W (Pi) or webcam (desktop).
All YOLO decoding happens on the MyriadX VPU — zero host CPU cost.
Video encoded by hardware MJPEG encoder.

Usage:
    python tools/detect.py                     # OAK-1W (Pi)
    python tools/detect.py --conf 0.5          # Higher confidence threshold
    python tools/detect.py --port 9090         # Different port

Open http://<device-ip>:8080 in a browser to see the dashboard.
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import depthai as dai

MODEL = "models/emine_yolov6s.rvc2.tar.xz"

parser = argparse.ArgumentParser(description="Drone detection dashboard")
parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
parser.add_argument("--fps", type=int, default=10, help="Camera FPS")
parser.add_argument("--port", type=int, default=8080, help="Dashboard port")
parser.add_argument("--jpeg-quality", type=int, default=70)
args = parser.parse_args()

# Shared state
latest_jpeg = b""
latest_dets = []
latest_fps = 0.0
lock = threading.Lock()

HTML = """<!DOCTYPE html>
<html>
<head><title>Drone Guardian</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#0a0a1a;display:flex;height:100vh;font-family:'Courier New',monospace;color:#eee}
  .video{flex:1;display:flex;align-items:center;justify-content:center}
  .wrap{position:relative;display:inline-block;line-height:0}
  .wrap img{max-width:100%;max-height:100vh;display:block;border:1px solid #1a2332}
  .wrap canvas{position:absolute;top:0;left:0;pointer-events:none}
  .panel{width:240px;padding:14px;background:#0d1117;border-left:1px solid #1a2332;
         overflow-y:auto;font-size:13px;display:flex;flex-direction:column;gap:10px}
  h2{color:#00ff88;font-size:1em;letter-spacing:2px}
  .card{background:#161b22;padding:10px;border-radius:4px;border:1px solid #1a2332}
  .card h3{font-size:.7em;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
  .card .val{font-size:1.2em}
  .det{padding:4px 0;border-bottom:1px solid #222}
  .det .label{color:#0f0;font-weight:bold}
  .det .conf{color:#aaa}
  .row{display:flex;gap:8px}
  .row .card{flex:1}
</style>
</head>
<body>
  <div class="video">
    <div class="wrap">
      <img id="stream" src="/stream">
      <canvas id="c"></canvas>
    </div>
  </div>
  <div class="panel">
    <h2>DRONE GUARDIAN</h2>
    <div class="row">
      <div class="card"><h3>FPS</h3><div class="val" id="fps">--</div></div>
      <div class="card"><h3>Drones</h3><div class="val" id="count">0</div></div>
    </div>
    <div class="card">
      <h3>Detections</h3>
      <div id="list" style="color:#555">none</div>
    </div>
  </div>
  <script>
    const img=document.getElementById('stream'),c=document.getElementById('c'),ctx=c.getContext('2d');
    function update(){
      fetch('/detections').then(r=>r.json()).then(d=>{
        const w=img.clientWidth,h=img.clientHeight;
        if(!w||!h)return;
        c.width=w;c.height=h;c.style.width=w+'px';c.style.height=h+'px';
        ctx.clearRect(0,0,w,h);
        let html='';
        document.getElementById('fps').textContent=d.fps.toFixed(0);
        document.getElementById('count').textContent=d.detections.length;
        for(const det of d.detections){
          const x1=Math.max(0,det.xmin*w),y1=Math.max(0,det.ymin*h);
          const x2=Math.min(w,det.xmax*w),y2=Math.min(h,det.ymax*h);
          const pct=Math.round(det.confidence*100);
          ctx.strokeStyle=pct>70?'#f00':'#0f0';ctx.lineWidth=2;
          ctx.strokeRect(x1,y1,x2-x1,y2-y1);
          ctx.fillStyle=ctx.strokeStyle;ctx.font='bold 14px monospace';
          ctx.fillText(det.label+' '+pct+'%',x1+2,y1-6);
          html+='<div class="det"><span class="label">'+det.label+'</span> <span class="conf">'+pct+'%</span></div>';
        }
        document.getElementById('list').innerHTML=html||'<div style="color:#555">none</div>';
      }).catch(()=>{});
    }
    setInterval(update,80);
  </script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            data = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            try:
                while True:
                    with lock:
                        jpeg = latest_jpeg
                    if not jpeg:
                        time.sleep(0.03)
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.03)
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif self.path == "/detections":
            with lock:
                data = json.dumps({"detections": latest_dets, "fps": latest_fps}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def log_message(self, *a):
        pass


# Start web server
server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
print(f"Dashboard: http://0.0.0.0:{args.port}")

# OAK Pipeline
with dai.Pipeline() as p:
    p.setXLinkChunkSize(0)
    cam = p.create(dai.node.Camera).build()

    # Camera at high res, ImageManip resizes for NN
    cam_preview = cam.requestOutput((1280, 720), dai.ImgFrame.Type.BGR888p, fps=args.fps)

    nn_archive = dai.NNArchive(MODEL)
    nn_size = nn_archive.getInputSize()
    manip = p.create(dai.node.ImageManip)
    manip.initialConfig.setOutputSize(nn_size[0], nn_size[1])
    manip.initialConfig.setFrameType(dai.ImgFrame.Type.BGR888p)
    manip.setMaxOutputFrameSize(nn_size[0] * nn_size[1] * 3)
    cam_preview.link(manip.inputImage)

    # DetectionNetwork — on-device YOLO decode
    det_nn = p.create(dai.node.DetectionNetwork).build(manip.out, nn_archive)
    det_nn.setConfidenceThreshold(args.conf)
    det_nn.setNumInferenceThreads(2)
    labels = det_nn.getClasses()

    # Hardware MJPEG encoder
    enc_out = cam.requestOutput((640, 480), dai.ImgFrame.Type.NV12, fps=args.fps)
    encoder = p.create(dai.node.VideoEncoder)
    encoder.setDefaultProfilePreset(args.fps, dai.VideoEncoderProperties.Profile.MJPEG)
    encoder.setQuality(args.jpeg_quality)
    enc_out.link(encoder.input)

    # Small non-blocking queues
    q_det = det_nn.out.createOutputQueue(maxSize=1, blocking=False)
    q_mjpeg = encoder.out.createOutputQueue(maxSize=1, blocking=False)

    p.start()
    print(f"OAK running — drone YOLOv6s 640x640, conf>{args.conf}")

    fps = 0.0
    frame_count = 0
    fps_time = time.monotonic()

    try:
        while p.isRunning():
            mjpeg = q_mjpeg.tryGet()
            if mjpeg is not None:
                with lock:
                    latest_jpeg = bytes(mjpeg.getData())

            msg = q_det.tryGet()
            if msg is None:
                time.sleep(0.001)
                continue

            frame_count += 1
            now = time.monotonic()
            if now - fps_time >= 2.0:
                fps = frame_count / (now - fps_time)
                frame_count = 0
                fps_time = now

            dets = []
            for d in msg.detections:
                dets.append({
                    "label": labels[d.label] if d.label < len(labels) else "drone",
                    "confidence": round(d.confidence, 3),
                    "xmin": round(max(0, d.xmin), 4),
                    "ymin": round(max(0, d.ymin), 4),
                    "xmax": round(min(1, d.xmax), 4),
                    "ymax": round(min(1, d.ymax), 4),
                })

            with lock:
                latest_dets = dets
                latest_fps = fps

            if dets:
                top = max(dets, key=lambda x: x["confidence"])
                print(f"\r  {top['label']} {top['confidence']:.0%} ({len(dets)} det) [{fps:.0f} fps]   ", end="", flush=True)

    except KeyboardInterrupt:
        print("\nDone.")

server.shutdown()

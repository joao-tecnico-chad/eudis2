"""Drone detection with tracking and servo activation.

Detects drones, tracks them spatially to avoid false triggers,
and fires the servo when a drone is confirmed >75% for >3 seconds.

Dashboard shows only the tracked drone box (ignores noise).

Usage:
    python tools/detect_and_fire.py
    python tools/detect_and_fire.py --model models/yolov6n_416.rvc2.tar.xz
    python tools/detect_and_fire.py --conf 0.5 --hold 5 --fire-conf 0.8
"""

import argparse
import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import depthai as dai

DEFAULT_MODEL = "models/yolov6n_416.rvc2.tar.xz"

parser = argparse.ArgumentParser(description="Drone detection + servo fire")
parser.add_argument("--model", default=DEFAULT_MODEL)
parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
parser.add_argument("--fire-conf", type=float, default=0.75, help="Confidence to trigger servo")
parser.add_argument("--hold", type=float, default=3.0, help="Seconds tracked before firing")
parser.add_argument("--fps", type=int, default=15)
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--no-servo", action="store_true", help="Disable servo (testing)")
parser.add_argument("--servo-gpio", type=int, default=18)
args = parser.parse_args()

# --- Servo (matches test_servo.py behavior) ---
# Home: 90 deg, Fire: 135 deg, Pulse: 900-2100 µs (HS-5085MG)
servo = None
servo_ready = True
HOME_ANGLE = 90.0
FIRE_ANGLE = 135.0
MIN_PULSE = 0.0009
MAX_PULSE = 0.0021


def angle_to_value(angle):
    """Map 0-180 degrees to gpiozero's -1..+1 range."""
    return max(-1.0, min(1.0, (angle / 90.0) - 1.0))


if not args.no_servo:
    try:
        from gpiozero import Servo as GpioServo
        from gpiozero.pins.pigpio import PiGPIOFactory
        try:
            factory = PiGPIOFactory()
            servo = GpioServo(args.servo_gpio, pin_factory=factory,
                              min_pulse_width=MIN_PULSE, max_pulse_width=MAX_PULSE)
        except Exception:
            servo = GpioServo(args.servo_gpio,
                              min_pulse_width=MIN_PULSE, max_pulse_width=MAX_PULSE)
        servo.value = angle_to_value(HOME_ANGLE)  # 90 deg home
        print(f"Servo on GPIO{args.servo_gpio} (home={HOME_ANGLE}°, fire={FIRE_ANGLE}°)")
    except ImportError:
        print("gpiozero not installed — servo disabled")


def fire_servo():
    global servo_ready
    if servo is None or not servo_ready:
        return
    servo_ready = False
    servo.value = angle_to_value(FIRE_ANGLE)  # 135 deg — fire
    print("\n*** SERVO FIRED ***")

    def rearm():
        global servo_ready
        time.sleep(2.0)
        servo.value = angle_to_value(HOME_ANGLE)  # 90 deg — home
        servo_ready = True
        print("*** REARMED ***")
    threading.Thread(target=rearm, daemon=True).start()


# --- Tracker ---
class DroneTracker:
    """Track a single drone spatially. Only counts if same drone stays in frame."""

    def __init__(self, match_dist=100, gap_sec=0.5):
        self.match_dist = match_dist
        self.gap_sec = gap_sec
        self.cx = 0.0
        self.cy = 0.0
        self.start_time = None
        self.last_seen = None
        self.ema_conf = 0.0
        self.best_det = None

    def update(self, detections, now):
        """Feed detections, return (tracked_det, hold_time, confirmed)."""
        # Filter: high confidence + centroid in center 50% of frame
        drones = []
        for d in detections:
            if d["confidence"] < args.conf:
                continue
            cx = (d["xmin"] + d["xmax"]) / 2
            cy = (d["ymin"] + d["ymax"]) / 2
            if 0.25 <= cx <= 0.75 and 0.25 <= cy <= 0.75:
                drones.append(d)
        if not drones:
            if self.last_seen and (now - self.last_seen) > self.gap_sec:
                self.reset()
            return None, 0, False

        # Always pick highest confidence detection
        best = max(drones, key=lambda d: d["confidence"])

        # Update track
        self.cx = (best["xmin"] + best["xmax"]) / 2
        self.cy = (best["ymin"] + best["ymax"]) / 2
        # Heavy smoothing (alpha=0.15) to handle confidence fluctuations
        self.ema_conf = 0.15 * best["confidence"] + 0.85 * self.ema_conf
        self.best_det = best
        self.last_seen = now

        if self.start_time is None:
            self.start_time = now
            self.ema_conf = best["confidence"]

        hold = now - self.start_time
        confirmed = hold >= args.hold and self.ema_conf >= args.fire_conf
        return best, hold, confirmed

    def reset(self):
        self.start_time = None
        self.last_seen = None
        self.ema_conf = 0.0
        self.best_det = None


tracker = DroneTracker()

# --- Shared state ---
latest_jpeg = b""
tracked_det = None
track_hold = 0.0
track_confirmed = False
track_ema = 0.0
latest_fps = 0.0
fired = False
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
  .panel{width:260px;padding:14px;background:#0d1117;border-left:1px solid #1a2332;
         overflow-y:auto;font-size:13px;display:flex;flex-direction:column;gap:10px}
  h2{color:#00ff88;font-size:1em;letter-spacing:2px}
  .card{background:#161b22;padding:10px;border-radius:4px;border:1px solid #1a2332}
  .card h3{font-size:.7em;color:#555;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px}
  .card .val{font-size:1.2em}
  .row{display:flex;gap:8px} .row .card{flex:1}
  .status{padding:8px;border-radius:4px;text-align:center;font-weight:bold;letter-spacing:1px}
  .s-idle{background:#1a2332;color:#4a9eff}
  .s-tracking{background:#332b00;color:#ffcc00;animation:pulse 1s infinite}
  .s-fire{background:#330000;color:#ff3333;animation:pulse .3s infinite}
  @keyframes pulse{50%{opacity:.6}}
</style>
</head>
<body>
  <div class="video"><div class="wrap">
    <img id="stream" src="/stream"><canvas id="c"></canvas>
  </div></div>
  <div class="panel">
    <h2>DRONE GUARDIAN</h2>
    <div id="status" class="status s-idle">SCANNING</div>
    <div class="row">
      <div class="card"><h3>FPS</h3><div class="val" id="fps">--</div></div>
      <div class="card"><h3>Conf</h3><div class="val" id="conf">--</div></div>
    </div>
    <div class="row">
      <div class="card"><h3>Hold</h3><div class="val" id="hold">0.0s</div></div>
      <div class="card"><h3>Need</h3><div class="val">""" + f"{args.hold:.0f}" + """s</div></div>
    </div>
    <div class="card"><h3>EMA Confidence</h3>
      <div style="background:#1a1a2e;border-radius:3px;height:20px;margin-top:4px">
        <div id="bar" style="background:#0f0;height:100%;border-radius:3px;width:0;transition:width .1s"></div>
      </div>
    </div>
  </div>
  <script>
    const img=document.getElementById('stream'),c=document.getElementById('c'),ctx=c.getContext('2d');
    function update(){
      fetch('/state').then(r=>r.json()).then(d=>{
        const w=img.clientWidth,h=img.clientHeight;
        if(!w||!h)return;
        c.width=w;c.height=h;c.style.width=w+'px';c.style.height=h+'px';
        ctx.clearRect(0,0,w,h);
        // Draw center zone (50%)
        ctx.strokeStyle='rgba(100,100,100,0.4)';ctx.lineWidth=1;ctx.setLineDash([5,5]);
        ctx.strokeRect(w*0.25,h*0.25,w*0.5,h*0.5);ctx.setLineDash([]);
        document.getElementById('fps').textContent=d.fps.toFixed(0);

        const st=document.getElementById('status');
        if(d.fired){st.textContent='FIRED';st.className='status s-fire';}
        else if(d.det && d.det.confidence>=0.7){
          const det=d.det;
          const x1=Math.max(0,det.xmin*w),y1=Math.max(0,det.ymin*h);
          const x2=Math.min(w,det.xmax*w),y2=Math.min(h,det.ymax*h);
          const pct=Math.round(det.confidence*100);
          ctx.strokeStyle=d.confirmed?'#f00':'#0f0';ctx.lineWidth=3;
          ctx.strokeRect(x1,y1,x2-x1,y2-y1);
          ctx.fillStyle=ctx.strokeStyle;ctx.font='bold 16px monospace';
          ctx.fillText('drone '+pct+'%',x1+2,y1-8);
          // Hold progress bar at bottom of box
          if(d.hold>0){
            const prog=Math.min(1,d.hold/""" + str(args.hold) + """);
            ctx.fillStyle=d.confirmed?'rgba(255,0,0,0.5)':'rgba(0,255,0,0.3)';
            ctx.fillRect(x1,y2-4,(x2-x1)*prog,4);
          }
          document.getElementById('conf').textContent=pct+'%';
          document.getElementById('hold').textContent=d.hold.toFixed(1)+'s';
          document.getElementById('bar').style.width=(d.ema*100)+'%';
          document.getElementById('bar').style.background=d.ema>""" + str(args.fire_conf) + """?'#f00':'#0f0';
          st.textContent=d.confirmed?'ARMED':'TRACKING '+d.hold.toFixed(1)+'s';
          st.className='status '+(d.confirmed?'s-fire':'s-tracking');
        } else {
          document.getElementById('conf').textContent='--';
          document.getElementById('hold').textContent='0.0s';
          document.getElementById('bar').style.width='0';
          st.textContent='SCANNING';st.className='status s-idle';
        }
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
        elif self.path == "/state":
            with lock:
                data = json.dumps({
                    "det": tracked_det,
                    "hold": track_hold,
                    "confirmed": track_confirmed,
                    "ema": track_ema,
                    "fps": latest_fps,
                    "fired": fired,
                }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_error(404)

    def log_message(self, *a):
        pass


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass

server = QuietServer(("0.0.0.0", args.port), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
print(f"Dashboard: http://0.0.0.0:{args.port}")

# --- OAK Pipeline ---
with dai.Pipeline() as p:
    p.setXLinkChunkSize(0)
    cam = p.create(dai.node.Camera).build()

    nn_archive = dai.NNArchive(args.model)
    nn_size = nn_archive.getInputSize()
    cam_nn = cam.requestOutput((nn_size[0], nn_size[1]), dai.ImgFrame.Type.BGR888p, fps=args.fps)

    det_nn = p.create(dai.node.DetectionNetwork).build(cam_nn, nn_archive)
    det_nn.setConfidenceThreshold(args.conf)
    det_nn.setNumInferenceThreads(2)
    labels = det_nn.getClasses()

    enc_out = cam.requestOutput((640, 480), dai.ImgFrame.Type.NV12, fps=args.fps)
    encoder = p.create(dai.node.VideoEncoder)
    encoder.setDefaultProfilePreset(args.fps, dai.VideoEncoderProperties.Profile.MJPEG)
    encoder.setQuality(70)
    enc_out.link(encoder.input)

    q_det = det_nn.out.createOutputQueue(maxSize=1, blocking=False)
    q_mjpeg = encoder.out.createOutputQueue(maxSize=1, blocking=False)

    p.start()
    print(f"Running — {args.model}")
    print(f"Fire: >{args.fire_conf:.0%} for >{args.hold}s | Servo: {'GPIO' + str(args.servo_gpio) if servo else 'disabled'}")

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

            det, hold, confirmed = tracker.update(dets, time.time())

            if confirmed and servo_ready:
                fire_servo()
                tracker.reset()  # reset hold timer so it has to re-confirm

            with lock:
                tracked_det = det
                track_hold = hold
                track_confirmed = confirmed
                track_ema = tracker.ema_conf
                latest_fps = fps
                fired = not servo_ready

            if det:
                status = "ARMED" if confirmed else f"tracking {hold:.1f}s"
                print(f"\r  drone {det['confidence']:.0%} ema={tracker.ema_conf:.0%} {status} [{fps:.0f}fps]   ", end="", flush=True)

    except KeyboardInterrupt:
        print("\nShutdown")

if servo:
    servo.value = angle_to_value(HOME_ANGLE)
    time.sleep(0.3)
    servo.detach()
server.shutdown()

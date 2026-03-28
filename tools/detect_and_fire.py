"""Drone Guardian — Detection, tracking, and servo activation.

Web dashboard with live parameter controls.
All detection runs on OAK-1W MyriadX VPU.

Usage:
    python tools/detect_and_fire.py
    python tools/detect_and_fire.py --model models/yolov5n_416.rvc2.tar.xz
    python tools/detect_and_fire.py --no-servo --no-baro
"""

import argparse
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import depthai as dai

DEFAULT_MODEL = "models/yolov6n_416.rvc2.tar.xz"
SETTINGS_FILE = Path(os.path.expanduser("~/.guardian_settings.json"))

parser = argparse.ArgumentParser(description="Drone Guardian")
parser.add_argument("--model", default=DEFAULT_MODEL)
parser.add_argument("--conf", type=float, default=0.3)
parser.add_argument("--fire-conf", type=float, default=0.75)
parser.add_argument("--hold", type=float, default=3.0)
parser.add_argument("--fps", type=int, default=15)
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--no-servo", action="store_true")
parser.add_argument("--no-baro", action="store_true")
parser.add_argument("--alt-min", type=float, default=1.0)
parser.add_argument("--servo-gpio", type=int, default=18)
args = parser.parse_args()

# --- Runtime settings (mutable from dashboard) ---
settings = {
    "conf": args.conf,
    "fire_conf": args.fire_conf,
    "hold": args.hold,
    "alt_min": args.alt_min,
    "zone": 0.5,
    "servo_enabled": not args.no_servo,
    "box_min_conf": 0.7,
}

# Load saved settings
if SETTINGS_FILE.exists():
    try:
        saved = json.loads(SETTINGS_FILE.read_text())
        settings.update(saved)
        print(f"Loaded settings from {SETTINGS_FILE}")
    except Exception:
        pass


def save_settings():
    try:
        SETTINGS_FILE.write_text(json.dumps(settings, indent=2))
    except Exception:
        pass


# --- Barometer ---
baro = None
baro_ref = 0.0
baro_alt = 0.0
alt_delta = 0.0
BARO_WINDOW = 20

if not args.no_baro:
    try:
        import board
        import busio
        import adafruit_bmp3xx
        from collections import deque

        i2c = busio.I2C(board.SCL, board.SDA)
        baro = adafruit_bmp3xx.BMP3XX_I2C(i2c)
        baro.pressure_oversampling = 32
        baro.temperature_oversampling = 2
        baro.filter_coefficient = 16
        alt_buf = deque(maxlen=BARO_WINDOW)
        for _ in range(BARO_WINDOW):
            alt_buf.append(baro.altitude)
            time.sleep(0.05)
        baro_ref = sum(alt_buf) / len(alt_buf)
        print(f"Barometer: ref={baro_ref:.1f}m")
    except (ImportError, Exception) as e:
        print(f"Barometer: {e}")
        baro = None

# --- Servo ---
servo = None
servo_ready = True
HOME_ANGLE = 90.0
FIRE_ANGLE = 180.0


def angle_to_value(angle):
    return max(-1.0, min(1.0, (angle / 90.0) - 1.0))


if settings["servo_enabled"]:
    try:
        from gpiozero import Servo as GpioServo
        from gpiozero.pins.pigpio import PiGPIOFactory
        try:
            factory = PiGPIOFactory()
            servo = GpioServo(args.servo_gpio, pin_factory=factory,
                              min_pulse_width=0.0005, max_pulse_width=0.0025)
            print(f"Servo: GPIO{args.servo_gpio} (pigpio)")
        except Exception as e:
            print(f"Servo pigpio failed: {e}")
            servo = GpioServo(args.servo_gpio,
                              min_pulse_width=0.0005, max_pulse_width=0.0025)
            print(f"Servo: GPIO{args.servo_gpio} (native fallback - PWM may not work!)")
        servo.value = angle_to_value(HOME_ANGLE)
    except (ImportError, Exception) as e:
        print(f"Servo: {e}")


def reset_altitude():
    global baro_ref, alt_delta
    if baro is not None:
        baro_ref = baro_alt
        alt_delta = 0.0


def move_servo(target_angle, speed=50):
    """Move servo to angle with stepping (matches test_servo.py behavior)."""
    current = [HOME_ANGLE]  # mutable for closure

    step_size = 0.5 + (speed / 100.0) * 4.5
    step_delay = 0.030 - (speed / 100.0) * 0.028
    direction = 1 if target_angle > current[0] else -1
    pos = current[0]

    while True:
        remaining = abs(target_angle - pos)
        if remaining < step_size:
            pos = target_angle
            servo.value = angle_to_value(pos)
            break
        pos += direction * step_size
        servo.value = angle_to_value(pos)
        time.sleep(step_delay)
    current[0] = target_angle


def fire_servo():
    global servo_ready
    if servo is None or not servo_ready or not settings["servo_enabled"]:
        return
    servo_ready = False
    print("\n*** FIRED ***")

    def fire_cycle():
        global servo_ready
        # Set directly and hold for consistent movement
        servo.value = angle_to_value(FIRE_ANGLE)
        time.sleep(0.5)  # let servo reach position
        servo.value = angle_to_value(FIRE_ANGLE)  # reinforce
        time.sleep(2.0)
        servo.value = angle_to_value(HOME_ANGLE)
        time.sleep(0.5)  # let servo reach home
        servo_ready = True
        print("*** REARMED ***")
    threading.Thread(target=fire_cycle, daemon=True).start()


# --- Tracker ---
class DroneTracker:
    def __init__(self, gap_sec=0.5):
        self.gap_sec = gap_sec
        self.cx = self.cy = 0.0
        self.start_time = self.last_seen = None
        self.ema_conf = 0.0
        self.best_det = None

    def update(self, detections, now):
        zone = settings["zone"]
        margin = (1 - zone) / 2
        drones = []
        for d in detections:
            if d["confidence"] < settings["conf"]:
                continue
            cx = (d["xmin"] + d["xmax"]) / 2
            cy = (d["ymin"] + d["ymax"]) / 2
            if margin <= cx <= 1 - margin and margin <= cy <= 1 - margin:
                drones.append(d)
        if not drones:
            if self.last_seen and (now - self.last_seen) > self.gap_sec:
                self.reset()
            return None, 0, False
        best = max(drones, key=lambda d: d["confidence"])
        self.cx = (best["xmin"] + best["xmax"]) / 2
        self.cy = (best["ymin"] + best["ymax"]) / 2
        self.ema_conf = 0.15 * best["confidence"] + 0.85 * self.ema_conf
        self.best_det = best
        self.last_seen = now
        if self.start_time is None:
            self.start_time = now
            self.ema_conf = best["confidence"]
        hold = now - self.start_time
        confirmed = hold >= settings["hold"] and self.ema_conf >= settings["fire_conf"]
        return best, hold, confirmed

    def reset(self):
        self.start_time = self.last_seen = None
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
<head>
<title>Drone Guardian</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  :root{--bg:#0b0e17;--panel:#0d1117;--card:#151b23;--border:#1e2a3a;--text:#c9d1d9;
    --dim:#484f58;--green:#2ea043;--yellow:#d29922;--red:#da3633;--blue:#388bfd}
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:var(--bg);color:var(--text);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    display:flex;height:100vh;overflow:hidden}
  @media(max-width:800px){body{flex-direction:column}
    .panel{width:100%!important;max-height:40vh;border-left:none!important;border-top:1px solid var(--border)}}

  .video{flex:1;display:flex;align-items:center;justify-content:center;background:#000}
  .wrap{position:relative;display:inline-block;line-height:0}
  .wrap{transform:rotate(-90deg)}
  .wrap img{max-width:100vh;max-height:100vw;display:block}
  .wrap canvas{position:absolute;top:0;left:0;pointer-events:none}

  .panel{width:300px;background:var(--panel);border-left:1px solid var(--border);
    overflow-y:auto;display:flex;flex-direction:column}
  .panel-header{padding:16px 18px;background:linear-gradient(135deg,#0d1117,#161b22);
    border-bottom:1px solid var(--border)}
  .panel-header h1{font-size:14px;font-weight:700;letter-spacing:3px;color:var(--green);margin:0}
  .panel-body{padding:12px;display:flex;flex-direction:column;gap:10px;flex:1}

  .status-bar{padding:10px;border-radius:6px;text-align:center;font-weight:700;font-size:13px;
    letter-spacing:2px;transition:all .3s}
  .st-scan{background:rgba(56,139,253,0.1);color:var(--blue);border:1px solid rgba(56,139,253,0.2)}
  .st-track{background:rgba(210,153,34,0.1);color:var(--yellow);border:1px solid rgba(210,153,34,0.3);
    animation:glow 1.5s ease-in-out infinite}
  .st-armed{background:rgba(218,54,51,0.15);color:var(--red);border:1px solid rgba(218,54,51,0.4);
    animation:glow .4s ease-in-out infinite}
  .st-fired{background:rgba(218,54,51,0.3);color:#ff6b6b;border:1px solid var(--red);
    animation:glow .2s ease-in-out infinite}
  @keyframes glow{50%{opacity:.7}}

  .metrics{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .metric{background:var(--card);padding:10px 12px;border-radius:6px;border:1px solid var(--border)}
  .metric label{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1px;display:block;margin-bottom:2px}
  .metric .v{font-size:18px;font-weight:600;font-variant-numeric:tabular-nums}

  .ema-bar{background:var(--card);border-radius:6px;border:1px solid var(--border);padding:10px 12px}
  .ema-bar label{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1px;display:block;margin-bottom:6px}
  .bar-track{background:#1a1f2e;border-radius:3px;height:8px;overflow:hidden}
  .bar-fill{height:100%;border-radius:3px;transition:width .15s,background .3s}

  .section-title{font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:2px;
    padding:6px 0 2px;border-top:1px solid var(--border);margin-top:4px}

  .control{display:flex;align-items:center;gap:8px;padding:4px 0}
  .control label{font-size:11px;color:var(--dim);min-width:55px}
  .control input[type=range]{flex:1;accent-color:var(--blue);height:4px}
  .control .cv{font-size:12px;color:var(--text);min-width:40px;text-align:right;font-variant-numeric:tabular-nums}

  .btn{padding:8px 12px;border-radius:6px;font-size:11px;font-weight:600;cursor:pointer;
    border:1px solid var(--border);font-family:inherit;letter-spacing:1px;transition:all .15s;width:100%}
  .btn:hover{filter:brightness(1.2)}
  .btn-blue{background:rgba(56,139,253,0.1);color:var(--blue);border-color:rgba(56,139,253,0.3)}
  .btn-red{background:rgba(218,54,51,0.1);color:var(--red);border-color:rgba(218,54,51,0.3)}
  .btn-green{background:rgba(46,160,67,0.1);color:var(--green);border-color:rgba(46,160,67,0.3)}

  .toggle{display:flex;align-items:center;justify-content:space-between;padding:6px 0}
  .toggle label{font-size:11px;color:var(--dim)}
  .switch{position:relative;width:36px;height:20px}
  .switch input{opacity:0;width:0;height:0}
  .slider{position:absolute;inset:0;background:#30363d;border-radius:10px;cursor:pointer;transition:.3s}
  .slider:before{content:'';position:absolute;height:14px;width:14px;left:3px;bottom:3px;
    background:#8b949e;border-radius:50%;transition:.3s}
  .switch input:checked+.slider{background:var(--green)}
  .switch input:checked+.slider:before{transform:translateX(16px);background:#fff}
</style>
</head>
<body>
  <div class="video"><div class="wrap">
    <img id="stream" src="/stream"><canvas id="c"></canvas>
  </div></div>
  <div class="panel">
    <div class="panel-header"><h1>DRONE GUARDIAN</h1></div>
    <div class="panel-body">
      <div id="status" class="status-bar st-scan">SCANNING</div>

      <div class="metrics">
        <div class="metric"><label>FPS</label><div class="v" id="fps">--</div></div>
        <div class="metric"><label>Confidence</label><div class="v" id="conf">--</div></div>
        <div class="metric"><label>Hold</label><div class="v" id="hold">0.0s</div></div>
        <div class="metric"><label>Altitude</label><div class="v" id="alt">+0.0m</div></div>
      </div>

      <div class="ema-bar">
        <label>EMA Confidence</label>
        <div class="bar-track"><div id="bar" class="bar-fill" style="width:0;background:var(--green)"></div></div>
      </div>

      <div class="section-title">Controls</div>

      <div class="control"><label>Detect</label>
        <input type="range" id="s_conf" min="5" max="95" value="" oninput="setSetting('conf',this.value/100)">
        <span class="cv" id="v_conf"></span></div>
      <div class="control"><label>Fire at</label>
        <input type="range" id="s_fire" min="30" max="99" value="" oninput="setSetting('fire_conf',this.value/100)">
        <span class="cv" id="v_fire"></span></div>
      <div class="control"><label>Hold</label>
        <input type="range" id="s_hold" min="1" max="10" step="0.5" value="" oninput="setSetting('hold',parseFloat(this.value))">
        <span class="cv" id="v_hold"></span></div>
      <div class="control"><label>Alt min</label>
        <input type="range" id="s_alt" min="0" max="10" step="0.5" value="" oninput="setSetting('alt_min',parseFloat(this.value))">
        <span class="cv" id="v_alt"></span></div>
      <div class="control"><label>Zone</label>
        <input type="range" id="s_zone" min="20" max="100" value="" oninput="setSetting('zone',this.value/100)">
        <span class="cv" id="v_zone"></span></div>
      <div class="control"><label>Box at</label>
        <input type="range" id="s_box" min="10" max="99" value="" oninput="setSetting('box_min_conf',this.value/100)">
        <span class="cv" id="v_box"></span></div>

      <div class="toggle"><label>Servo enabled</label>
        <label class="switch"><input type="checkbox" id="servo_toggle" onchange="setSetting('servo_enabled',this.checked)">
        <span class="slider"></span></label></div>

      <div style="display:flex;gap:8px">
        <button class="btn btn-blue" onclick="fetch('/reset-alt')">RESET ALT</button>
        <button class="btn btn-red" onclick="fetch('/manual-fire')">TEST FIRE</button>
      </div>
    </div>
  </div>

  <script>
    const img=document.getElementById('stream'),c=document.getElementById('c'),ctx=c.getContext('2d');
    let S={};

    function loadSettings(){
      fetch('/settings').then(r=>r.json()).then(s=>{
        S=s;
        document.getElementById('s_conf').value=s.conf*100;
        document.getElementById('s_fire').value=s.fire_conf*100;
        document.getElementById('s_hold').value=s.hold;
        document.getElementById('s_alt').value=s.alt_min;
        document.getElementById('s_zone').value=s.zone*100;
        document.getElementById('s_box').value=s.box_min_conf*100;
        document.getElementById('servo_toggle').checked=s.servo_enabled;
        updateLabels();
      });
    }
    function updateLabels(){
      document.getElementById('v_conf').textContent=Math.round(S.conf*100)+'%%';
      document.getElementById('v_fire').textContent=Math.round(S.fire_conf*100)+'%%';
      document.getElementById('v_hold').textContent=S.hold+'s';
      document.getElementById('v_alt').textContent=S.alt_min+'m';
      document.getElementById('v_zone').textContent=Math.round(S.zone*100)+'%%';
      document.getElementById('v_box').textContent=Math.round(S.box_min_conf*100)+'%%';
    }
    function setSetting(k,v){
      S[k]=v; updateLabels();
      fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(S)});
    }

    function update(){
      fetch('/state').then(r=>r.json()).then(d=>{
        const w=img.clientWidth,h=img.clientHeight;
        if(!w||!h)return;
        c.width=w;c.height=h;c.style.width=w+'px';c.style.height=h+'px';
        ctx.clearRect(0,0,w,h);

        // Draw zone
        const zm=(1-S.zone)/2;
        ctx.strokeStyle='rgba(56,139,253,0.2)';ctx.lineWidth=1;ctx.setLineDash([6,4]);
        ctx.strokeRect(w*zm,h*zm,w*S.zone,h*S.zone);ctx.setLineDash([]);

        document.getElementById('fps').textContent=d.fps.toFixed(0);
        document.getElementById('alt').textContent=(d.alt_delta>=0?'+':'')+d.alt_delta.toFixed(1)+'m';

        const st=document.getElementById('status');
        if(d.fired){
          st.textContent='FIRED';st.className='status-bar st-fired';
        } else if(d.det && d.det.confidence>=S.box_min_conf){
          const det=d.det;
          const x1=Math.max(0,det.xmin*w),y1=Math.max(0,det.ymin*h);
          const x2=Math.min(w,det.xmax*w),y2=Math.min(h,det.ymax*h);
          const pct=Math.round(det.confidence*100);
          const clr=d.confirmed?'var(--red)':'var(--green)';
          ctx.shadowColor=d.confirmed?'#da3633':'#2ea043';ctx.shadowBlur=10;
          ctx.strokeStyle=clr;ctx.lineWidth=2;ctx.strokeRect(x1,y1,x2-x1,y2-y1);
          ctx.shadowBlur=0;
          ctx.fillStyle=clr;ctx.font='bold 13px -apple-system,sans-serif';
          ctx.fillText('DRONE '+pct+'%%',x1+4,y1-6);
          if(d.hold>0){
            const prog=Math.min(1,d.hold/S.hold);
            ctx.fillStyle=d.confirmed?'rgba(218,54,51,0.4)':'rgba(46,160,67,0.25)';
            ctx.fillRect(x1,y2-3,(x2-x1)*prog,3);
          }
          document.getElementById('conf').textContent=pct+'%%';
          document.getElementById('hold').textContent=d.hold.toFixed(1)+'s';
          st.textContent=d.confirmed?'ARMED':'TRACKING '+d.hold.toFixed(1)+'s';
          st.className='status-bar '+(d.confirmed?'st-armed':'st-track');
        } else {
          document.getElementById('conf').textContent='--';
          document.getElementById('hold').textContent='0.0s';
          st.textContent=d.airborne?'SCANNING':'GROUNDED';
          st.className='status-bar st-scan';
        }
        const ema=d.ema||0;
        document.getElementById('bar').style.width=(ema*100)+'%%';
        document.getElementById('bar').style.background=ema>=S.fire_conf?'var(--red)':'var(--green)';
      }).catch(()=>{});
    }
    loadSettings();
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
                    "alt_delta": round(alt_delta, 2),
                    "airborne": (baro is None) or (alt_delta > settings["alt_min"]),
                }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(data)
        elif self.path == "/settings":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(settings).encode())
        elif self.path == "/reset-alt":
            reset_altitude()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            print("\n  Altitude reset")
        elif self.path == "/manual-fire":
            fire_servo()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/settings":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                new = json.loads(body)
                settings.update(new)
                save_settings()
                print(f"\n  Settings updated: conf={settings['conf']:.0%} fire={settings['fire_conf']:.0%} hold={settings['hold']}s zone={settings['zone']:.0%}")
            except Exception:
                pass
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

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
    det_nn.setConfidenceThreshold(0.1)  # low threshold, filter in software for live adjustment
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
    print(f"Running — {args.model} ({nn_size[0]}x{nn_size[1]})")

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

            if baro is not None:
                alt_buf.append(baro.altitude)
                baro_alt = sum(alt_buf) / len(alt_buf)
                alt_delta = baro_alt - baro_ref

            is_airborne = (baro is None) or (alt_delta > settings["alt_min"])
            det, hold, confirmed = tracker.update(dets, time.time())

            if confirmed and servo_ready and is_airborne:
                fire_servo()
                tracker.reset()

            with lock:
                tracked_det = det
                track_hold = hold
                track_confirmed = confirmed
                track_ema = tracker.ema_conf
                latest_fps = fps
                fired = not servo_ready

            if det:
                air = "AIR" if is_airborne else "GND"
                st = "ARMED" if confirmed else f"{hold:.1f}s"
                print(f"\r  {det['confidence']:.0%} ema={tracker.ema_conf:.0%} {st} {air} [{fps:.0f}fps]   ", end="", flush=True)

    except KeyboardInterrupt:
        print("\nShutdown")

if servo:
    servo.value = angle_to_value(HOME_ANGLE)
    time.sleep(0.3)
    servo.detach()
server.shutdown()

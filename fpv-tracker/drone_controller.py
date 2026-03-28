"""
Drone Chase Controller v2
=========================
Fixes:
  1. Crosshair correctly centered to actual frame dimensions
  2. Roll/Pitch/Throttle/Yaw update live from PID
  3. Stick boxes explained + made clearer with labels
  4. Bbox scales with simulated distance
  5. Bbox color: green=tracking, yellow=warning, red=critical
  6. Audio beeps based on distance from center

Usage:
  python drone_controller.py --simulate
  python drone_controller.py --list-ports
  python drone_controller.py --port /dev/tty.usbserial-XXXX
"""

import cv2
import numpy as np
import serial
import serial.tools.list_ports
import time
import argparse
import threading
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

# ── Audio ────────────────────────────────────────────────────────────────────

_last_beep_time = 0.0
_beep_lock      = threading.Lock()

def beep_async():
    """Non-blocking system beep."""
    global _last_beep_time
    def _do():
        try:
            import platform, subprocess
            sys = platform.system()
            if sys == "Darwin":
                subprocess.run(["afplay", "/System/Library/Sounds/Tink.aiff"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=0.5)
            elif sys == "Windows":
                import winsound; winsound.Beep(880, 60)
            else:
                subprocess.run(["paplay", "/usr/share/sounds/alsa/Front_Left.wav"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=0.5)
        except:
            pass
    threading.Thread(target=_do, daemon=True).start()

# ── CRSF ─────────────────────────────────────────────────────────────────────

CRSF_SYNC          = 0xC8
CRSF_TYPE_CHANNELS = 0x16
CRSF_MAX           = 1811
CRSF_MIN           = 172

def us_to_crsf(us):
    return int((us - 1000) / 1000 * (CRSF_MAX - CRSF_MIN) + CRSF_MIN)

def crsf_crc8(data):
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0xD5) if (crc & 0x80) else (crc << 1)
            crc &= 0xFF
    return crc

def build_crsf_packet(channels_us):
    ch = [max(CRSF_MIN, min(CRSF_MAX, us_to_crsf(v))) for v in channels_us]
    bits, bc, payload = 0, 0, bytearray()
    for val in ch:
        bits |= (val << bc); bc += 11
        while bc >= 8:
            payload.append(bits & 0xFF); bits >>= 8; bc -= 8
    inner = bytes([CRSF_TYPE_CHANNELS]) + bytes(payload)
    return bytes([CRSF_SYNC, len(inner) + 1]) + inner + bytes([crsf_crc8(inner)])

# ── PID ──────────────────────────────────────────────────────────────────────

@dataclass
class PID:
    kp: float; ki: float; kd: float
    out_min: float = -400.0; out_max: float = 400.0
    _i: float = 0.0; _pe: float = 0.0; _lt: float = 0.0

    def reset(self): self._i = self._pe = self._lt = 0.0

    def compute(self, err, now):
        if self._lt == 0.0:
            self._lt = now; self._pe = err; return 0.0
        dt = now - self._lt
        if dt <= 0: return 0.0
        self._i += err * dt
        max_i = (self.out_max - self.out_min) / max(self.kp, 1e-6)
        self._i = max(-max_i, min(max_i, self._i))
        out = self.kp * err + self.ki * self._i + self.kd * (err - self._pe) / dt
        self._pe = err; self._lt = now
        return max(self.out_min, min(self.out_max, out))

# ── Controller ───────────────────────────────────────────────────────────────

class DroneChaseController:
    def __init__(self, frame_w=640, frame_h=480, base_throttle=1500,
                 simulate=True, serial_port=None, baud=420000):
        self.frame_w = frame_w; self.frame_h = frame_h
        self.cx = frame_w // 2; self.cy = frame_h // 2
        self.base_throttle = base_throttle
        self.simulate = simulate

        self.pid_roll     = PID(kp=1.2, ki=0.05, kd=0.3)
        self.pid_pitch    = PID(kp=1.2, ki=0.05, kd=0.3)
        self.pid_yaw      = PID(kp=0.4, ki=0.0,  kd=0.1)
        self.pid_throttle = PID(kp=0.8, ki=0.02, kd=0.2, out_min=-300, out_max=300)

        self.target_size_ratio = 0.15
        self.dead_zone_px      = 15

        self.channels    = [1500] * 16
        self.armed       = False
        self.roll_us     = 1500
        self.pitch_us    = 1500
        self.throttle_us = 1000
        self.yaw_us      = 1500
        self.error_x     = 0.0
        self.error_y     = 0.0

        self.ser = None
        if not simulate and serial_port:
            try:
                self.ser = serial.Serial(serial_port, baud, timeout=0.1)
                print(f"Connected: {serial_port} @ {baud}")
            except Exception as e:
                print(f"Serial failed: {e} — simulation mode.")
                self.simulate = True

    def arm(self):
        print("Arming..."); self.channels[2] = 1000; self.channels[3] = 1800
        self._send(); time.sleep(2.0); self.channels[3] = 1500
        self._send(); self.armed = True; print("Armed.")

    def disarm(self):
        print("Disarming."); self.channels[2] = 1000; self.channels[3] = 1000
        self._send(); time.sleep(1.0); self.channels[3] = 1500
        self._send(); self.armed = False

    def update(self, bbox):
        now = time.time()
        if bbox is None:
            self.pid_roll.reset(); self.pid_pitch.reset(); self.pid_yaw.reset()
            self.roll_us = self.pitch_us = self.yaw_us = 1500
            self.error_x = self.error_y = 0.0
            self.channels[0] = self.channels[1] = self.channels[3] = 1500
            self._send(); return

        x, y, w, h = bbox
        tcx = x + w // 2; tcy = y + h // 2
        ex = (tcx - self.cx) if abs(tcx - self.cx) > self.dead_zone_px else 0.0
        ey = (tcy - self.cy) if abs(tcy - self.cy) > self.dead_zone_px else 0.0
        nx = ex / (self.frame_w / 2); ny = ey / (self.frame_h / 2)
        dist_err = self.target_size_ratio - w / self.frame_w

        ro = self.pid_roll.compute(nx, now)
        po = self.pid_pitch.compute(ny, now)
        yo = self.pid_yaw.compute(nx, now)
        to = self.pid_throttle.compute(dist_err, now)

        self.roll_us     = int(np.clip(1500 + ro, 1000, 2000))
        self.pitch_us    = int(np.clip(1500 - po, 1000, 2000))
        self.yaw_us      = int(np.clip(1500 + yo, 1000, 2000))
        self.throttle_us = int(np.clip(self.base_throttle + to, 1000, 2000))
        self.error_x     = nx; self.error_y = ny

        self.channels[0] = self.roll_us
        self.channels[1] = self.pitch_us
        self.channels[2] = self.throttle_us if self.armed else 1000
        self.channels[3] = self.yaw_us
        self._send()

    def _send(self):
        if not self.simulate and self.ser and self.ser.is_open:
            try: self.ser.write(build_crsf_packet(self.channels))
            except: pass

    def close(self):
        if self.ser and self.ser.is_open:
            self.disarm(); self.ser.close()

# ── Zone helper ───────────────────────────────────────────────────────────────

# Zone thresholds (normalised distance from centre: 0 = centre, 1 = edge).
# Green  = small central area  (on-target)
# Yellow = medium warning band
# Red    = large edge area      (losing target)
_ZONE_GREEN_MAX  = 0.35   # green when norm <= 0.35
_ZONE_YELLOW_MAX = 0.65   # yellow when 0.35 < norm < 0.65, red beyond

def get_zone(bbox, frame_w, frame_h):
    """Return (colour_bgr, label, norm) based on bbox distance from centre.

    norm uses max(|dx|/half_w, |dy|/half_h): 0 = centre, 1 = edge.
    """
    if bbox is None:
        return (100,100,100), "NO TARGET", 0.0
    x, y, w, h = bbox
    tcx = x + w//2; tcy = y + h//2
    cx = frame_w / 2; cy = frame_h / 2
    norm = max(abs(tcx - cx) / cx, abs(tcy - cy) / cy) if cx and cy else 0.0
    if norm < _ZONE_GREEN_MAX:  return (0,210,80),  "TRACKING",  norm
    if norm < _ZONE_YELLOW_MAX: return (0,200,255), "WARNING",   norm
    return                      (0,60,255),  "CRITICAL",  norm

# ── Draw ──────────────────────────────────────────────────────────────────────

def draw_overlay(frame, bbox, ctrl, fw, fh):
    cx = fw // 2; cy = fh // 2

    # Crosshair at true center
    cv2.line(frame,   (cx-28, cy), (cx+28, cy), (255,255,255), 1)
    cv2.line(frame,   (cx, cy-28), (cx, cy+28), (255,255,255), 1)
    cv2.circle(frame, (cx, cy), 50, (255,255,255), 1)
    cv2.circle(frame, (cx, cy), 3,  (255,255,255), -1)

    color, zone, frac = get_zone(bbox, fw, fh)

    if bbox is not None:
        x, y, w, h = bbox
        tcx = x + w//2; tcy = y + h//2

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.circle(frame, (tcx,tcy), 4, color, -1)
        cv2.arrowedLine(frame, (tcx,tcy), (cx,cy), color, 2, tipLength=0.15)

        err_px = int(math.hypot(tcx-cx, tcy-cy))
        mx, my = (tcx+cx)//2, (tcy+cy)//2
        cv2.putText(frame, f"{err_px}px", (mx+8,my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,255), 1)
        cv2.putText(frame, zone, (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    # Arm badge
    arm_col = (0,210,80) if ctrl.armed else (0,80,255)
    cv2.putText(frame, "ARMED" if ctrl.armed else "DISARMED",
                (12,32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, arm_col, 2)

    # Channel bars
    def ch_bar(label, val, yp):
        cv2.putText(frame, f"{label}: {val}us", (12,yp),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210,210,210), 1)
        bx, bw, bh = 148, 80, 7; by = yp - 7
        cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), (50,50,50), -1)
        fill = int((val-1000)/1000*bw); mid = bw//2
        if fill >= mid:
            cv2.rectangle(frame,(bx+mid,by),(bx+fill,by+bh),(0,190,90),-1)
        else:
            cv2.rectangle(frame,(bx+fill,by),(bx+mid,by+bh),(0,100,210),-1)
        cv2.line(frame,(bx+mid,by-1),(bx+mid,by+bh+1),(160,160,160),1)

    ch_bar("Roll    ", ctrl.roll_us,     66)
    ch_bar("Pitch   ", ctrl.pitch_us,    86)
    ch_bar("Throttle", ctrl.throttle_us, 106)
    ch_bar("Yaw     ", ctrl.yaw_us,      126)
    cv2.putText(frame, f"Err X: {ctrl.error_x:+.3f}  Err Y: {ctrl.error_y:+.3f}",
                (12,148), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160,160,160), 1)

    _draw_sticks(frame, ctrl, fw, fh)

    # Legend
    for i, txt in enumerate(["SPACE: arm/disarm","ESC: quit","S: list ports"]):
        cv2.putText(frame, txt, (fw-185, fh-14-i*17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (110,110,110), 1)
    return frame


def _draw_sticks(frame, ctrl, fw, fh):
    """
    Two RC stick visualizers.
    LEFT  stick: X=Yaw,  Y=Throttle  (dot rests at bottom-center when disarmed)
    RIGHT stick: X=Roll, Y=Pitch     (dot rests at center when no target)
    """
    SZ = 46; PAD = 12; BOT = fh - SZ - PAD - 20
    LX = SZ + PAD + 8
    RX = LX + 2*SZ + PAD + 8

    def stick(cx, cy, xus, yus, title, sub):
        x1,y1 = cx-SZ, cy-SZ; x2,y2 = cx+SZ, cy+SZ
        cv2.rectangle(frame,(x1-1,y1-1),(x2+1,y2+1),(70,70,70),1)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(20,20,20),-1)
        cv2.line(frame,(cx,y1),(cx,y2),(45,45,45),1)
        cv2.line(frame,(x1,cy),(x2,cy),(45,45,45),1)
        dx = int(np.interp(xus,[1000,2000],[-SZ+6,SZ-6]))
        dy = int(np.interp(yus,[1000,2000],[ SZ-6,-SZ+6]))
        cv2.circle(frame,(cx+dx,cy+dy),6,(0,210,80),-1)
        cv2.circle(frame,(cx+dx,cy+dy),6,(0,255,100),1)
        cv2.putText(frame,title,(x1,y2+14),cv2.FONT_HERSHEY_SIMPLEX,0.38,(180,180,180),1)
        cv2.putText(frame,sub,  (x1,y2+26),cv2.FONT_HERSHEY_SIMPLEX,0.30,(100,100,100),1)

    stick(LX, BOT, ctrl.yaw_us, ctrl.throttle_us, "LEFT STICK",  "X=Yaw  Y=Thr")
    stick(RX, BOT, ctrl.roll_us, ctrl.pitch_us,   "RIGHT STICK", "X=Roll Y=Pitch")

# ── Simulated target ──────────────────────────────────────────────────────────

def simulate_bbox(fw, fh, t):
    """
    Drifts across frame including near edges. Size oscillates (approaching/retreating).
    Swap this out for teammate's YOLO bbox.
    """
    tcx = int(fw * 0.5 + math.sin(t * 0.4) * fw * 0.42)
    tcy = int(fh * 0.5 + math.cos(t * 0.3) * fh * 0.38)
    sf  = 0.05 + 0.09 * (0.5 + 0.5 * math.sin(t * 0.18))
    w   = int(fw * sf); h = int(fh * sf * 0.75)
    x   = max(0, min(fw-w, tcx - w//2))
    y   = max(0, min(fh-h, tcy - h//2))
    return (x, y, w, h)

# ── Audio manager ─────────────────────────────────────────────────────────────

class AudioManager:
    """Zone-based beep urgency.

    Green  (norm < 0.20): silence — on target.
    Yellow (0.20 – 0.50): moderate beep rate for course-correction cues.
    Red    (>= 0.50):     rapid high-frequency beeps — target near edge.
    """
    def __init__(self): self._last = 0.0

    def update(self, bbox, fw, fh):
        if bbox is None: return
        x,y,w,h = bbox; tcx=x+w//2; tcy=y+h//2
        cx = fw / 2; cy = fh / 2
        norm = max(abs(tcx - cx) / cx, abs(tcy - cy) / cy) if cx and cy else 0.0

        # Green zone — on target, stay silent
        if norm < _ZONE_GREEN_MAX: return

        # Yellow zone — moderate beep rate
        if norm < _ZONE_YELLOW_MAX:
            t = (norm - _ZONE_GREEN_MAX) / (_ZONE_YELLOW_MAX - _ZONE_GREEN_MAX)
            interval = 0.85 - (0.85 - 0.30) * t   # 0.85s → 0.30s
        else:
            # Red zone — rapid beeps, faster toward the edge
            t = min((norm - _ZONE_YELLOW_MAX) / (1.0 - _ZONE_YELLOW_MAX), 1.0)
            interval = 0.25 - 0.18 * t             # 0.25s → 0.07s

        now = time.time()
        if now - self._last >= interval:
            beep_async(); self._last = now

# ── Main ──────────────────────────────────────────────────────────────────────

def list_ports():
    ports = serial.tools.list_ports.comports()
    if not ports: print("No serial ports found.")
    else:
        print("Available ports:")
        for p in ports: print(f"  {p.device}  —  {p.description}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",       default=None)
    ap.add_argument("--baud",       type=int, default=420000)
    ap.add_argument("--camera",     type=int, default=1)
    ap.add_argument("--width",      type=int, default=640)
    ap.add_argument("--height",     type=int, default=480)
    ap.add_argument("--throttle",   type=int, default=1500)
    ap.add_argument("--simulate",   action="store_true")
    ap.add_argument("--list-ports", action="store_true", dest="lp")
    args = ap.parse_args()

    if args.lp: list_ports(); return

    simulate = args.simulate or args.port is None
    if simulate:
        print("SIMULATION MODE — no drone commands sent.")
        print("Pass --port /dev/tty.usbserial-XXXX to go live.\n")

    ctrl  = DroneChaseController(frame_w=args.width, frame_h=args.height,
                                 base_throttle=args.throttle,
                                 simulate=simulate,
                                 serial_port=args.port, baud=args.baud)
    audio = AudioManager()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"Camera {args.camera} failed, trying 0..."); cap = cv2.VideoCapture(0)

    # Read actual frame dimensions
    ret, frame = cap.read()
    if ret:
        args.height, args.width = frame.shape[:2]
        ctrl.frame_w = args.width;  ctrl.frame_h = args.height
        ctrl.cx = args.width//2;    ctrl.cy = args.height//2

    print("Controls: SPACE=arm/disarm  ESC=quit  S=list ports")
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret: frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)

        # ── SWAP THIS for teammate's YOLO output ──────────────────────────
        # results = yolo_model.predict(frame, conf=0.4)
        # if len(results[0].boxes):
        #     b = results[0].boxes[0].xywh[0].cpu().numpy()
        #     bbox = (int(b[0]-b[2]/2), int(b[1]-b[3]/2), int(b[2]), int(b[3]))
        # else:
        #     bbox = None
        bbox = simulate_bbox(args.width, args.height, time.time()-t0)
        # ─────────────────────────────────────────────────────────────────

        ctrl.update(bbox)
        audio.update(bbox, args.width, args.height)
        frame = draw_overlay(frame, bbox, ctrl, args.width, args.height)
        cv2.imshow("drone controller", frame)

        key = cv2.waitKey(1) & 0xFF
        if   key == 27:        break
        elif key == ord(' '):  ctrl.disarm() if ctrl.armed else ctrl.arm()
        elif key == ord('s'):  list_ports()

    ctrl.close(); cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

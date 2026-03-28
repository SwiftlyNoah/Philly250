"""
Drone Chase Controller v5
=========================
Changes from v4:
  - Removed all speech/voice cues — beep only
  - Beep generated in pure Python (numpy + wave) — no afplay/paplay needed
  - Two distinct tones: HIGH (880 Hz, short) for RED, LOW (440 Hz, longer) for YELLOW
  - Beep intervals: RED every 0.4s, YELLOW every 0.9s, GREEN silent
  - Beep playback: macOS afplay, Windows winsound, Linux aplay — all non-blocking
  - PID uses raw pixel error (not normalized) so bars move visibly in simulation
  - PID debug panel shown top-right

Usage:
  python drone_controller.py --simulate
  python drone_controller.py --list-ports
  python drone_controller.py --port /dev/tty.usbserial-XXXX
"""

import cv2
import numpy as np
import time
import argparse
import threading
import math
import platform
import subprocess
import wave
import tempfile
import os
from dataclasses import dataclass, field

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ── Tone generation ───────────────────────────────────────────────────────────

def _make_wav(freq_hz, duration_s, sample_rate=44100):
    """Generate a sine-wave WAV file, return its temp path."""
    t    = np.linspace(0, duration_s, int(sample_rate * duration_s), False)
    # Fade in/out to avoid click
    fade = int(sample_rate * 0.005)
    tone = np.sin(2 * np.pi * freq_hz * t)
    if fade > 0 and len(tone) > 2 * fade:
        tone[:fade]  *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)
    pcm = (tone * 32767).astype(np.int16)
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    with wave.open(tmp.name, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return tmp.name

# Pre-generate both tones once at startup
_TONE_RED    = _make_wav(freq_hz=1200, duration_s=0.06)   # short high-pitch  — RED
_TONE_YELLOW = _make_wav(freq_hz=520,  duration_s=0.15)   # longer lower-pitch — YELLOW

_SYS = platform.system()

def _play_wav(path):
    """Non-blocking WAV playback — no external dependency on macOS/Windows,
    falls back to aplay on Linux, then terminal bell."""
    def _do():
        try:
            if _SYS == "Darwin":
                subprocess.run(["afplay", path],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1)
            elif _SYS == "Windows":
                import winsound
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                # Try aplay, then paplay, then terminal bell
                for player in (["aplay", "-q", path], ["paplay", path]):
                    result = subprocess.run(player,
                                            stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL,
                                            timeout=1)
                    if result.returncode == 0:
                        return
                print('\a', end='', flush=True)   # terminal bell fallback
        except Exception:
            print('\a', end='', flush=True)
    threading.Thread(target=_do, daemon=True).start()

# ── Audio manager ─────────────────────────────────────────────────────────────

# Zone thresholds (normalised distance from centre: 0 = centre, 1 = edge).
_ZONE_GREEN_MAX  = 0.35   # green when norm <= 0.35
_ZONE_YELLOW_MAX = 0.65   # yellow when 0.35 < norm < 0.65, red beyond

class AudioCue:
    """
    Beep-only urgency feedback (centre-based norm metric).
      CRITICAL (norm >= 0.65): HIGH tone (1200 Hz), every 0.4s
      WARNING  (0.35 < norm < 0.65): LOW tone (520 Hz), every 0.9s
      TRACKING (norm <= 0.35): silent
    """
    def __init__(self):
        self._last_beep = 0.0

    def update(self, bbox, fw, fh):
        if bbox is None:
            return
        x, y, w, h = bbox
        tcx = x + w // 2; tcy = y + h // 2
        cx = fw / 2; cy = fh / 2
        norm = max(abs(tcx - cx) / cx, abs(tcy - cy) / cy) if cx and cy else 0.0
        now = time.time()

        if norm >= _ZONE_YELLOW_MAX:              # RED / CRITICAL
            if now - self._last_beep >= 0.4:
                _play_wav(_TONE_RED)
                self._last_beep = now
        elif norm > _ZONE_GREEN_MAX:              # YELLOW / WARNING
            if now - self._last_beep >= 0.9:
                _play_wav(_TONE_YELLOW)
                self._last_beep = now
        # GREEN / TRACKING: no beep

# ── CRSF ──────────────────────────────────────────────────────────────────────

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

# ── PID ───────────────────────────────────────────────────────────────────────

@dataclass
class PID:
    kp: float; ki: float; kd: float
    out_min: float = -500.0; out_max: float = 500.0
    _i:      float = field(default=0.0, repr=False)
    _pe:     float = field(default=0.0, repr=False)
    _lt:     float = field(default=0.0, repr=False)
    p_term:  float = field(default=0.0, repr=False)
    i_term:  float = field(default=0.0, repr=False)
    d_term:  float = field(default=0.0, repr=False)
    output:  float = field(default=0.0, repr=False)

    def reset(self):
        self._i = self._pe = self._lt = 0.0
        self.p_term = self.i_term = self.d_term = self.output = 0.0

    def compute(self, err, now):
        if self._lt == 0.0:
            # First call: return P-only immediately so bars move on frame 1
            self._lt    = now
            self._pe    = err
            self.p_term = self.kp * err
            self.output = max(self.out_min, min(self.out_max, self.p_term))
            return self.output
        dt = now - self._lt
        if dt <= 0:
            return self.output
        self._i    += err * dt
        max_i       = (self.out_max - self.out_min) / max(self.kp, 1e-6)
        self._i     = max(-max_i, min(max_i, self._i))
        self.p_term = self.kp * err
        self.i_term = self.ki * self._i
        self.d_term = self.kd * (err - self._pe) / dt
        out         = self.p_term + self.i_term + self.d_term
        self._pe    = err
        self._lt    = now
        self.output = max(self.out_min, min(self.out_max, out))
        return self.output

# ── Controller ────────────────────────────────────────────────────────────────

class DroneChaseController:
    def __init__(self, frame_w=640, frame_h=480, base_throttle=1500,
                 simulate=True, serial_port=None, baud=420000):
        self.frame_w = frame_w; self.frame_h = frame_h
        self.cx = frame_w // 2; self.cy = frame_h // 2
        self.base_throttle = base_throttle
        self.simulate      = simulate

        # Gains for RAW PIXEL error:
        #   200px off-center → kp=1.75 → ~350 us output — clearly visible on bars
        self.pid_roll     = PID(kp=1.75, ki=0.08, kd=0.40, out_min=-500, out_max=500)
        self.pid_pitch    = PID(kp=1.75, ki=0.08, kd=0.40, out_min=-500, out_max=500)
        self.pid_yaw      = PID(kp=0.60, ki=0.0,  kd=0.12, out_min=-300, out_max=300)
        # Throttle: dist_err is float ~0.0–0.15; kp=1200 → 0.10 err → 120 us
        self.pid_throttle = PID(kp=1200, ki=20.0, kd=80.0, out_min=-350, out_max=350)

        self.target_size_ratio = 0.15
        self.dead_zone_px      = 8

        self.channels    = [1500] * 16
        self.armed       = False
        self.roll_us     = 1500
        self.pitch_us    = 1500
        self.throttle_us = base_throttle
        self.yaw_us      = 1500
        self.error_x_px  = 0.0
        self.error_y_px  = 0.0
        self.error_x_n   = 0.0   # normalized [-1,+1] for HUD
        self.error_y_n   = 0.0

        self.ser = None
        if not simulate and serial_port and HAS_SERIAL:
            try:
                self.ser = serial.Serial(serial_port, baud, timeout=0.1)
                print(f"Connected: {serial_port} @ {baud}")
            except Exception as e:
                print(f"Serial failed: {e} — simulation mode.")
                self.simulate = True

    def arm(self):
        print("Arming...")
        self.channels[2] = 1000; self.channels[3] = 1800
        self._send(); time.sleep(2.0)
        self.channels[3] = 1500; self._send()
        self.armed = True; print("Armed.")

    def disarm(self):
        print("Disarming.")
        self.channels[2] = 1000; self.channels[3] = 1000
        self._send(); time.sleep(1.0)
        self.channels[3] = 1500; self._send()
        self.armed = False

    def update(self, bbox):
        now = time.time()
        if bbox is None:
            for p in (self.pid_roll, self.pid_pitch, self.pid_yaw, self.pid_throttle):
                p.reset()
            self.roll_us = self.pitch_us = self.yaw_us = 1500
            self.throttle_us = self.base_throttle
            self.error_x_px = self.error_y_px = 0.0
            self.error_x_n  = self.error_y_n  = 0.0
            self.channels[0] = self.channels[1] = self.channels[3] = 1500
            self._send(); return

        x, y, w, h = bbox
        tcx = x + w // 2; tcy = y + h // 2

        raw_ex = float(tcx - self.cx)
        raw_ey = float(tcy - self.cy)
        ex = raw_ex if abs(raw_ex) > self.dead_zone_px else 0.0
        ey = raw_ey if abs(raw_ey) > self.dead_zone_px else 0.0

        dist_err = self.target_size_ratio - w / self.frame_w

        ro = self.pid_roll.compute(ex, now)
        po = self.pid_pitch.compute(ey, now)
        yo = self.pid_yaw.compute(ex, now)
        to = self.pid_throttle.compute(dist_err, now)

        self.roll_us     = int(np.clip(1500 + ro, 1000, 2000))
        self.pitch_us    = int(np.clip(1500 - po, 1000, 2000))
        self.yaw_us      = int(np.clip(1500 + yo, 1000, 2000))
        self.throttle_us = int(np.clip(self.base_throttle + to, 1000, 2000))
        self.error_x_px  = ex
        self.error_y_px  = ey
        self.error_x_n   = ex / (self.frame_w / 2)
        self.error_y_n   = ey / (self.frame_h / 2)

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

# ── Zone ──────────────────────────────────────────────────────────────────────

def get_zone(bbox, frame_w, frame_h):
    """Return (colour_bgr, label, norm) based on bbox distance from centre.

    norm uses max(|dx|/half_w, |dy|/half_h): 0 = centre, 1 = edge.
    """
    if bbox is None:
        return (100, 100, 100), "NO TARGET", 0.0
    x, y, w, h = bbox
    tcx = x + w // 2; tcy = y + h // 2
    cx = frame_w / 2; cy = frame_h / 2
    norm = max(abs(tcx - cx) / cx, abs(tcy - cy) / cy) if cx and cy else 0.0
    if norm <= _ZONE_GREEN_MAX:  return (0, 210, 80),  "TRACKING",  norm
    if norm < _ZONE_YELLOW_MAX:  return (0, 200, 255), "WARNING",   norm
    return                       (0, 60, 255),  "CRITICAL",  norm

# ── Draw ──────────────────────────────────────────────────────────────────────

def draw_overlay(frame, bbox, ctrl, fw, fh):
    cx = fw // 2; cy = fh // 2

    cv2.line(frame,   (cx - 28, cy), (cx + 28, cy), (255, 255, 255), 1)
    cv2.line(frame,   (cx, cy - 28), (cx, cy + 28), (255, 255, 255), 1)
    cv2.circle(frame, (cx, cy), 50, (255, 255, 255), 1)
    cv2.circle(frame, (cx, cy),  3, (255, 255, 255), -1)

    color, zone, frac = get_zone(bbox, fw, fh)

    if bbox is not None:
        x, y, w, h = bbox
        tcx = x + w // 2; tcy = y + h // 2
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (tcx, tcy), 4, color, -1)
        cv2.arrowedLine(frame, (tcx, tcy), (cx, cy), color, 2, tipLength=0.15)
        err_px = int(math.hypot(tcx - cx, tcy - cy))
        mx, my = (tcx + cx) // 2, (tcy + cy) // 2
        cv2.putText(frame, f"{err_px}px", (mx + 8, my),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(frame, zone, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    arm_col = (0, 210, 80) if ctrl.armed else (0, 80, 255)
    cv2.putText(frame, "ARMED" if ctrl.armed else "DISARMED",
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, arm_col, 2)

    def ch_bar(label, val, yp):
        active = abs(val - 1500) > 5
        col    = (0, 220, 255) if active else (200, 200, 200)
        cv2.putText(frame, f"{label}: {val}us", (12, yp),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)
        bx, bw, bh = 160, 90, 8; by = yp - 8
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (50, 50, 50), -1)
        fill = int((val - 1000) / 1000 * bw)
        mid  = bw // 2
        if fill > mid:
            cv2.rectangle(frame, (bx + mid, by), (bx + fill, by + bh), (0, 190, 90), -1)
        elif fill < mid:
            cv2.rectangle(frame, (bx + fill, by), (bx + mid, by + bh), (0, 100, 210), -1)
        cv2.line(frame, (bx + mid, by - 1), (bx + mid, by + bh + 1), (160, 160, 160), 1)
        if active:
            cv2.rectangle(frame, (bx - 1, by - 1), (bx + bw + 1, by + bh + 1), col, 1)

    ch_bar("Roll    ", ctrl.roll_us,     66)
    ch_bar("Pitch   ", ctrl.pitch_us,    86)
    ch_bar("Thr[SIM]" if not ctrl.armed else "Throttle", ctrl.throttle_us, 106)
    ch_bar("Yaw     ", ctrl.yaw_us,      126)

    cv2.putText(frame,
                f"Err X: {ctrl.error_x_n:+.3f} ({int(ctrl.error_x_px):+d}px)  "
                f"Err Y: {ctrl.error_y_n:+.3f} ({int(ctrl.error_y_px):+d}px)",
                (12, 148), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (160, 160, 160), 1)

    _draw_pid_debug(frame, ctrl, fw)
    _draw_sticks(frame, ctrl, fw, fh)

    for i, txt in enumerate(["SPACE: arm/disarm", "ESC: quit", "S: list ports"]):
        cv2.putText(frame, txt, (fw - 195, fh - 14 - i * 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (110, 110, 110), 1)
    return frame


def _draw_pid_debug(frame, ctrl, fw):
    px = fw - 315; py = 18
    cv2.putText(frame, "── PID DEBUG ──", (px, py),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80, 180, 255), 1)

    def row(label, pid, yp):
        cv2.putText(frame,
                    f"{label}  P:{pid.p_term:+6.1f}  I:{pid.i_term:+5.1f}"
                    f"  D:{pid.d_term:+5.1f}  >{pid.output:+6.1f}",
                    (px, yp), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (150, 210, 150), 1)

    row("ROL", ctrl.pid_roll,     py + 14)
    row("PIT", ctrl.pid_pitch,    py + 26)
    row("YAW", ctrl.pid_yaw,      py + 38)
    row("THR", ctrl.pid_throttle, py + 50)


def _draw_sticks(frame, ctrl, fw, fh):
    SZ = 52; PAD = 16
    BOT = fh - SZ - PAD - 30
    LX  = SZ + PAD + 10
    RX  = LX + 2 * SZ + PAD + 18

    def pad(scx, scy, xus, yus, title, sub):
        x1, y1 = scx - SZ, scy - SZ
        x2, y2 = scx + SZ, scy + SZ
        cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (70, 70, 70), 1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 20, 20), -1)
        cv2.line(frame, (scx, y1), (scx, y2), (45, 45, 45), 1)
        cv2.line(frame, (x1, scy), (x2, scy), (45, 45, 45), 1)
        margin = SZ - 8
        dx = int(np.interp(xus, [1000, 2000], [-margin,  margin]))
        dy = int(np.interp(yus, [1000, 2000], [ margin, -margin]))
        dev = math.hypot(dx, dy) / margin
        dot_col = ((0, 210, 80)   if dev < 0.25 else
                   (0, 200, 255)  if dev < 0.60 else
                   (0, 60,  255))
        cv2.circle(frame, (scx + dx, scy + dy), 7, dot_col, -1)
        cv2.circle(frame, (scx + dx, scy + dy), 7, (255, 255, 255), 1)
        cv2.putText(frame, f"X:{xus}", (x1,      y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1)
        cv2.putText(frame, f"Y:{yus}", (x2 - 44, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (80, 80, 80), 1)
        cv2.putText(frame, title, (x1, y2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        cv2.putText(frame, sub,   (x1, y2 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100, 100, 100), 1)

    pad(LX, BOT, ctrl.yaw_us,  ctrl.throttle_us, "LEFT STICK",  "X=Yaw  Y=Thr")
    pad(RX, BOT, ctrl.roll_us, ctrl.pitch_us,    "RIGHT STICK", "X=Roll Y=Pitch")

# ── Simulated target ──────────────────────────────────────────────────────────

def simulate_bbox(fw, fh, t):
    tcx = int(fw * 0.5 + math.sin(t * 0.4) * fw * 0.42)
    tcy = int(fh * 0.5 + math.cos(t * 0.3) * fh * 0.38)
    sf  = 0.05 + 0.09 * (0.5 + 0.5 * math.sin(t * 0.18))
    w   = int(fw * sf); h = int(fh * sf * 0.75)
    x   = max(0, min(fw - w, tcx - w // 2))
    y   = max(0, min(fh - h, tcy - h // 2))
    return (x, y, w, h)

# ── Ports ─────────────────────────────────────────────────────────────────────

def list_ports():
    if not HAS_SERIAL:
        print("pyserial not installed."); return
    ports = serial.tools.list_ports.comports()
    if not ports: print("No serial ports found.")
    else:
        print("Available ports:")
        for p in ports: print(f"  {p.device}  —  {p.description}")

# ── Main ──────────────────────────────────────────────────────────────────────

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
    audio = AudioCue()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        print(f"Camera {args.camera} failed, trying 0...")
        cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    if ret:
        args.height, args.width = frame.shape[:2]
        ctrl.frame_w = args.width;  ctrl.frame_h = args.height
        ctrl.cx = args.width // 2;  ctrl.cy = args.height // 2

    print("Controls: SPACE=arm/disarm  ESC=quit  S=list ports")
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)

        # ── SWAP for teammate's YOLO output ──────────────────────────────
        # results = yolo_model.predict(frame, conf=0.4)
        # if len(results[0].boxes):
        #     b = results[0].boxes[0].xywh[0].cpu().numpy()
        #     bbox = (int(b[0]-b[2]/2), int(b[1]-b[3]/2), int(b[2]), int(b[3]))
        # else:
        #     bbox = None
        bbox = simulate_bbox(args.width, args.height, time.time() - t0)
        # ─────────────────────────────────────────────────────────────────

        ctrl.update(bbox)
        audio.update(bbox, args.width, args.height)
        frame = draw_overlay(frame, bbox, ctrl, args.width, args.height)
        cv2.imshow("drone controller", frame)

        key = cv2.waitKey(1) & 0xFF
        if   key == 27:        break
        elif key == ord(' '):  ctrl.disarm() if ctrl.armed else ctrl.arm()
        elif key == ord('s'):  list_ports()

    # Cleanup temp tone files
    for f in (_TONE_RED, _TONE_YELLOW):
        try: os.unlink(f)
        except: pass

    ctrl.close(); cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

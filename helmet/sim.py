"""Traffic simulator: scripted cars through the REAL pipeline, no camera or YOLO needed.

    python -m helmet.main --sim                  # then open http://<ip>:8080/view
    python -m helmet.main --sim --demo-person    # teammates walking/jogging at you instead of cars
    python -m helmet.main --sim --collision      # plus the front camera: walking at a chair, BRAKE

Each scene moves cars around the rider in metres. They are projected to boxes with the same
pinhole model the perception code inverts, jittered like YOLO boxes, and handed to the real
tracker, TTC, alert policy, OLED HUD and Arduino. So alerts, buzzes and the web view behave
exactly as they would with a camera. Passing cars keep going past the rider, out of the
camera's view, and the fitted side ultrasonic sensors (SONAR_MOUNT) get simulated echoes, so
the camera -> sensor hand-off can be tested too. Uses: building the web view without hardware, and a
guaranteed demo backup if the room's lighting defeats the camera.
"""
import math
import random
import time

import cv2
import numpy as np

from . import config as cfg
from .perception import Detection, focal_px

W, H = cfg.CAPTURE_WIDTH, cfg.CAPTURE_HEIGHT
FPS = 15.0
CAM_HEIGHT_M = 1.5            # helmet camera height above the road
CAR_W, CAR_H = 1.8, 1.45      # rear view of a car, metres
PERSON_W, PERSON_H = 0.5, 1.75


CAM_MIN_Z = 0.6              # closer than this (or beside/ahead) the rear camera can't see it
FRONT_CAM_HEIGHT_M = 1.6      # helmet front camera; objects stand on the floor ahead
FRONT_SIZE = {"chair": (0.5, 0.9), "person": (PERSON_W, PERSON_H)}   # width, height seen face-on (m)
FRONT_EVERY = 2               # front YOLO runs on every 2nd frame: it shares the model with the rear


def _approach(x0, x1, z0, speed, until=0.6):
    """Road user closing at `speed` m/s from z0, drifting laterally x0 -> x1 by the time it arrives.
    `until` < 0 means it keeps going past the rider (overtakes) until it is that far ahead."""
    def pos(t):
        z = z0 - speed * t
        if z < until:
            return None
        return x0 + (x1 - x0) * (1 - max(z, 0.0) / z0), z
    return pos, (z0 - until) / speed


def _hold(x, z):
    return (lambda t: (x, z)), None


# (title, [car position functions], duration s)
def _scenes():
    s = []
    p, d = _approach(0.0, 0.0, 34.0, 9.0)
    s.append(("Car closing fast from directly behind", [p], d + 1.5))
    p, d = _approach(-2.6, -2.6, 32.0, 8.0, until=-3.0)
    s.append(("Car passing on the left, normal gap", [p], d + 1.5))
    p, d = _approach(-3.0, 0.0, 32.0, 9.0)
    s.append(("Car cutting into your lane from the left", [p], d + 1.5))
    p, d = _approach(1.6, 1.6, 32.0, 10.0, until=-3.0)
    s.append(("Close pass on the right", [p], d + 1.5))
    p, d = _approach(-1.5, -1.5, 32.0, 9.0, until=-3.0)
    s.append(("Close pass on the left: the side sensor measures the gap", [p], d + 1.5))
    far, _ = _approach(0.4, 0.4, 26.0, 1.5, until=14.0)
    near, d = _approach(-2.8, -2.8, 30.0, 8.0, until=-3.0)
    s.append(("Two cars: one slow behind, one passing left", [far, near], d + 1.5))
    p, _ = _hold(0.2, 9.0)
    s.append(("Car following at a steady 9 m: no alert", [p], 6.0))
    return s


def _demo_scenes():
    """The stationary demo: teammates walking and jogging at the rider (README calibration step 7)."""
    s = []
    p, d = _approach(0.0, 0.0, 10.0, 2.6, until=0.8)
    s.append(("Teammate jogging straight at you", [p], d + 1.5))
    p, d = _approach(-1.5, -1.5, 9.0, 1.4, until=-1.0)
    s.append(("Teammate walking past on your left (1.5 m)", [p], d + 1.5))
    p, d = _approach(-2.2, 0.0, 9.0, 1.6, until=0.8)
    s.append(("Teammate cutting into your path from the left", [p], d + 1.5))
    p, d = _approach(0.95, 0.95, 9.0, 1.8, until=-1.0)
    s.append(("Teammate brushing past on your right", [p], d + 1.5))
    p, d = _approach(-0.95, -0.95, 9.0, 1.8, until=-1.0)
    s.append(("Teammate brushing past on your left: the side sensor measures the gap", [p], d + 1.5))

    def step_in(t):                          # appears beside you, where the camera can't see
        if t > 5.0:
            return None
        x = -2.6 + min(t, 1.5) / 1.5 * 1.5 if t < 3.5 else -1.1 - (t - 3.5) / 1.5 * 1.5
        return x, 0.1
    s.append(("Teammate steps in beside you: only the side sensor can see there", [step_in], 5.5))
    p, _ = _hold(0.3, 4.0)
    s.append(("Teammate standing still 4 m behind: no alert", [p], 5.0))
    return s


def _walk_to(x, z0, speed, stop_at):
    """The rider (or both of you) closing at `speed` from z0 ahead; stops `stop_at` metres short."""
    return lambda t: (x, max(stop_at, z0 - speed * t))


def _front_scenes():
    """Front camera, rider's view: x = metres to your right, z = metres ahead.
    (title, label, position function, duration s)"""
    return [
        ("Walking at a chair: BRAKE before you reach it", "chair", _walk_to(0.0, 9.0, 1.4, 0.9), 9.0),
        ("Walking past a chair 1.5 m to your right: no BRAKE", "chair", _walk_to(1.5, 9.0, 1.4, -1.0), 7.0),
        ("Standing 3 m from a chair: no alert", "chair", _walk_to(0.1, 3.0, 0.0, 3.0), 5.0),
        ("Person walking at you while you walk: BRAKE", "person", _walk_to(0.3, 12.0, 2.6, 1.2), 6.0),
        ("Backing away from a chair: no alert", "chair", lambda t: (0.0, 1.5 + 1.0 * t), 5.0),
    ]


class FrontScenario:
    def __init__(self, seed=11):
        self.scenes = _front_scenes()
        self.rng = random.Random(seed)
        self.idx, self.t_scene = 0, 0.0

    def title(self):
        return self.scenes[self.idx][0]

    def step(self, dt):
        """Advance time; returns [(label, x, z)] for objects ahead."""
        self.t_scene += dt
        if self.t_scene > self.scenes[self.idx][3]:
            self.idx, self.t_scene = (self.idx + 1) % len(self.scenes), 0.0
        _, label, pos, _ = self.scenes[self.idx]
        x, z = pos(self.t_scene)
        return [(label, x, z)] if z > 0.3 else []

    def boxes(self, objects):
        """Front-camera boxes (not mirrored: image right = your right), with YOLO-like jitter."""
        f = focal_px(W)
        out = []
        for label, x, z in objects:
            size_w, size_h = FRONT_SIZE[label]
            wj = 0.04 if label == "person" else 0.02
            w = f * size_w / z * (1 + self.rng.uniform(-wj, wj))
            h = f * size_h / z
            cx = W / 2 + f * x / z + self.rng.uniform(-0.02, 0.02) * w
            y2 = H / 2 + f * FRONT_CAM_HEIGHT_M / z + self.rng.uniform(-0.015, 0.015) * h
            y1 = H / 2 + f * (FRONT_CAM_HEIGHT_M - size_h) / z + self.rng.uniform(-0.015, 0.015) * h
            x1, x2 = max(0.0, cx - w / 2), min(W - 1.0, cx + w / 2)
            y1, y2 = max(0.0, y1), min(H - 1.0, y2)
            if x2 - x1 >= 4 and y2 > y1:
                out.append(Detection(x1, y1, x2, y2, 0.9, label))
        return out


class Scenario:
    def __init__(self, seed=7, people=False):
        self.people = people
        self.scenes = _demo_scenes() if people else _scenes()
        self.label, self.size = ("person", (PERSON_W, PERSON_H)) if people else ("car", (CAR_W, CAR_H))
        self.half = (PERSON_W / 2, 0.2) if people else (CAR_W / 2, 2.2)   # footprint: half width, half length
        self.rng = random.Random(seed)
        self.idx, self.t_scene = 0, 0.0

    def title(self):
        return self.scenes[self.idx][0]

    def step(self, dt):
        """Advance time; returns [(x, z), ...] for cars currently in the scene."""
        self.t_scene += dt
        title, cars, dur = self.scenes[self.idx]
        if self.t_scene > dur:
            self.idx = (self.idx + 1) % len(self.scenes)
            self.t_scene = 0.0
            title, cars, dur = self.scenes[self.idx]
        return [p for p in (f(self.t_scene) for f in cars) if p is not None]

    def boxes(self, cars):
        """Rider-view boxes (x right = rider's right), far first, YOLO-like jitter."""
        f = focal_px(W)
        size_w, size_h = self.size
        wj = 0.04 if self.people else 0.02                     # arm swing makes a walker's width noisier
        out = []
        for x, z in sorted(cars, key=lambda c: -c[1]):
            if z < CAM_MIN_Z:
                continue                                        # beside or past you: out of the rear camera's view
            # YOLO box jitter scales with box size: ~2% of width, ~1.5% of height per edge
            w = f * size_w / z * (1 + self.rng.uniform(-wj, wj))
            h = f * size_h / z
            cx = W / 2 + f * x / z + self.rng.uniform(-0.02, 0.02) * w
            y2 = H / 2 + f * CAM_HEIGHT_M / z + self.rng.uniform(-0.015, 0.015) * h
            y1 = H / 2 + f * (CAM_HEIGHT_M - size_h) / z + self.rng.uniform(-0.015, 0.015) * h
            x1, x2 = max(0.0, cx - w / 2), min(W - 1.0, cx + w / 2)
            y1, y2 = max(0.0, y1), min(H - 1.0, y2)
            if x2 - x1 >= 4 and y2 > y1:
                out.append(Detection(x1, y1, x2, y2, 0.9, self.label))
        return out


    def sonar(self, objects):
        """Simulated side-sensor echoes: nearest object surface inside each fitted sensor's
        +/-15 deg cone, within 3 m (metres or None), with a little noise and the odd dropout."""
        out = {}
        hw, hl = self.half
        for name, (zone, yaw) in cfg.SONAR_MOUNT.items():
            side = -1 if zone == "LEFT" else 1
            sx, best = side * cfg.SONAR_OFFSET_M, None
            for off in range(-15, 16, 3):
                a = math.radians(yaw + off)
                dx, dz = side * math.cos(a), math.sin(a)
                for x, z in objects:
                    t = _ray_box(sx, 0.0, dx, dz, x - hw, x + hw, z - hl, z + hl)
                    if t is not None and (best is None or t < best):
                        best = t
            # A real HC-SR04 reads nothing closer than a few cm (e.g. a scripted car overlapping the rider)
            if best is not None and cfg.SONAR_MIN_M <= best <= cfg.SONAR_MAX_CM / 100.0 and self.rng.random() > 0.05:
                out[name] = round(max(cfg.SONAR_MIN_M, best + self.rng.uniform(-0.01, 0.01)), 3)
            else:
                out[name] = None
        return out


def _ray_box(ox, oz, dx, dz, x0, x1, z0, z1):
    """Distance along a ray (origin o, unit direction d) to an axis-aligned box, or None."""
    tmin, tmax = 0.0, math.inf
    for o, d, lo, hi in ((ox, dx, x0, x1), (oz, dz, z0, z1)):
        if abs(d) < 1e-9:
            if not lo <= o <= hi:
                return None
            continue
        ta, tb = (lo - o) / d, (hi - o) / d
        tmin, tmax = max(tmin, min(ta, tb)), min(tmax, max(ta, tb))
    return tmin if tmin <= tmax else None


def render(dets, title):
    """A simple rear-camera-like picture so the camera inset shows the same scene."""
    img = np.zeros((H, W, 3), np.uint8)
    img[: H // 2] = (70, 58, 48)                               # sky
    img[H // 2:] = (62, 62, 62)                                # road
    vp = (W // 2, H // 2)
    for x_edge in (-6.0, 6.0):
        cv2.line(img, vp, (int(W / 2 + focal_px(W) * x_edge / 2.0), H), (150, 150, 150), 2)
    for d in dets:                                             # far first, so near ones cover them
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        if d.label == "person":
            bw = x2 - x1
            cv2.rectangle(img, (x1 + bw // 5, y1 + bw // 2), (x2 - bw // 5, y2), (190, 170, 150), -1)
            cv2.circle(img, ((x1 + x2) // 2, y1 + bw // 4), max(2, bw // 4), (160, 190, 220), -1)
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), (205, 205, 210), -1)
        wh = max(1, (y2 - y1) // 3)
        cv2.rectangle(img, (x1 + (x2 - x1) // 6, y1 + wh // 4), (x2 - (x2 - x1) // 6, y1 + wh), (60, 50, 45), -1)
        lh = max(1, (y2 - y1) // 8)
        for lx in (x1 + 2, x2 - max(3, (x2 - x1) // 6)):
            cv2.rectangle(img, (lx, y1 + wh + lh), (lx + max(2, (x2 - x1) // 7), y1 + wh + 2 * lh), (40, 40, 230), -1)
    cv2.putText(img, f"SIMULATOR: {title}", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def render_front(dets, title):
    """A simple indoor front-camera picture: wall, floor, and the scripted chair or person."""
    img = np.zeros((H, W, 3), np.uint8)
    img[: H // 2] = (88, 80, 72)                               # wall
    img[H // 2:] = (60, 70, 80)                                # floor
    for d in dets:
        x1, y1, x2, y2 = int(d.x1), int(d.y1), int(d.x2), int(d.y2)
        bw, bh = x2 - x1, y2 - y1
        if d.label == "person":
            cv2.rectangle(img, (x1 + bw // 5, y1 + bw // 2), (x2 - bw // 5, y2), (190, 170, 150), -1)
            cv2.circle(img, ((x1 + x2) // 2, y1 + bw // 4), max(2, bw // 4), (160, 190, 220), -1)
            continue
        cv2.rectangle(img, (x1, y1), (x2, y1 + bh // 2), (40, 70, 120), -1)                  # back rest
        cv2.rectangle(img, (x1, y1 + bh // 2), (x2, y1 + bh // 2 + max(2, bh // 10)), (50, 90, 150), -1)  # seat
        for lx in (x1, x2 - max(2, bw // 10)):
            cv2.rectangle(img, (lx, y1 + bh // 2), (lx + max(2, bw // 10), y2), (30, 50, 90), -1)      # legs
    cv2.putText(img, f"SIMULATOR (front): {title}", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                cv2.LINE_AA)                                   # bottom: the front preview's status lines are at the top
    return img


class SimSource:
    """Stands in for the camera. Frames come out as the camera would give them (flipped if
    MIRROR_VIEW), so main.py's mirror step turns them back into the rider's view."""
    is_live = False

    def __init__(self, seed=7):
        self.scenario = Scenario(seed, people=cfg.DEMO_PERSON_AS_VEHICLE)
        self.current = []                  # rider-view detections for the latest frame
        self.objects = []                  # every road user's (x, z), including ones the camera can't see
        self.paused, self.ended = False, False
        self.t, self._last = 0.0, None
        self.front_engine = None           # set by forward.enable_forward for --sim --collision
        self.front = FrontScenario()
        self.frame_count = 0

    @property
    def title(self):
        if self.front_engine is None:
            return self.scenario.title()
        return f"{self.scenario.title()} | front: {self.front.title()}"

    def read(self, timeout=0.1):
        now = time.monotonic()
        if self._last is None:
            self._last = now - 1 / FPS
        wait = self._last + 1 / FPS - now
        if wait > 0:
            time.sleep(wait)
        now = time.monotonic()
        dt, self._last = now - self._last, now
        if self.paused:
            return None, None
        self.t += dt
        self.objects = self.scenario.step(dt)
        self.current = self.scenario.boxes(self.objects)
        frame = render(self.current, self.scenario.title())
        if cfg.MIRROR_VIEW:
            frame = cv2.flip(frame, 1)
        if self.front_engine is not None:
            self.step_front(dt)
        return frame, self.t

    def step_front(self, dt):
        idx = self.front.idx
        dets = self.front.boxes(self.front.step(dt))
        if self.front.idx != idx:
            self.front_engine.reset()      # scripted scene cut: don't carry one object's history into the next
        front = render_front(dets, self.front.title())
        self.front_engine.on_frame(front, self.t)
        self.frame_count += 1
        if self.frame_count % FRONT_EVERY == 0:
            self.front_engine.on_detections(dets, self.t, None, front)

    def sonar_readings(self):
        return self.scenario.sonar(self.objects)

    def toggle_pause(self):
        self.paused = not self.paused

    def stale_for(self):
        return 0.0

    def release(self):
        pass


class SimDetector:
    """Returns the scripted boxes for the frame SimSource just produced (no YOLO)."""
    model_name = "simulator (no camera, no YOLO)"
    last_ms = 0.0

    def __init__(self, source):
        self.source = source

    def detect(self, frame):
        return list(self.source.current)

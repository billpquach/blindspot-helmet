"""Camera-only front warning (no calibration): timing, false alarms, outputs and wiring.
Run: python -m tests.test_front_mono (no YOLO download, cameras, GPIO, or network).

The timing tests are the demo's evidence: simulated approaches with YOLO-like box jitter
at the rates the front camera really gets (it shares YOLO with the rear, so 5-7.5 detections
per second), and BRAKE must fire before the marker mode's physical stopping distance.
"""
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

import cv2
import numpy as np

from helmet import config as cfg
from helmet import snapshot
from helmet.collision import CollisionPolicy, Settings, stopping_distance
from helmet.forward import DualLiveSource, DualReplaySource
from helmet.front_mono import MonoFrontEngine
from helmet.hud import Hud, render
from helmet.outputs import Beeper
from helmet.sim import FrontScenario, SimSource
from tests.test_collision import calibration
from tools.eval_front import QUIET, approach

BLANK = np.zeros((480, 640, 3), np.uint8)


class TimingTests(unittest.TestCase):
    def test_brake_fires_before_the_stopping_distance(self):
        for hz, speeds in ((7.5, (1.0, 1.4, 2.0, 2.6)), (5.0, (1.0, 1.4, 2.0))):
            for v in speeds:
                need = stopping_distance(v, Settings())
                for seed in range(10):
                    gap, _ = approach("chair", 0.0, 10.0, v, hz, seed)
                    self.assertIsNotNone(gap, f"no BRAKE at {v} m/s, {hz} Hz, seed {seed}")
                    self.assertGreaterEqual(gap, need, f"late BRAKE at {v} m/s, {hz} Hz, seed {seed}")

    def test_person_walking_at_you_and_head_sway(self):
        gap, _ = approach("person", 0.3, 12.0, 2.6, seed=3)
        self.assertGreaterEqual(gap, stopping_distance(2.6, Settings()))
        for seed in range(5):
            gap, _ = approach("chair", 0.0, 10.0, 1.4, seed=seed, sway_px=60)
            self.assertGreaterEqual(gap, stopping_distance(1.4, Settings()))

    def test_no_brake_when_stationary_receding_or_passing(self):
        for name, (label, x, z0, v, stop) in QUIET.items():
            for seed in range(10):
                _, rows = approach(label, x, z0, v, seed=seed, stop_at=stop, seconds=7.0, sway_px=30)
                self.assertNotIn("BRAKE", {r["state"] for r in rows}, f"{name}, seed {seed}")
                self.assertTrue(any(r["obstacles"] for r in rows), f"{name}: object never tracked")
                if v <= 0:
                    self.assertNotIn("CAUTION", {r["state"] for r in rows}, f"{name}, seed {seed}")

    def test_stopping_clears_after_the_hold(self):
        _, rows = approach("chair", 0.0, 8.0, 1.4, stop_at=1.2, seconds=10.0)
        states = [r["state"] for r in rows]
        self.assertIn("BRAKE", states)
        self.assertEqual(states[-1], "CLEAR")               # you stopped short: the warning goes away


class EngineTests(unittest.TestCase):
    def test_result_matches_marker_fields_and_is_json(self):
        _, rows = approach("chair", 0.0, 8.0, 1.4)
        marker_fields = set(CollisionPolicy(calibration()).result)
        for r in rows:
            self.assertLessEqual(marker_fields, set(r))
            json.dumps(r, allow_nan=False)
        brake = next(r for r in rows if r["state"] == "BRAKE")
        self.assertEqual((brake["mode"], brake["detected_label"], brake["display_asset"]), ("camera-only", "chair", "tree"))
        self.assertTrue(brake["valid"] and brake["on_path"] and brake["trend"] == "closing")
        self.assertAlmostEqual(brake["speed_mps"], 1.4, delta=0.5)   # rough by design
        self.assertEqual(brake["obstacles"][0]["tier"], 3)

    def test_camera_loss_holds_brake_briefly_then_unavailable(self):
        engine = MonoFrontEngine(lambda r: None)
        scene, t = FrontScenario(0), 0.0
        while engine.snapshot()["state"] != "BRAKE":
            engine.on_detections(scene.boxes([("chair", 0.0, max(0.5, 8 - 1.4 * t))]), t, None, BLANK)
            t += 1 / 7.5
        engine.unavailable(t + 0.1, "FRONT CAMERA OFFLINE")
        self.assertEqual(engine.snapshot()["state"], "BRAKE")          # bounded hold, never silently dropped
        engine.unavailable(t + 1.0, "FRONT CAMERA OFFLINE")
        self.assertEqual(engine.snapshot()["state"], "UNAVAILABLE")
        self.assertFalse(engine.snapshot()["valid"])
        engine.on_detections([], t - 5, None, BLANK)                    # time went backwards: fresh start
        self.assertEqual(engine.snapshot()["state"], "CLEAR")
        self.assertEqual(engine.snapshot()["reason"], "NO OBJECT AHEAD")

    def test_log_and_preview(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "front.jsonl"
            engine = MonoFrontEngine(lambda r: None, log_path=str(path))
            scene = FrontScenario(0)
            for i in range(20):
                engine.on_frame(BLANK, i / 7.5)
                engine.on_detections(scene.boxes([("chair", 0.0, 6 - 0.2 * i)]), i / 7.5, None, BLANK)
            self.assertEqual(engine.preview().shape, BLANK.shape)
            engine.close()
            lines = [json.loads(ln) for ln in path.read_text().splitlines()]
            self.assertEqual(lines[0]["type"], "session")
            self.assertEqual(lines[0]["mode"], "camera-only")
            self.assertEqual(len(lines), 21)


class OutputTests(unittest.TestCase):
    def test_snapshot_draws_every_front_object(self):
        _, rows = approach("chair", 0.0, 8.0, 1.4)
        brake = next(r for r in rows if r["state"] == "BRAKE")
        brake["obstacles"].append({"id": "front:99", "label": "person", "display_asset": None, "x": 2.0, "z": 6.0,
                                   "tier": 0, "ttc": None, "on_path": False})
        s = snapshot.build([], 1, hud_state={}, collision=brake)
        chair, person = s["front_obstacles"]
        self.assertLess(chair["z"], 0)
        self.assertEqual((chair["display_asset"], chair["tier"], chair["approx"]), ("tree", 3, True))
        self.assertEqual((person["label"], person["display_asset"], person["z"]), ("person", None, -6.0))
        self.assertEqual(s["collision"]["state"], "BRAKE")
        self.assertEqual(s["view_range_m"], {"behind": cfg.VIEW_BEHIND_M, "ahead": cfg.VIEW_AHEAD_M})
        json.dumps(s, allow_nan=False)

    def test_oled_brake_and_stale_feed_clears_objects(self):
        _, rows = approach("chair", 0.0, 8.0, 1.4)
        brake = next(r for r in rows if r["state"] == "BRAKE")
        self.assertTrue(np.array_equal(render({}, collision=brake), render({}, collision={"state": "BRAKE"})))
        hud = Hud(enabled=False)
        hud.update_collision(brake)
        self.assertTrue(hud.collision["obstacles"])
        hud._collision_updated -= 1
        self.assertEqual(hud.collision["state"], "UNAVAILABLE")
        self.assertEqual(hud.collision["obstacles"], [])


class GrowingChairDetector:
    """Stands in for YOLO: the rear sees nothing, the front sees a chair we walk toward at 1.4 m/s."""
    model_name, last_ms = "stand-in", 1.0

    def __init__(self):
        self.scene, self.start, self.front_maps = FrontScenario(0), None, []

    def detect(self, frame, class_map=None):
        if class_map is None:
            return []
        self.front_maps.append(class_map)
        self.start = self.start or time.monotonic()
        z = max(0.5, 6.0 - 1.4 * (time.monotonic() - self.start) * 3)   # 3x speed-up: the test stays short
        return self.scene.boxes([("chair", 0.0, z)])


class RuntimeTests(unittest.TestCase):
    def test_live_dual_cameras_reach_brake(self):
        class Camera:
            def read(self, timeout):
                time.sleep(.02)
                return BLANK.copy(), time.monotonic()
            def stale_for(self):
                return 0
            def release(self):
                pass
        engine, detector = MonoFrontEngine(lambda r: None), GrowingChairDetector()
        source = DualLiveSource(Camera(), Camera(), detector, engine)
        try:
            deadline = time.monotonic() + 4
            while time.monotonic() < deadline and engine.snapshot()["state"] != "BRAKE":
                source.read(timeout=.05)
            self.assertEqual(engine.snapshot()["state"], "BRAKE")
            self.assertEqual(detector.front_maps[0], cfg.FRONT_CLASSES)
            self.assertIsNotNone(engine.camera_frame()[0])
        finally:
            source.release()

    def test_replay_reaches_brake_from_recorded_clips(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [str(Path(folder) / f"{side}.avi") for side in ("rear", "front")]
            for path in paths:
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (640, 480))
                for _ in range(60):
                    writer.write(BLANK)
                writer.release()
            scene, calls = FrontScenario(0), [0]

            class Detector:
                model_name, last_ms = "stand-in", 1.0
                def detect(self, frame, class_map=None):
                    if class_map is None:
                        return []
                    calls[0] += 1
                    return scene.boxes([("chair", 0.0, max(0.5, 7.0 - 1.4 * calls[0] / 10))])
            rows = []
            engine = MonoFrontEngine(rows.append)
            src = DualReplaySource(*paths, Detector(), engine, realtime=False)
            while not src.ended:
                src.read()
            src.release()
            self.assertIn("BRAKE", {r["state"] for r in rows})

    def test_simulator_front_scenes(self):
        src = SimSource()
        engine = src.front_engine = MonoFrontEngine(lambda r: None)
        seen = {}
        for _ in range(int(sum(sc[3] for sc in src.front.scenes) * 15)):
            src.t += 1 / 15
            src.step_front(1 / 15)
            seen.setdefault(src.front.title(), set()).add(engine.snapshot()["state"])
        titles = [sc[0] for sc in src.front.scenes]
        self.assertIn("BRAKE", seen[titles[0]])            # walking at a chair
        self.assertNotIn("BRAKE", seen[titles[1]])         # walking past it
        self.assertNotIn("BRAKE", seen[titles[2]])         # standing still
        self.assertIn("BRAKE", seen[titles[3]])            # person walking at you
        self.assertNotIn("BRAKE", seen[titles[4]])         # backing away
        self.assertIn("| front: ", src.title)


class CommandLineTests(unittest.TestCase):
    def parse(self, *argv):
        from helmet import main as M
        with patch.object(sys, "argv", ["main", *argv]), redirect_stderr(io.StringIO()):
            return M.parse_args()

    def test_collision_needs_no_calibration(self):
        self.assertIsNone(self.parse("--collision").collision_calibration)
        self.assertTrue(self.parse("--sim", "--collision").sim)
        self.assertTrue(self.parse("--sim").collision, "--sim scripts the front camera too")
        self.assertFalse(self.parse("--sim", "--rear-only").collision)
        for bad in (["--sim", "--collision", "--collision-calibration", "c.json"], ["--collision-calibration", "c.json"],
                    ["--sim", "--collision", "--video", "a.mp4"], ["--collision", "--video", "rear.avi"],
                    ["--rear-only"], ["--sim", "--collision", "--rear-only"]):
            with self.assertRaises(SystemExit, msg=bad):
                self.parse(*bad)

    def test_pair_recording_without_calibration(self):
        from tools import record_pair

        class Camera:
            def read(self, timeout):
                time.sleep(.01)
                return BLANK, time.monotonic()
            def stale_for(self):
                return 0
            def release(self):
                pass
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            out = Path(folder) / "take"
            with patch.object(sys, "argv", ["record_pair", "--output", str(out), "--seconds", ".12"]), \
                    patch.object(record_pair, "open_camera", return_value=Camera()):
                record_pair.main()
            self.assertTrue((out / "timestamps.json").exists())
            self.assertFalse((out / "calibration.json").exists())

    def test_main_sim_collision_serves_front_and_rear(self):
        from helmet import main as M
        port = 18917
        argv = ["main", "--sim", "--collision", "--headless", "--stream", str(port), "--no-gemini", "--no-hud",
                "--port", "/dev/null-no-arduino"]
        cfg.TTS_ENABLED = False
        out = io.StringIO()
        plays = []

        class RecordingOut:
            status = "recording"

            def play(self, kind, side):
                plays.append((kind, side))

            def close(self):
                pass

        def run():
            with patch.object(sys, "argv", argv), patch.object(cfg, "BEEP_ENABLED", True), \
                    patch.object(M, "Beeper", lambda: Beeper(RecordingOut())), redirect_stdout(out):
                M.main()
        th = threading.Thread(target=run, daemon=True)
        th.start()
        try:
            deadline, snap = time.monotonic() + 15, {}
            while time.monotonic() < deadline:
                time.sleep(0.3)
                try:
                    snap = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/v1/state", timeout=2).read())
                except OSError:
                    continue
                if (snap.get("collision") or {}).get("state") == "BRAKE" and snap["front_obstacles"]:
                    break
            self.assertEqual(snap["collision"]["state"], "BRAKE", snap.get("collision"))
            self.assertEqual(snap["collision"]["mode"], "camera-only")
            self.assertLess(snap["front_obstacles"][0]["z"], 0)
            self.assertIn("| front: ", snap["scene"])
            self.assertIn("camera-only (no calibration)", out.getvalue())
            deadline = time.monotonic() + 2                  # the main loop reads BRAKE on its next pass
            while ("front", "B") not in plays and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertIn(("front", "B"), plays)
            self.assertIn("Beeps: recording", out.getvalue())
        finally:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/key?c=q", timeout=3)
            th.join(timeout=5)
        self.assertFalse(th.is_alive())


if __name__ == "__main__":
    unittest.main()

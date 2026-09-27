"""2.5D view: snapshot content, simulator scenes, and the /view, /static and /state endpoints.
Run:  python -m tests.test_view      (no camera, YOLO, Arduino or browser needed)"""
import io
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from contextlib import redirect_stdout
from types import SimpleNamespace

import tempfile

from helmet import config as cfg
cfg.INCIDENT_DIR = tempfile.mkdtemp(prefix="incidents-test-")   # keep test clips out of the repo
from helmet import main as M
from helmet import snapshot
from helmet.perception import Tracker, update_metrics
from helmet.risk import AlertPolicy
from helmet.sim import Scenario, W, H
from helmet.streamer import Streamer

PORT = 8766


def _track(**kw):
    base = dict(id=1, label="car", lat_m=-1.5, dist_m=12.0, alongside=False, zone="LEFT", shown_tier=2,
                ttc=3.2, pred_lat_m=-1.2, on_path=False, matched_now=True, reason="blind spot, TTC 3.2s")
    base.update(kw)
    tr = SimpleNamespace(**base)
    tr.coasting = lambda t: False
    return tr


def test_snapshot_content():
    link = SimpleNamespace(status="Arduino on /dev/ttyACM0", sonar={"SL": 1.234, "SR": None, "BL": None, "BR": None},
                           sonar_fresh=lambda: True)
    tracks = [_track(),
              _track(id=2, alongside=True, dist_m=6.0, zone="RIGHT", lat_m=1.4, ttc=math.inf, pred_lat_m=None),
              _track(id=3, lat_m=None),                                   # not measured yet: skipped
              _track(id=4, matched_now=False)]
    tracks[3].coasting = lambda t: True                                   # lost: skipped
    caps = deque([(time.monotonic() - 1.0, "local", "Car left!")])
    s = snapshot.build(tracks, 5.0, hud_state={"LEFT": 2}, link=link, captions=caps, scene="demo")
    ids = [c["id"] for c in s["cars"]]
    assert ids == [1, 2], ids
    c1, c2 = s["cars"]
    assert (c1["x"], c1["z"], c1["tier"], c1["ttc"], c1["path"]) == (-1.5, 12.0, 2, 3.2, -1.2)
    assert c2["z"] == snapshot.ALONGSIDE_Z and c2["ttc"] is None and c2["path"] is None
    assert s["hud"] == {"LEFT": 2} and s["sonar"]["SL"] == 1.23 and s["scene"] == "demo"
    assert s["caption"]["text"] == "Car left!"
    json.dumps(s)                                                         # must serialise (no inf/NaN)
    print(f"  snapshot: {len(json.dumps(s))} bytes for 2 cars, alongside drawn at z={snapshot.ALONGSIDE_Z}")


def test_sim_scenes_trigger_the_right_alerts():
    fps = 15.0
    sc, trk, pol = Scenario(seed=3), Tracker(), AlertPolicy()
    per_scene = {}
    t = 0.0
    while True:
        t += 1 / fps
        title = sc.title()
        dets = sc.boxes(sc.step(1 / fps))
        if sc.title() != title and sc.idx == 0:
            break                                                         # one full loop
        trk.update(dets, t)
        for tr in trk.tracks:
            if tr.matched_now:
                update_metrics(tr, t, W, W / 2, False, frame_h=H)
        for f in pol.evaluate(trk.tracks, t, False):
            per_scene.setdefault(sc.title(), []).append((f.tier, f.zone, f.reason))
    got = lambda key: [x for k, v in per_scene.items() if key in k for x in v]
    assert any(t == 3 and z == "CENTER" for t, z, _ in got("directly behind"))
    assert not any(t == 3 for t, _, _ in got("normal gap")) and any(t == 2 and z == "LEFT" for t, z, _ in got("normal gap"))
    assert any(t == 3 and "cutting in" in r for t, _, r in got("cutting into"))
    assert any(t == 3 and z == "RIGHT" for t, z, _ in got("Close pass on the right"))
    assert not got("steady 9 m"), "false alert on a car following at a steady distance"
    for k, v in per_scene.items():
        print(f"  {k}: {sorted({(tier, zone) for tier, zone, _ in v})}")


def test_demo_person_sim_and_snapshot():
    # --sim --demo-person: teammates walking/jogging, drawn as people in the view
    sc, trk, pol = Scenario(seed=3, people=True), Tracker(), AlertPolicy()
    per_scene, labels, t = {}, set(), 0.0
    while True:
        t += 1 / 15
        title = sc.title()
        dets = sc.boxes(sc.step(1 / 15))
        if sc.title() != title and sc.idx == 0:
            break
        labels |= {d.label for d in dets}
        trk.update(dets, t)
        for tr in trk.tracks:
            if tr.matched_now:
                update_metrics(tr, t, W, W / 2, False, frame_h=H)
        for f in pol.evaluate(trk.tracks, t, False):
            per_scene.setdefault(sc.title(), []).append((f.tier, f.zone, f.reason))
    got = lambda key: [x for k, v in per_scene.items() if key in k for x in v]
    assert labels == {"person"}
    assert any(t == 3 and z == "CENTER" for t, z, _ in got("jogging straight"))
    assert got("walking past") and not any(t == 3 for t, _, _ in got("walking past"))   # README step 7
    assert any(t == 3 and "cutting in" in r for t, _, r in got("cutting into"))
    assert any(t == 3 and z == "RIGHT" for t, z, _ in got("brushing past"))
    assert not got("standing still")
    saved, cfg.DEMO_PERSON_AS_VEHICLE = cfg.DEMO_PERSON_AS_VEHICLE, True
    try:
        s = snapshot.build([_track(label="person")], 1.0, hud_state={})
    finally:
        cfg.DEMO_PERSON_AS_VEHICLE = saved
    assert s["demo_person"] is True and s["cars"][0]["label"] == "person"
    for k, v in per_scene.items():
        print(f"  {k}: {sorted({(tier, zone) for tier, zone, _ in v})}")


def test_endpoints():
    st = Streamer(PORT)
    try:
        get = lambda p: urllib.request.urlopen(f"http://127.0.0.1:{PORT}{p}", timeout=3)
        page = get("/view").read().decode()
        assert "three.module.min.js" in page and "EventSource('/state')" in page
        js = get("/static/three.module.min.js")
        assert js.headers["Content-Type"] == "text/javascript" and len(js.read()) > 500_000
        for bad in ("/static/../config.py", "/static/%2e%2e/config.py", "/static/nope.js"):
            try:
                get(bad)
                raise AssertionError(f"{bad} should be 404")
            except urllib.error.HTTPError as e:
                assert e.code == 404
        try:
            get("/api/v1/state")
            raise AssertionError("no snapshot yet: should be 503")
        except urllib.error.HTTPError as e:
            assert e.code == 503
        state = snapshot.build([_track(id=9)], 1.0, hud_state={"LEFT": 2},
                               collision={"state": "BRAKE", "reason": "BRAKING BOUNDARY", "valid": True,
                                          "target_id": "front:marker:0", "x": 0.0, "z": 3.0,
                                          "on_path": True, "ttc_s": 2.0},
                               incidents={"latest": "rear-incident", "recording": True})
        st.publish_state(state)
        polled = json.loads(get("/api/v1/state").read())            # what the iOS app polls
        assert polled["cars"][0]["id"] == 9
        assert polled["collision"]["state"] == "BRAKE"
        assert polled["front_obstacles"][0]["display_asset"] == "tree"
        assert polled["incidents"] == {"latest": "rear-incident", "recording": True}
        r = get("/state")
        lines = []
        while len([ln for ln in lines if ln.startswith(b"data:")]) < 1:
            lines.append(r.readline().strip())
        data = json.loads([ln for ln in lines if ln.startswith(b"data:")][0][5:])
        assert data["cars"][0]["id"] == 9
        assert data == polled, "browser and iOS must receive the same combined snapshot"
        r.close()
        print("  /view, /static (no path traversal), /state SSE: ok")
    finally:
        st.close()


def test_main_sim_serves_live_state():
    sys.argv = ["main", "--sim", "--headless", "--stream", str(PORT + 1), "--no-gemini", "--no-hud",
                "--port", "/dev/null-no-arduino"]
    cfg.TTS_ENABLED = False
    cfg.BEEP_ENABLED = False
    out = io.StringIO()

    def run():
        with redirect_stdout(out):
            M.main()
    th = threading.Thread(target=run, daemon=True)
    th.start()
    time.sleep(3.5)
    r = urllib.request.urlopen(f"http://127.0.0.1:{PORT + 1}/state", timeout=3)
    snaps = []
    while len(snaps) < 5:
        ln = r.readline().strip()
        if ln.startswith(b"data:"):
            snaps.append(json.loads(ln[5:]))
    r.close()
    urllib.request.urlopen(f"http://127.0.0.1:{PORT + 1}/key?c=q", timeout=3)
    th.join(timeout=5)
    ts = [s["t"] for s in snaps]
    assert all(b > a for a, b in zip(ts, ts[1:])), "snapshots must move forward in time"
    assert any(s["cars"] for s in snaps) and snaps[-1]["scene"], snaps[-1]
    assert snaps[-1].get("collision") and "| front: " in snaps[-1]["scene"], "--sim scripts the front camera too"
    assert "2.5D view: http://" in out.getvalue()
    rate = (len(ts) - 1) / (ts[-1] - ts[0])
    print(f"  main --sim: /state live at ~{rate:.0f} Hz, scene '{snaps[-1]['scene']}', {len(snaps[-1]['cars'])} car(s)")
    assert 5 <= rate <= 20
    assert not th.is_alive()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("ALL PASSED")

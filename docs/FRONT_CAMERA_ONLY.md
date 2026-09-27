# Front BRAKE warning, no calibration

`--collision` without a calibration file runs the front camera through the same pipeline as the
rear: YOLO, the tracker, time to contact from box growth, and a predicted path. When something
ahead is in your path and you'll reach it in 2.8 s or less, the OLED shows BRAKE. The 3D view
draws it ahead of the bike (a chair is drawn as a tree) with rear traffic behind. You don't need
a marker, a checkerboard or an alignment photo.

The marker mode ([COLLISION_DEMO.md](COLLISION_DEMO.md)) measures exact metres. Use it with
`--collision-calibration`.

## Run it

On the Pi, stop any other run first, because each camera can only be opened once:

```bash
python -m helmet.main --collision --demo-person --headless --stream 8080 --no-gemini
```

Open `http://<pi-ip>:8080/view` on a laptop or phone on the same network. `--demo-person` makes
the rear camera count walking teammates as traffic, for an indoor demo. The front camera always
looks for people.

| Goal | Command |
|---|---|
| No hardware at all (demo backup) | `python -m helmet.main --sim` (add `--demo-person` for teammates behind) |
| Record both cameras for a replay test | `python -m tools.record_pair --output take_01 --seconds 20` |
| Replay a recording | `python -m helmet.main --collision --video take_01/rear.avi --front-video take_01/front.avi --replay-timestamps take_01/timestamps.json --headless --stream 8080 --no-gemini` |
| Save every front decision | add `--collision-log front.jsonl` |
| Regenerate the evidence table below | `python -m tools.eval_front` |
| Run the tests | `python -m tests.test_front_mono` |

## How it works

1. YOLO looks for people, bikes, cars, motorbikes, buses, trucks and chairs in the front image
   (`FRONT_CLASSES` in `config.py`). The front and rear cameras take turns on one model, so each
   gets about half the frame rate.
2. The rear's tracker keeps an ID on each object across frames.
3. Time to contact comes from how fast a box grows. A box that doubles in width means you've
   halved the distance, whatever the lens or the object's real size, so TTC = 2 × area / (rate
   of area growth) needs no calibration. Camera-only forward collision warnings in cars use the
   same principle. The growth has to be consistent over about a second, which rejects YOLO's box
   jitter.
4. The sideways offset is (box centre − image centre) / box width × typical width, which doesn't
   depend on distance either. A line fitted through it predicts where the object will be when
   you reach it. An object counts as in your path if it's inside your 1.2 m corridor now or will
   be at contact.
5. The warning is based on time, not metres. Stopping distance divided by speed is almost
   constant at demo speeds. With the marker mode's formula (1 s reaction, 0.25 s delay, braking
   at 1.5 m/s², 0.75 m margin):

   | Speed | Stopping distance | As time to contact |
   |---|---|---|
   | 1.0 m/s (slow walk) | 2.33 m | 2.3 s |
   | 1.4 m/s (walk) | 3.15 m | 2.3 s |
   | 2.0 m/s (jog) | 4.58 m | 2.3 s |

   BRAKE fires at 2.8 s: the 2.3 s, plus about 0.5 s for the front's detection rate and a
   2-frame confirmation. CAUTION fires at 4.5 s (orange in the view, like the rear's blind-spot
   tier). BRAKE stays up for at least 0.8 s. When it starts, you hear three quick, higher
   beeps in both ears (the rear's are two lower ones on the threat's side). A BRAKE that
   flickers off and on beeps at most once every 4 s, a recent rear beep doesn't hold it back,
   and CAUTION stays silent. The settings are the `BEEP_*` lines in `config.py`.
6. Distance comes from the box width and a typical width per class (chair 0.5 m, person 0.5 m,
   car 1.8 m). It's rough (±20 to 30%) and only displayed, marked `~` in the view. No decision
   uses it.

The code is in `helmet/front_mono.py` (engine and policy). It reuses the tracker, TTC and path
code in `helmet/perception.py`.

## Demo script

1. Put a chair in open floor space. Start 7 to 8 m away, facing it.
2. Walk at a normal pace straight at it, looking where you're going.
   - About 4.5 s out, the tree turns orange and the banner says "Chair ahead".
   - About 2.8 s out (3.5 to 4 m at a walk), the OLED says BRAKE, you hear three quick beeps
     and the tree turns red.
   - Stop, and the warning clears.
3. Walk past the chair 1.5 m to one side. Nothing should fire.
4. Stand still 3 m from it. Nothing should fire.
5. Have a teammate walk at you. You get BRAKE, with a person model in the view.
6. Rear alerts keep working the whole time. A teammate approaching from behind (with
   `--demo-person`) shows up behind the bike in the same 3D view.

If the room or the camera misbehaves, run `--sim`. It scripts all of the above
through the real engine and the real 3D view.

## Test cases

### Automated (no hardware; `python -m tests.test_front_mono`)

| What it checks | Test |
|---|---|
| BRAKE fires before the stopping distance at 1.0, 1.4, 2.0 and 2.6 m/s (7.5 detections/s) and at 1.0 to 2.0 m/s (5/s), 10 seeds each | `test_brake_fires_before_the_stopping_distance` |
| A person walking at you, and ±60 px of head sway (about ±5°), still brake on time | `test_person_walking_at_you_and_head_sway` |
| No BRAKE when standing still, backing away, or walking past on either side | `test_no_brake_when_stationary_receding_or_passing` |
| Stopping short clears the warning after the hold | `test_stopping_clears_after_the_hold` |
| Losing the camera holds BRAKE briefly and never drops it silently; time going backwards (a replay loop) starts fresh | `test_camera_loss_holds_brake_briefly_then_unavailable` |
| Same output fields as marker mode, valid JSON, decision log written | `test_result_matches_marker_fields_and_is_json`, `test_log_and_preview` |
| Every tracked object ahead reaches the 3D view, a stale feed clears them, and the OLED shows BRAKE | `OutputTests` |
| BRAKE beeps once when it starts, a flickering BRAKE doesn't beep again within 4 s, a recent rear beep doesn't hold it back, and the `--sim` run plays it | `tests/test_beeps.py`, `test_main_sim_collision_serves_front_and_rear` |
| Live dual-camera threads, paired-video replay and the simulator scenes reach BRAKE when they should and not otherwise | `RuntimeTests` |
| `--collision` needs no calibration, `--sim` turns the front on (`--rear-only` turns it off), and `--sim` serves front and rear together end to end | `CommandLineTests` |

### Simulated evidence (`python -m tools.eval_front`, 20 runs per row)

These are scripted approaches with YOLO-like box jitter, fed through the real engine. On time
means BRAKE fired before the stopping distance.

| Detections/s | Speed | Start | Stopping distance | On time | Gap at BRAKE (min / median) |
|---|---|---|---|---|---|
| 7.5 | 1.0 m/s | 10 m | 2.33 m | 20/20 | 2.50 / 2.77 m |
| 7.5 | 1.4 m/s | 10 m | 3.15 m | 20/20 | 3.54 / 3.91 m |
| 7.5 | 2.0 m/s | 10 m | 4.58 m | 20/20 | 5.43 / 5.43 m |
| 7.5 | 2.6 m/s | 10 m | 6.25 m | 20/20 | 6.58 / 6.93 m |
| 5.0 | 1.0 m/s | 10 m | 2.33 m | 20/20 | 2.50 / 2.70 m |
| 5.0 | 1.4 m/s | 10 m | 3.15 m | 20/20 | 3.54 / 3.82 m |
| 5.0 | 2.0 m/s | 10 m | 4.58 m | 20/20 | 4.90 / 5.30 m |
| 5.0 | 2.6 m/s | 10 m | 6.25 m | 0/20 | 5.54 / 5.54 m |
| 5.0 | 2.6 m/s | 14 m | 6.25 m | 20/20 | 6.42 / 6.94 m |

The failing row isn't a threshold problem. TTC needs about 8 detections of history (1.6 s at
5/s) before its first estimate, and from 10 m at a jog that comes too late. From 14 m it's on
time, so the run-up has to cover about 1.5 s of tracking plus the stopping distance.

False BRAKEs, 20 runs each with ±30 px of head sway: 0/20 standing 3 m from a chair, 0/20 with a
person standing 4 m ahead, 0/20 walking past 1.5 m to the right, 0/20 walking past 1.0 m to the
left, and 0/20 backing away.

The simulator always detects the object, so real detection range still has to be measured on
the Pi (L0 below).

### Live acceptance on the Pi (fill in the results column at the venue)

| ID | Setup | Pass if | Result |
|---|---|---|---|
| L0 | Walk backwards away from the chair watching the front box on `http://<pi-ip>:8080/` | Note the distance where the box disappears. It must exceed the stopping distance plus 1.5 s of walking (about 5.3 m at 1.4 m/s) | |
| L1 | Walk at the chair from 7 m at a normal pace, 5 times; tape a mark at 3.15 m and film the OLED and the mark | BRAKE before the mark every time | |
| L2 | Stand 3 m from the chair for 30 s, looking around normally | No BRAKE, no CAUTION | |
| L3 | Walk past the chair 1.5 m to each side, 3 times | No BRAKE | |
| L4 | Back away from the chair | No alert | |
| L5 | A teammate walks at you while you walk at them | BRAKE, with a person model in the view | |
| L6 | Power off, disconnect the front camera's ribbon, boot and start the app | OLED shows FRONT LOST, view says "Front camera unavailable", rear alerts work normally | |
| L7 | A teammate approaches from behind while the front is clear, then both at once | Rear arrows and haptics for the rear. With both, BRAKE takes over the OLED and the rear buzz still fires | |
| L8 | Both cameras running, open `http://<pi-ip>:8080/api/v1/state` | `collision.front_yolo_hz` is at least 5 (front detections per second) | |

## Limitations (say these before a judge asks)

- Distances are rough (±20 to 30%), so they're only displayed. The timing comes from box growth,
  which is much more reliable.
- Straight ahead is assumed to be the centre of the image, so look where you're going. If you
  stare at an object off to the side, one you'll miss can look like one you'll hit, and the
  reverse.
- YOLO has to recognise the object. The COCO classes cover people, vehicles and chairs, but not
  potholes, bollards or car doors.
- The front shares one model with the rear and gets half the frame rate. It needs about 1.5 s of
  tracking before its first TTC, which sets the minimum run-up.
- A covered or blacked-out lens reads as "nothing ahead", not as a fault. Only a camera that
  stops sending frames shows FRONT LOST.
- Low objects drop out of view up close. A level helmet camera loses a 0.9 m chair at about
  1.3 m. BRAKE is held for 0.8 s, and by then you should have stopped.
- There's no ego speed and no road validation. All the evidence here is from walking speed,
  indoors, or the simulator.

## Roadmap

### Next few days

- Give the front camera its own inference slot or a larger `IMGSZ` for more detection range. A
  Raspberry Pi AI HAT (Hailo) could run both cameras at full rate.
- Detect a blocked lens: treat a front image that is almost uniform (lens covered, mud) as FRONT
  LOST instead of "nothing ahead". The threshold needs tuning on real footage so a dark room
  doesn't trigger it.
- Draw the predicted path ribbon for front objects as well. The data (`pred_lat_m`) is already
  computed.
- Record every rehearsal (`tools.record_pair` plus `--collision-log`) into a labelled clip set,
  and replay it as a regression test on each change.

### Medium term

- Add an IMU to the helmet to tell head turns from bike heading, and get speed from a wheel
  sensor or the phone's GPS. With real speed, the fixed 2.8 s becomes a true stopping distance
  for each speed.
- Estimate where you're actually heading from optical flow (the focus of expansion) instead of
  assuming the image centre. That removes the "look where you're going" limit.
- Range objects from the ground plane (box bottom plus camera height) to cross-check the
  width-based distance.
- Fine-tune YOLO on footage from a cyclist's point of view, with potholes, bollards and opening
  car doors, which COCO lacks.

### Long term

- Warn about the door zone: a parked car with a door starting to open, a leading cause of urban
  cyclist crashes.
- Build a near-miss map from the incident reports the helmet already saves (clip, Gemini summary,
  and location from the phone).
- Validate on real roads: a labelled on-road dataset, measured photon-to-OLED latency, and
  performance at night and in rain, before claiming anything beyond a demo.

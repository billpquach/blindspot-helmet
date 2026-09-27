Created by: Sion King

# Blind-spot helmet

A bike helmet that watches the road behind you. A rear camera runs YOLO on a Raspberry Pi, and
the helmet warns you with left or right vibration, a short beep in your earbuds for the most
urgent threats, a rear light aimed at the driver, a transparent heads-up display and short
spoken alerts. Gemini writes descriptions and incident reports on the side. It never decides whether an alert fires.

`--collision` adds the front camera. It uses the same box-growth time to contact as the rear,
so it needs no calibration. When something ahead is 2.8 s away, the display shows BRAKE and
you hear three quick beeps. The 3D view draws front and rear traffic together. The guide,
test cases and roadmap are in [FRONT_CAMERA_ONLY.md](docs/FRONT_CAMERA_ONLY.md), and `--sim`
runs both cameras with no hardware. [COLLISION_DEMO.md](docs/COLLISION_DEMO.md) covers an older
marker mode that measures exact metres but needs calibration. The rear-only commands below work without either.

```
FAST PATH (every frame, ~70-120 ms, no network)
camera ─► mirror ─► global-motion ─► YOLO ─► tracker ─► TTC / zone ─► policy ─► motors + light + "Truck left!"
                                                                          │
SLOW PATH (events only, ~1-2 s, optional)                                 ▼ submit (non-blocking)
                            GeminiGateway: budget ▸ min interval ▸ coalesce ▸ expire ▸ circuit breaker
                                                                          │
                          drained next frame through guardrails ◄─────────┘  ("Box truck passing close on your left")
```

## Raspberry Pi (current target)

The Pi replaces the laptop. The Arduino stays: it's an independent watchdog that turns the
light into a normal bike light if the Pi crashes, and it handles the NeoPixel timing.

```bash
git clone <your repo> ~/blindspot-helmet && cd ~/blindspot-helmet
bash scripts/setup_pi.sh             # apt + venv + pip + NCNN export + all tests (needs internet)
. .venv/bin/activate
python -m helmet.main --video clip.mp4 --loop     # works with NO camera, Arduino, or headphones
```

Then open the printed `http://<pi-ip>:8080` on a phone or laptop on the same network. It shows
the live overlay and has a button for every key command. On the Pi's own desktop, add
`--window` for the normal OpenCV window.

On a Pi (`config.IS_PI`), these defaults change automatically:

- The detector is `yolo26n.pt`, exported once to NCNN (`yolo26n_imgsz320_ncnn_model/`) with
  `IMGSZ = 320`. Benchmark with `python -m tools.bench yolo --video clip.mp4` and use the
  largest size that keeps you at 8+ FPS end to end.
- A Pi Camera Module is used if one is present, otherwise the USB webcam. The Camera Module 3
  Wide is light and sees about 120°. Force a choice with `--camera-source usb|picamera2`.
- It runs headless with the web view, and skips drawing entirely when nobody is watching.
- Speech (`espeak-ng`) and beeps go to the default audio output: AirPods, or any Bluetooth or
  USB speaker. Pair it in the desktop's Bluetooth menu or with `bluetoothctl`, then press `b`
  (Test beeps on the web page). You should hear two beeps in your left ear, two in your right,
  then three quicker, higher ones in both.
- A HIGH rear alert beeps twice on its side, and a front BRAKE beeps three times. The same
  warning beeps at most every 4 s, and a rear beep never follows another beep within 1 s, so a
  busy road doesn't turn it into a nag. MEDIUM and LOW are silent (vibration and the HUD only).
  The settings are the `BEEP_*` lines in `config.py`.
- Bluetooth earbuds go to sleep after a few seconds of silence and cut off the start of the
  next sound. So the app keeps the audio output open and plays silence between beeps. The
  startup line should say `Beeps: default output, kept awake`. If it names a player such as
  `aplay` instead, install `sounddevice` (in `requirements.txt`), or the beeps get a 0.25 s
  silent lead-in (`BEEP_LEAD_IN_S`) to survive the wake-up.

To start at boot with no laptop, see `deploy/helmet.service`.

Power is the most common Pi problem. On a normal 5V/3A supply or power bank, the Pi 5 caps
total USB current at 600 mA. The C920 plus the Arduino with motors and LEDs can exceed that,
and the symptom is a camera or serial port that randomly drops. Use the official 27 W supply
for the stationary demo. On a power bank, the Camera Module (CSI, not USB) removes the webcam's
draw. Adding `usb_max_current_enable=1` to `/boot/firmware/config.txt` lifts the cap at your
own risk.

Heat is the next one. Continuous YOLO throttles an uncooled Pi 5 within minutes, so fit the
Active Cooler and watch `vcgencmd measure_temp`.

## Setup (laptop)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                       # paste your Gemini key
python -m tests.test_synthetic && python -m tests.test_gateway && python -m tests.test_headless   # no hardware needed
```

Flash `firmware/helmet_arduino/helmet_arduino.ino` with the Arduino IDE. Install Adafruit
NeoPixel from the Library Manager, and pick board "Arduino Nano" with processor "ATmega328P"
(or "ATmega328P (Old Bootloader)" for most clones). On power-up you should feel LEFT, then
RIGHT.

## Run

```bash
python -m helmet.main                          # live camera, auto-detects the Arduino
python -m helmet.main --demo-person            # stationary demo: walking teammates count as vehicles
python -m helmet.main --video clip.mp4 --loop  # recorded footage (real-time paced, video timestamps)
python -m helmet.main --no-gemini              # prove it works fully offline
python -m helmet.main --agent                  # Gemini tool calling
python -m helmet.main --log run.csv            # per-frame metrics for calibration
python -m helmet.main --model yolo26n.pt       # swap detector
python -m helmet.main --sim                    # both cameras scripted, no camera/YOLO; open http://<ip>:8080/view
python -m helmet.main --sim --rear-only        # scripted rear traffic only
python -m helmet.main --collision              # add the front camera: BRAKE warning, no calibration
```

| Key | Action | Key | Action |
|---|---|---|---|
| q / Esc | quit | f | simulate camera failure |
| g | "What's behind me?" (Gemini, or local answer offline) | k | kill heartbeat → Arduino failsafe |
| v | voice question (3 s, laptop mic) | x | Gemini on/off |
| t | haptic test: left, right, both | a | describe ↔ agent mode |
| b | beep test: left, right, front BRAKE | | |
| l | cycle light override | m | mirror on/off |
| p | cycle sensitivity profile | c | hot-reload `config.py` |
| space | pause video | r | record the debug window to MP4 |
| d | hide/show metrics panel | | |

The handlebar button (D3 to GND) does the same as `g`. When the Pi runs headless, the web view
has a button for each of these commands.

## Wiring (Arduino Uno + Pi, no soldering)

The Pi and the Uno connect only through the Uno's USB cable (USB-A on the Pi, USB-B on the
Uno), which carries power, serial and a shared ground. The OLED is the only part wired to the
Pi's pins (see the HUD section).

Run the Uno's 5V pin to the breadboard's + rail and a GND pin to the − rail. Every part below
takes power from those rails.

| Part | Part pin | Uno pin |
|---|---|---|
| Ultrasonic SL (left side, pointing out ~5° back) | TRIG / ECHO | D7 / D6 |
| Ultrasonic SR (right side, pointing out ~5° back) | TRIG / ECHO | D11 / D10 |
| Ultrasonic BL (back, left of centre, optional) | TRIG / ECHO | A0 / A1 |
| Ultrasonic BR (back, right of centre, optional) | TRIG / ECHO | A2 / A3 |
| Left vibration motor | via NPN (below), or module IN | D5 |
| Right vibration motor | via NPN, or module IN | D9 |
| Left buzzer (optional, muted unless `BUZZERS_ENABLED`) | I/O (3-pin module), or via NPN | D4 |
| Right buzzer (optional) | I/O, or via NPN | D12 |
| Optional button | one leg | D3 (other leg → GND) |
| Optional NeoPixel rear light | DIN | D2 (330 Ω in series if you have one) |
| All ultrasonics, modules | VCC / GND | + rail / − rail |

The Uno is 5V, so the HC-SR04 ECHO pins connect directly, without the voltage dividers a Pi
would need. Mark each wired sensor `true` in `SONAR_FITTED` in the sketch (SL and SR are on by
default). Only fitted sensors are pinged, so one sensor updates about 33 times a second and two
about 16. TRIG and ECHO are not in number order (left: TRIG D7, ECHO D6). Swapping them gives
no readings and makes the sensor's ECHO fight an Arduino output.

A bare 2-wire vibration disc needs one NPN transistor (2N2222, PN2222 or S8050). Never drive a
motor straight from a pin; it draws more than a pin can give.
```
Uno D5 ──[1 kΩ]── B (base)
                  C (collector) ── motor red wire ... motor other wire ── + rail (5V)
                  E (emitter)   ── − rail (GND)
diode (1N4148/1N4001) across the motor: striped end (cathode) to the 5V side
```
A bare 2-pin buzzer is wired the same way (buzzer + to 5V, − to the collector, no diode).
Buzzer and vibration modules with 3 pins (VCC, GND, IN/I/O) have the transistor built in, so IN
goes straight to the Uno pin.

Check the wiring before mounting. Flash the sketch from the Pi with
`bash scripts/flash_arduino.sh` (no Arduino IDE needed; `bash scripts/flash_arduino.sh
hcsr04_test` for the sensor test), then run `python -m tools.bench serial` on the Pi. You should
feel left then right at power-up. `l` and `r` buzz each side, and `u` prints the four
distances (wave a hand in front of each sensor). If you wired the optional buzzers, `z1`
unmutes them: a strong buzz then chirps twice and `c` chirps both. Total draw is roughly 250 mA from the Pi's USB (see
the power note above).

### Standalone two-sensor test

`firmware/hcsr04_test/hcsr04_test.ino` tests both helmet sensors with the same wiring as the
helmet firmware:

| Sensor | TRIG | ECHO |
|---|---|---|
| Left | D7 | D6 |
| Right | D11 | D10 |

Connect both sensors' VCC pins to the breadboard's + rail (Arduino 5V) and both GND pins to the
− rail (Arduino GND). Upload the sketch in the Arduino IDE or run
`bash scripts/flash_arduino.sh hcsr04_test`, then open Serial Monitor at 9600 baud. Each line
shows separate LEFT and RIGHT distances (or `no echo`) and echo success percentages. Pings
alternate with a 65 ms quiet gap, and the built-in LED lights if either sensor reads under
50 cm.

You don't need to rewire to go back to `helmet_arduino.ino`. Close Serial Monitor before using
the helmet app or bench tool.

## Camera mounting

With the Pi on top of the helmet, the rear camera is mounted upside down (its cable runs down
the back of the helmet) and the front camera the normal way up. In `helmet/config.py`:

| Setting | Default | Meaning |
|---|---|---|
| `CAMERA_INDEX` | `1` | which camera is the rear one (verify with `python -m tools.bench check`) |
| `CAMERA_PORT` | `"i2c@80000"` | Pi only: picks the rear camera by its connector, which overrides `CAMERA_INDEX` (numbers can reorder, connectors can't) |
| `CAMERA_ROTATE_180` | `True` | rear camera upside down; rotated in the camera itself (no CPU cost) |
| `FRONT_CAMERA_INDEX` | `0` | the helmet-front camera |
| `FRONT_CAMERA_PORT` | `"i2c@88000"` | Pi only: the front camera's connector |
| `FRONT_CAMERA_ROTATE_180` | `False` | front camera upright |
| `MIRROR_VIEW` | `True` | applied after the rotation, so rider-left shows on the left |

Check with `python -m tools.bench check`. It labels each Camera Module rear or front and saves
its snapshot (`check_csi0.jpg`, `check_csi1.jpg`) as the app will see it. If the rear snapshot
shows the front view, swap the two connector settings (or the two index settings for USB
cameras). Then do calibration step 1 (a teammate on your left must show up as `LEFT`).

## Camera + ultrasonic fusion

The rear camera sees about ±33° either side of straight back, so a car right beside you is
outside its view. The side ultrasonic sensors see exactly there and measure the gap to a few
cm. The camera says what it is and that it approached; the sensor says how close it passed.

Mount one sensor per side at the widest point of the helmet, pointing straight out and level
(at most about 10 to 15° back; the default config assumes 5°). A car's flat side reflects sound
away at steep angles, so sensors angled further back miss cars. Tilt them slightly up, never
down, and check that each reads `---` with nobody around. A steady reading means it sees your
shoulder or backpack. Tell the software which sensors are fitted in `helmet/config.py`:
`SONAR_MOUNT = {"SL": ("LEFT", 5.0), "SR": ("RIGHT", 5.0)}` (remove one that isn't wired).

`helmet/fusion.py` range-gates and median-filters each sensor's readings, and drops them while
your head is turning. An echo that hasn't moved for 3 s is background (a wall, your backpack)
and is ignored. A moving echo becomes a contact, linked to the camera track that was just on
that side (at the frame edge, or lost from view in the last 1.5 s):

| Situation | Alert |
|---|---|
| The camera saw it approach, the sensor measures it beside you inside the close-pass distance (1.0 m, 0.4 m for people) | HIGH "close pass 0.32 m measured" |
| Same, but a wider gap | MED "beside you 1.35 m measured" |
| Moving echo under 1.5 m the camera never saw | MED "object beside you" (never HIGH: it can't tell a car from a pole) |
| Camera fault | sensor contacts keep giving MED side alerts |

Every pass is logged, for example `[PASS] car passed on your left at 0.72 m`. In the 2.5D view
the car moves up beside the bike with a "0.72 m gap" label, and an echo the camera never saw is
drawn as an orange column. `--sim` and `--sim --demo-person` include hand-off scenes with
simulated echoes.

## 2.5D view (phone or laptop)

Open `http://<pi-ip>:8080/view` (the startup line prints it). It's a chase view, like a car's
driver-assist display: your bike heads up the screen (north), objects the front camera sees
(`--collision`) are ahead of it, and rear traffic comes up from the bottom (south). Each tracked
object sits where the pipeline measured it, tinted by alert tier (grey tracked, yellow
approaching, orange blind spot, red danger).

Only the last 12 m behind and 8 m ahead are drawn (`VIEW_BEHIND_M` and `VIEW_AHEAD_M` in
`config.py`), because one camera's distance is too rough beyond that. Farther objects are
hidden unless they're alerting; those are pinned at the edge with their time to contact.

The blind-spot zones and screen edges light up from the same state the OLED shows, with the
same arrows and the same 4 Hz blink for HIGH, so the phone, OLED and buzz always agree. Cars
show their time to contact, a car cutting in shows its predicted path, and the ultrasonic
sensors draw arcs beside the bike. Front and rear camera previews sit in the bottom-right
corner, and the Hide cameras / Show cameras button toggles both. The rear preview uses the same
rotation and mirror settings as detection; the front uses its own rotation setting. The full
debug overlay is at `/`.

Live runs (including `--demo-person`) open the front camera only while its preview is watched.
Choose it with `--front-camera` (default `FRONT_CAMERA_INDEX`). Collision mode reuses its
existing front capture. A missing or busy camera shows "Camera unavailable", and simulation or
rear-only video replay shows "No front feed in this run". The previews are also available on
their own at `/stream.mjpg?camera=front` and `/stream.mjpg?camera=rear` (append `&t=...` if you
set `STREAM_TOKEN`).

The Pi sends a ~700-byte JSON snapshot per frame over `/state` (server-sent events), and the
phone draws the scene with three.js. three.js is bundled, so the view works on a hotspot with
no internet. Video is sent only while someone watches the camera previews or the debug feed.

Without hardware, `python -m helmet.main --sim` runs scripted traffic (car behind, normal left
pass, car cutting in, close pass right, two cars, steady follower) through the real tracker,
alerts, OLED and Arduino, with scripted front scenes (a chair, a person walking at you) through
the front BRAKE engine at the same time. Both camera tiles on `/view` show the simulated feeds.
`--rear-only` leaves the front out. It's useful for working on the view and as a demo backup.

With `--demo-person`, people are drawn as walking figures instead of cars, the banner says
"Person...", and a DEMO MODE badge shows. `--sim --demo-person` rehearses the stationary demo
with no hardware: teammates jogging at you, walking past on the 1.5 m line, cutting in,
brushing past, and standing still (which must stay quiet).

On a laptop, add `--stream 8080` to any run (`--video clip.mp4 --stream 8080`). Distance comes
from one camera (±20 to 30%); the side and time to contact are the reliable parts.

## Incident reports (clip + Gemini analysis)

The Pi keeps the last few seconds of rear video in memory (JPEGs, about 6 MB). When a HIGH
alert fires, it saves a clip from 6 s before to 4 s after, attaches the measured facts, and
asks Gemini for a short report. The API key stays on the Pi; the phone only downloads results.

Each incident is saved in `incidents/<id>/` as `clip.mp4` (H.264, plays on iPhone), `thumb.jpg`
(the frame the alert fired on) and `meta.json`. The metadata holds the side, the vehicle, the
measured gap from the side sensor, the time to contact when the warning fired, a distance
trace, the alert events, a local summary, and Gemini's report.

More alerts from the same vehicle extend the clip, up to 20 s. A different vehicle alerting in
the meantime gets its own incident afterwards, and the same vehicle can't open a new one for
15 s. The rider can also press `i` (or "Mark incident" on the web page, or send
`POST /api/v1/incidents` from the app) to save the last 6 s and the next 4 s by hand.

Gemini gets 6 frames from the clip plus the measured facts and returns JSON: `classification`
(close_pass, near_miss, aggressive_overtake, tailgating, normal_pass, false_alarm, unclear),
`severity` 1-5, `vehicle`, `maneuver`, `summary` and `confidence`. The prompt tells it to treat
the sensor numbers as authoritative and never to call a pass "safe". Without a key, reports
still get the local summary (`analysis_status: "not_configured"`).

Clips are encoded with `ffmpeg` (installed by `setup_pi.sh`, or `pip install imageio-ffmpeg`)
at low priority on a background thread, so detection isn't slowed. Without ffmpeg, OpenCV writes
an MPEG-4 file that browsers may not play. To try it with no hardware, run
`python -m helmet.main --sim`, wait for the red scenes, then open
`http://<pi-ip>:8080/api/v1/incidents`.

API for the iOS app (all paths relative to `http://<pi-ip>:8080`). If `STREAM_TOKEN` is set,
add `?t=<word>` to the incident requests; `/api/v1/state` doesn't need it:

| Request | Returns |
|---|---|
| `GET /api/v1/state` | live snapshot; `incidents.latest` changes when a new report is saved, `incidents.recording` is true while one is being captured |
| `GET /api/v1/incidents` | `{"incidents": [report, ...], "enabled": true, "recording": false}`, newest first |
| `GET /api/v1/incidents/<id>` | one report (below) |
| `GET /api/v1/incidents/<id>/clip.mp4` | the clip; supports `Range` (AVPlayer needs it) |
| `GET /api/v1/incidents/<id>/thumb.jpg` | thumbnail |
| `POST /api/v1/incidents` | 202; the rider marks an incident now |
| `POST /api/v1/incidents/<id>/analyze` | 202 re-runs Gemini; 409 if no key or budget left |

```json
{"id": "20260927-143102-l", "time": "2026-09-27T14:31:02", "kind": "auto", "severity": "HIGH",
 "zone": "LEFT", "side": "left", "label": "car", "reason": "close pass 0.25 m measured",
 "measured_clearance_m": 0.25, "ttc_at_alert_s": 1.4, "min_ttc_s": 0.6, "duration_s": 10.0,
 "local_summary": "Car passed close on your left (measured gap 0.25 m, warned 1.4 s before contact).",
 "analysis_status": "done",
 "analysis": {"classification": "close_pass", "severity": 4, "vehicle": "white SUV",
              "maneuver": "overtook without moving over", "summary": "...", "confidence": "medium"},
 "url": "/api/v1/incidents/20260927-143102-l",
 "clip_url": "/api/v1/incidents/20260927-143102-l/clip.mp4",
 "thumb_url": "/api/v1/incidents/20260927-143102-l/thumb.jpg"}
```

`analysis_status` is `pending`, `done`, `failed` (see `analysis_error`), `budget_exhausted`,
`not_configured` or `not_requested`. Show `analysis.summary` when it's `done`, otherwise
`local_summary`. `kind` is `auto`, or `manual` for a rider mark (`severity: "MARKED"`). The Pi
has no GPS, so the phone should attach its own location when it first sees a new id.

Settings are in `config.py` under `INCIDENT_*`.

## Transparent OLED HUD (Pi)

The HUD puts big arrows in the rider's peripheral vision for MED and HIGH threats: an arrow on
the threat's side (outline for MED, filled and blinking for HIGH), chevrons for "behind", an X
when the camera is down, and nothing at all (fully transparent) when it's clear. The debug
overlay shows the same picture top-left, labelled HUD, so you can check it without looking at
the panel.

Wiring for the 1.51" transparent OLED (SSD1309, SPI by default) to the Pi's 40-pin header:

| OLED | Pi pin | | OLED | Pi pin |
|---|---|---|---|---|
| VCC | 3.3V (pin 1) | | CS | GPIO8 / CE0 (pin 24) |
| GND | GND (pin 6) | | DC | GPIO25 (pin 22) |
| DIN | GPIO10 / MOSI (pin 19) | | RST | GPIO27 (pin 13) |
| CLK | GPIO11 / SCLK (pin 23) | | | |

Enable SPI once (`sudo raspi-config nonint do_spi 0`, then reboot). On a Pi 5, luma needs
`pip install rpi-lgpio` for the DC and RST pins, because the old RPi.GPIO doesn't support the
Pi 5. The startup line `HUD:` says whether the panel was found. If it wasn't, the HUD runs
preview-only and everything else works as usual. To use I2C instead (a resistor change on the
back of some boards), wire SDA/SCL to pins 3/5 and set `HUD_INTERFACE = "i2c"`.

Arrows are drawn from the rider's point of view. `HUD_ROTATE_180 = True` is for a panel mounted
upside down. If the rider sees arrows pointing the wrong way, they're usually reading the glass
from the back: set `HUD_MIRROR = True`. Check with a teammate on your left; the arrow must point
to your left.

## Serial protocol (57600 baud, newline-terminated)

`L<n>` `R<n>` `B<n>` haptics (0 stop, 1 gentle, 2 medium, 3 strong, 4 fault) ·
`M<n>` light 0 normal / 1 alert / 2 danger (sent every 250 ms, which doubles as the heartbeat) ·
`C<n>` n short chirps (1 to 3) on both buzzers, no vibration · `Z1`/`Z0` buzzers on / muted ·
`F1`/`F0` host fault · `X` all off · `?` status.
The buzzers are optional. The Pi mutes them at startup unless `BUZZERS_ENABLED = True`, since
the beeps go to the earbuds. When they're on, a strong buzz (3) also chirps twice on its side,
40 ms each, and a side that just chirped waits 1 s before chirping again (`CHIRPS_FOR_LEVEL`
in the sketch).
The board replies `READY`, `BTN`, `FAILSAFE`, `LINK OK` or `ERR <line>`.
If no command arrives for 1.5 s, the light becomes a normal flashing bike light and both motors
give a long buzz, with a beep.

## Calibration procedure

Tune by watching numbers, not by guessing. Every metric is in the side panel:
`box` (width x height in pixels, for `FOCAL_PX`), `grow` (late/early size ratio), `cons`
(fraction of frames that grew), `TTC`, `dist`, `lat` (lateral offset, − = your left), `->`
(predicted `lat` when it reaches you; `PATH` means heading into your lane), `clr` (passing
clearance), `[area|h|w]` (which box size TTC used), zone, tier and the reason. A tier in
brackets `(raw HIGH)` is what this frame says. The box only turns that colour once the tier
holds for `CONFIRM_FRAMES`, which is also when an alert actually fires. Run with
`--demo-person --no-gemini --log calib.csv`, edit `helmet/config.py`, and press `c`.

Your lane is the shaded band, the "behind" corridor: bike half-width `RIDER_HALF_WIDTH_M` plus
`CORRIDOR_MARGIN_M` each side (1.2 m total by default). The band is drawn in perspective from
`CAMERA_HEIGHT_M`, so set that for your setup (about 1.6 on a helmet, 0.8 for a table demo).
The exact test is the short cyan bar across each box's top. It shows your lane at that object's
distance, and a box that overlaps it is zone `CENTER`. The arrow on a box points to where it
will be sideways at contact.

To set up, tape marks on the floor behind the helmet at 2, 4, 6, 8 and 10 m on a centre line,
plus a parallel line 1.5 m to the left and right. Put the helmet at head height (about 1.6 m),
with the camera level and pointing straight back.

1. Check orientation. A teammate stands on the LEFT line. The box must be on the left half,
   with negative `lat` and zone `LEFT`. If it isn't, press `m` and set `MIRROR_VIEW` to match.
   Press `t` while wearing the helmet: you must feel left, right, then both.
2. Find the jitter floor. The teammate stands still at 4 m for 10 s while you watch `grow` and
   `cons`. Set `MIN_GROWTH_RATIO` a little above the highest `grow` you see (typically 1.05 to
   1.08) and `MIN_CONSISTENCY` above the highest `cons` (typically 0.55 to 0.65). TTC should
   read `inf` or more than 20 s nearly all the time.
3. Check TTC accuracy. The teammate walks from 10 m straight at the camera at a steady pace
   while someone films the screen. TTC should count down about 1 s per second and reach about
   0 at the helmet. If it reads consistently high, it's lagging: shorten `HISTORY_LEN`. If it's
   jumpy, lengthen it. Inside about 3.5 m the feet leave the frame and `[area]`/`[h]` switches
   to `[w]` (width); TTC should keep counting down through that switch.
4. Check lateral offset. Stand on the 1.5 m side line at 3, 6 and 9 m. `lat` should read about
   ±1.5 m at every distance, since it doesn't depend on distance. If it's biased, the camera is
   yawed; if it's scaled, fix `CLASS_WIDTH_M` for that class.
5. Set distance (optional). At a measured distance D, read the box width w in pixels (the first
   number after `box`): `FOCAL_PX = w * D / real_width_m`. A real car (1.8 m) in a parking lot
   works best.
6. Test shake. Wear the helmet, nod and turn your head while the teammate stands still at 5 m.
   `SHAKY` should light up during the movement. No MED or HIGH may fire, and the zone must not
   flip. Raise `SHAKE_GATE_FRAC` if normal riding posture keeps triggering SHAKY.
7. Check the tiers. Jog at the camera from 10 m: HIGH (red, both motors, two beeps, strobe,
   "Person behind!") should fire about 2 to 3 s out. Walk past on the left line: MED left, not
   HIGH. Walk past 0.95 m to the left: HIGH left ("close pass"; 0.7 m is inside the lane, so that one
   is HIGH "behind"). Start 2 m left and walk diagonally into the centre line: HIGH "cutting in"
   before you reach it. Walk in and stand still half out of frame at the edge: MED "alongside"
   at most, never HIGH.
8. Try real traffic. Run each recorded clip with `--video clip.mp4 --log clipN.csv` (without
   `--demo-person`). For each clip, count passes detected, correct side, false alerts and missed
   HIGHs. These are your pitch numbers.
9. Repeat at the venue. Redo steps 2 and 7 in the demo room, because lighting and background
   change the jitter.

Commit the final values with a message like "calibrated at venue".

## Troubleshooting

- Pi, `numpy.dtype size changed` when importing picamera2: the venv's NumPy is newer than the
  one apt built picamera2 against. Run `pip install "numpy<2"` in the venv (older OS), or use a
  USB camera.
- Pi, serial permission denied: log out and back in after setup (dialout group).
- Pi, web view unreachable: check that you're on the same Wi-Fi as the Pi. Hackathon networks
  often isolate clients, so use a phone hotspot for both. Set `STREAM_TOKEN` so strangers can't
  press your buttons.
- Pi, camera "Device or resource busy": another program still has the camera open. Stop the
  other run (`pkill -f helmet.main`), and stop the auto-start service if you installed it
  (`systemctl --user stop helmet`), since it restarts itself after a kill.
- Low FPS: `python -m tools.bench yolo` compares models and sizes. Try `yolo26n.pt` (faster on
  CPU), `IMGSZ = 320`, or OpenVINO on Intel:
  `yolo export model=yolov8n.pt format=openvino imgsz=416`, then `--model yolov8n_openvino_model/`.
- No Arduino: use a data-capable USB cable, install the CH340 driver on Windows or macOS, close
  the Arduino IDE serial monitor (only one program can hold the port), or pass `--port`.
- Speech clips the first word (Bluetooth power saving): set `TTS_PREFIX = "Hey. "`.
- No beeps: check the startup line after `Beeps:`, press `b`, and make sure the earbuds are the
  default output (`wpctl status` on the Pi). `[ALERT] ... + beep` in the log means a beep was
  sent. An alert without `+ beep` was held back by a cooldown.
- Don't use the bone-conduction headset's mic. It switches Bluetooth into call mode and wrecks
  the audio. Use the laptop mic for `v`.
- Gemini 429 errors: raise `GEMINI_MIN_INTERVAL_S` to 60 / (your RPM in AI Studio).
- The code calls Gemini through `generate_content`. The newer Interactions API also exists but
  isn't needed.

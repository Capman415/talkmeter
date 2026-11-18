# Talk Meter v0.1

A small classroom web app for Christopher Parsons (Steady Bots / Benton Grade School). Park it in a corner of the Smartboard. While you are talking, it watches for *other* voices and fills a riveted meter. At the end of class it reports how many times chatter started and how quickly it usually began.

This is a first, simple version: a gauge, live counts, and an end-of-class report. A later phase can swap in a cartoon cat that gets sicker as chatter builds, with a hospital bed, a kid-appropriate gravestone “class mark,” and a revive button. The code already has a character hook for that swap; v0.1 only ships the meter gauge.

## Privacy

- No audio is recorded or sent anywhere.
- No names, and no attempt to identify students.
- Only a numeric voice profile (MFCC averages), settings, and class counts are stored in this browser’s `localStorage`.
- The privacy line is shown on every screen.

## How to run it

Open `index.html` from a local web server or (later, with your approval) GitHub Pages. A bare `file://` page may not be able to use the microphone or the optional service worker.

```bash
cd talkmeter
python3 -m http.server 8080
```

Then open http://localhost:8080/ in Chrome or another current Chromium browser.

On a Smartboard, open the page, use **Full screen** (or the `F` key), pick **Size** and **Corner**, and leave the window in the corner of the board. The default microphone is used unless you pick another one. An always-connected Onn USB mic is fine if it is the system default.

First-time setup:

1. Allow the microphone and confirm the right device.
2. Calibrate room noise in a quiet moment (music off).
3. Read the on-screen script in your normal teaching voice for 10 seconds. That builds the fingerprint. Use **Re-enroll** if you change rooms, mics, or your voice.
4. Start class. **End class** when you are done to see the report. Trends stay in this browser and can be exported as CSV or cleared.

After the first load, the optional service worker lets the page keep working offline.

## What v0.1 can do

- Voice-activity detection and a simple teacher fingerprint (averaged MFCCs, plus short vowel snapshots).
- The gauge rises only when you are talking *and* additional non-teacher speech is present (including overlap).
- Rough voice-count labels: **1–2 voices**, **a few**, **lots**. These are honest estimates, labeled as such — not a headcount.
- Chatter-start count, and the time from when you begin talking until chatter starts. End-of-class line, for example: “Chatter started 9 times, usually about 8 seconds after you began.”
- Sensitivity slider and a calibrate-room-noise button.
- Best-effort ignoring of steady tones and simple held-chord loops (Smartboard music). This is not perfect.

## Limits (please read)

These were checked with **synthetic** speech-like signals and tones on this computer. They are **not** yet validated with a real microphone, an Onn USB mic, a Promethean Smartboard, classroom acoustics, or real student voices.

- The “teacher” model is a lightweight MFCC profile, not a neural speaker model. Similar voices, a sore throat, a different mic distance, or a very noisy room can confuse it.
- Overlap detection is approximate. Mixed teacher-plus-class audio can be harder than a clearly different second voice.
- The 3-level voice count is rough and can under- or over-estimate a busy class.
- Music rejection works best on **steady** tones and static pads. Songs with vocals, sudden track changes, or speech-like music can still look like chatter. Turn board music down when you can, and use room-noise calibration with the music off.
- Pitch tracking can jump octaves on some voices; timbre matching is trusted more than pitch when the fingerprint is close.
- Previews and screenshots are offline fixture renders, not live classroom captures.

## Tests

Synthetic tests (no microphone) live in `tests/run.js`:

```bash
node tests/run.js
```

They cover enrollment vs a different synthetic speaker, teacher-alone (no false chatter), overlap chatter and delay, two separate starts, child-only (no count when you are silent), steady-tone and simple-loop rejection, summary wording, and storage that never keeps audio.

## Publishing

Do **not** push, create a GitHub repo, or publish this app until Christopher explicitly approves that release. This folder is a local git repo only.

## License notes

Lilita One and Rammetto One are under the SIL Open Font License 1.1 (see `assets/fonts/README.md`).

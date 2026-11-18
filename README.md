# Talk Meter v0.2

A small classroom web app for Christopher Parsons (Steady Bots / Benton Grade School). Park it in a corner of the Smartboard. While you are talking, it watches for *other* voices and fills a riveted meter. At the end of class it reports how many times chatter started and how quickly it usually began.

v0.2 adds **SpeechNet**, a tiny in-browser speech gate (same idea as Level 4 ClapNet) so the meter is meant to move only for human voices — children and adults — not claps, chairs, music, or other room noise. A later phase can swap in a cartoon cat that gets sicker as chatter builds, with a hospital bed, a kid-appropriate gravestone “class mark,” and a revive button. The code already has a character hook for that swap; v0.1 only ships the meter gauge.

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

## SpeechNet (v0.2)

The Level 4 clap detector uses a fast DSP trigger plus a tiny int8 **ClapNet** MLP in plain JavaScript (no TensorFlow.js). Talk Meter v0.2 uses the same recipe for voices:

- Offline training on openly licensed FSD50K / Freesound clips (CC0 and CC-BY 3.0) already on this box, using exclusive labels so mixed speech+clap recordings do not poison a class. Laughter is trained as `other`, so it does not fill the meter by default.
- A compact **SpeechNet** (128-d log-mel summary → 32 → 16 → 4, int8 weights, **about 13 KB**) shipped in `assets/speechnet.json`.
- Plain JS inference in `js/speechnet.js`. Nothing is uploaded. After the first load, the model is cached by the service worker for offline use.
- About **0.3 ms per 512 ms window** on a development CPU, well under ClapNet’s 12 ms budget. On a weaker Smartboard panel, a 10–30× slowdown would still be a few milliseconds, at about 12 scores per second.
- A smoothed speech score gates the existing teacher / chatter logic: a frame can count as chatter only when SpeechNet is confident it is hearing **speech**, not music or clapping. Settings include an optional **“what I hear”** label: Voices / Music / Clapping / Other / Quiet.

Held-out results on real (non-classroom) recordings, clip-level speech vs non-speech: **85.6% accuracy**, speech precision **0.67**, recall **0.70** (42 of 60 speech clips, 189 of 210 non-speech). Details and the 4-class confusion table are in `docs/speechnet_v0.2.md`. These are not classroom / Onn-mic / Promethean numbers.

A full YAMNet (or similar) model was considered as a live classifier, the way the clap research used YAMNet as an *offline teacher*. For the Smartboard we ship the tiny student, like ClapNet, instead of a multi-megabyte TensorFlow.js model.

## What the app can do


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
- SpeechNet reduces false chatter from claps, tones, and many non-voice noises, but it is not perfect. Music with vocals, very short interjections, overlapping classroom noise, and laughter can still be confused. Turn board music down when you can, and use the “what I hear” label while you tune sensitivity.
- Pitch tracking can jump octaves on some voices; timbre matching is trusted more than pitch when the fingerprint is close.
- Previews and screenshots are offline fixture renders, not live classroom captures.

## Tests

Synthetic tests (no microphone) live in `tests/run.js`, including SpeechNet checks:

```bash
node tests/run.js
```

`node tests/run.js` currently reports 32 passing synthetic checks. They cover enrollment vs a different synthetic speaker, SpeechNet speech vs sine/music/clicks, teacher-alone (no false chatter), overlap chatter and delay, two separate starts, child-only (no count when you are silent), steady-tone, chord-loop and click rejection under the teacher, summary wording, and storage that never keeps audio.

## Publishing

v0.1 is already live. Do **not** push or publish v0.2 until Christopher explicitly approves that release.

## License notes

Lilita One and Rammetto One are under the SIL Open Font License 1.1 (see `assets/fonts/README.md`).

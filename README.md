# Talk Meter v0.3

A small classroom web app for Christopher Parsons (Steady Bots / Benton Grade School). Park it in a corner of the Smartboard. While you are talking, it watches for *other* voices and fills a riveted meter. At the end of class it reports how many times chatter started and how quickly it usually began.

v0.3 retrains **SpeechNet** on more openly licensed speech, music (including vocals), and noise, and adds an optional **Learn my room** 30-second setup step that stores only numeric thresholds. The meter is still meant to move only for human voices — children and adults — not claps, chairs, music, or other room noise. A later phase can swap in a cartoon cat that gets sicker as chatter builds, with a hospital bed, a kid-appropriate gravestone “class mark,” and a revive button. The code already has a character hook for that swap; the shipped view is still the meter gauge.

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
2. Calibrate room noise in a quiet moment (music off). Optionally use **Learn my room** (30 seconds) so leftover HVAC or board hum is less likely to count as voices. Only a couple of numbers are saved, never the sound.
3. Read the on-screen script in your normal teaching voice for 10 seconds. That builds the fingerprint. Use **Re-enroll** if you change rooms, mics, or your voice.
4. Start class. **End class** when you are done to see the report. Trends stay in this browser and can be exported as CSV or cleared.

After the first load, the optional service worker lets the page keep working offline.

## SpeechNet (v0.3)

The Level 4 clap detector uses a fast DSP trigger plus a tiny int8 **ClapNet** MLP in plain JavaScript (no TensorFlow.js). Talk Meter uses the same recipe for voices:

- Offline training on openly licensed FSD50K / Freesound (CC0 and CC-BY 3.0 only), LibriSpeech (CC-BY 4.0), MUSAN (CC-BY 4.0, with sung/music-with-vocals labeled as music), plus cheap room-reverb (OpenSLR 28, Apache 2.0, augmentation only), USB-mic EQ, and speech-over-noise mixes. ESC-50 (CC-BY-NC), DEMAND (CC-BY-SA), and research-only kids corpora were skipped. Licenses are listed in `docs/speechnet_v0.3.md`.
- A compact **SpeechNet** (128-d log-mel summary → 64 → 32 → 4, int8 weights, **about 21 KB**) shipped in `assets/speechnet.json`.
- Plain JS inference in `js/speechnet.js`. Nothing is uploaded. After the first load, the model is cached by the service worker for offline use.
- About **0.6–0.8 ms per 512 ms window** on a development CPU (first call a few ms), well under ClapNet’s 12 ms budget. On a weaker Smartboard panel, a 10–30× slowdown would still be a few milliseconds, at about 12 scores per second.
- A smoothed speech score (default threshold **0.50**) gates the existing teacher / chatter logic. Settings include an optional **“what I hear”** label and **Learn my room**.

On the **same frozen FSD50K held-out set** as v0.2, the published majority-vote speech vs non-speech score went from accuracy 0.856 / precision 0.667 / recall 0.700 to 0.844 / 0.629 / 0.733 (not an improvement). The live **speech gate**, which is what fills the meter, is better: accuracy **0.874**, precision **0.717**, recall **0.717** (43 of 60 speech clips, 193 of 210 non-speech, 17 false speech vs 21 in v0.2 majority vote). A second held-out set (LibriSpeech test + MUSAN holdout) is easier (clip accuracy 0.931) and should not be compared directly to the FSD number. Details, licenses, and per-class false-alarm rates are in `docs/speechnet_v0.3.md`. These are still not classroom / Onn-mic / Promethean numbers.

A full YAMNet (or similar) model was considered as a live classifier, the way the clap research used YAMNet as an *offline teacher*. For the Smartboard we ship the tiny student, like ClapNet, instead of a multi-megabyte TensorFlow.js model.

## What the app can do


- Voice-activity detection and a simple teacher fingerprint (averaged MFCCs, plus short vowel snapshots).
- The gauge rises only when you are talking *and* additional non-teacher speech is present (including overlap).
- Rough voice-count labels: **1–2 voices**, **a few**, **lots**. These are honest estimates, labeled as such — not a headcount.
- Chatter-start count, and the time from when you begin talking until chatter starts. End-of-class line, for example: “Chatter started 9 times, usually about 8 seconds after you began.”
- Sensitivity slider, a 3-second room-noise calibrate, and an optional 30-second **Learn my room** step (numbers only).
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

`node tests/run.js` currently reports 37 passing synthetic checks. They cover enrollment vs a different synthetic speaker, SpeechNet speech vs sine/music/clicks, Learn-my-room threshold math (no audio stored), teacher-alone (no false chatter), overlap chatter and delay, two separate starts, child-only (no count when you are silent), steady-tone, chord-loop and click rejection under the teacher, summary wording, and storage that never keeps audio.

## Publishing

v0.1 is already live. Do **not** push or publish v0.2 or v0.3 until Christopher explicitly approves that release.

## License notes

Lilita One and Rammetto One are under the SIL Open Font License 1.1 (see `assets/fonts/README.md`).

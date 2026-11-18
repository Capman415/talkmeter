# SpeechNet v0.3

A tiny in-browser speech gate for Talk Meter, same recipe as Level 4 **ClapNet** and Talk Meter v0.2, trained on more openly licensed audio so the meter is less likely to treat any sound as a voice, and more likely to catch kids and adults talking.

## Licenses (checked before use)

| Source | What we used it for | License | Notes |
|---|---|---|---|
| FSD50K / Freesound (already on this box + extra previews) | speech, child speech, music, clapping, coughs, footsteps, doors, fans, singing, etc. | **CC0** or **CC-BY 3.0** only | BY-NC and sampling+ clips were skipped. Attribution is in the Freesound/FSD50K metadata; only numeric weights are shipped. |
| LibriSpeech (`dev-clean` train, `test-clean` / `test-other` held-out) | adult read speech | **CC-BY 4.0** | OpenSLR 12, derived from LibriVox public-domain audiobooks. Panayotov et al. |
| MUSAN (speech, music, noise) | adult speech, music including vocals as *music*, room-like noise | **CC-BY 4.0** | OpenSLR 17, Snyder, Chen, Povey 2015. Music with vocals is labeled **music**, not speech. |
| OpenSLR 28 room impulse responses | reverb augmentation only | **Apache 2.0** | Not used as a class; never shipped. |
| Synthetic tones, formant speech, clicks, rumble | in-app sine/chord tests and HVAC-like noise | generated here | Not a third-party corpus. |

**Skipped (license too tight or unclear for a public derived model):**

- **ESC-50** — CC-BY-NC. Non-commercial. We do not use it.
- **DEMAND** — CC-BY-SA 3.0. ShareAlike corpus; skipped. MUSAN noise covers a similar need.
- **CSLU Kids**, **MyST**, other research-only kids corpora — licenses typically do not allow redistribution of a derived model in a public app. Skipped.
- **CHILDES** and similar — often CC-BY-NC-SA. Skipped.
- **Common Voice** teens/children — CC0 would have been ideal, but a practical licensed subset was not pulled this round (the public English bundles are very large / often gated). Child speech in this model still comes from FSD50K `Child_speech_and_kid_speaking` (CC0/CC-BY).

Mixture labels: speech over classroom-like noise = **speech**. Music with vocals / singing = **music**. Speech mixed under louder music (negative SNR) = **music**; speech over quieter music = **speech**.

The official comparison test is the **frozen v0.2 FSD50K held-out clip list** in `tools/fsd_heldout_v02.json` (those IDs are never used for training). A second held-out set uses LibriSpeech official test sets and a hash-held-out slice of MUSAN.

## Runtime

- Architecture: 128-d log-mel summary → **64 → 32 → 4**, int8 weights (`(128, 64) → (64, 32) → (32, 4)`).
- Shipped file: `assets/speechnet.json` is **21801 bytes** (21.3 KB), under the 100 KB budget.
- Gate default after this training pass: P(speech) ≥ **0.50**, and at least 0.05 above P(music) and P(clapping), plus the existing tone guard and optional **Learn my room** threshold bump (numbers only, no audio saved).
- Speed on a development CPU: first call a few milliseconds, then about **0.6–0.8 ms** per 512 ms window (well under the 12 ms ClapNet budget). About 12 scores/sec. A 10–30× slower Smartboard should still be a few milliseconds per score.

## Before / after (same frozen FSD50K held-out, 270 clips)

v0.2 published a **clip-level majority-vote** speech vs non-speech score. The live app actually uses a **speech gate** (P(speech) above a threshold and above music/clapping). Both are shown below. Majority vote on this same test did **not** beat v0.2; the gate, which is what fills the meter, did.

| Metric (270 frozen clips) | v0.2 majority vote | v0.3 majority vote | v0.3 speech gate (th=0.50) |
|---|---:|---:|---:|
| Speech vs non-speech accuracy | 0.856 | 0.844 | **0.874** |
| Speech precision | 0.667 | 0.629 | **0.717** |
| Speech recall | 0.700 | 0.733 | **0.717** |
| True speech / missed speech | 42 / 18 | 44 / 16 | 43 / 17 |
| False speech / true non-speech | 21 / 189 | 26 / 184 | **17 / 193** |

The gate was chosen for fewer false alarms (the original classroom complaint) without dropping recall below v0.2. Laughter is still the noisiest non-speech class (9 of 23 laughter clips gated as speech). Music and clapping false alarms are low on this set.

## Learn my room

Setup can optionally listen for **30 seconds** (HVAC can stay on, nobody talking, board music off). SpeechNet scores those windows, then stores only numbers in `localStorage`: a noise floor and a voice threshold of `max(default, p95(P(speech)) + 0.10)`, capped at 0.70. The audio is discarded immediately. If the room already looks quiet, the default threshold is kept. This is cheap — one 30-second pass, no extra model — and is meant to damp leftover fan or board hum if it looks a bit like speech. It cannot fix a badly confused live classroom; that still needs a real-mic test.


## Same FSD50K held-out as v0.2 (windows)

              precision    recall  f1-score   support

      speech      0.544     0.685     0.607       349
       music      0.374     0.496     0.426       117
    clapping      0.789     0.413     0.543       254
       other      0.647     0.640     0.643       664

    accuracy                          0.598      1384
   macro avg      0.589     0.558     0.555      1384
weighted avg      0.624     0.598     0.597      1384


| true \ pred | speech | music | clapping | other |
|---|---|---|---|---|
| speech | 239 | 33 | 5 | 72 |
| music | 11 | 58 | 0 | 48 |
| clapping | 25 | 12 | 105 | 112 |
| other | 164 | 52 | 23 | 425 |

## Same FSD50K held-out as v0.2 (clip-level majority vote)

v0.2 baseline, speech vs non-speech: accuracy **0.856**, speech precision **0.667**, recall **0.700** (42 / 18 / 21 / 189).

              precision    recall  f1-score   support

      speech      0.629     0.733     0.677        60
       music      0.516     0.485     0.500        33
    clapping      0.842     0.516     0.640        62
       other      0.618     0.704     0.659       115

    accuracy                          0.641       270
   macro avg      0.651     0.610     0.619       270
weighted avg      0.659     0.641     0.639       270


| true \ pred | speech | music | clapping | other |
|---|---|---|---|---|
| speech | 44 | 6 | 1 | 9 |
| music | 0 | 16 | 0 | 17 |
| clapping | 4 | 2 | 32 | 24 |
| other | 22 | 7 | 5 | 81 |

### Speech vs non-speech (clip-level, comparable)

              precision    recall  f1-score   support

      speech      0.629     0.733     0.677        60
  non-speech      0.920     0.876     0.898       210

    accuracy                          0.844       270
   macro avg      0.774     0.805     0.787       270
weighted avg      0.855     0.844     0.849       270


| true \ pred | speech | non-speech |
|---|---|---|
| speech | 44 | 16 |
| non-speech | 26 | 184 |

### Window-level speech gates on this same test

- th=0.30: precision 0.502, recall 0.762 (tp=266 fp=264 tn=771 fn=83)
- th=0.35: precision 0.522, recall 0.739 (tp=258 fp=236 tn=799 fn=91)
- th=0.40: precision 0.542, recall 0.708 (tp=247 fp=209 tn=826 fn=102)
- th=0.45: precision 0.562, recall 0.670 (tp=234 fp=182 tn=853 fn=115)
- th=0.50: precision 0.587, recall 0.645 (tp=225 fp=158 tn=877 fn=124)
- th=0.55: precision 0.614, recall 0.607 (tp=212 fp=133 tn=902 fn=137)

### False-alarm rate per non-speech class (clip-level, predicted speech)

| class / source | false alarms | n | FA rate |
|---|---:|---:|---:|
| Cheering | 1 | 1 | 1.000 |
| Crowd | 3 | 3 | 1.000 |
| Laughter | 11 | 23 | 0.478 |
| Tap | 4 | 35 | 0.114 |
| Door | 3 | 29 | 0.103 |
| Cutlery_and_silverware | 1 | 10 | 0.100 |
| Finger_snapping | 1 | 14 | 0.071 |
| Clapping | 2 | 38 | 0.053 |
| Applause | 0 | 8 | 0.000 |
| Drum | 0 | 33 | 0.000 |
| Hands | 0 | 1 | 0.000 |
| Typing | 0 | 15 | 0.000 |

### False-alarm rate per non-speech class (clip-level speech **gate** at th=0.50)

| class / source | false alarms | n | FA rate |
|---|---:|---:|---:|
| Crowd | 2 | 3 | 0.667 |
| Laughter | 9 | 23 | 0.391 |
| Tap | 3 | 35 | 0.086 |
| Door | 2 | 29 | 0.069 |
| Clapping | 1 | 38 | 0.026 |
| Applause | 0 | 8 | 0.000 |
| Cheering | 0 | 1 | 0.000 |
| Cutlery_and_silverware | 0 | 10 | 0.000 |
| Drum | 0 | 33 | 0.000 |
| Finger_snapping | 0 | 14 | 0.000 |
| Hands | 0 | 1 | 0.000 |
| Typing | 0 | 15 | 0.000 |

## Second held-out set (LibriSpeech test + MUSAN holdout)

              precision    recall  f1-score   support

      speech      0.932     0.974     0.952       380
       music      0.918     0.838     0.876        80
    clapping      0.000     0.000     0.000         0
       other      0.768     0.671     0.716        79

    accuracy                          0.909       539
   macro avg      0.654     0.621     0.636       539
weighted avg      0.906     0.909     0.906       539


| true \ pred | speech | music | clapping | other |
|---|---|---|---|---|
| speech | 370 | 1 | 0 | 9 |
| music | 6 | 67 | 0 | 7 |
| clapping | 0 | 0 | 0 | 0 |
| other | 21 | 5 | 0 | 53 |

              precision    recall  f1-score   support

      speech      0.932     0.974     0.952       380
  non-speech      0.930     0.830     0.877       159

    accuracy                          0.931       539
   macro avg      0.931     0.902     0.915       539
weighted avg      0.931     0.931     0.930       539


| true \ pred | speech | non-speech |
|---|---|---|
| speech | 370 | 10 |
| non-speech | 27 | 132 |

### False-alarm rate by source (new held-out, predicted speech)

| class / source | false alarms | n | FA rate |
|---|---:|---:|---:|
| MUSAN-noise | 21 | 79 | 0.266 |
| MUSAN-music | 6 | 80 | 0.075 |

## Honest limits

These numbers are still not a Benton classroom / Onn USB mic / Promethean Smartboard session. Expect a drop in a live room. Likely remaining failure cases: music with very speech-like vocals, very short interjections, overlapping class noise, laughter (trained as *other*), and child speech that is far from the FSD child clips and closer to shouts or play noise. **Learn my room** can raise the threshold if leftover HVAC or board hum looks like speech, but it cannot invent a better model. Real-classroom listening is still required before trusting the counts.

# SpeechNet v0.2

A tiny in-browser speech gate for Talk Meter, built the same way as Level 4 **ClapNet**:

- Offline training on openly licensed FSD50K / Freesound clips already on this box (CC0 and CC-BY 3.0), plus a few synthetic speech and tone windows so in-app test signals behave.
- A small MLP (128-32-16-4) on 128-d log-mel summaries with int8 weights, shipped as `assets/speechnet.json`.
- Plain JavaScript inference (`js/speechnet.js`). No TensorFlow.js, no ONNX runtime, nothing leaves the device.
- YAMNet-style class idea (Speech / Child speech / Music / Clapping / other) but a student small enough for a Smartboard browser, like ClapNet (~tens of KB, well under 12 ms).

**Runtime:** `assets/speechnet.json` is **13 KB**. Inference is about **0.3–1.2 ms** per 512 ms window on a development CPU (first call can be a few ms). That is well under the 12 ms ClapNet budget. At ~12 windows per second, this is a small fraction of a core here; a 10–30× slower Smartboard panel should still be a few milliseconds per score. A simple tone guard (very stable log-mel over the window) stops held sines from being labeled as speech.

**Gate:** smoothed P(speech) ≥ 0.40, and at least 0.05 above P(music) and P(clapping).


**Window:** 512 ms at 16 kHz, 32-band log-mel, 16 frames (32 ms hop).  
**Labels:** `speech` (adult Speech, Child speech, Chatter, Crowd), `music` (Music, Drum), `clapping` (Clapping, Applause, Hands, Finger snapping), `other` (taps, knocks, typing, door, cutlery, laughter, cheering). Laughter is in `other` so it does not fill the meter by default.

## Held-out window-level results

              precision    recall  f1-score   support

      speech      0.559     0.630     0.592       351
       music      0.451     0.492     0.471       122
    clapping      0.532     0.618     0.572       254
       other      0.670     0.574     0.618       666

    accuracy                          0.589      1393
   macro avg      0.553     0.578     0.563      1393
weighted avg      0.598     0.589     0.590      1393


| true \ pred | speech | music | clapping | other |
|---|---|---|---|---|
| speech | 221 | 8 | 34 | 88 |
| music | 28 | 60 | 7 | 27 |
| clapping | 19 | 5 | 157 | 73 |
| other | 127 | 60 | 97 | 382 |

## Held-out clip-level results (majority vote, real recordings only)

              precision    recall  f1-score   support

      speech      0.667     0.700     0.683        60
       music      0.667     0.606     0.635        33
    clapping      0.597     0.694     0.642        62
       other      0.705     0.643     0.673       115

    accuracy                          0.663       270
   macro avg      0.659     0.661     0.658       270
weighted avg      0.667     0.663     0.663       270


| true \ pred | speech | music | clapping | other |
|---|---|---|---|---|
| speech | 42 | 0 | 7 | 11 |
| music | 4 | 20 | 3 | 6 |
| clapping | 3 | 2 | 43 | 14 |
| other | 14 | 8 | 19 | 74 |

## Speech vs non-speech (clip-level)

              precision    recall  f1-score   support

      speech      0.667     0.700     0.683        60
  non-speech      0.913     0.900     0.906       210

    accuracy                          0.856       270
   macro avg      0.790     0.800     0.795       270
weighted avg      0.858     0.856     0.857       270


| true \ pred | speech | non-speech |
|---|---|---|
| speech | 42 | 18 |
| non-speech | 21 | 189 |

These numbers are from openly licensed recorded clips, not a classroom Promethean / Onn-mic session. Expect a drop in a live room. Music with vocals, overlapping classroom noise, and very short interjections are the likely failure cases.

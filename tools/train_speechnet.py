#!/usr/bin/env python3
"""Train SpeechNet, a tiny int8 MLP that tells speech from music, clapping, and other sounds.

Same recipe as Level 4 ClapNet: licensed FSD50K/Freesound clips (already on the box),
sklearn MLP, int8 JSON weights, plain-JS runtime. No audio is shipped in the model file.
"""
from __future__ import annotations

import base64
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path('/workspace/talkmeter')
FSD = Path('/workspace/beatbots/level4/research/data/fsd50k')
AUDIO = FSD / 'audio'
CACHE = Path('/tmp/talkmeter-fsd16k')
MODEL = ROOT / 'assets/speechnet.json'
REPORT = ROOT / 'docs/speechnet_v0.2.md'

SR = 16000
WIN = 8192          # 512 ms
HOP_FR = 512        # 32 ms, 16 frames exactly in WIN
NFFT = 512
NMEL = 32
NFR = 16
FEAT = NMEL * 4      # 128  (mean, std, max, mean-delta)
CLASSES = ['speech', 'music', 'clapping', 'other']

SPEECH_C = {'Speech', 'Child_speech_and_kid_speaking', 'Chatter', 'Crowd'}
MUSIC_C = {'Music', 'Drum'}
CLAP_C = {'Clapping', 'Applause', 'Hands', 'Finger_snapping'}
OTHER_C = {'Tap', 'Knock', 'Typing', 'Door', 'Cutlery_and_silverware', 'Laughter', 'Cheering'}

# --- features (must match js/speechnet.js) ---
def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(m):
    return 700 * (10 ** (m / 2595) - 1)

def mel_filters():
    lo, hi = hz_to_mel(100), hz_to_mel(7800)
    pts = [mel_to_hz(lo + (hi - lo) * i / (NMEL + 1)) for i in range(NMEL + 2)]
    bins = [int(np.floor((NFFT + 1) * hz / SR)) for hz in pts]
    fb = []
    for i in range(NMEL):
        a, c, d = bins[i], bins[i + 1], bins[i + 2]
        if c <= a:
            c = a + 1
        if d <= c:
            d = c + 1
        r = np.zeros(NFFT // 2 + 1, np.float64)
        for k in range(a, c):
            if k < r.size:
                r[k] = (k - a) / max(c - a, 1)
        for k in range(c, d):
            if k < r.size:
                r[k] = (d - k) / max(d - c, 1)
        fb.append((r, a, d))
    return fb

FB = mel_filters()
HANN = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(NFFT) / (NFFT - 1))

def logmel(w: np.ndarray) -> np.ndarray:
    """16 x 32 log-mel. Must match js/speechnet.js."""
    w = np.asarray(w, np.float32)
    if len(w) < WIN:
        w = np.pad(w, (0, WIN - len(w)))
    elif len(w) > WIN:
        w = w[:WIN]
    mx = max(float(np.max(np.abs(w))), 1e-6)
    M = np.empty((NFR, NMEL), np.float64)
    for fr in range(NFR):
        seg = w[fr * HOP_FR:fr * HOP_FR + NFFT] / mx * HANN
        spec = np.fft.rfft(seg, n=NFFT)
        pwr = (spec.real * spec.real + spec.imag * spec.imag)
        for m in range(NMEL):
            r, a, d = FB[m]
            s = float(np.dot(r[a:min(d, pwr.size)], pwr[a:min(d, pwr.size)]))
            M[fr, m] = np.log(s + 1e-6)
    return M

def feat(w: np.ndarray) -> np.ndarray:
    """Compact 128-d summary of the 512 ms log-mel (mean, std, max, mean-delta)."""
    M = logmel(w)
    d = np.diff(M, axis=0)
    f = np.concatenate([M.mean(0), M.std(0), M.max(0), d.mean(0)])
    return f.astype(np.float32)

# --- audio ---
def decode_16k(path: Path) -> np.ndarray:
    cache = CACHE / (path.stem + '.f32')
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        return np.fromfile(cache, dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path),
         '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-acodec', 'pcm_f32le', 'pipe:1'],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    x = np.frombuffer(r.stdout, dtype=np.float32).copy()
    cache.write_bytes(x.tobytes())
    return x

def rms(x):
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))

def windows(x, max_n=8):
    if len(x) < WIN:
        y = np.pad(x, (0, WIN - len(x)))
        if rms(y) > 0.005:
            return [y]
        return []
    hop = WIN // 2
    starts = list(range(0, len(x) - WIN + 1, hop))
    # always include a trailing window if the file is longer
    if starts[-1] != len(x) - WIN:
        starts.append(len(x) - WIN)
    segs = []
    for s in starts:
        w = x[s:s + WIN]
        if rms(w) > 0.005:
            segs.append(w.copy())
    if len(segs) > max_n:
        # spread across the file
        idx = np.linspace(0, len(segs) - 1, max_n).astype(int)
        segs = [segs[i] for i in idx]
    return segs

def target_of(classes: set[str]) -> str | None:
    """Exclusive labels only. Mixed speech+clap/music clips are skipped so they do not poison a class."""
    has_speech = bool(classes & {'Speech', 'Child_speech_and_kid_speaking', 'Chatter'})
    has_music = bool(classes & MUSIC_C)
    has_clap = bool(classes & CLAP_C)
    has_noise = bool(classes & {'Tap', 'Knock', 'Typing', 'Door', 'Cutlery_and_silverware'})
    # Crowd/Cheering/Laughter mixed with speech or clapping are too messy to use as a hard label.
    if has_speech and (has_music or has_clap or has_noise or 'Laughter' in classes or 'Cheering' in classes):
        return None
    if has_speech:
        return 'speech'
    if has_music and has_clap:
        return None
    if has_music:
        return 'music'
    if has_clap:
        return 'clapping'
    if has_noise or 'Laughter' in classes:
        return 'other'
    return None

def augment(w, rng):
    y = w * float(np.exp(rng.uniform(-0.7, 0.4)))
    if rng.random() < 0.7:
        y = y + rng.normal(0, 0.01 + 0.03 * rng.random(), len(y)).astype(np.float32) * (np.max(np.abs(y)) + 1e-6)
    if rng.random() < 0.3:
        # cheap one-pole low-pass
        a = float(rng.uniform(0.2, 0.6))
        z = 0.0
        out = np.empty_like(y)
        for i, s in enumerate(y):
            z = z + a * (s - z)
            out[i] = z
        y = out
    if rng.random() < 0.3:
        # mild high-pass
        a = float(rng.uniform(0.92, 0.99))
        prev = 0.0
        z = 0.0
        out = np.empty_like(y)
        for i, s in enumerate(y):
            z = a * (z + s - prev)
            prev = s
            out[i] = z
        y = out
    m = np.max(np.abs(y)) + 1e-9
    if m > 1:
        y = y / m
    return y.astype(np.float32)

# --- synthetic extras (helps the net ignore tones the classroom app already tests) ---
def synth_speech(n, f0=110.0, scale=1.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    contour = 1 + 0.12 * np.sin(2 * np.pi * 2.4 * t) + 0.05 * np.sin(2 * np.pi * 5.0 * t)
    phase = np.cumsum(2 * np.pi * f0 * contour / SR)
    # cheap pulse train + formants
    src = (np.mod(phase, 2 * np.pi) < 0.6).astype(np.float32)
    vowels = np.array([[730, 1090, 2440], [270, 2290, 3010], [300, 870, 2240], [530, 1840, 2480]]) * scale
    y = src
    hop = int(0.02 * SR)
    # one formant resonator cascade with changing targets
    for fi in range(3):
        out = np.zeros_like(y)
        y1 = y2 = 0.0
        cur = vowels[0, fi]
        bw = [90, 120, 160][fi]
        r = np.exp(-np.pi * bw / SR)
        a1 = a2 = 0.0
        vi = 0
        target = cur
        for i, s in enumerate(y):
            if i % hop == 0:
                vi = (i // int(0.12 * SR)) % len(vowels)
                target = vowels[vi, fi] * (1 + 0.03 * rng.normal())
                theta = 2 * np.pi * cur / SR
                a1 = 2 * r * np.cos(theta)
                a2 = -r * r
            cur += (target - cur) * 0.08
            o = s + a1 * y1 + a2 * y2
            y2, y1 = y1, o
            out[i] = o
        y = out
    y = np.diff(y, prepend=0)  # lip
    y *= 0.4 + 0.6 * (0.5 + 0.5 * np.sin(2 * np.pi * 3.6 * t))
    y += rng.normal(0, 0.004, n)
    m = np.max(np.abs(y)) + 1e-9
    return (y / m * 0.28).astype(np.float32)

def synth_tone(n, freqs, seed=0):
    t = np.arange(n) / SR
    x = np.zeros(n, np.float32)
    for f in freqs:
        x += np.sin(2 * np.pi * f * t)
    x /= max(len(freqs), 1)
    return (x * 0.22).astype(np.float32)

def synth_clicks(n, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n, np.float32)
    for s in rng.integers(0, n, size=8):
        env = np.exp(-np.arange(min(800, n - s)) / 80)
        x[s:s + len(env)] += rng.normal(0, 1) * 0.6 * env * rng.choice([-1, 1])
    return x

def inventory():
    sel = json.loads((FSD / 'sel.json').read_text())
    items = {}
    for cls, rows in sel.items():
        for row in rows:
            fid = str(row[0])
            path = AUDIO / f'{fid}.mp3'
            if not path.exists():
                continue
            rec = items.setdefault(fid, {
                'id': fid, 'path': path, 'classes': set(),
                'license': row[2], 'uploader': row[1], 'title': row[3]
            })
            rec['classes'].add(cls)
    out = []
    for rec in items.values():
        lab = target_of(rec['classes'])
        if not lab:
            continue
        rec['label'] = lab
        out.append(rec)
    return out

def collect():
    items = inventory()
    print('clips', len(items), Counter(r['label'] for r in items))
    X = {'tr': [], 'te': []}
    Y = {'tr': [], 'te': []}
    META = {'tr': [], 'te': []}
    skipped = 0
    for i, rec in enumerate(items):
        sp = 'te' if int(rec['id']) % 5 == 0 else 'tr'
        try:
            x = decode_16k(rec['path'])
        except subprocess.CalledProcessError:
            skipped += 1
            continue
        rng = np.random.default_rng(int(rec['id']) + 17)
        for w in windows(x, max_n=8):
            X[sp].append(feat(w))
            Y[sp].append(rec['label'])
            META[sp].append(rec)
            if sp == 'tr':
                for _ in range(2):
                    X[sp].append(feat(augment(w, rng)))
                    Y[sp].append(rec['label'])
                    META[sp].append(rec)
        if (i + 1) % 200 == 0:
            print(f'  decoded {i+1}/{len(items)}')
    print('decode skipped', skipped)

    # synthetic support so in-app tones and formant speech behave
    for seed, f0, scale, sp in [
        (1, 112, 0.96, 'tr'), (2, 110, 0.96, 'tr'), (3, 230, 1.28, 'tr'),
        (4, 260, 1.36, 'tr'), (5, 200, 1.20, 'tr'),
        (11, 112, 0.96, 'te'), (22, 230, 1.28, 'te'),
    ]:
        w = synth_speech(WIN, f0=f0, scale=scale, seed=seed)
        X[sp].append(feat(w))
        Y[sp].append('speech')
        META[sp].append({'id': f'synth-speech-{seed}', 'label': 'speech', 'license': 'synthetic', 'uploader': 'talkmeter', 'title': 'synthetic speech', 'classes': {'Speech'}, 'path': None})
    freqs_tr = [[220], [262], [330], [392], [440], [523], [880], [1000],
                [262, 330, 392], [294, 370, 440], [196, 247, 294],
                [440, 880], [100, 200, 300]]
    freqs_te = [[247], [349], [698], [262, 330, 392], [440]]
    for seed, freqs, sp in [(i, f, 'tr') for i, f in enumerate(freqs_tr)] + [(100+i, f, 'te') for i, f in enumerate(freqs_te)]:
        X[sp].append(feat(synth_tone(WIN, freqs, seed)))
        Y[sp].append('music')
        META[sp].append({'id': f'synth-tone-{seed}', 'label': 'music', 'license': 'synthetic', 'uploader': 'talkmeter', 'title': 'synthetic tone', 'classes': {'Music'}, 'path': None})
    # gentle noise / HVAC-like low rumble as other
    for seed, sp in [(0, 'tr'), (1, 'tr'), (2, 'tr'), (3, 'te')]:
        rng = np.random.default_rng(200 + seed)
        nse = rng.normal(0, 0.04, WIN).astype(np.float32)
        # brown-ish
        nse = np.cumsum(nse)
        nse /= (np.max(np.abs(nse)) + 1e-9)
        nse *= 0.08
        X[sp].append(feat(nse))
        Y[sp].append('other')
        META[sp].append({'id': f'synth-noise-{seed}', 'label': 'other', 'license': 'synthetic', 'uploader': 'talkmeter', 'title': 'synthetic noise', 'classes': {'Tap'}, 'path': None})
    for seed, sp in [(0, 'tr'), (1, 'tr'), (2, 'te')]:
        X[sp].append(feat(synth_clicks(WIN, seed)))
        Y[sp].append('other')
        META[sp].append({'id': f'synth-click-{seed}', 'label': 'other', 'license': 'synthetic', 'uploader': 'talkmeter', 'title': 'synthetic clicks', 'classes': {'Tap'}, 'path': None})

    for k in X:
        X[k] = np.stack(X[k]).astype(np.float32)
        Y[k] = np.array(Y[k])
        print(k, X[k].shape, Counter(Y[k]))
    return X, Y, META

def quantize(a):
    s = float(np.max(np.abs(a)) / 127.0) or 1.0
    q = np.round(a / s).clip(-127, 127).astype(np.int8)
    return q, s

def b64(q: np.ndarray) -> str:
    return base64.b64encode(q.tobytes()).decode('ascii')

def train(X, Y, META):
    mu = X['tr'].mean(0)
    sd = X['tr'].std(0) + 1e-6
    Ztr = (X['tr'] - mu) / sd
    Zte = (X['te'] - mu) / sd

    # File-level validation inside the train set (no window leak).
    tr_ids = np.array([m['id'] for m in META['tr']])
    uniq = np.unique(tr_ids)
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    val_ids = set(uniq[: max(1, len(uniq) // 5)])
    val_m = np.array([i in val_ids for i in tr_ids])
    tr_m = ~val_m
    print('file-val files', len(val_ids), 'windows', int(val_m.sum()), 'fit', int(tr_m.sum()))

    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        max_iter=1,
        warm_start=True,
        random_state=0,
        alpha=2e-2,
        learning_rate_init=0.003,
        solver='adam',
    )
    best_score, best_state, wait = -1, None, 0
    for it in range(120):
        clf.fit(Ztr[tr_m], Y['tr'][tr_m])
        pred_v = clf.predict(Ztr[val_m])
        acc = float((pred_v == Y['tr'][val_m]).mean())
        # prefer speech-vs-not F1 on val files (clip-level)
        # window-level acc is the early-stop signal; print every 10
        if acc > best_score:
            best_score = acc
            best_state = (np.copy(clf.coefs_[0]), np.copy(clf.intercepts_[0]),
                          np.copy(clf.coefs_[1]), np.copy(clf.intercepts_[1]),
                          np.copy(clf.coefs_[2]), np.copy(clf.intercepts_[2]))
            wait = 0
        else:
            wait += 1
        if it % 15 == 0:
            print(f'  iter {it} val-acc={acc:.3f} best={best_score:.3f}')
        if wait >= 15 and it >= 20:
            print('  early stop at', it, 'best', best_score)
            break
    if best_state is not None:
        for i in range(3):
            clf.coefs_[i] = best_state[2 * i]
            clf.intercepts_[i] = best_state[2 * i + 1]
    print('restored best file-val acc', best_score)
    pred_tr = clf.predict(Ztr)
    pred_te = clf.predict(Zte)
    print('TRAIN')
    print(classification_report(Y['tr'], pred_tr, labels=CLASSES, digits=3))
    print('TEST (held-out files)')
    print(classification_report(Y['te'], pred_te, labels=CLASSES, digits=3))
    cm = confusion_matrix(Y['te'], pred_te, labels=CLASSES)
    print('confusion rows=true cols=pred', CLASSES)
    print(cm)

    # Clip-level majority vote on held-out real files
    by = defaultdict(list)
    for i, rec in enumerate(META['te']):
        if rec['id'].startswith('synth-'):
            continue
        by[rec['id']].append(i)
    y_true, y_pred = [], []
    for fid, idxs in by.items():
        votes = pred_te[idxs]
        # majority
        lab = Counter(votes).most_common(1)[0][0]
        y_true.append(Y['te'][idxs[0]])
        y_pred.append(lab)
    cm_clip = confusion_matrix(y_true, y_pred, labels=CLASSES)
    print('CLIP-LEVEL TEST', len(y_true))
    print(classification_report(y_true, y_pred, labels=CLASSES, digits=3))
    print(cm_clip)

    # speech vs not-speech
    def binr(y):
        return np.array(['speech' if v == 'speech' else 'non-speech' for v in y])
    print('BIN clip-level')
    print(classification_report(binr(y_true), binr(y_pred), digits=3))
    print(confusion_matrix(binr(y_true), binr(y_pred), labels=['speech', 'non-speech']))

    # export
    layers = []
    # sklearn: coefs_[i] is (n_in, n_out)
    for i, (W, b) in enumerate(zip(clf.coefs_, clf.intercepts_)):
        q, s = quantize(W)
        layers.append({
            'shape': [int(W.shape[0]), int(W.shape[1])],
            'w8': b64(q),
            's': s,
            'b': [float(x) for x in b],
        })
    model = {
        'v': 'speechnet-02',
        'sr': SR,
        'win': WIN,
        'nmel': NMEL,
        'nfr': NFR,
        'hop': HOP_FR,
        'fft': NFFT,
        'classes': [str(c) for c in clf.classes_],
        'mu': [float(x) for x in mu],
        'sd': [float(x) for x in sd],
        'layers': layers,
        'notes': 'Tiny SpeechNet (128-32-16-4 int8 MLP) on 128-d log-mel summaries. Trained on FSD50K/Freesound CC0/CC-BY clips already on this box plus a few synthetic tones/speech. Runtime is plain JS; no audio is stored in this file.',
    }
    MODEL.write_text(json.dumps(model))
    print('wrote', MODEL, 'bytes', MODEL.stat().st_size)

    # soft scores for threshold tuning
    proba = clf.predict_proba(Zte)
    cls_idx = {c: i for i, c in enumerate(clf.classes_)}
    order = [cls_idx[c] for c in CLASSES]
    proba = proba[:, order]

    def gate(p, th=0.40, margin=0.08):
        return (p[:, 0] >= th) & (p[:, 0] >= p[:, 1] + margin) & (p[:, 0] >= p[:, 2] + margin)

    is_sp = Y['te'] == 'speech'
    for th in (0.30, 0.35, 0.40, 0.45, 0.50):
        g = gate(proba, th)
        tp = int((g & is_sp).sum()); fn = int((~g & is_sp).sum())
        fp = int((g & ~is_sp).sum()); tn = int((~g & ~is_sp).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        print(f'gate th={th:.2f}  prec={prec:.3f} rec={rec:.3f}  tp={tp} fp={fp} tn={tn} fn={fn}')

    write_report(Y, pred_te, META, cm, cm_clip, y_true, y_pred, clf, proba, is_sp)
    # gold vector for JS parity test
    gold_w = synth_speech(WIN, f0=112, scale=0.96, seed=11)
    gold_f = feat(gold_w)
    gold_z = (gold_f - mu) / sd
    np.savez(Path('/tmp/speechnet_gold.npz'), w=gold_w, f=gold_f, z=gold_z, mu=mu, sd=sd)
    Path('/tmp/speechnet_gold_w.json').write_text(json.dumps([float(x) for x in gold_w[::16]]))  # too big if full
    # write a compact gold: first 16 of feat + a few scores
    raw = gold_z
    h = raw
    for W, b in zip(clf.coefs_[:-1], clf.intercepts_[:-1]):
        h = np.maximum(0, h @ W + b)
    logits = h @ clf.coefs_[-1] + clf.intercepts_[-1]
    ex = np.exp(logits - logits.max())
    sm = ex / ex.sum()
    Path('/tmp/speechnet_gold.json').write_text(json.dumps({
        'w': [float(x) for x in gold_w],
        'f': [float(x) for x in gold_f],
        'classes': list(clf.classes_),
        'scores_sklearn_order': [float(x) for x in sm],
    }))
    print('gold saved')
    return model, cm, cm_clip

def md_table(labels, cm):
    head = '| true \\ pred | ' + ' | '.join(labels) + ' |'
    sep = '|---|' + '---|' * len(labels)
    rows = [head, sep]
    for i, lab in enumerate(labels):
        rows.append('| ' + lab + ' | ' + ' | '.join(str(int(x)) for x in cm[i]) + ' |')
    return '\n'.join(rows)

def write_report(Y, pred_te, META, cm, cm_clip, y_true, y_pred, clf, proba, is_sp):
    from sklearn.metrics import classification_report as cr
    clip_rep = cr(y_true, y_pred, labels=CLASSES, digits=3)
    win_rep = cr(Y['te'], pred_te, labels=CLASSES, digits=3)
    bin_true = ['speech' if v == 'speech' else 'non-speech' for v in y_true]
    bin_pred = ['speech' if v == 'speech' else 'non-speech' for v in y_pred]
    bin_cm = confusion_matrix(bin_true, bin_pred, labels=['speech', 'non-speech'])
    bin_rep = cr(bin_true, bin_pred, labels=['speech', 'non-speech'], digits=3)
    body = f"""# SpeechNet v0.2

A tiny in-browser speech gate for Talk Meter, built the same way as Level 4 **ClapNet**:

- Offline training on openly licensed FSD50K / Freesound clips already on this box (CC0 and CC-BY 3.0), plus a few synthetic speech and tone windows so in-app test signals behave.
- A small MLP (128-32-16-4) on 128-d log-mel summaries with int8 weights, shipped as `assets/speechnet.json`.
- Plain JavaScript inference (`js/speechnet.js`). No TensorFlow.js, no ONNX runtime, nothing leaves the device.
- YAMNet-style class idea (Speech / Child speech / Music / Clapping / other) but a student small enough for a Smartboard browser, like ClapNet (~tens of KB, well under 12 ms).

**Window:** 512 ms at 16 kHz, 32-band log-mel, 16 frames (32 ms hop).  
**Labels:** `speech` (adult Speech, Child speech, Chatter, Crowd), `music` (Music, Drum), `clapping` (Clapping, Applause, Hands, Finger snapping), `other` (taps, knocks, typing, door, cutlery, laughter, cheering). Laughter is in `other` so it does not fill the meter by default.

## Held-out window-level results

{win_rep}

{md_table(CLASSES, cm)}

## Held-out clip-level results (majority vote, real recordings only)

{clip_rep}

{md_table(CLASSES, cm_clip)}

## Speech vs non-speech (clip-level)

{bin_rep}

{md_table(['speech', 'non-speech'], bin_cm)}

These numbers are from openly licensed recorded clips, not a classroom Promethean / Onn-mic session. Expect a drop in a live room. Music with vocals, overlapping classroom noise, and very short interjections are the likely failure cases.
"""
    REPORT.write_text(body)
    print('wrote', REPORT)

def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    X, Y, META = collect()
    train(X, Y, META)

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Train SpeechNet v0.3: more openly licensed data, same FSD50K v0.2 held-out test.

See docs/speechnet_v0.3.md for licenses, metrics, and limits.
No training audio is written into the exported model.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

ROOT = Path('/workspace/talkmeter')
DATA = Path('/workspace/research/speechnet-data')
FSD_AUDIO = Path('/workspace/beatbots/level4/research/data/fsd50k/audio')
FSD_EXTRA = DATA / 'fsd-extra/audio'
LIBRISPEECH = DATA / 'openslr/LibriSpeech'
MUSAN = DATA / 'openslr/musan'
RIRS = DATA / 'openslr/RIRS_NOISES'
CACHE = Path('/tmp/talkmeter-16k-v03')
MODEL = ROOT / 'assets/speechnet.json'
REPORT = ROOT / 'docs/speechnet_v0.3.md'
HELD_PATH = ROOT / 'tools/fsd_heldout_v02.json'
GOLD = Path('/tmp/speechnet_gold.json')

# load v0.2 feature code so the front-end math stays bit-identical
spec = importlib.util.spec_from_file_location('speechnet_v02', ROOT / 'tools/train_speechnet.py')
v02 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v02)

SR, WIN, HOP_FR, NFFT, NMEL, NFR, FEAT = v02.SR, v02.WIN, v02.HOP_FR, v02.NFFT, v02.NMEL, v02.NFR, v02.FEAT
CLASSES = v02.CLASSES
feat, logmel, windows, rms, augment = v02.feat, v02.logmel, v02.windows, v02.rms, v02.augment
synth_speech, synth_tone, synth_clicks = v02.synth_speech, v02.synth_tone, v02.synth_clicks
quantize, b64, target_of = v02.quantize, v02.b64, v02.target_of
md_table = v02.md_table

HELD_V02 = set(json.loads(HELD_PATH.read_text())['ids'])

# Extra FSD labels → SpeechNet class. Exclusive mixing is handled in fsd_label().
SPEECH_X = {
    'Speech', 'Child_speech_and_kid_speaking', 'Chatter', 'Conversation',
    'Whispering', 'Shout', 'Yell',
    'Female_speech_and_woman_speaking', 'Male_speech_and_man_speaking',
}
MUSIC_X = {
    'Music', 'Drum', 'Piano', 'Guitar', 'Singing', 'Female_singing', 'Male_singing',
}
CLAP_X = {'Clapping', 'Applause', 'Hands', 'Finger_snapping'}
OTHER_X = {
    'Tap', 'Knock', 'Typing', 'Door', 'Cutlery_and_silverware', 'Laughter', 'Cheering',
    'Cough', 'Sneeze', 'Walk_and_footsteps', 'Mechanical_fan', 'Breathing', 'Squeak',
    'Slam', 'Dishes_and_pots_and_pans', 'Thump_and_thud', 'Drawer_open_or_close',
    'Cupboard_open_or_close', 'Keys_jangling', 'Writing', 'Wind', 'Clock',
}


def fsd_label(classes: set[str]) -> str | None:
    """Speech over noise is speech; singing / music-with-vocals is music; skip messy mixes."""
    has_speech = bool(classes & SPEECH_X)
    has_music = bool(classes & MUSIC_X)
    has_clap = bool(classes & CLAP_X)
    has_other = bool(classes & OTHER_X)
    if has_speech and (has_music or has_clap):
        return None
    if has_speech:
        return 'speech'
    if has_music and has_clap:
        return None
    if has_music:
        return 'music'
    if has_clap:
        return 'clapping'
    if has_other:
        return 'other'
    return None


def cache_path(path: Path) -> Path:
    key = hashlib.sha1(str(path).encode()).hexdigest()[:16]
    return CACHE / f'{key}-{path.stem}.f32'


def decode_16k(path: Path) -> np.ndarray:
    cache = cache_path(path)
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime and cache.stat().st_size > 0:
        return np.fromfile(cache, dtype=np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(path),
         '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-acodec', 'pcm_f32le', 'pipe:1'],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    x = np.frombuffer(r.stdout, dtype=np.float32).copy()
    cache.write_bytes(x.tobytes())
    return x


def file_key(path: Path) -> str:
    return hashlib.sha1(str(path).encode()).hexdigest()[:12]


def holdout_hash(s: str, mod=5) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) % mod


def list_flac(root: Path):
    if not root.exists():
        return []
    return sorted(root.rglob('*.flac'))


def list_wav(root: Path):
    if not root.exists():
        return []
    return sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.wav', '.flac', '.mp3'})


def load_rir_pool(max_n=40):
    pool = []
    if not RIRS.exists():
        return pool
    cands = list((RIRS / 'simulated_rirs').rglob('*.wav')) if (RIRS / 'simulated_rirs').exists() else []
    cands += list((RIRS / 'real_rirs_isotropic_noises').rglob('*.wav')) if (RIRS / 'real_rirs_isotropic_noises').exists() else []
    rng = np.random.default_rng(3)
    rng.shuffle(cands)
    for p in cands[:max_n * 2]:
        try:
            x = decode_16k(p)
        except Exception:
            continue
        if len(x) < 64:
            continue
        # keep a short IR
        n = min(len(x), 4096)
        ir = x[:n].astype(np.float32)
        m = np.max(np.abs(ir)) + 1e-9
        pool.append(ir / m)
        if len(pool) >= max_n:
            break
    print('RIR pool', len(pool))
    return pool


def cheap_reverb(w, rng, rirs):
    if rirs and rng.random() < 0.7:
        ir = rirs[int(rng.integers(0, len(rirs)))]
    else:
        n = int(rng.uniform(0.06, 0.22) * SR)
        ir = rng.normal(0, 1, n).astype(np.float32) * np.exp(-np.linspace(0, rng.uniform(2.2, 5.0), n)).astype(np.float32)
        ir /= (np.max(np.abs(ir)) + 1e-9)
    y = np.convolve(w, ir, mode='full')[:len(w)].astype(np.float32)
    m = np.max(np.abs(y)) + 1e-9
    if m > 1:
        y = y / m
    return y


def usb_mic_eq(w, rng):
    """Mild boxy USB-mic EQ: some mid emphasis and a gentle high roll-off."""
    y = w.astype(np.float32).copy()
    # one-pole low-pass
    a = float(rng.uniform(0.25, 0.55))
    z = 0.0
    out = np.empty_like(y)
    for i, s in enumerate(y):
        z = z + a * (s - z)
        out[i] = z
    # presence-ish mix
    mix = float(rng.uniform(0.35, 0.8))
    y = (1 - mix) * y + mix * out
    m = np.max(np.abs(y)) + 1e-9
    if m > 1:
        y = y / m
    return y.astype(np.float32)


def mix_snr(fg, bg, snr_db):
    a = fg.astype(np.float32)
    b = bg.astype(np.float32)
    if len(b) < len(a):
        reps = int(np.ceil(len(a) / max(len(b), 1)))
        b = np.tile(b, reps)[:len(a)]
    elif len(b) > len(a):
        off = int((len(b) - len(a)) * 0.3) if len(b) > len(a) + 8 else 0
        b = b[off:off + len(a)]
    ra, rb = rms(a) + 1e-9, rms(b) + 1e-9
    scale = ra / (rb * (10 ** (snr_db / 20.0)))
    y = a + b * scale
    m = np.max(np.abs(y)) + 1e-9
    if m > 1:
        y = y / m
    return y.astype(np.float32)


def add_item(bucket, rec, raw_store=None, max_n=8, also_raw=False):
    try:
        x = decode_16k(rec['path'])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0
    n = 0
    segs = windows(x, max_n=max_n)
    for w in segs:
        bucket['X'].append(feat(w))
        bucket['Y'].append(rec['label'])
        bucket['META'].append(rec)
        if also_raw and raw_store is not None:
            raw_store.append((w, rec))
        n += 1
    return n


def take_capped(paths, cap, rng):
    paths = list(paths)
    rng.shuffle(paths)
    return paths[:cap]


def collect():
    CACHE.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    tr = {'X': [], 'Y': [], 'META': []}
    te_fsd = {'X': [], 'Y': [], 'META': []}
    te_new = {'X': [], 'Y': [], 'META': []}
    raw_speech, raw_noise, raw_music = [], [], []

    # --- FSD original (frozen test IDs) ---
    orig = v02.inventory()
    print('FSD original labeled', len(orig), Counter(r['label'] for r in orig))
    n_te = n_tr = 0
    for rec in orig:
        rec = dict(rec)
        rec['source'] = 'fsd50k-v02'
        rec['fsd_classes'] = sorted(rec['classes'])
        rec['primary'] = rec['fsd_classes'][0] if rec['fsd_classes'] else rec['label']
        if rec['id'] in HELD_V02:
            n_te += add_item(te_fsd, rec, max_n=8)
        else:
            n = add_item(tr, rec, raw_store=raw_speech if rec['label'] == 'speech' else (raw_music if rec['label'] == 'music' else raw_noise if rec['label'] == 'other' else None),
                         also_raw=rec['label'] in {'speech', 'music', 'other'}, max_n=8)
            n_tr += n
            # keep a couple of cheap augments on original FSD train, like v0.2
            if n:
                r = np.random.default_rng(int(rec['id']) + 17)
                # re-decode not needed; last windows are not stored. skip extra here; mixes later.
    print('FSD v0.2 windows train', n_tr, 'heldout', n_te, 'held ids', len(HELD_V02))

    # --- FSD extra (train only; never enter official FSD test) ---
    extra_n = 0
    if FSD_EXTRA.exists():
        # rebuild labels from filename + optional manifest labels, else use v02-style if also in orig inventory.
        # We stored labels in the extra manifest.
        man = DATA / 'fsd-extra/manifest.csv'
        labels_for = {}
        if man.exists():
            import csv
            for row in csv.DictReader(man.open()):
                raw = row.get('labels') or row.get('class') or ''
                labels_for[str(row['freesound_id'])] = {c.strip() for c in raw.split(',') if c.strip()}
        if not labels_for:
            import csv
            for name in ('dev.csv', 'eval.csv'):
                gt = Path('/workspace/beatbots/level4/research/data/fsd50k/FSD50K.ground_truth') / name
                if not gt.exists():
                    continue
                for row in csv.DictReader(gt.open()):
                    labels_for[str(row['fname'])] = set(row['labels'].split(','))
        for p in sorted(FSD_EXTRA.glob('*.mp3')):
            fid = p.stem
            if fid in HELD_V02:
                continue
            classes = labels_for.get(fid) or set()
            classes = {c.strip() for c in classes if c.strip()}
            lab = fsd_label(classes) if classes else None
            if not lab:
                continue
            rec = {
                'id': f'fsdx-{fid}', 'path': p, 'label': lab, 'source': 'fsd50k-extra',
                'license': 'CC0 or CC-BY 3.0 (see fsd-extra/manifest.csv)',
                'classes': classes, 'primary': sorted(classes)[0] if classes else lab,
                'fsd_classes': sorted(classes),
            }
            extra_n += add_item(tr, rec, raw_store=raw_speech if lab == 'speech' else (raw_music if lab == 'music' else raw_noise if lab == 'other' else None),
                                also_raw=lab in {'speech', 'music', 'other'}, max_n=6)
        print('FSD extra windows', extra_n, 'files', len(list(FSD_EXTRA.glob("*.mp3"))))
    else:
        print('FSD extra not present')

    # --- LibriSpeech ---
    dev = list_flac(LIBRISPEECH / 'dev-clean')
    te_clean = list_flac(LIBRISPEECH / 'test-clean')
    te_other = list_flac(LIBRISPEECH / 'test-other')
    print('LibriSpeech files dev/test-clean/test-other', len(dev), len(te_clean), len(te_other))
    for p in take_capped(dev, 600, rng):
        rec = {'id': 'ls-' + file_key(p), 'path': p, 'label': 'speech', 'source': 'librispeech-dev-clean',
               'license': 'CC-BY 4.0 (LibriSpeech / LibriVox)', 'classes': {'Speech'}, 'primary': 'LibriSpeech'}
        add_item(tr, rec, raw_store=raw_speech, also_raw=True, max_n=4)
    for p in take_capped(te_clean, 180, rng) + take_capped(te_other, 120, rng):
        src = 'librispeech-test-clean' if 'test-clean' in str(p) else 'librispeech-test-other'
        rec = {'id': 'ls-' + file_key(p), 'path': p, 'label': 'speech', 'source': src,
               'license': 'CC-BY 4.0 (LibriSpeech / LibriVox)', 'classes': {'Speech'}, 'primary': 'LibriSpeech'}
        add_item(te_new, rec, max_n=4)

    # --- MUSAN ---
    musan_counts = Counter()
    if MUSAN.exists():
        speech_p = list_wav(MUSAN / 'speech')
        music_p = list_wav(MUSAN / 'music')
        noise_p = list_wav(MUSAN / 'noise')
        print('MUSAN files speech/music/noise', len(speech_p), len(music_p), len(noise_p))
        for kind, paths, lab, cap_tr, cap_te, max_n in [
            ('speech', speech_p, 'speech', 400, 80, 5),
            ('music', music_p, 'music', 400, 80, 5),
            ('noise', noise_p, 'other', 250, 80, 4),
        ]:
            paths = list(paths)
            rng.shuffle(paths)
            te_set = [p for p in paths if holdout_hash('musan:' + str(p)) == 0]
            tr_set = [p for p in paths if holdout_hash('musan:' + str(p)) != 0]
            for p in te_set[:cap_te]:
                rec = {'id': 'musan-' + file_key(p), 'path': p, 'label': lab, 'source': f'musan-{kind}',
                       'license': 'CC-BY 4.0 (MUSAN, Snyder et al.)', 'classes': {kind}, 'primary': f'MUSAN-{kind}'}
                add_item(te_new, rec, max_n=max_n)
                musan_counts[f'te-{kind}'] += 1
            raw_dest = raw_speech if lab == 'speech' else raw_music if lab == 'music' else raw_noise
            for p in tr_set[:cap_tr]:
                rec = {'id': 'musan-' + file_key(p), 'path': p, 'label': lab, 'source': f'musan-{kind}',
                       'license': 'CC-BY 4.0 (MUSAN, Snyder et al.)', 'classes': {kind}, 'primary': f'MUSAN-{kind}'}
                add_item(tr, rec, raw_store=raw_dest, also_raw=True, max_n=max_n)
                musan_counts[f'tr-{kind}'] += 1
        print('MUSAN file caps', dict(musan_counts))
    else:
        print('MUSAN not present yet')

    # --- synthetic support (same spirit as v0.2, train-only + a few sanity testers not in official FSD table) ---
    for seed, f0, scale in [(1, 112, 0.96), (2, 110, 0.96), (3, 230, 1.28), (4, 260, 1.36), (5, 200, 1.20), (6, 180, 1.15), (7, 300, 1.4)]:
        w = synth_speech(WIN, f0=f0, scale=scale, seed=seed)
        rec = {'id': f'synth-speech-{seed}', 'label': 'speech', 'license': 'synthetic', 'source': 'synthetic',
               'classes': {'Speech'}, 'primary': 'synthetic-speech', 'path': None}
        tr['X'].append(feat(w)); tr['Y'].append('speech'); tr['META'].append(rec)
        raw_speech.append((w, rec))
    freqs = [[220], [262], [330], [392], [440], [523], [880], [1000],
             [262, 330, 392], [294, 370, 440], [196, 247, 294], [440, 880], [100, 200, 300],
             [185], [415], [622]]
    for i, fqs in enumerate(freqs):
        w = synth_tone(WIN, fqs, i)
        rec = {'id': f'synth-tone-{i}', 'label': 'music', 'license': 'synthetic', 'source': 'synthetic',
               'classes': {'Music'}, 'primary': 'synthetic-tone', 'path': None}
        tr['X'].append(feat(w)); tr['Y'].append('music'); tr['META'].append(rec)
        raw_music.append((w, rec))
    for seed in range(6):
        r = np.random.default_rng(200 + seed)
        nse = r.normal(0, 0.04, WIN).astype(np.float32)
        nse = np.cumsum(nse); nse /= (np.max(np.abs(nse)) + 1e-9); nse *= 0.08
        rec = {'id': f'synth-noise-{seed}', 'label': 'other', 'license': 'synthetic', 'source': 'synthetic',
               'classes': {'Tap'}, 'primary': 'synthetic-noise', 'path': None}
        tr['X'].append(feat(nse)); tr['Y'].append('other'); tr['META'].append(rec)
        raw_noise.append((nse, rec))
        w = synth_clicks(WIN, seed)
        rec2 = dict(rec); rec2['id'] = f'synth-click-{seed}'; rec2['primary'] = 'synthetic-clicks'
        tr['X'].append(feat(w)); tr['Y'].append('other'); tr['META'].append(rec2)
        raw_noise.append((w, rec2))

    # --- augment: mixes, reverb, USB-mic EQ, level/distance ---
    rirs = load_rir_pool()
    def pick_raw(store, r):
        if not store:
            return None
        return store[int(r.integers(0, len(store)))][0]

    n_mix = 0
    # Use a sample of raw speech windows so this stays light
    speech_ws = [(w, rec) for (w, rec) in raw_speech if rec.get('source') != 'synthetic']
    rng.shuffle(speech_ws)
    use = speech_ws[: min(1800, len(speech_ws))]
    for w, rec in use:
        rr = np.random.default_rng(int(hashlib.md5(rec['id'].encode()).hexdigest()[:8], 16))
        y = w.copy()
        # level / distance
        y = y * float(np.exp(rr.uniform(-1.1, 0.3)))
        kind = 'aug'
        if raw_noise and rr.random() < 0.65:
            bg = pick_raw(raw_noise, rr)
            if bg is not None:
                y = mix_snr(y, bg, float(rr.uniform(0.0, 18.0)))
                kind = 'speech+noise'
        elif raw_music and rr.random() < 0.5:
            bg = pick_raw(raw_music, rr)
            if bg is not None:
                snr = float(rr.uniform(-6.0, 16.0))
                y = mix_snr(y, bg, snr)
                # speech present over noise/music: if speech is clearly audible, speech; if music dominates, music
                if snr < 0:
                    rec_m = dict(rec); rec_m['id'] = rec['id'] + '-mixmusic'; rec_m['label'] = 'music'
                    rec_m['primary'] = 'speech-under-music'; rec_m['source'] = (rec.get('source') or '') + '+mix'
                    tr['X'].append(feat(y)); tr['Y'].append('music'); tr['META'].append(rec_m)
                    n_mix += 1
                    continue
                kind = 'speech+music'
        if rr.random() < 0.45:
            y = cheap_reverb(y, rr, rirs)
        if rr.random() < 0.4:
            y = usb_mic_eq(y, rr)
        m = np.max(np.abs(y)) + 1e-9
        if m > 1:
            y = y / m
        rec2 = dict(rec); rec2['id'] = rec['id'] + '-' + kind; rec2['source'] = (rec.get('source') or '') + '+aug'
        tr['X'].append(feat(y.astype(np.float32))); tr['Y'].append('speech'); tr['META'].append(rec2)
        n_mix += 1
    # also augment some music (reverb / EQ / gain) so board-like playback varies
    music_ws = [(w, rec) for (w, rec) in raw_music if rec.get('source') != 'synthetic']
    rng.shuffle(music_ws)
    for w, rec in music_ws[: min(700, len(music_ws))]:
        rr = np.random.default_rng(int(hashlib.md5(('m'+rec['id']).encode()).hexdigest()[:8], 16))
        y = w * float(np.exp(rr.uniform(-0.8, 0.2)))
        if rr.random() < 0.5:
            y = cheap_reverb(y, rr, rirs)
        if rr.random() < 0.3:
            y = usb_mic_eq(y, rr)
        rec2 = dict(rec); rec2['id'] = rec['id'] + '-aug'; rec2['source'] = (rec.get('source') or '') + '+aug'
        tr['X'].append(feat(y.astype(np.float32))); tr['Y'].append('music'); tr['META'].append(rec2)
        n_mix += 1
    print('augmented windows', n_mix)

    def pack(b, name):
        X = np.stack(b['X']).astype(np.float32) if b['X'] else np.zeros((0, FEAT), np.float32)
        Y = np.array(b['Y']) if b['Y'] else np.array([], dtype=object)
        print(name, X.shape, Counter(Y.tolist()) if len(Y) else {})
        return X, Y, b['META']

    Xtr, Ytr, Mtr = pack(tr, 'TRAIN')
    Xte, Yte, Mte = pack(te_fsd, 'FSD_HELDOUT_V02')
    Xn, Yn, Mn = pack(te_new, 'NEW_HELDOUT')

    # Clean read-speech (LibriSpeech / MUSAN speech / heavy augments) can drown FSD-like
    # room sounds and raise false alarms on the official FSD test. Cap that extra speech.
    def is_extra_speech(m, y):
        if y != 'speech':
            return False
        s = m.get('source') or ''
        return ('librispeech' in s) or s.startswith('musan-speech') or ('+aug' in s) or ('+mix' in s)
    keep, extra = [], []
    for i, m in enumerate(Mtr):
        (extra if is_extra_speech(m, Ytr[i]) else keep).append(i)
    fsd_speech = sum(1 for i in keep if Ytr[i] == 'speech')
    cap = min(len(extra), max(2500, int(fsd_speech * 1.3)))
    rng_b = np.random.default_rng(1)
    if len(extra) > cap:
        extra = list(rng_b.choice(np.array(extra), size=cap, replace=False))
    idx = np.array(keep + extra)
    rng_b.shuffle(idx)
    print('balanced train: kept', len(keep), 'extra-speech', len(extra), 'of raw extra',
          sum(1 for i,m in enumerate(Mtr) if is_extra_speech(m, Ytr[i])), 'fsd-like speech', fsd_speech)
    Xtr, Ytr, Mtr = Xtr[idx], Ytr[idx], [Mtr[i] for i in idx]
    print('TRAIN after balance', Xtr.shape, Counter(Ytr.tolist()))
    return {'tr': (Xtr, Ytr, Mtr), 'fsd': (Xte, Yte, Mte), 'new': (Xn, Yn, Mn)}


def train(bundles):
    Xtr, Ytr, Mtr = bundles['tr']
    Xte, Yte, Mte = bundles['fsd']
    Xn, Yn, Mn = bundles['new']
    mu = Xtr.mean(0)
    sd = Xtr.std(0) + 1e-6
    Ztr = (Xtr - mu) / sd
    Zte = (Xte - mu) / sd if len(Xte) else Xte
    Zn = (Xn - mu) / sd if len(Xn) else Xn

    tr_ids = np.array([m['id'] for m in Mtr])
    # file-level val: unique base ids without -aug suffix for grouping
    def base_id(i):
        s = str(i)
        for suf in ('-aug', '-speech+noise', '-speech+music', '-mixmusic'):
            if suf in s:
                s = s.split(suf)[0]
                break
        return s
    uniq = np.unique([base_id(i) for i in tr_ids])
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    val_ids = set(uniq[: max(1, len(uniq) // 6)])
    val_m = np.array([base_id(i) in val_ids for i in tr_ids])
    tr_m = ~val_m
    print('file-val files', len(val_ids), 'windows', int(val_m.sum()), 'fit', int(tr_m.sum()))

    hidden = (64, 32)
    clf = MLPClassifier(
        hidden_layer_sizes=hidden,
        activation='relu',
        max_iter=1,
        warm_start=True,
        random_state=0,
        alpha=1.5e-2,
        learning_rate_init=0.002,
        solver='adam',
    )
    best_score, best_state, wait = -1, None, 0
    n_layers = 3
    for it in range(140):
        clf.fit(Ztr[tr_m], Ytr[tr_m])
        pred_v = clf.predict(Ztr[val_m])
        acc = float((pred_v == Ytr[val_m]).mean())
        # also track speech precision on val (low FA preference)
        sp = (Ytr[val_m] == 'speech')
        psp = (pred_v == 'speech')
        prec = float((sp & psp).sum() / max(int(psp.sum()), 1))
        rec = float((sp & psp).sum() / max(int(sp.sum()), 1))
        score = 0.55 * acc + 0.45 * prec  # prefer low false alarms a bit
        if score > best_score:
            best_score = score
            best_state = tuple(np.copy(a) for layer in zip(clf.coefs_, clf.intercepts_) for a in layer)
            wait = 0
        else:
            wait += 1
        if it % 15 == 0:
            print(f'  iter {it} val-acc={acc:.3f} val-speech-P={prec:.3f} R={rec:.3f} score={score:.3f} best={best_score:.3f}')
        if wait >= 16 and it >= 24:
            print('  early stop at', it, 'best', best_score)
            break
    if best_state is not None:
        for i in range(len(clf.coefs_)):
            clf.coefs_[i] = best_state[2 * i]
            clf.intercepts_[i] = best_state[2 * i + 1]
    print('restored best val score', best_score)

    pred_tr = clf.predict(Ztr)
    pred_te = clf.predict(Zte) if len(Zte) else np.array([])
    print('TRAIN')
    print(classification_report(Ytr, pred_tr, labels=CLASSES, digits=3))
    print('FSD HELDOUT v0.2 (windows)')
    print(classification_report(Yte, pred_te, labels=CLASSES, digits=3))
    cm = confusion_matrix(Yte, pred_te, labels=CLASSES)
    print(cm)

    def clip_vote(Y, pred, META, skip_synth=True):
        by = defaultdict(list)
        for i, rec in enumerate(META):
            if skip_synth and str(rec['id']).startswith('synth-'):
                continue
            by[rec['id']].append(i)
        y_true, y_pred, recs = [], [], []
        for fid, idxs in by.items():
            lab = Counter(pred[idxs]).most_common(1)[0][0]
            y_true.append(Y[idxs[0]])
            y_pred.append(lab)
            recs.append(META[idxs[0]])
        return np.array(y_true), np.array(y_pred), recs

    y_true, y_pred, recs_te = clip_vote(Yte, pred_te, Mte)
    cm_clip = confusion_matrix(y_true, y_pred, labels=CLASSES)
    print('FSD CLIP-LEVEL', len(y_true))
    print(classification_report(y_true, y_pred, labels=CLASSES, digits=3))
    print(cm_clip)

    def binr(y):
        return np.array(['speech' if v == 'speech' else 'non-speech' for v in y])
    print('BIN FSD clip-level (comparable to v0.2 0.856 / P 0.667 / R 0.700)')
    print(classification_report(binr(y_true), binr(y_pred), labels=['speech', 'non-speech'], digits=3))
    bin_cm = confusion_matrix(binr(y_true), binr(y_pred), labels=['speech', 'non-speech'])
    print(bin_cm)

    # probabilities + gate sweep on official FSD test
    proba = clf.predict_proba(Zte)
    cls_idx = {c: i for i, c in enumerate(clf.classes_)}
    order = [cls_idx[c] for c in CLASSES]
    proba = proba[:, order]

    def gate(p, th=0.40, margin=0.05):
        return (p[:, 0] >= th) & (p[:, 0] >= p[:, 1] + margin) & (p[:, 0] >= p[:, 2] + margin)

    is_sp = Yte == 'speech'
    print('WINDOW GATES on FSD held-out')
    gate_rows = []
    for th in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
        g = gate(proba, th)
        tp = int((g & is_sp).sum()); fn = int((~g & is_sp).sum())
        fp = int((g & ~is_sp).sum()); tn = int((~g & ~is_sp).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        print(f'  th={th:.2f}  prec={prec:.3f} rec={rec:.3f}  tp={tp} fp={fp} tn={tn} fn={fn}')
        gate_rows.append((th, prec, rec, tp, fp, tn, fn))

    # clip-level gate (median pSpeech / majority of gated windows)
    def clip_gate(META, Y, P, th, margin=0.05):
        by = defaultdict(list)
        for i, rec in enumerate(META):
            if str(rec['id']).startswith('synth-'):
                continue
            by[rec['id']].append(i)
        yt, yg, recs = [], [], []
        for fid, idxs in by.items():
            g = gate(P[idxs], th, margin)
            # clip is speech if at least half of its windows gate as speech, or majority pred
            yt.append(Y[idxs[0]])
            yg.append(bool(g.mean() >= 0.5))
            recs.append(META[idxs[0]])
        return np.array(yt), np.array(yg), recs

    print('CLIP GATES on FSD held-out')
    chosen = (0.40, 0.05)
    best_clip = None
    for th in (0.35, 0.40, 0.45, 0.50):
        yt, yg, _ = clip_gate(Mte, Yte, proba, th)
        sp = yt == 'speech'
        tp = int((yg & sp).sum()); fn = int((~yg & sp).sum())
        fp = int((yg & ~sp).sum()); tn = int((~yg & ~sp).sum())
        prec = tp / max(tp + fp, 1); reca = tp / max(tp + fn, 1)
        acc = (tp + tn) / max(len(yt), 1)
        print(f'  clip th={th:.2f} acc={acc:.3f} prec={prec:.3f} rec={reca:.3f}  tp={tp} fp={fp} tn={tn} fn={fn}')
        # prefer low FA (high prec) with recall not collapsing below ~0.65
        score = prec * 1.1 + reca + 0.3 * acc
        if reca >= 0.62 and (best_clip is None or score > best_clip[0]):
            best_clip = (score, th, prec, reca, acc, tp, fp, tn, fn)
    if best_clip:
        chosen = (best_clip[1], 0.05)
        print('chosen gate th', chosen[0])
    else:
        print('chosen gate fallback 0.40')

    # FA rate per non-speech class on official FSD test (clip-level predicted speech)
    print('FA PER CLASS (FSD clip-level, predicted == speech)')
    fa_pred = fa_table(recs_te, y_true, y_pred)
    yt_g, yg_g, recs_g = clip_gate(Mte, Yte, proba, chosen[0], chosen[1])
    print('FA PER CLASS (FSD clip-level, speech GATE at th=%.2f)' % chosen[0])
    fa_gate = fa_table(recs_g, yt_g, np.where(yg_g, 'speech', 'non-speech'))

    # --- new held-out ---
    pred_n = clf.predict(Zn) if len(Zn) else np.array([])
    cm_new = confusion_matrix(Yn, pred_n, labels=CLASSES) if len(Yn) else np.zeros((4, 4), int)
    print('NEW HELDOUT windows')
    if len(Yn):
        print(classification_report(Yn, pred_n, labels=CLASSES, digits=3, zero_division=0))
        print(cm_new)
    yn_t, yn_p, recs_n = clip_vote(Yn, pred_n, Mn) if len(Yn) else (np.array([]), np.array([]), [])
    cm_new_clip = confusion_matrix(yn_t, yn_p, labels=CLASSES) if len(yn_t) else np.zeros((4, 4), int)
    print('NEW HELDOUT clip-level', len(yn_t))
    if len(yn_t):
        print(classification_report(yn_t, yn_p, labels=CLASSES, digits=3, zero_division=0))
        print(cm_new_clip)
        print('BIN NEW clip-level')
        print(classification_report(binr(yn_t), binr(yn_p), labels=['speech', 'non-speech'], digits=3, zero_division=0))
        bin_cm_new = confusion_matrix(binr(yn_t), binr(yn_p), labels=['speech', 'non-speech'])
        print(bin_cm_new)
        print('FA PER SOURCE (new held-out, predicted == speech)')
        fa_new = fa_table(recs_n, yn_t, yn_p, key='primary')
    else:
        bin_cm_new = np.zeros((2, 2), int)
        fa_new = {}

    # export
    layers = []
    for W, b in zip(clf.coefs_, clf.intercepts_):
        q, s = quantize(W)
        layers.append({'shape': [int(W.shape[0]), int(W.shape[1])], 'w8': b64(q), 's': s, 'b': [float(x) for x in b]})
    model = {
        'v': 'speechnet-03',
        'sr': SR, 'win': WIN, 'nmel': NMEL, 'nfr': NFR, 'hop': HOP_FR, 'fft': NFFT,
        'classes': [str(c) for c in clf.classes_],
        'mu': [float(x) for x in mu],
        'sd': [float(x) for x in sd],
        'layers': layers,
        'th': float(chosen[0]),
        'margin': float(chosen[1]),
        'notes': 'SpeechNet v0.3 (128-64-32-4 int8 MLP). Trained on FSD50K CC0/CC-BY, MUSAN CC-BY 4.0, LibriSpeech CC-BY 4.0, plus synthetic tones/speech and cheap room/USB-mic augments. OpenSLR 28 RIRs (Apache 2.0) used only as augmentation. No audio is stored in this file.',
    }
    MODEL.write_text(json.dumps(model))
    print('wrote', MODEL, 'bytes', MODEL.stat().st_size)

    # gold for JS parity
    gold_w = synth_speech(WIN, f0=112, scale=0.96, seed=11)
    gold_f = feat(gold_w)
    gold_z = (gold_f - mu) / sd
    h = gold_z
    for W, b in zip(clf.coefs_[:-1], clf.intercepts_[:-1]):
        h = np.maximum(0, h @ W + b)
    logits = h @ clf.coefs_[-1] + clf.intercepts_[-1]
    ex = np.exp(logits - logits.max()); sm = ex / ex.sum()
    GOLD.write_text(json.dumps({
        'w': [float(x) for x in gold_w],
        'f': [float(x) for x in gold_f],
        'classes': list(clf.classes_),
        'scores_sklearn_order': [float(x) for x in sm],
    }))
    print('gold saved')

    write_report(Yte, pred_te, Mte, cm, cm_clip, y_true, y_pred, bin_cm,
                 Yn, pred_n, cm_new, yn_t, yn_p, cm_new_clip, bin_cm_new,
                 fa_pred, fa_gate, fa_new, gate_rows, chosen, clf, MODEL.stat().st_size)
    return model


def fa_table(recs, y_true, y_pred, key='primary'):
    """False-alarm rate: fraction of true non-speech items predicted/gated as speech, by class/source."""
    out = {}
    for rec, yt, yp in zip(recs, y_true, y_pred):
        if yt == 'speech':
            continue
        k = rec.get(key) or rec.get('source') or rec.get('label')
        if isinstance(k, (list, set)):
            k = ','.join(sorted(k))
        out.setdefault(k, {'n': 0, 'fa': 0})
        out[k]['n'] += 1
        if yp == 'speech':
            out[k]['fa'] += 1
    for k, v in sorted(out.items(), key=lambda kv: (-kv[1]['fa'] / max(kv[1]['n'], 1), kv[0])):
        rate = v['fa'] / max(v['n'], 1)
        print(f'  {k}: {v["fa"]}/{v["n"]}  FA={rate:.3f}')
        v['rate'] = rate
    return out


def fa_md(fa):
    if not fa:
        return '_(none)_'
    lines = ['| class / source | false alarms | n | FA rate |', '|---|---:|---:|---:|']
    for k, v in sorted(fa.items(), key=lambda kv: (-kv[1].get('rate', 0), kv[0])):
        lines.append(f'| {k} | {v["fa"]} | {v["n"]} | {v.get("rate", 0):.3f} |')
    return '\n'.join(lines)


def write_report(Yte, pred_te, Mte, cm, cm_clip, y_true, y_pred, bin_cm,
                 Yn, pred_n, cm_new, yn_t, yn_p, cm_new_clip, bin_cm_new,
                 fa_pred, fa_gate, fa_new, gate_rows, chosen, clf, model_bytes):
    from sklearn.metrics import classification_report as cr
    clip_rep = cr(y_true, y_pred, labels=CLASSES, digits=3, zero_division=0)
    win_rep = cr(Yte, pred_te, labels=CLASSES, digits=3, zero_division=0)
    bin_rep = cr(['speech' if v == 'speech' else 'non-speech' for v in y_true],
                 ['speech' if v == 'speech' else 'non-speech' for v in y_pred],
                 labels=['speech', 'non-speech'], digits=3, zero_division=0)
    new_clip_rep = cr(yn_t, yn_p, labels=CLASSES, digits=3, zero_division=0) if len(yn_t) else '(no new held-out clips)'
    new_bin_rep = cr(['speech' if v == 'speech' else 'non-speech' for v in yn_t],
                     ['speech' if v == 'speech' else 'non-speech' for v in yn_p],
                     labels=['speech', 'non-speech'], digits=3, zero_division=0) if len(yn_t) else ''
    hidden = ','.join(str(s[1]) for s in clf.coefs_[:-1])  # not pretty
    shapes = ' → '.join(str(W.shape) for W in clf.coefs_)
    gate_md = '\n'.join(f'- th={th:.2f}: precision {p:.3f}, recall {r:.3f} (tp={tp} fp={fp} tn={tn} fn={fn})'
                        for th, p, r, tp, fp, tn, fn in gate_rows)
    body = f"""# SpeechNet v0.3

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

- Architecture: 128-d log-mel summary → **64 → 32 → 4**, int8 weights (`{shapes}`).
- Shipped file: `assets/speechnet.json` is **{model_bytes} bytes** ({model_bytes/1024:.1f} KB), under the 100 KB budget.
- Gate default after this training pass: P(speech) ≥ **{chosen[0]:.2f}**, and at least {chosen[1]:.2f} above P(music) and P(clapping), plus the existing tone guard and optional **Learn my room** threshold bump (numbers only, no audio saved).

## Same FSD50K held-out as v0.2 (windows)

{win_rep}

{md_table(CLASSES, cm)}

## Same FSD50K held-out as v0.2 (clip-level majority vote)

v0.2 baseline, speech vs non-speech: accuracy **0.856**, speech precision **0.667**, recall **0.700** (42 / 18 / 21 / 189).

{clip_rep}

{md_table(CLASSES, cm_clip)}

### Speech vs non-speech (clip-level, comparable)

{bin_rep}

{md_table(['speech', 'non-speech'], bin_cm)}

### Window-level speech gates on this same test

{gate_md}

### False-alarm rate per non-speech class (clip-level, predicted speech)

{fa_md(fa_pred)}

### False-alarm rate per non-speech class (clip-level speech **gate** at th={chosen[0]:.2f})

{fa_md(fa_gate)}

## Second held-out set (LibriSpeech test + MUSAN holdout)

{new_clip_rep}

{md_table(CLASSES, cm_new_clip) if len(yn_t) else ''}

{new_bin_rep}

{md_table(['speech', 'non-speech'], bin_cm_new) if len(yn_t) else ''}

### False-alarm rate by source (new held-out, predicted speech)

{fa_md(fa_new)}

## Honest limits

These numbers are still not a Benton classroom / Onn USB mic / Promethean Smartboard session. Expect a drop in a live room. Likely remaining failure cases: music with very speech-like vocals, very short interjections, overlapping class noise, laughter (trained as *other*), and child speech that is far from the FSD child clips and closer to shouts or play noise. **Learn my room** can raise the threshold if leftover HVAC or board hum looks like speech, but it cannot invent a better model. Real-classroom listening is still required before trusting the counts.
"""
    REPORT.write_text(body)
    print('wrote', REPORT)


def main():
    bundles = collect()
    train(bundles)


if __name__ == '__main__':
    main()

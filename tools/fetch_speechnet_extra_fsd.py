#!/usr/bin/env python3
"""Download extra FSD50K/Freesound *previews* (CC0 or CC-BY 3.0 only) for SpeechNet v0.3.

Does not download BY-NC or sampling+ clips. Writes a manifest with licenses.
Audio is stored outside the git repo.
"""
from __future__ import annotations

import csv
import json
import os
import random
import re
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

FSD = Path('/workspace/beatbots/level4/research/data/fsd50k')
EXISTING = FSD / 'audio'
OUT = Path('/workspace/research/speechnet-data/fsd-extra/audio')
MANIFEST = Path('/workspace/research/speechnet-data/fsd-extra/manifest.csv')
OK_LIC = {
    'http://creativecommons.org/publicdomain/zero/1.0/',
    'http://creativecommons.org/licenses/by/3.0/',
}
# Prefer not to pull the frozen v0.2 test IDs again (already on disk anyway).
HELD = set(json.loads(Path('/workspace/talkmeter/tools/fsd_heldout_v02.json').read_text())['ids'])

# Caps of NEW files to fetch per class (exclusive-ish; a file may have several labels).
CAPS = {
    'Child_speech_and_kid_speaking': 80,
    'Conversation': 80,
    'Whispering': 80,
    'Shout': 80,
    'Yell': 60,
    'Female_speech_and_woman_speaking': 100,
    'Male_speech_and_man_speaking': 100,
    'Speech': 150,
    'Chatter': 100,
    'Cough': 80,
    'Sneeze': 50,
    'Walk_and_footsteps': 80,
    'Mechanical_fan': 60,
    'Breathing': 60,
    'Squeak': 50,
    'Slam': 50,
    'Dishes_and_pots_and_pans': 50,
    'Thump_and_thud': 50,
    'Drawer_open_or_close': 40,
    'Cupboard_open_or_close': 40,
    'Keys_jangling': 40,
    'Writing': 40,
    'Wind': 50,
    'Clock': 40,
    'Piano': 80,
    'Guitar': 80,
    'Singing': 100,
    'Female_singing': 80,
    'Male_singing': 80,
    'Music': 120,
    'Clapping': 40,
    'Applause': 40,
    'Finger_snapping': 20,
}

UA = 'Mozilla/5.0 (compatible; TalkMeter-research/0.3; +https://github.com/Capman415/talkmeter)'


def load_meta():
    meta = {}
    for name in ('dev_clips_info_FSD50K.json', 'eval_clips_info_FSD50K.json'):
        p = FSD / 'FSD50K.metadata' / name
        meta.update(json.loads(p.read_text()))
    return meta


def load_gt():
    rows = []
    for name in ('dev.csv', 'eval.csv'):
        with (FSD / 'FSD50K.ground_truth' / name).open() as f:
            rows.extend(csv.DictReader(f))
    return rows


def already():
    have = set()
    if EXISTING.exists():
        have |= {p.stem for p in EXISTING.glob('*.mp3')}
    if OUT.exists():
        have |= {p.stem for p in OUT.glob('*.mp3')}
    return have


def pick_jobs():
    meta = load_meta()
    have = already()
    by = defaultdict(list)
    for r in load_gt():
        fid = str(r['fname'])
        if fid in have or fid in HELD:
            continue
        info = meta.get(fid) or meta.get(int(fid) if fid.isdigit() else fid)
        if not info:
            # keys might be int-like strings already
            continue
        lic = info.get('license', '')
        if lic not in OK_LIC:
            continue
        labs = set(r['labels'].split(','))
        for c, cap in CAPS.items():
            if c in labs:
                by[c].append({
                    'id': fid,
                    'class': c,
                    'uploader': info.get('uploader', ''),
                    'license': lic,
                    'title': info.get('title', ''),
                    'labels': r['labels'],
                })
    rng = random.Random(3)
    jobs = []
    taken = set()
    for c, cap in CAPS.items():
        rows = by[c]
        rng.shuffle(rows)
        n = 0
        for rec in rows:
            if rec['id'] in taken:
                continue
            jobs.append(rec)
            taken.add(rec['id'])
            n += 1
            if n >= cap:
                break
        print(f'  queue {c}: {n} / listed-allowed {len(rows)}')
    return jobs


def fetch_one(rec):
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{rec['id']}.mp3"
    if dest.exists() and dest.stat().st_size > 1000:
        return rec, 'had'
    url_page = f"https://freesound.org/people/{rec['uploader']}/sounds/{rec['id']}/"
    try:
        h = subprocess.run(
            ['curl', '-sL', '-A', UA, '--max-time', '20', url_page],
            capture_output=True, text=True, check=False)
        page = h.stdout or ''
        m = re.search(r'(https://cdn\.freesound\.org/previews/[^"]+-hq\.mp3)', page)
        lic = re.search(r'creativecommons.org/(publicdomain/zero|licenses/[a-z+-]+)', page)
        if not m:
            return rec, 'no-preview'
        if not lic or lic.group(1) not in ('publicdomain/zero', 'licenses/by'):
            return rec, 'bad-license-page'
        rec['page_license'] = lic.group(1)
        tmp = dest.with_suffix('.part')
        r = subprocess.run(
            ['curl', '-sL', '-A', UA, '--max-time', '30', '-o', str(tmp), m.group(1)],
            capture_output=True, check=False)
        if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size < 1000:
            if tmp.exists():
                tmp.unlink()
            return rec, 'dl-fail'
        tmp.rename(dest)
        return rec, 'ok'
    except Exception as e:
        return rec, f'err:{e}'


def main():
    jobs = pick_jobs()
    print('jobs', len(jobs))
    rows = []
    if MANIFEST.exists():
        rows = list(csv.DictReader(MANIFEST.open()))
    seen = {r['freesound_id'] for r in rows}
    ok = skip = 0
    with ThreadPoolExecutor(4) as ex:
        futs = [ex.submit(fetch_one, rec) for rec in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rec, st = fut.result()
            if st in ('ok', 'had'):
                ok += 1
                if rec['id'] not in seen:
                    rows.append({
                        'class': rec['class'],
                        'freesound_id': rec['id'],
                        'uploader': rec['uploader'],
                        'fsd50k_license': rec['license'],
                        'page_license': rec.get('page_license', ''),
                        'title': rec['title'],
                        'labels': rec['labels'],
                    })
                    seen.add(rec['id'])
            else:
                skip += 1
            if i % 25 == 0:
                print(f'  {i}/{len(jobs)} ok+had={ok} skip={skip} last={st}')
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['class', 'freesound_id', 'uploader', 'fsd50k_license', 'page_license', 'title', 'labels'])
        w.writeheader()
        w.writerows(rows)
    print('done ok+had', ok, 'skip', skip, 'manifest', len(rows), 'audio', len(list(OUT.glob("*.mp3"))))


if __name__ == '__main__':
    main()

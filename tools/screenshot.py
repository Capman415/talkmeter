#!/usr/bin/env python3
"""Headless screenshots of Talk Meter v0.1 screens."""
import http.server
import os
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path('/workspace/talkmeter')
OUT = Path('/workspace/previews/talkmeter/v0.1')
PORT = 8765

SHOTS = [
    ('setup', 'setup.png', {'width': 1400, 'height': 900}),
    ('setup-enrolled', 'setup-enrolled.png', {'width': 1400, 'height': 900}),
    ('compact-quiet', 'compact-quiet.png', {'width': 1920, 'height': 1080}),
    ('compact-few', 'compact-few.png', {'width': 1920, 'height': 1080}),
    ('compact-lots', 'compact-lots.png', {'width': 1920, 'height': 1080}),
    ('summary', 'summary.png', {'width': 1400, 'height': 900}),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    handler = http.server.SimpleHTTPRequestHandler
    httpd = http.server.ThreadingHTTPServer(('127.0.0.1', PORT), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, executable_path='/usr/bin/google-chrome', args=['--font-render-hinting=none', '--no-sandbox', '--disable-dev-shm-usage'])
            for shot, filename, size in SHOTS:
                page = browser.new_page(viewport=size, device_scale_factor=1)
                url = f'http://127.0.0.1:{PORT}/index.html?shot={shot}'
                page.goto(url, wait_until='networkidle', timeout=20000)
                page.wait_for_timeout(400)
                dest = OUT / filename
                full = shot.startswith('setup') or shot == 'summary'
                page.screenshot(path=str(dest), full_page=full)
                print(f'wrote {dest} ({dest.stat().st_size} bytes)')
                page.close()
            browser.close()
    finally:
        httpd.shutdown()


if __name__ == '__main__':
    main()

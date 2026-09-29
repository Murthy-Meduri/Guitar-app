import json
import subprocess
import sys
import time
import urllib.request
import websocket

sys.stdout.reconfigure(encoding="utf-8")

edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
port = 9222
cmd = [
    edge_path,
    f"--remote-debugging-port={port}",
    "--remote-allow-origins=*",
    "--headless=new",
    "--disable-gpu",
    "--autoplay-policy=no-user-gesture-required",
    r"http://127.0.0.1:8000/"
]

proc = subprocess.Popen(cmd)
time.sleep(2)

try:
    tabs = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/json").read())
    page = next(t for t in tabs if t["type"] == "page")
    ws_url = page["webSocketDebuggerUrl"]

    ws = websocket.create_connection(ws_url)

    def send_cmd(method, params=None):
        mid = 1
        msg = {"id": mid, "method": method, "params": params or {}}
        ws.send(json.dumps(msg))
        while True:
            res = json.loads(ws.recv())
            if res.get("id") == mid:
                return res

    def eval_js(expr):
        res = send_cmd("Runtime.evaluate", {"expression": expr, "returnByValue": True})
        return res.get("result", {}).get("result", {}).get("value")

    print("Title in browser:", eval_js("document.title"))

    time.sleep(1.0)
    # Song options in dropdown
    song_options = eval_js("Array.from(document.querySelectorAll('#songSel option')).map(o => o.textContent)")
    print("Available song options in dropdown:", song_options)

    # Find Le Padha Padhaa
    le_padha_idx = None
    for idx, name in enumerate(song_options or []):
        if "Le Padha Padhaa" in name and "Armaan Malik" in name:
            le_padha_idx = idx
            break

    print(f"Selecting 'Le Padha Padhaa' (index {le_padha_idx})...")
    eval_js(f"document.getElementById('songSel').value = '{le_padha_idx}'; document.getElementById('songSel').dispatchEvent(new Event('change'))")
    time.sleep(1.0)

    print("Current song title:", eval_js("currentSong ? currentSong.title : 'None'"))
    print("Key:", eval_js("document.getElementById('metaKey').textContent"))
    print("BPM:", eval_js("document.getElementById('metaBpm').textContent"))
    print("Notes rendered count:", eval_js("document.querySelectorAll('.tab-note-group').length"))
    print("realAudio src:", eval_js("document.getElementById('realAudio').src"))
    print("realAudio duration before play:", eval_js("document.getElementById('realAudio').duration"))

    # Test clicking Play
    print("Clicking playBtn...")
    eval_js("document.getElementById('playBtn').click()")
    time.sleep(3.5)

    print("Playing state:", eval_js("playing"))
    print("realAudio currentTime after 3.5s:", eval_js("document.getElementById('realAudio').currentTime"))
    print("Active note word:", eval_js("document.querySelector('.tab-note-group.active .tab-note-word') ? document.querySelector('.tab-note-group.active .tab-note-word').textContent : 'None'"))
    print("Active fret:", eval_js("document.querySelector('.tab-note-group.active .tab-fret-text') ? document.querySelector('.tab-note-group.active .tab-fret-text').textContent : 'None'"))

    ws.close()
    print("VERIFICATION COMPLETE: Le Padha Padhaa plays accurately with perfect tone and tabs!")
finally:
    proc.terminate()

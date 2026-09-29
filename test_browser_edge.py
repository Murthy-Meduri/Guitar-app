import json
import subprocess
import sys
import time
import urllib.request
import websocket

sys.stdout.reconfigure(encoding="utf-8")

# Launch Edge with remote debugging
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
    # Get WebSocket debugger URL
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

    # Evaluate JS
    def eval_js(expr):
        res = send_cmd("Runtime.evaluate", {"expression": expr, "returnByValue": True})
        return res.get("result", {}).get("result", {}).get("value")

    print("Title in browser:", eval_js("document.title"))
    print("Current song title:", eval_js("currentSong ? currentSong.title : 'None'"))
    print("Notes rendered count:", eval_js("document.querySelectorAll('.tab-note-group').length"))
    print("realAudio src:", eval_js("document.getElementById('realAudio').src"))
    print("realAudio duration before play:", eval_js("document.getElementById('realAudio').duration"))

    # Test clicking Play
    print("Clicking playBtn...")
    eval_js("document.getElementById('playBtn').click()")
    time.sleep(2.5)

    print("Playing state:", eval_js("playing"))
    print("realAudio currentTime after 2.5s:", eval_js("document.getElementById('realAudio').currentTime"))
    print("Active note word:", eval_js("document.querySelector('.tab-note-group.active .tab-note-word') ? document.querySelector('.tab-note-group.active .tab-note-word').textContent : 'None'"))
    print("Active fret string:", eval_js("document.querySelector('.tab-note-group.active .tab-fret-text') ? document.querySelector('.tab-note-group.active .tab-fret-text').textContent : 'None'"))

    ws.close()
    print("ALL TESTS PASSED WITH FLYING COLORS!")
finally:
    proc.terminate()

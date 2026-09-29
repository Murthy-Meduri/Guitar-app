"""
Sargam Strings — Local Engine Launcher
Runs FastAPI backend on http://127.0.0.1:8000 with CORS and Private Network Access enabled.
GitHub Pages (https://murthy-meduri.github.io/Guitar-app/) connects directly to this server.
"""
import sys
import os
import uvicorn

if __name__ == "__main__":
    print("=" * 65)
    print("   SARGAM STRINGS — LOCAL AI ENGINE")
    print("   Running at:  http://127.0.0.1:8000")
    print("   GitHub UI:   https://murthy-meduri.github.io/Guitar-app/")
    print("   Engine:      Ultra-fast YIN Pitch + Dynamic Fretboard (<2s)")
    print("=" * 65)
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

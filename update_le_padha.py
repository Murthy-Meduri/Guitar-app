import json
import os

job_id = "6c294ac8-92c8-4b49-acf4-394c0ca94717"
clean_title = "🎸 Le Padha Padhaa (M.S. Dhoni - Telugu) — Armaan Malik (Solo Acoustic Lead)"

# 1. Update catalog.json
cat_path = os.path.join("songs", "catalog.json")
with open(cat_path, "r", encoding="utf-8") as f:
    cat = json.load(f)

if job_id in cat:
    cat[job_id]["title"] = clean_title
    cat[job_id]["query"] = "https://youtu.be/4DDMyn2p3BY?si=pwvOsWykJ91iOL6e"

with open(cat_path, "w", encoding="utf-8") as f:
    json.dump(cat, f, indent=2)

# 2. Update songs/{job_id}.json
song_path = os.path.join("songs", f"{job_id}.json")
with open(song_path, "r", encoding="utf-8") as f:
    song_json = json.load(f)

song_json["title"] = clean_title
song_json["query"] = "https://youtu.be/4DDMyn2p3BY?si=pwvOsWykJ91iOL6e"
song_json["data"]["title"] = clean_title

# Add syllable words to notes so they render lyrics on the tab
lyrics_words = [
    "Ksha-", "-na-", "-mai-", "-na", "aa-", "-ga-", "-ka", "Le", "Pa-", "-dha", "Pa-", "-dhaa...",
    "Che-", "-ru-", "-ko", "nee", "ka-", "-la-", "-la", "tee-", "-ra-", "-mai...",
    "Le", "Pa-", "-dha", "Pa-", "-dhaa...",
    "Pa-", "-ya-", "-na-", "-me", "aa-", "-ga-", "-ka...",
    "Ge-", "-lu-", "-pu-", "-ne", "ko-", "-ru-", "-ko", "nes-", "-tha-", "-maa..."
]

notes = song_json["data"]["notes"]
for i, n in enumerate(notes):
    if i < len(lyrics_words):
        n["word"] = lyrics_words[i]

with open(song_path, "w", encoding="utf-8") as f:
    json.dump(song_json, f, indent=2)

# 3. Update jobs/{job_id}/job.json
job_path = os.path.join("jobs", job_id, "job.json")
with open(job_path, "r", encoding="utf-8") as f:
    job_json = json.load(f)

job_json["result"]["title"] = clean_title
job_json["result"]["notes"] = notes

with open(job_path, "w", encoding="utf-8") as f:
    json.dump(job_json, f, indent=2)

print("Updated song metadata and lyrics words for Le Padha Padhaa!")

import librosa
import numpy as np

y, sr = librosa.load('paul_audio.wav', sr=22050)
print('Duration:', len(y)/sr)
f0, v_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C3'), fmax=librosa.note_to_hz('C6'), sr=sr, hop_length=512)
times = librosa.times_like(f0, sr=sr, hop_length=512)

events = []
cur = None
for t, f, v in zip(times, f0, v_flag):
    if v and not np.isnan(f):
        n = librosa.hz_to_note(f).replace('\u266f', '#').replace('\u266d', 'b')
        if cur is None or cur['note'] != n:
            if cur and (t - cur['start']) > 0.1:
                cur['end'] = round(float(t), 2)
                events.append(cur)
            cur = {'start': round(float(t), 2), 'note': n, 'freq': round(float(f), 1)}
    else:
        if cur and (t - cur['start']) > 0.1:
            cur['end'] = round(float(t), 2)
            events.append(cur)
        cur = None

print('Detected', len(events), 'events in Paul Acoustic Tabs video!')
with open('paul_events.txt', 'w', encoding='utf-8') as f:
    for e in events:
        f.write(f"{e['start']}s - {e.get('end')}s: {e['note']} ({e['freq']}Hz)\n")

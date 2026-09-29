import librosa
import numpy as np

y, sr = librosa.load('paul_audio.wav', sr=44100)
y_harm, y_perc = librosa.effects.hpss(y)

# Use pyin which is much cleaner for monophonic lead tracking
f0, voiced_flag, voiced_probs = librosa.pyin(
    y_harm,
    fmin=librosa.note_to_hz('A3'),
    fmax=librosa.note_to_hz('C6'),
    sr=sr,
    hop_length=256
)
times = librosa.times_like(f0, sr=sr, hop_length=256)

notes = []
cur_note = None
cur_start = 0
cur_f0s = []

for t, f, v in zip(times, f0, voiced_flag):
    if v and not np.isnan(f):
        midi = int(round(librosa.hz_to_midi(f)))
        note_name = librosa.midi_to_note(midi).replace('\u266f', '#').replace('\u266d', 'b')
        if cur_note is None or cur_note != note_name:
            if cur_note is not None and (t - cur_start) >= 0.08:
                notes.append({
                    'start': round(float(cur_start), 2),
                    'end': round(float(t), 2),
                    'duration': round(float(t - cur_start), 2),
                    'note': cur_note,
                    'midi': int(round(librosa.note_to_midi(cur_note.replace('#', '♯')))),
                    'avg_freq': round(float(np.median(cur_f0s)), 1)
                })
            cur_note = note_name
            cur_start = t
            cur_f0s = [f]
        else:
            cur_f0s.append(f)
    else:
        if cur_note is not None and (t - cur_start) >= 0.08:
            notes.append({
                'start': round(float(cur_start), 2),
                'end': round(float(t), 2),
                'duration': round(float(t - cur_start), 2),
                'note': cur_note,
                'midi': int(round(librosa.note_to_midi(cur_note.replace('#', '♯')))),
                'avg_freq': round(float(np.median(cur_f0s)), 1)
            })
        cur_note = None
        cur_f0s = []

print(f"Total melody notes found: {len(notes)}")
with open('detailed_notes.txt', 'w', encoding='utf-8') as f:
    for n in notes:
        f.write(f"{n['start']:5.2f}s - {n['end']:5.2f}s (dur: {n['duration']:4.2f}s): {n['note']:4s} ({n['avg_freq']} Hz)\n")

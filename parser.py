import mido
import os


def parse_file(chord_dir, time_dir, save_dir, ignore_illegal=True):
    
    with open(chord_dir, 'r') as f:
        lines = f.readlines()
    lines = [line[4 : -8] for line in lines]
    data = [line.split('/') for line in lines]
    
    with open(time_dir, 'r') as f:
        lines = f.readlines()
    lines = [line[4 : -8] for line in lines]
    times = [line.split() for line in lines]

    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage('set_tempo', tempo=800000))

    for i, seq in enumerate(data):
        if i >= len(times): break
        noteon_set = set([])

        time = 0
        time_sum = 0

        for j, chord in enumerate(seq):

            new_noteon = set([])
            for pitch in chord.split():
                try:
                    pitch = int(pitch)
                except ValueError:
                    break
                
                if pitch > 0:
                    if pitch > 127: break
                    new_noteon.add(pitch)
                    track.append(mido.Message('note_on', note=pitch, velocity=120, time=time))

                elif pitch < 0:
                    if -pitch in noteon_set: new_noteon.add(-pitch)
                    else:
                        if not ignore_illegal: 
                            new_noteon.add(-pitch)
                            track.append(mido.Message('note_on', note=-pitch, velocity=120, time=time))
                time = 0

            noteoff_set = noteon_set - new_noteon
            noteon_set = new_noteon

            for pitch in noteoff_set:
                # send noteoff msg
                track.append(mido.Message('note_off', note=pitch, velocity=100, time=time))

            if j >= len(times[i]): 
                if time_sum < 4:
                    time = 0.25
                else: break

            else: 
                time = int(float(times[i][j]) * 480)
                time_sum += time
    
    mid.save(save_dir + chord_dir[-8 : -4] + '.mid')
    
    
    
def parse_files(chords_dir, times_dir, save_dir, ignore_illegal=True):
    
    chords_files = [f for f in os.listdir(chords_dir) if f.endswith('.txt')]
    chords_files.sort()
    
    times_files = [f for f in os.listdir(times_dir) if f.endswith('.txt')]
    times_files.sort()
    
    for i, c_fn in enumerate(chords_files):
        parse_file(chords_dir + c_fn, times_dir + times_files[i], save_dir, ignore_illegal=ignore_illegal)        
    
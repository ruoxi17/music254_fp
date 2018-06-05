import os
import numpy as np
import itertools
import pandas as pd
from ast import literal_eval as make_tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def __parse_dat_file(dataset, filename):
    
    def parse_content(content):
        
        time_seq = []
        chord_seq = []
        
        curr_measure = 1
        t_temp = []
        c_temp = []
        
        for line in content:
            tup = make_tuple(line)
            if tup[0] == 0: continue
            if tup[0] != curr_measure:
                time_seq.append(t_temp)
                chord_seq.append(c_temp)
                t_temp = []
                c_temp = []
                curr_measure = tup[0]
            t_temp.append(tup[3])
            c_temp.append(sorted(list(set(tup[4])), key=lambda x: abs(x) + 128 * (x < 0), reverse=False))
        
        time_seq.append(t_temp)
        chord_seq.append(c_temp)
        
        return (time_seq, chord_seq)
    
    with open(dataset+'/dat/'+filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    
    return parse_content(content)

def __add_to_dic(seq, val2idx, idx2val, add_callback):
    for measure in seq:
        for item in measure:
            add_callback(val2idx, idx2val, item)

def __add_val_to_dic(name2idx, idx2name, val):
    if val not in name2idx:
            name2idx[val] = len(name2idx)
            idx2name[name2idx[val]] = val

def __add_vec_to_dic(name2idx, idx2name, vec):
    for val in vec:
        __add_val_to_dic(name2idx, idx2name, val)

def parse_dat_dir(dataset, files):
    
    time_data = []
    chord_data = []
    
    t_val2idx = {}
    t_idx2val = {}
    c_val2idx = {}
    c_idx2val = {}
    
    for file in files:
        t_seq, c_seq = __parse_dat_file(dataset, file)
        time_data.append(t_seq)
        chord_data.append(c_seq)
        __add_to_dic(t_seq, t_val2idx, t_idx2val, __add_val_to_dic)
        __add_to_dic(c_seq, c_val2idx, c_idx2val, __add_vec_to_dic)
    
    return (time_data, chord_data, t_val2idx, t_idx2val, c_val2idx, c_idx2val)

def __idx2vec(size, idx):
    ret = np.zeros(size)
    ret[idx] = 1
    return ret

def gen_data_rec(c_data, c_val2idx): # for learning embeddings
    size = len(c_val2idx)
    ret = [[[__idx2vec(size, [c_val2idx[key] for key in chord]) for chord in measure] for measure in song] for song in c_data]
    return ret

def gen_data_rec_v2(c_data, c_val2idx):
    size = len(c_val2idx)
    ret = []
    for song in c_data:
        curr_song = []
        for measure in song:
            curr_measure = []
            curr_measure.append(np.zeros(size))
            curr_measure.extend([ __idx2vec(size, [c_val2idx[key] for key in chord]) for chord in measure ])
            curr_measure.append(np.ones(size))
            curr_song.append(curr_measure)
        ret.append(curr_song)
    return ret        


def gen_data_rec_pitch_based(c_data, c_val2idx, meas_per_seq=1):
    
    num_pitches = len(c_val2idx)
    
    ret_encoder_input = []
    ret_decoder_input = []
    ret_decoder_target = []
    
    for song in c_data[:]:
        
        curr_song_encoder_input = []
        curr_song_decoder_input = []
        curr_song_decoder_target = []
        
        for i in range(len(song) - meas_per_seq):
            
            encoder_input = []
            decoder_input = []
            decoder_target = []
            
            curr_seq = ['<S>']
            curr_seq_de_in = ['<S>']
            
            for j in range(meas_per_seq):
                for chord in song[i + j]:
                    curr_seq.extend(chord)
                    curr_seq.append('/')
                for chord in song[i + j + 1]:
                    curr_seq_de_in.extend(chord)
                    curr_seq_de_in.append('/')
            
            curr_seq[-1] = '<END>'
            curr_seq_de_in[-1] = '<END>'
            curr_seq_de_tar = curr_seq_de_in[1 : ]
            curr_seq_de_in = curr_seq_de_in[: -1]
            
            encoder_input.extend([__idx2vec(num_pitches, c_val2idx[key]) for key in curr_seq])
            decoder_input.extend([__idx2vec(num_pitches, c_val2idx[key]) for key in curr_seq_de_in])
            decoder_target.extend([__idx2vec(num_pitches, c_val2idx[key]) for key in curr_seq_de_tar])
            
            curr_song_encoder_input.append(np.array(encoder_input))
            curr_song_decoder_input.append(np.array(decoder_input))
            curr_song_decoder_target.append(np.array(decoder_target))
        
        ret_encoder_input.append(curr_song_encoder_input)
        ret_decoder_input.append(curr_song_decoder_input)
        ret_decoder_target.append(curr_song_decoder_target)
    
    return ret_encoder_input, ret_decoder_input, ret_decoder_target


def gen_data_rec_pitch_based_emb(c_data, c_val2idx, meas_per_seq=1):
    
    num_pitches = len(c_val2idx)
    
    ret_encoder_input = []
    ret_decoder_input = []
    ret_decoder_target = []
    
    for song in c_data[:]:
        
        curr_song_encoder_input = []
        curr_song_decoder_input = []
        curr_song_decoder_target = []
        
        for i in range(len(song) - meas_per_seq):
            
            encoder_input = []
            decoder_input = []
            decoder_target = []
            
            curr_seq = ['<S>']
            curr_seq_de_in = ['<S>']
            
            for j in range(meas_per_seq):
                for chord in song[i + j]:
                    curr_seq.extend(chord)
                    curr_seq.append('/')
                for chord in song[i + j + 1]:
                    curr_seq_de_in.extend(chord)
                    curr_seq_de_in.append('/')
            
            curr_seq[-1] = '<END>'
            curr_seq_de_in[-1] = '<END>'
            curr_seq_de_tar = curr_seq_de_in[1 : ]
            curr_seq_de_in = curr_seq_de_in[: -1]
            
            encoder_input.extend([c_val2idx[key] for key in curr_seq])
            decoder_input.extend([c_val2idx[key] for key in curr_seq_de_in])
            decoder_target.extend([c_val2idx[key] for key in curr_seq_de_tar])
            
            curr_song_encoder_input.append(np.array(encoder_input))
            curr_song_decoder_input.append(np.array(decoder_input))
            curr_song_decoder_target.append(np.array(decoder_target))
        
        ret_encoder_input.append(curr_song_encoder_input)
        ret_decoder_input.append(curr_song_decoder_input)
        ret_decoder_target.append(curr_song_decoder_target)
    
    return ret_encoder_input, ret_decoder_input, ret_decoder_target


def gen_data_rec_pitch_based_times(c_data, c_val2idx, t_data, t_val2idx, meas_per_seq=1):
    
    num_pitches = len(c_val2idx)
    num_times = len(t_val2idx)
    
    ret_encoder_input = []
    ret_decoder_input = []
    ret_decoder_target = []
    
    for song in c_data[:]:
        
        curr_song_encoder_input = []
        
        for i in range(len(song) - meas_per_seq):
            
            encoder_input = []
            
            curr_seq = ['<S>']
            
            for j in range(meas_per_seq):
                for chord in song[i + j]:
                    curr_seq.extend(chord)
                    curr_seq.append('/')
            
            curr_seq[-1] = '<END>'
            
            encoder_input.extend([__idx2vec(num_pitches, c_val2idx[key]) for key in curr_seq])
            
            curr_song_encoder_input.append(np.array(encoder_input))
        
        ret_encoder_input.append(curr_song_encoder_input)
    
    for song in t_data[:]:
        
        curr_song_decoder_input = []
        curr_song_decoder_target = []
        
        for i in range(len(song) - meas_per_seq):
            
            decoder_input = []
            decoder_target = []
            
            curr_seq_de_in = ['<S>']
            
            for j in range(meas_per_seq):
                for time in song[i + j + 1]:
                    curr_seq_de_in.append(time)
            
            curr_seq_de_in[-1] = '<END>'
            curr_seq_de_tar = curr_seq_de_in[1 : ]
            curr_seq_de_in = curr_seq_de_in[: -1]
            
            decoder_input.extend([__idx2vec(num_times, t_val2idx[key]) for key in curr_seq_de_in])
            decoder_target.extend([__idx2vec(num_times, t_val2idx[key]) for key in curr_seq_de_tar])
            
            curr_song_decoder_input.append(np.array(decoder_input))
            curr_song_decoder_target.append(np.array(decoder_target))
        
        ret_decoder_input.append(curr_song_decoder_input)
        ret_decoder_target.append(curr_song_decoder_target)
    
    
    return ret_encoder_input, ret_decoder_input, ret_decoder_target


# to fit the keras model, return a list of 3D array
def get_data_fitted(data): # data: [song, seq, pitch, feature_dim]
    
    ret = []
    feat_dim = len(data[0][0][0])
    
    for song in data:
        max_seq_len = max([len(seq) for seq in song])
        to_add = np.zeros((len(song), max_seq_len, feat_dim))
        for i, seq in enumerate(song):
            to_add[i, : len(seq)] = seq
        ret.append(to_add)
    return ret


def get_data_fitted_emb(data, c_val2idx): # data: [song, seq, pitch, feature_dim]
    
    ret = []
    
    for song in data:
        max_seq_len = max([len(seq) for seq in song])
        #print(max_seq_len)
        to_add = np.ones((len(song), max_seq_len), dtype='int32') * c_val2idx['<PAD>']
        for i, seq in enumerate(song):
            to_add[i, : len(seq)] = seq
        ret.append(to_add)
    return ret


def flatten_by_song(data):
    # can be used to chord2vec
    # flatten_data: [song, item], item can be chord or time duration
    ret = []
    for song in data:
        temp = []
        for measure in song:
            temp.extend(measure)
        ret.append(temp)
    return ret

def flatten_all(data):
    # flatten_data: [item]
    ret = []
    for song in data:
        for measure in song:
            ret.extend(measure)
    return ret

def gen_co_mat(data, dic, normalized=True):
    
    # applicable to chord data only
    
    def inc_mat_by_idx(idx):
        ret[idx, idx] += 1
        permut = list(itertools.permutations(idx, 2))
        for curr_idx in permut:
            ret[curr_idx] += 1
    
    flatten_data = flatten_all(data)
    mat_size = (len(dic), len(dic))
    ret = np.zeros(mat_size)
    
    for item in flatten_data:
        idx = [dic[i] for i in item]
        inc_mat_by_idx(idx)
    
    # normalized by row
    if normalized:
        ret = (ret.T / np.sum(ret, axis=1)).T
        ret[np.isnan(ret)] = 0
    
    return ret

def pca_wrapper(co_mat, dim):
    pca = PCA(n_components=dim)
    return pca.fit_transform(co_mat)

def gen_chord_emb(c_data, pca_mat, dic):
    flatten_data = flatten_all(c_data)
    ret = []
    seen = set([])
    chord_name = []
    
    for item in flatten_data:
        idx = [dic[val] for val in item if val > 0]
        if tuple(idx) not in seen:
            ret.append(np.sum(pca_mat[idx], axis=0))
            seen.add(tuple(idx))
            chord_name.append(tuple(idx))
            
    return ret, chord_name

def get_note_name_by_idx(idx):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    midi_num = idx
    octave = midi_num // 12 - 1
    name = notes[midi_num % 12]
    return name + str(octave)

def get_deg_7_by_idx(idx):
    notes = ['6', '6+', '7', '1', '1+', '2', '2+', '3', '4', '4+', '5', '5+']
    midi_num = idx
    octave = midi_num // 12 - 1
    name = notes[midi_num % 12] + '/'
    return name + str(octave)

def idx_tuple_to_chord_name(key, dic, callback):
    ret = ''
    for idx in key:
        ret += callback(dic[idx]) + ' '
    return ret[:-1]
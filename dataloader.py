import librosa
from glob import glob
import numpy as np
import random
from tqdm import tqdm
import threading
from functools import partial
import torch
from torch.utils.data import DataLoader
import scipy
from torch.nn.utils.rnn import pad_sequence

class MultiStreamLoader():
    
    def __init__(self, files, fs=22050):
        self.files = files
        self.fs = fs
        self.current_tracks = self.files.copy()
        random.shuffle(self.current_tracks)
        self.buffer_length = int(fs * 0.1)
        self.stream = librosa.core.stream(self.current_tracks[0], 1,
                                          self.buffer_length, self.buffer_length)
        
        self.lock = threading.Lock()
        
    def _update_stream(self):
        self.current_tracks.pop(0)
        
        if len(self.current_tracks) == 0:
            self.current_tracks = self.files.copy()
            random.shuffle(self.current_tracks)
        
        self.stream = librosa.core.stream(self.current_tracks[0], 1,
                                          self.buffer_length, self.buffer_length)
        
    def _get_stream(self):
        try:
            return next(self.stream)
        except StopIteration:
            self._update_stream()
            return next(self.stream)
        
    def get(self, sample_length, sec=0):
        with self.lock:
            buffer = self._get_stream()
    #         while len(buffer) < sec * self.fs:
    #             buffer = np.concatenate((buffer, self._get_stream()), axis=0)
            while len(buffer) < sample_length:
                buffer = np.concatenate((buffer, self._get_stream()), axis=0)
        return buffer[:sample_length]
    
    
def concat_variable_length_files(speech_files, anchor=6.05):
    speech_file_lengths = [(f, librosa.core.get_duration(filename=f)) for f in tqdm(speech_files)]
    speech_file_lengths.sort(key=lambda x: x[1]) 
    
    idx = np.argmin([abs(l[1] - anchor) for l in speech_file_lengths])
    
    if idx % 2 != 1:
        idx +=1 

    folding_files = speech_file_lengths[:idx]
    left_files = [(s[0], s[1]) for s in speech_file_lengths[idx:]]

    def fold(input_list):
        center = int((len(input_list) - 1) / 2)
        x = input_list[:center]
        y = input_list[center:][::-1]

        return [((a[0], b[0]), (a[1], b[1])) for a, b in zip(x, y)]

    folded_files = fold(folding_files)

    merge_files = folded_files + left_files
    
    return merge_files


def load_wavs(wavs):
    if isinstance(wavs, str):
        wavs = [wavs]
    wavs = list(wavs) # change tuple to list to support shuffling
    random.shuffle(wavs)
        
    for i, wav in enumerate(wavs):
        _y, sr = librosa.core.load(wav, sr=22050, mono=True)
        if i == 0:
            y = _y
        else:
            y = np.concatenate([y, _y], axis=0)

    return y


def get_zxx_and_log_spectrogram(audio, fs=22050, nseg=2040, nsc=510):
    _, _, Zxx = scipy.signal.stft(audio, fs=fs, nperseg=nseg, 
                                         noverlap=nseg-nsc)
    
    Zxx = Zxx.T

    Sxx = np.abs(Zxx)
    log_spectrogram = 20 * np.log10(np.maximum(Sxx, 1e-8))
    
    log_spectrogram = (log_spectrogram + 160) / 160
    
    Zxx_tensor = zxx_to_complex_tensor(Zxx)
    
    return Zxx_tensor, torch.tensor(log_spectrogram)
    
#     return Zxx, torch.tensor(log_spectrogram)


def zxx_to_complex_tensor(Zxx):
    complex_tensor = np.stack([Zxx.real, Zxx.imag], axis=-1)
    return torch.tensor(complex_tensor)


def complex_tensor_to_zxx(complex_tensor):
    complex_array = complex_tensor.numpy()
    return complex_array[:, :, :, 0] + 1j * complex_array[:, :, :, 1]


def complex_tensor_to_audio(complex_tensor):
    # B, T, F
#     audio = [scipy.signal.istft(c.T, fs=22050, nperseg=2048, noverlap=2048 - 512)[1] 
#          for c in complex_tensor_to_zxx(complex_tensor)]
    audio = [scipy.signal.istft(c.T, fs=22050, nperseg=2040, noverlap=2040 - 510)[1] 
             for c in complex_tensor_to_zxx(complex_tensor)]
    
    return audio


def zxx_to_audio(zxx_tensor):
    # B, T, F
    audio = [scipy.signal.istft(c.T, fs=22050, nperseg=2040, noverlap=2040 - 510)[1] 
         for c in zxx_tensor]
    
    return audio


def get_next_complete_length(length):  
    for num in range(1, 30):
        for i in range(6):
            num = 2 * num + 3
        if num >= length:
            return num
    assert False, f'Unable to find complete length larger than {length}'

    
def pad_to_complete_length(tensor, axis):
    length = tensor.shape[axis]
    complete_length = get_next_complete_length(length)
    padding_length = complete_length - length
    
    if padding_length == 0:
        return tensor
    else:
        padding_shape = list(tensor.shape)
        padding_shape[axis] = padding_length
        pad = torch.zeros(padding_shape)
        return torch.cat((tensor, pad), axis)
    

def sample_mixed_audio(file_tuple, background_stream_loader):
    
    Zxx_signal_list = list()
    Zxx_noise_list = list()
    Zxx_mixed_list = list()
    log_spectrogram_signal_list = list()
    log_spectrogram_noise_list = list()
    log_spectrogram_mixed_list = list()
    
    for files in file_tuple:
        y = load_wavs(files)
        background = background_stream_loader.get(len(y))
        
        signal_weight = min(np.random.normal(0.6, 0.4/3), 1)
        noise_weight = max(min(np.random.normal(0.6, 0.4/3), 1-signal_weight), 0)
        
        y = y * signal_weight
        background = background * noise_weight
        mixed = y + background
        
        Zxx_signal, log_spectrogram_signal = get_zxx_and_log_spectrogram(y)
        Zxx_noise, log_spectrogram_noise = get_zxx_and_log_spectrogram(background)
        Zxx_mixed, log_spectrogram_mixed = get_zxx_and_log_spectrogram(mixed)
        
        Zxx_signal_list.append(Zxx_signal)
        Zxx_noise_list.append(Zxx_noise)
        Zxx_mixed_list.append(Zxx_mixed)
        log_spectrogram_signal_list.append(log_spectrogram_signal)
        log_spectrogram_noise_list.append(log_spectrogram_noise)
        log_spectrogram_mixed_list.append(log_spectrogram_mixed)
        
    Zxxs = (pad_to_complete_length(pad_sequence(Zxx_signal_list, 
                                                batch_first=True), axis=1), 
            pad_to_complete_length(pad_sequence(Zxx_noise_list, 
                                                batch_first=True), axis=1), 
            pad_to_complete_length(pad_sequence(Zxx_mixed_list, 
                                                batch_first=True), axis=1))
    
    log_spectrograms = (pad_to_complete_length(pad_sequence(log_spectrogram_signal_list, 
                                                            batch_first=True).unsqueeze_(1), axis=2), 
                        pad_to_complete_length(pad_sequence(log_spectrogram_noise_list, 
                                                            batch_first=True).unsqueeze_(1), axis=2), 
                        pad_to_complete_length(pad_sequence(log_spectrogram_mixed_list, 
                                                            batch_first=True).unsqueeze_(1), axis=2))
        
#     Zxxs.shape[1]
#     log_spectrograms.shape[1]
    
    
#     log_spectrograms.shape
    # torch.Size([4, 321, 1025, 2])
    # torch.Size([4, 1, 321, 1025])
        
    return Zxxs, log_spectrograms

def get_data_loader(speech_files, background_stream_loader):

    dataloader = DataLoader(speech_files, batch_size=4, 
                            shuffle=True, num_workers=4,
                            collate_fn=partial(sample_mixed_audio, background_stream_loader=background_stream_loader))
    
    return dataloader
    
    
if __name__ == "__main__":
    wav_files = sorted(glob('./data/background/YD/*.wav'))
    background_stream_loader = MultiStreamLoader(wav_files)
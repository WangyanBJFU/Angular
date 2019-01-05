import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os
import wave


def wav_show(wave_data, fs):
    time = np.arange(0, len(wave_data)) * (1.0 / fs)
    plt.plot(time, wave_data)
    plt.show()


def read_wav_data(filename):
    wav = wave.open(filename,"rb") 
    num_frame = wav.getnframes()
    num_channel=wav.getnchannels() 
    framerate=wav.getframerate() 
    num_sample_width=wav.getsampwidth() 
    str_data = wav.readframes(num_frame) 
    wav.close() 
    wave_data = np.fromstring(str_data, dtype = np.short) 
    wave_data.shape = -1, num_channel 
    wave_data = wave_data.T 
    #wave_data = wave_data 
    return wave_data, framerate


if(__name__ == '__main__'):
    wave_data, fs = read_wav_data('wangyan.wav')
    wav_show(wave_data[0],fs)

                                                                                                                                         35,0-1        All

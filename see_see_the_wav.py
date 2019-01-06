import numpy as np
import matplotlib.pyplot as plt
import time
import math
import os
import wave

'''
自己录了个“啊~~~~”转成wav之后想看看声音波形，
随便试了试
如果用python3 plt生成的画布上只有坐标和指针没图像的话，"The Gtk3Agg backend is known to not work on Python 3.x with pycairo."
就执行一句
sudo pip3 install cairocffi就能显示了
'''

path = '/home/wy/wav_show/audio'
dirs = os.listdir(path)
print("Files in this folder are:")
for file in dirs:
    print(file)


def wav_show(wave_data, framerate):
    time = np.arange(0, len(wave_data)) * (1.0 / framerate)
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
    t0=time.time()
    wave_data, framerate = read_wav_data('wangyan.wav')
    t1=time.time()
    wav_show(wave_data[0],framerate)
    print("Reading wav data's cost time is:",t1-t0)
                                                                                                                                         35,0-1        All

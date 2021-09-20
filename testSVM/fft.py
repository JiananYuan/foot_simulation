# fft.py -- 信号fft变换

import numpy as np
from scipy import fftpack
from scipy import signal
from fileManager import write_data
Fs = 20000      # 采样频率
begin = 0       # 开始截取的频率
end = 240       # 结束截取的频率


# 获得窗函数
def getWindow(type, L):
    if type == 'hanning':
        return signal.windows.hann(L)
    if type == 'hamming':
        return signal.windows.hamming(L)


def get_fft_processed_data(signal):
    # 获取窗函数
    win = getWindow("hanning", len(signal))
    # 与窗函数相乘
    win_signal = np.multiply(np.array(signal), np.array(win)).tolist()
    F = fftpack.fft(win_signal)
    # 得到分解波的时间序列
    f = fftpack.fftfreq(len(signal), 1.0/Fs)
    # 能量大小
    mask = np.where(f >= 0)
    freq = abs(F[mask]/len(signal))
    return f[mask], freq


def fft(data, path, inter_step_frequency):
    # data是二维张量
    rec = []
    for item in data:
        fmask, freq = get_fft_processed_data(item)
        freq = freq.tolist()
        tmp = freq[begin:end]
        tmp.append(inter_step_frequency)
        write_data(path, tmp)
        rec.append(tmp)
    return rec

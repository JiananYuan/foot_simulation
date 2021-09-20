# signal_segment.py -- 信号切段

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from fileManager import write_data, read_data

L = 5000                 # 窗函数大小（一个信号约5000点长）
small_L = 600            # 探测信号起点的移动长度
sample_freq = 20000      # 采样频率
step_len = 2500          # 窗函数移动的步长
test_fit_X = []          # 查看切段效果的X坐标
test_fit_Y = []          # 查看切段效果的Y坐标
energy_factor = 1.005    # 能量阈值影响因子
small_factor = 2.5       # 小段能量阈值影响因子


# 巴特沃兹滤波 参数：(滤波类型，滤波器阶数，目标频率阈值，待处理数据)
def filter(type, step, des_freq, raw_data):
    global sample_freq
    b, a = signal.butter(step, 2*des_freq/sample_freq, type)
    filterData = signal.filtfilt(b, a, raw_data)
    return filterData


# 归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.mean(data)) / _range


# 计算标准差
def standard_var(data):
    return np.sqrt(np.var(data))


# 计算均值
def mean(data):
    return np.mean(data)


# 重构切段算法
def segment(data):
    global test_fit_Y
    global test_fit_X
    # 定义rec存储切段之后的信号，是信号的集合
    rec = []

    # 先对一个信号长度的噪声进行采样计算能量阈值
    noise = data[0:L]
    # 噪声能量值，大于这个值之后的一定范围可被认为是有效信号
    noise_energy = 0
    for i in noise:
        noise_energy = noise_energy + i**2

    # 同理计算small范围内的能量值
    small_noise = data[0:small_L]
    small_noise_energy = 0
    for i in small_noise:
        small_noise_energy = small_noise_energy + i**2

    cursor = 0
    # 探测第1个信号的起点
    while cursor <= len(data)-L:
        # 截取长度为L的信号
        tmp_segment = data[cursor:cursor+L]
        # 时域能量计算
        energy = 0
        for i in tmp_segment:
            energy = energy + i**2
        # 检测到一个有效信号（可能是部分信号）
        if energy >= noise_energy * energy_factor:
            # 下面这一步是增强程序的鲁棒性，万一后面程序检索不到信号，则以第1个信号作为代表
            rec.append(data[cursor:cursor + L])
            # test_fit_Y.append(data[cursor:cursor + L])
            # test_fit_X.append([j for j in range(cursor, cursor + L)])

            # 细粒度探测信号起始点
            while True:
                tmp_segment = data[cursor:cursor+small_L]
                energy = 0
                for i in tmp_segment:
                    energy = energy + i*i
                # 缩小步长，细粒度探测一个有效信号的开端
                if energy < small_noise_energy * energy_factor:
                    cursor = cursor + small_L
                else:
                    cursor = cursor + small_L//2
                    break
            # 这个时候的cursor已经近似为第1个信号开端
            break
        else:
            # 没探测到有效信号，则滑动L
            cursor = cursor + L

    # 下面探测之后信号区间端点
    # 步进时间差 和 步进次数
    inter_step_interval_time = 0
    inter_step_count = 0

    sig = []
    last_energy = 0x7fffffff
    # 最大信号段能量值，初始化为0
    max_energy = 0
    # 检验用 #########################################
    begin_pos = 0
    tag = True
    # ##### #########################################
    while cursor <= len(data)-small_L:
        # 检验用 #####################################
        if tag:
            tag = False
            begin_pos = cursor
        # ##### #####################################
        tmp_segment = data[cursor:cursor+small_L]
        energy = 0
        for i in tmp_segment:
            energy = energy + i*i
        # 属于同个信号
        if energy < last_energy * small_factor:
            sig.extend(tmp_segment)
            cursor = cursor + small_L
            last_energy = energy
            # for i in range(0, len(test_fit_X)):
            #     print(len(test_fit_X[i]), end="--")
            #     print(len(test_fit_Y[i]))
        # 跨越到了下个信号
        else:
            tmp_segment = data[cursor:cursor+small_L//2]
            sig.extend(tmp_segment)
            energy = 0
            for i in sig:
                energy = energy + i**2
            if energy > max_energy:
                # 累加步进频率
                inter_step_interval_time += (cursor + small_L//2 - begin_pos) * (1/sample_freq)
                inter_step_count = inter_step_count + 1
                max_energy = energy
                rec.clear()
                for i in sig:
                    rec.append(i)
                rec = [rec]
                # 检验用 #################################################
                # test_fit_X.clear()
                # test_fit_Y.clear()
                # test_fit_X.append([j for j in range(begin_pos, cursor+small_L//2)])
                # # test_fit_Y.append(sig)
                # for i in sig:
                #     test_fit_Y.append(i)
                # test_fit_Y = [test_fit_Y]
                # ##### #################################################
            sig.clear()
            last_energy = 0x7ffffff
            cursor = cursor + small_L//2
            tag = True

    # 论文中最后一个维度要加上inter_step_frequency
    return rec, 1 / (inter_step_interval_time / inter_step_count)


def old_segment(data):
    global test_fit_Y
    global test_fit_X
    # 先对一个信号长度的噪声进行采样计算能量阈值
    noise = data[0:L]
    # 噪声能量值，大于这个值之后的一定范围可被认为是有效信号
    noise_energy = 0
    for i in noise:
        noise_energy = noise_energy + i*i
    # rec存储切段之后的信号，是信号的集合
    rec = []
    cursor = 0
    # 最大信号段能量值，初始化为0
    max_energy = 0
    # 游标向前滑动
    while cursor <= len(data)-L:
        # 根据窗口大小截取长度为L的信号
        tmp_segment = data[cursor:cursor+L]
        # 时域能量计算
        energy = 0
        for i in tmp_segment:
            energy = energy + i**2
        if energy >= noise_energy*energy_factor:

            # 大于能量阈值，加入切段信号序列集合中
            # 只取能量最高的段
            if energy > max_energy:
                max_energy = energy
                rec.clear()
                rec.append(tmp_segment)
                # 检验用
                test_fit_X.clear()
                test_fit_Y.clear()
                test_fit_X.append([j for j in range(cursor, cursor+L)])
                test_fit_Y.append(tmp_segment)
                # 此处有信号，游标移动一个信号的长度
            cursor = cursor + L
        else:
            cursor = cursor + step_len
    return rec


def save_to_csv(save_path, data):
    # data是二维张量
    for item in data:
        write_data(save_path, item)
    return


def signal_segment(path, save_path):
    data = read_data(path)            # 读取原时域数据，存放于data
    data = data[0:len(data)-1]        # 最后一个值为nan，将它去掉
    data = normalization(data)        # 归一化
    data = filter('lowpass', 3, 70, data)    # 巴特沃兹滤波(滤波类型，滤波器阶数，目标频率阈值，待处理数据)
    segment_signals, inter_step_frequency = segment(data)   # 信号切段
    # 检验切段效果使用以下代码 ###############################
    # plt.plot(data)
    # plt.plot(test_fit_X[0], test_fit_Y[0], color='red')
    # plt.show()
    # #################### ###############################
    save_to_csv(save_path, segment_signals)
    return segment_signals, inter_step_frequency

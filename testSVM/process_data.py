from signal_segment import signal_segment
from fft import fft
import os
from train_model import train_model

# 喂给OneClassSVM的训练特征（features：二维张量）
features = []
# 每人50组训练数据
end_index = 50


def process_train():
    name = ['hwl', 'pzc', 'yjn', 'zl', 'zyh']
    for item in name:
        print("Processing " + item + "'s data...")
        # 测试者数据集路径
        path = './collected_data/' + item + '/'
        # 遍历路径下面所有的csv文件
        for root, dirs, files in os.walk(path):
            for index, csv in enumerate(files):
                print("正在训练的文件：" + str(csv))
                # raw_data的数据路径、切段后数据的存储路径、fft变换之后数据的存储路径
                collected_path = path + str(csv)
                segment_path = './segment_data/' + item + '/' + str(csv)
                fft_path = './fft_data/' + item + '/' + str(csv)
                # 信号切段并保存
                signals, inter_step_frequency = signal_segment(collected_path, segment_path)
                # fft变换并存入特征
                for _f in fft(signals, fft_path, inter_step_frequency):  # 最后一个参数传入步进频率，也是作为一个特征
                    # _f.append()
                    # print(_f)
                    # 只取一部分作为训练样本
                    if index < end_index:
                        features.append(_f)

    # 构造train_label标签
    train_label = [0]*50 + [1]*50 + [2]*50 + [3]*50 + [4]*50
    # 训练OneClassSVM模型
    train_model(features, train_label)
    print("Success!")

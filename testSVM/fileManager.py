import pandas as pd
import os

start_index = 50


def read_data(path):
    df = pd.read_csv(path, header=None)
    rec = []
    for i in df:
        rec.append(df.iloc[0, i])
    return rec


def write_data(path, data):
    df = pd.DataFrame(columns=data)
    df.to_csv(path, index=False, sep=',', mode='a')  # 追加方式


def load_test_data():
    dir_path = './fft_data/'
    test_data = []
    test_label = []
    for root, dirs, files in os.walk(dir_path):  # ./fft_data/
        # 取出每个人的文件夹
        for _dir in dirs:
            test_path = dir_path + str(_dir) + '/'  # ./fft_data/yjn
            for _root, _dirs, _files in os.walk(test_path):
                # 取出每个人的csv文件，每人取10组数据
                for index, file in enumerate(_files):
                    if index < start_index:
                        continue
                    csv_path = test_path + str(file)  # ./fft_data/yjn/1.csv
                    tmp = read_data(csv_path)
                    test_data.append(tmp)
                    label_num = 0
                    if str(_dir) == 'hwl':
                        label_num = 0
                    elif str(_dir) == 'pzc':
                        label_num = 1
                    elif str(_dir) == 'yjn':
                        label_num = 2
                    elif str(_dir) == 'zl':
                        label_num = 3
                    elif str(_dir) == 'zyh':
                        label_num = 4
                    test_label.append(label_num)

    return test_data, test_label


def load_adjust_fft_data(name):
    dir_path = './fft_data/'
    adjust_data = []
    adjust_path = dir_path + name + '/'
    for _root, _dirs, _files in os.walk(adjust_path):
        # 取出每个人的csv文件
        for index, file in enumerate(_files):
            if index < 50:
                csv_path = adjust_path + str(file)
                tmp = read_data(csv_path)
                adjust_data.append(tmp)
    return adjust_data

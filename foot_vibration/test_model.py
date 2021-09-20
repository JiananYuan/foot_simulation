from fileManager import load_test_data
from fileManager import load_adjust_fft_data
import joblib
from train_model import train_model
import os

# 来一个信号，就去做比对，对得上就ok
# 何谓“对得上”
# 对于合法用户，所有模型识别之后，至少有一个模型可以接纳，且匹配数据标签
# 对于非法用户，没有一个模型接纳这个信号


def test_model():
    test_data, test_label = load_test_data()
    test_size = len(test_data)
    acc = 0
    name = ['pzc', 'yjn', 'zl', 'zyh']  # 四个合法用户
    N = len(name)
    for index, person in enumerate(name):
        print('Attacking ' + person + "\'s model...")
        model = joblib.load('model/' + person + '.pickle')
        for j in range(test_size):
            pre = model.predict([test_data[j]])
            print(pre)
            if pre == [1] and test_label[j] == person:
                acc = acc + 1
            if pre == [-1] and test_label[j] != person:
                acc = acc + 1
    print("footPrintID的精度为：" + str(acc / (test_size*N)))


def adjust_model():
    name = ['pzc', 'yjn', 'zyh', 'zl', 'hwl']
    for item in name:
        print("Reading " + item + "'s fft data...")
        features = load_adjust_fft_data(item)
        # 训练OneClassSVM模型
        train_model(features, item)
        print("Success in building " + item + "'s model!")
    test_model()

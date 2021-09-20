# main.py -- 调用入口
from process_data import process_train
from test_model import test_model
from test_model import adjust_model


if __name__ == '__main__':
    op = int(input('选择模式：1-训练模式  2-测试模式  3-调参模式  \n'))
    if op == 1:
        print('训练模式开启...')
        process_train()
    if op == 2:
        print('测试模式开启...')
        test_model()
    if op == 3:
        print('调参模式开启...')
        adjust_model()

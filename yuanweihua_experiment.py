import os
import csv
import numpy as np
from pprint import pprint

import mindspore as ms
from mindspore import nn
from mindspore import context
from mindspore import dataset
from mindspore.train.callback import LossMonitor

def create_dataset(data):

    # Todo 分别将Iris-setosa，Iris-versicolor，Iris-virginica对应为0，1，2三类
    label_map = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    X = np.array([[float(x) for x in s[:-1]] for s in data[:150]], np.float32)
    Y = np.array([label_map[s[-1]] for s in data[:150]], np.int32)

    # Todo Using random choice and split dataset into train set and validation set by 8:2.
    train_idx = np.random.choice(150, 120, replace=False)
    test_idx = np.array(list(set(range(150)) - set(train_idx)))
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    # Convert the training data to MindSpore Dataset.
    XY_train = list(zip(X_train, Y_train))
    ds_train = dataset.GeneratorDataset(XY_train, ['x', 'y'])
    ds_train = ds_train.shuffle(buffer_size=120).batch(32, drop_remainder=True)
    XY_test = list(zip(X_test, Y_test))
    ds_test = dataset.GeneratorDataset(XY_test, ['x', 'y'])
    ds_test = ds_test.batch(30)

    return ds_train, ds_test


def softmax_regression(ds_train, ds_test):
    net = nn.Dense(4, 3)

    # Todo 使用交叉熵损失计算
    loss = nn.loss.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # Todo 使用动量优化器优化参数，其中学习率设置为0.05，动量设置为0.9
    opt = nn.optim.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)

    model = ms.train.Model(net, loss, opt, metrics={'acc', 'loss'})
    model.train(25, ds_train, callbacks=[LossMonitor(per_print_times=ds_train.get_dataset_size())], dataset_sink_mode=False)
    metrics = model.eval(ds_test)
    print(metrics)


if __name__ == "__main__":


    with open('iris.data') as csv_file:
        data = list(csv.reader(csv_file, delimiter=','))
    pprint(data[0:5]); pprint(data[50:55]); pprint(data[100:105]) # print some samples

    softmax_regression(*create_dataset(data))
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mnist_loader
import network

if __name__ == "__main__":
    #先加载 MNIST 数据。
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    #设置一个有 30 个隐藏层神经元的 Network。我们在导入如上所列的名为 network 的 Python 程序后做
    net = network.Network([784, 30, 10])

    #使用随机梯度下降来从 MNIST training_data 学习超过 30 次迭代期,小批量数据大小为 10,学习速率 η = 3.0
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
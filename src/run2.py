#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mnist_loader
import network2

if __name__ == "__main__":
    #先加载 MNIST 数据。
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    net = network2.Network([784, 30, 10], cost = network2.CrossEntropyCost)
    net.large_weight_initializer()

    net.SGD(training_data, 30, 10, 0.5, evaluation_data=test_data, monitor_evaluation_accuracy=True)


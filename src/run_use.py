#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle
import os
import numpy as np

import mnist_loader

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def feedforward(weights, biases, a):
    """Return the output of the network if ``a`` is input."""
    for b, w in zip(biases, weights):
        a = sigmoid(np.dot(w, a)+b)
    return a

if __name__ == "__main__":
    #先加载 MNIST 数据。
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    if os.path.exists('params'):  
        f=open('params')
        weights = cPickle.load(f)        
        biases = cPickle.load(f)

        test_results = [(np.argmax(feedforward(weights, biases, x)), y) for (x, y) in test_data]
        fit_num = sum(int(x == y) for (x, y) in test_results)
        print "{0} / {1}".format(fit_num, len(test_data))
    else:
        print "There is no param file!"
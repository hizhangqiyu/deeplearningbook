import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

LEARNING_RATE = 0.01
STEPS = 1000
BATCHES = 128

w = np.random.normal(size=(784, 10))
b = np.random.normal(size=(1, 10))

# record loss for each step
loss_r = []

def softmax(x):
    ex = np.sum(np.exp(x), axis=1)  # 1000x1
    return x / ex  # 1000x10

def relu(x):
    return x * (x > 0)

def sigmoid(x):
    return 1 / (np.exp(-1 * x))

def ff(x):
    f = softmax(np.dot(x, w) + b) # 1000x10
    return f

# MSE
def loss_mse(y, y_):
    return np.sum((y - y_) ** 2) / 2

# cross-entropy
def loss_cross_entropy(y, y_):
    return -1 * np.mean(np.sum(y_ * np.log(y), axis=1))

#
def bp(x, y_, y):



for step in range(STEPS):
    batch_xs, batch_ys = mnist.train.next_batch(BATCHES)

    y_ = ff(batch_xs)


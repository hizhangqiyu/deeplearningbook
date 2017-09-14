import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../data/mnist", one_hot=True)

LEARNING_RATE = 0.01
STEPS = 100001
BATCHES = 128

w = np.random.normal(size=(784, 10))
b = np.random.normal(size=(1, 10))

# record loss for each step
loss_r = []

def softmax(x):
    if len(x.shape) > 1:
        shiftx = x - np.amax(x, axis=1)[:,None]
        exps = np.exp(shiftx)
        x = exps / np.sum(exps, axis=1)[:,None]
    else:
        shiftx = x - np.amax(x)
        exps = np.exp(shiftx)
        x = exps / np.sum(exps)
    return x

# MSE
def loss_mse(y, y_):
    return np.sum((y - y_) ** 2) / 2

# cross-entropy
def loss_cross_entropy(y, y_):
    cost = - np.sum(np.log(y_[y == 1])) / BATCHES
    return cost

def accuracy(y, y_):
    acc = np.equal(np.argmax(y, axis=1), np.argmax(y_, axis=1))
    acc = np.sum(acc) / BATCHES
    return acc

for step in range(STEPS):
    batch_xs, batch_ys = mnist.train.next_batch(BATCHES)

    # feedforward networks
    f = np.dot(batch_xs, w) + b
    a = softmax(f)
    y_ = a

    # backpropagetion
    grad_a = (a - batch_ys) / BATCHES
    grad_w = np.dot(batch_xs.T, grad_a)
    grad_b = np.sum(grad_a, axis=0, keepdims=True)

    w -= LEARNING_RATE * grad_w
    b -= LEARNING_RATE * grad_b

    loss_r.append(loss_cross_entropy(batch_ys, y_))
    if step % 5000 == 0:
        acc = accuracy(batch_ys, y_)
        print("step %r accuracy=%r" % (step, acc))

plt.plot(loss_r[:], range(STEPS))
plt.show()
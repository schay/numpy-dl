import numpy as np
import utils
from mnist_data_provider import MNIST
from Modules import Fully, Softmax, Cross_entropy, Squared, Relu, Sigmoid, Drop_out, SGD
from graph import Graph
import matplotlib.pyplot as plt


# MNISTデータの読み込み
mnist = MNIST()
n_output = 10
# ave, _ = utils.get_ave_var(mnist.train_X)  # 正規化
# var = 1  # 正規化
# mnist.map_x(utils.normalize, ave, var)  # 正規化
inputs_shape = np.prod(mnist.img_shape)
mnist.map_x(np.reshape, [-1, inputs_shape])
mnist.map_y(utils.one_hot, n_output)


# パラメータ
s_batch = 100
epsilon = 0.0001
gamma = 0.99
epoch = 0
weight_scale = 0.001
keep_prob = 0.5


# 層の定義
f1 = Fully(100, weight_scale)
f2 = Fully(30, weight_scale)
f2_d = Fully(100, weight_scale)
f1_d = Fully(inputs_shape, weight_scale)
drop_out = Drop_out()
op = SGD(epsilon, gamma)
logits = [f1, Relu(), f2, Relu(), f2_d, Relu(), f1_d, Sigmoid()]
loss_func = Squared()

graph = Graph(inputs_shape, logits, loss_func, op)


# 事前学習
n_train = mnist.train_data_num
losses_train = []
losses_test = []

for e in range(epoch):
    loss = 0
    drop_out(keep_prob)
    for s in range(mnist.train_data_num // s_batch):
        x, _ = mnist.train_batch(s_batch)

        y = graph.forward(x)
        loss, delta = graph.loss_delta(y, x)
        graph.backward(delta)
        graph.update_weight()

    op.down_epsilon()
    drop_out(1)

    train_x, _ = mnist.train_all()
    train_y = graph.forward(train_x)
    loss, _ = graph.loss_delta(train_y, train_x)
    losses_train.append(loss)
    print('epoch', e+1)
    print('train:', '\tloss:', loss)

    test_x, _ = mnist.test_all()
    test_y = graph.forward(test_x)
    loss, _ = graph.loss_delta(test_y, test_x)
    losses_test.append(loss)

    print('test:', '\tloss:', loss)


# 学習
epsilon = 0.001
epoch = 30

f3 = Fully(n_output, weight_scale)
logits = [f1, Relu(), f2, Relu(), f3, Softmax()]
loss_func = Cross_entropy()
op = SGD(epsilon, gamma)
graph = Graph(inputs_shape, logits, loss_func, op)

losses_train = []
losses_test = []


for e in range(epoch):
    loss = 0
    drop_out(keep_prob)
    for s in range(mnist.train_data_num // s_batch):
        x, y_ = mnist.train_batch(s_batch)

        y = graph.forward(x)
        loss, delta = graph.loss_delta(y, y_)
        graph.backward(delta)
        graph.update_weight()

    op.down_epsilon()
    drop_out(1)

    train_x, train_y_ = mnist.train_all()
    train_y = graph.forward(train_x)
    loss, _ = graph.loss_delta(train_y, train_y_)
    loss /= mnist.train_data_num
    losses_train.append(loss)
    print('epoch', e+1)
    print('train:', '\tloss:', loss, '\taccuracy:', utils.predict(train_y, train_y_))

    test_x, test_y_ = mnist.test_all()
    test_y = graph.forward(test_x)
    loss, _ = graph.loss_delta(test_y, test_y_)
    loss /= mnist.test_data_num
    losses_test.append(loss)

    print('test:', '\tloss:', loss, '\taccuracy:', utils.predict(test_y, test_y_))

plt.plot(np.arange(epoch), losses_train)
plt.plot(np.arange(epoch), losses_test)
plt.show()

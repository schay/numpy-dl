import numpy as np
from mnist_data_provider import MNIST
from Modules import Fully, Softmax, Cross_entropy, Relu

mnist = MNIST()

# パラメータ
shape = {2, 28, 28, 1}
n_output = 10
s_batch = 10
epsilon = 0.01
lam = 0.0001
gamma = 0.9


f1 = Fully(28*28, 100)
f2 = Fully(100, 10)
rl = Relu()
sm = Softmax()
ce = Cross_entropy()


for i in range(6000):
    # 入力
    iterator = i * s_batch
    x = mnist.train_X[iterator:iterator+s_batch]
    sy_ = mnist.train_Y[iterator:iterator+s_batch]

    x = np.reshape(x, [-1, 28*28])
    x = x.astype(np.float32)
    y_ = np.zeros([s_batch, n_output], dtype=np.float32)
    y_[np.arange(s_batch), sy_] = 1  # one hot

    # 順伝播
    z1 = f1.forward(x)
    # z1d = rl.forward(z1)
    z2 = f2.forward(z1)
    y = sm.forward(z2)
    loss = ce.loss(y, y_)

    # 逆伝播
    # delta = ce.delta(y, y_)
    delta_y = y - y_
    # delta_y = sm.backward(delta)
    delta_z2 = f2.backward(delta_y)
    # delta_z1d = rl.backward(delta_z2)
    delta_z1 = f1.backward(delta_z2)

    # 重みの更新
    f2.update_weight()
    f1.update_weight()

    # print(f2.weights[:3, :3])
    print(y_)
    print(y)

x = mnist.test_X
x = np.reshape(x, [-1, 28*28])
x = x.astype(np.float32)
z1 = f1.forward(x)
z2 = f2.forward(z1)
y = sm.forward(z2)

sy_ = mnist.test_Y
s_batch = np.shape(x)[0]
y_ = np.zeros([s_batch, n_output], dtype=np.float32)
y_[np.arange(s_batch), sy_] = 1
accuracy = len(np.where(y.argmax(axis=1) == y_.argmax(axis=1))[0]) / s_batch

print(y.argmax(axis=1)[:10])
print(y_.argmax(axis=1)[:10])
print('accuracy :', accuracy)

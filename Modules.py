"""
Layerと活性化関数、誤差関数、Optimizer全てここで定義してます。
"""

import numpy as np
from interface import Module, Layer, Activator, Loss, Optimizer


class Fully(Layer):

    def __init__(self, outputs_shape, weight_scale):
        super().__init__()
        self.inputs = None
        self.outputs_shape = outputs_shape
        self.weights = None
        self.bias = None
        self.scale = weight_scale

    def build(self, inputs_shape):
        self.weights = np.random.normal(0, self.scale, [inputs_shape, self.outputs_shape])
        self.bias = np.zeros([self.outputs_shape])
        return self.outputs_shape

    def forward(self, inputs):
        self.inputs = inputs
        outputs = np.matmul(self.inputs, self.weights) + self.bias
        return outputs

    def backward(self, delta):
        self.delta = delta
        return delta.dot(self.weights.T)

    def update_weight(self, optimizer):
        self.weights -= optimizer(self.inputs, self.delta)
        self.bias -= optimizer(np.ones(np.shape(self.inputs)[0]), self.delta)


class Convolution(Layer):
    """
    できていません
    """

    def __init__(self, kernel_shape, stride, output_channels):
        super().__init__()
        self.kernel_shape = kernel_shape
        self.stride = stride
        self.input_channels = None
        self.output_channels = output_channels
        self.inputs = None
        self.outputs_shape = None
        self.weights = None
        self.bias = None

    def build(self, inputs_shape):
        self.input_channels = inputs_shape[-1]
        weights_shape = [self.kernel_shape, self.kernel_shape, self.input_channels, self.output_channels]
        self.weights = np.random.normal(0, 0.0001, weights_shape)
        self.bias = np.random.normal(0, 0.0001, [self.output_channels])
        x_shape = (inputs_shape[0] - self.kernel_shape + self.stride)/self.stride
        y_shape = (inputs_shape[1] - self.kernel_shape + self.stride)/self.stride
        return [x_shape, y_shape, self.output_channels]

    def forward(self, inputs):
        super().forward(inputs)

    def backward(self, delta):
        super().backward(delta)

    def update_weight(self, epsilon=0.001):
        raise NotImplementedError


class Sigmoid(Activator):

    def __init__(self):
        super().__init__()
        self.outputs = None

    def forward(self, inputs):
        self.outputs = 1.0 / (1.0 + np.exp(-inputs))
        return self.outputs

    def backward(self, delta):
        return delta * self.outputs * (1.0 - self.outputs)


class Softmax(Activator):

    def __init__(self):
        super().__init__()
        self.outputs = None

    def forward(self, inputs):
        inputs -= np.max(inputs)
        exp = np.exp(inputs)
        self.outputs = exp / (exp.sum(axis=1, keepdims=True))
        return self.outputs

    def backward(self, delta):
        return delta * self.outputs * (1.0 - self.outputs)


class Relu(Activator):

    def __init__(self):
        super().__init__()
        self.outputs = None

    def forward(self, inputs):
        self.outputs = np.maximum(inputs, 0.0)
        return self.outputs

    def backward(self, delta):
        return delta * np.vectorize(lambda _x: 0.0 if _x < 0.0 else 1.0)(self.outputs)


class Cross_entropy(Loss):

    def __init__(self):
        super().__init__()
        self.y = None
        self.y_ = None

    def loss(self, y, y_):
        return np.sum(-y_ * np.log(y))

    def delta(self, y, y_):
        return (y - y_) / (y * (1.0 - y))


class Squared(Loss):
    """
    二乗誤差
    """

    def __init__(self):
        super().__init__()
        self.y = None
        self.y_ = None

    def loss(self, y, y_):
        return (y - y_).T.dot(y - y_).sum()

    def delta(self, y, y_):
        return y - y_


class Drop_out(Module):
    """
    ドロップアウトの実装
    バッチごとに同じユニットを無視する。
    無視するユニットを0にしている。
    """

    def __init__(self):
        super().__init__()

    def __call__(self, keep_prob):
        self.keep_prob = keep_prob

    def forward(self, inputs):
        batch_scalar = np.prod(np.shape(inputs[0]))
        drop_p = np.zeros(batch_scalar)
        drop_p[:round(batch_scalar * self.keep_prob).astype(np.int64)] = 1
        np.random.shuffle(drop_p)
        return inputs * drop_p / self.keep_prob


class SGD(Optimizer):

    def __init__(self, epsilon, gamma):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma

    def __call__(self, inputs, delta):
        grad = inputs.T.dot(delta)
        return self.epsilon * grad / np.shape(inputs)[0]

    def down_epsilon(self):
        self.epsilon = self.gamma * self.epsilon


class Momentum(Optimizer):
    """
    できていません
    """

    def __init__(self, epsilon, gamma, alpha=0.9):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, inputs, delta):
        grad = inputs.T.dot(delta)
        return self.epsilon * grad / np.shape(inputs)[0]  # ここ何になるか

    def down_epsilon(self):
        self.epsilon = self.gamma * self.epsilon

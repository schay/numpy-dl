

class Module:
    """
    ネットワークの構造に入るも
    """

    def __init__(self):
        pass

    def build(self, inputs_shape):
        return inputs_shape

    def forward(self, inputs):
        return inputs

    def backward(self, delta):
        return delta


class Layer(Module):
    """
    重みの更新が必要なもの
    """

    def __init__(self):
        super().__init__()
        self.weights = None
        self.bias = None
        self.delta = None

    def update_weight(self, epsilon):
        pass


class Activator(Module):

    def __init__(self):
        super().__init__()


class Loss:

    def __init__(self):
        pass

    def loss(self, y, y_):
        raise NotImplementedError

    def delta(self, y, y_):
        raise NotImplementedError


class Optimizer:

    def __init__(self):
        pass


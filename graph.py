import interface


class Graph:
    """
    グラフの作成と順伝搬、逆伝搬、重みの更新を一括で行います。
    """

    def __init__(self, inputs_shape, logits, loss, optimizer):
        self.inputs_shape = inputs_shape
        self._graph = logits
        self.loss = loss
        self.optimizer = optimizer

        self._build_graph(inputs_shape)

    def _build_graph(self, inputs_shape):
        shape = inputs_shape
        for g in self._graph:
            shape = g.build(shape)

    def forward(self, prev_outputs):
        outputs = None
        for g in self._graph:
            outputs = g.forward(prev_outputs)
            prev_outputs = outputs
        return outputs

    def backward(self, loss_delta):
        next_delta = loss_delta
        for g in reversed(self._graph):
            delta = g.backward(next_delta)
            next_delta = delta

    def loss_delta(self, y, y_):
        return self.loss.loss(y, y_), self.loss.delta(y, y_)

    def update_weight(self):
        for g in reversed(self._graph):
            if issubclass(g.__class__, interface.Layer):
                g.update_weight(self.optimizer)

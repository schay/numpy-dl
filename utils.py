import numpy as np


def normalize(data, ave, var):
    return (data - ave) / var


def get_ave_var(data):
    ave = np.average(data, axis=0)
    var = np.var(data, axis=0)
    return ave, var


def one_hot(indices, depth, dtype=np.float32):
    batch_size = np.shape(indices)[0]
    y_ = np.zeros([batch_size, depth], dtype=dtype)
    y_[np.arange(batch_size), indices] = 1
    return y_


def predict(y, y_):
    s_batch = np.shape(y)[0]
    return len(np.where(y.argmax(axis=1) == y_.argmax(axis=1))[0]) / s_batch

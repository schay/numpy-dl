import os
import urllib
import gzip
import numpy as np


class Batch_iterator:
    def __init__(self, shuffle):
        self.shuffle = shuffle
        self.iterator = 0
        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None
        self.train_data_num = None
        self.test_data_num = None

    def train_batch(self, batch_size):
        if self.iterator == 0 and self.shuffle:
            self._shuffle_train_data()
        it = np.arange(self.iterator, self.iterator + batch_size, dtype=np.int32)
        self.iterator += batch_size
        if self.iterator > self.train_data_num - batch_size:
            self.iterator = 0
        return self.train_X[it], self.train_Y[it]

    def train_all(self):
        return self.train_X, self.train_Y

    def test_all(self):
        return self.test_X, self.test_Y

    def _shuffle_train_data(self):
        shuffle = np.random.permutation(np.arange(self.train_data_num))
        self.train_X = self.train_X[shuffle]
        self.train_Y = self.train_Y[shuffle]

    def map_x(self, func, *args):
        self.train_X = func(self.train_X, *args)
        self.test_X = func(self.test_X, *args)

    def map_y(self, func, *args):
        self.train_Y = func(self.train_Y, *args)
        self.test_Y = func(self.test_Y, *args)


class MNIST(Batch_iterator):
    def __init__(self, shuffle=True):
        super().__init__(shuffle)
        self.train_data_num = 60000
        self.test_data_num = 10000
        self.img_shape = np.array([28, 28])

        # mnistファイルのダウンロード
        self.dataset_dir = os.path.dirname(__file__) + '/data'
        self.url_base = 'http://yann.lecun.com/exdb/mnist/'
        self.key_file = {
            'train_img': 'train-images-idx3-ubyte.gz',
            'train_label': 'train-labels-idx1-ubyte.gz',
            'test_img': 't10k-images-idx3-ubyte.gz',
            'test_label': 't10k-labels-idx1-ubyte.gz'
        }
        self.download_mnist()

        # mnistデータのロード
        self.train_X = self._load_image('train_img')
        self.train_Y = self._load_label('train_label')
        self.test_X = self._load_image('test_img')
        self.test_Y = self._load_label('test_label')

        self.next_batch = 0

    def _download(self, filename):
        file_path = self.dataset_dir + '/' + filename
        if os.path.exists(file_path):
            return print('already exist')
        print('Downloading ' + filename + ' ...')
        urllib.request.urlretrieve(self.url_base + filename, file_path)
        print('Done')

    def download_mnist(self):
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)

        for v in self.key_file.values():
            self._download(v)

    def _load_image(self, key):
        filename = self.key_file[key]
        file_path = self.dataset_dir + '/' + filename
        with gzip.open(file_path, 'rb') as f:
            image = np.frombuffer(f.read(), np.uint8, offset=16)
        return image.reshape(-1, self.img_shape[0], self.img_shape[1])

    def _load_label(self, key):
        filename = self.key_file[key]
        file_path = self.dataset_dir + '/' + filename
        with gzip.open(file_path, 'rb') as f:
            label = np.frombuffer(f.read(), np.uint8, offset=8)
        return label

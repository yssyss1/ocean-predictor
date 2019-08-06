from keras.utils import Sequence
import numpy as np


class SimpleGenerator(Sequence):
    """
    데이터 제너레이터
    data_x, data_y에서 매 iteration마다 batch_size만큼 가져옴
    """

    def __init__(self, data_x, data_y, batch_size, max_value, min_value):
        super(SimpleGenerator, self).__init__()
        self.data_x = data_x
        self.data_y = data_y
        self.batch_size = batch_size
        self.min_value = min_value
        self.max_value = max_value

    def __len__(self):
        return int(len(self.data_x) / self.batch_size)

    def __getitem__(self, idx):
        x_batch_train = (np.array(self.data_x[idx * self.batch_size:(idx + 1) * self.batch_size]) - self.min_value) / (
                    self.max_value - self.min_value)
        y_batch_train = (np.array(self.data_y[idx * self.batch_size:(idx + 1) * self.batch_size]) - self.min_value) / (
                    self.max_value - self.min_value)
        return x_batch_train, y_batch_train

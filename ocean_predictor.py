from predictor_model import ae_convlstm_model, naive_convlstm_model
from util import dataset_split
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from data_generator import SimpleGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.utils import plot_model
from save_callback import CustomCallback
import time


class OceanPredictor:
    def __init__(self, model_type, input_shape, look_back, look_forward, dataset, label, weight_path=None, exp_name=None, model_to_png=True):
        """
        :param model_type: 생성할 모델 종류
        :param input_shape: 입력 데이터 사이즈
        :param look_back: 시퀀스의 길이 ( 과거 몇 개의 데이터를 이용해 미래를 예측할 때 '과거 몇 개의 데이터'에 해당하는 길이 )
        :param look_forward: 예측하고자하는 값의 범위 ( 과거 몇 개의 데이터를 이용해 미래를 예측할 때 '미래'에 해당하는 길이 )
        :param dataset: 전체 데이터셋 ( train, validation, test split하지 않은 전체 데이터셋 )
        :param label: 데이터셋 라벨 ( sst, u10, v10 ... )
        :param weight_path: load할 weight 경로 - 없다면 처음부터 학습
        :param exp_name: experiment 이름 - 학습시 weight, loss curve, prediction 결과를 저장하기 위한 root folder로 없다면 현재 시각으로 폴더가 생성됨
        :param model_to_png: 생성된 모델을 png파일로 출력할지에 대한 플래그
        """
        self.model_type = model_type
        self.input_shape = input_shape
        self.image_shape = input_shape[-3:]
        self.look_back = look_back
        self.model = self.build_model()
        self.model.summary()
        self.dataset_arr = dataset_split(dataset, look_back, look_forward)
        self.max_list = np.max(self.get_data('train')[0], axis=(0, 1, 2, 3))
        self.min_list = np.min(self.get_data('train')[0], axis=(0, 1, 2, 3))
        self.save_path = os.path.join('.', exp_name) if exp_name is not None else os.path.join('.', str(time.time()))
        self.label = label
        os.makedirs(self.save_path, exist_ok=True)

        if weight_path is not None:
            if not os.path.exists(weight_path):
                raise FileNotFoundError('{} is not exist!'.format(weight_path))

            print("Load weight from {}".format(weight_path))
            self.model.load_weights(weight_path)

        if model_to_png:
            plot_model(self.model, os.path.join(self.save_path, 'model.png'), show_shapes=True)

    def build_model(self):
        """
        model type에 따라 모델을 반환해주는 함수
        """
        if self.model_type == 'naive_convlstm_model':
            return naive_convlstm_model(self.input_shape, self.image_shape[-1])
        elif self.model_type == 'ae_convlstm_model':
            return ae_convlstm_model(self.input_shape, self.image_shape[-1])
        else:
            raise ValueError('Check your model_type! This class only supports ["naive_convlstm_model", "ae_convlstm_model"] now')

    def get_data(self, data_type):
        """
        data_type(train, validation, test)에 따라 h5 파일에서 해당 범위의 데이터를 읽어오는 함수
        :param data_type: 데이터 종류 - train, validation, test
        """
        if data_type == 'train':
            return self.dataset_arr[0], self.dataset_arr[1]
        elif data_type == 'validation':
            return self.dataset_arr[2], self.dataset_arr[3]
        elif data_type == 'test':
            return self.dataset_arr[4], self.dataset_arr[5]
        else:
            raise ValueError('Check type argument!')

    def plot_one_point(self, y_gt, y_pred, lat, lon, plot_name, idx):
        """
        특정 위도 경도의 실제값, 예측값 비교 그래프를 그리는 함수
        :param y_gt: 실제값
        :param y_pred: 예측값
        :param lat: 위도
        :param lon: 경도
        :param plot_name: 그래프 저장 이름
        :param idx: 데이터 종류 번호 ( sst, u10, v10 ... )
        """
        save_path = os.path.join(self.save_path, self.label[idx].upper())
        os.makedirs(save_path, exist_ok=True)
        plt.clf()
        plt.plot(np.arange(y_gt.shape[0]), y_gt[:, lat, lon], 'b', label='real value')
        plt.plot(np.arange(y_pred.shape[0]), y_pred[:, lat, lon], 'r', label='prediction')
        plt.xlabel('Time')
        plt.ylabel(self.label[idx].upper())
        plt.legend()
        plt.savefig(os.path.join(save_path, '{}.png'.format(plot_name)))

    def calculate_mae_percent(self, prediction, gt, label):
        """
        원래 스케일, 정규화된 스케일에 대한 mae를 구해서 출력해주는 함수
        :param prediction: 예측값
        :param gt: 실제값
        :param label: 데이터 종류 ( sst, u10, v10 ... )
        """
        mae_original = np.average(np.abs((prediction-gt)), axis=(0, 1, 2))
        prediction_normalize = (prediction - self.min_list) / (self.max_list - self.min_list)
        gt_normalize = (gt - self.min_list) / (self.max_list - self.min_list)
        mae_normalize = np.average(np.abs(prediction_normalize-gt_normalize), axis=(0, 1, 2))
        print('{} result\n original scale: {} normalized scale: {}'.format(label, mae_original, mae_normalize))

    def compile(self, optimizer='adam', loss='mae'):
        """
        모델을 compile(optimizer와 loss function을 정의하는 작업)하기 위한 함수
        :param optimizer: 손실함수를 최적화 하기 위한 optimizer 종류
        :param loss: 손실함수 종류 ( mae, mse, cross-entropy ... )
        """
        self.model.compile(optimizer=optimizer, loss=loss)

    def train(self, batch, epochs):
        """
        train 데이터셋에 대해 모델을 학습시키는 함수.
        h5로 읽은 데이터셋을 제너레이터를 통해 배치사이즈만큼 가져오는 형태로 만듦. 데이터셋이 클 경우 메모리에 다 올라가지 않는 경우를 방지하기 위함
        train, validation 데이터셋에 대해 각각 제너레이터를 정의해서 keras 내부 함수인 fit_generator에 넣어줌
        :param batch: 배치 사이즈
        :param epochs: 전체 데이터셋에 대해 학습시키는 횟수
        """
        x_train, y_train = self.get_data('train')
        x_validation, y_validation = self.get_data('validation')
        validation_gen = SimpleGenerator(x_validation, y_validation, batch, self.max_list, self.min_list)
        train_gen = SimpleGenerator(x_train, y_train, batch, self.max_list, self.min_list)
        self.model.fit_generator(train_gen, len(train_gen), epochs=epochs, callbacks=[CustomCallback(self.model, self.save_path, validation_gen), ReduceLROnPlateau(monitor='loss')])

    def prediction(self, print_train_data=False, batch_size=10):
        """
        학습된 weight를 이용해 실제값, 예측값 비교 그래프와 mae 계산하기 위한 함수
        :param print_train_data: train 데이터에 대한 예측값, 실제값 비교 그래프와 mae를 계산여부 플래그
        :param batch_size: 배치 사이즈 ( 예측값 계산을 위한 배치사이즈 )
        """
        x_test, y_test = self.get_data('test')
        x_test = (np.array(x_test) - self.min_list) / (self.max_list - self.min_list)
        y_test = (np.array(y_test) - self.min_list) / (self.max_list - self.min_list)
        y_gt_arr = []
        y_prediction_arr = []

        for batch_idx in tqdm(range(len(x_test) // batch_size)):
            x_batch_test = x_test[batch_idx*batch_size:(batch_idx + 1)*batch_size]
            y_gt_arr.append(y_test[batch_idx*batch_size:(batch_idx + 1)*batch_size])
            y_prediction_arr.append(self.model.predict_on_batch(x_batch_test))

        y_gt_arr = np.array(y_gt_arr).reshape((-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])) * (self.max_list - self.min_list) + self.min_list
        y_prediction_arr = np.array(y_prediction_arr).reshape((-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])) * (self.max_list - self.min_list) + self.min_list

        for idx in tqdm(range(y_gt_arr.shape[-1])):
            for y in range(0, self.image_shape[0]):
                for x in range(0, self.image_shape[1]):
                    self.plot_one_point(y_gt_arr[..., idx], y_prediction_arr[..., idx], y, x, '{}_{}_{}_test'.format(self.label[idx],  y, x), idx)

        self.calculate_mae_percent(y_prediction_arr, y_gt_arr, 'test')

        if print_train_data:
            x_train, y_train = self.get_data('train')
            # 11680개가 8년치 데이터를 의미함. 4 * 365 * 8 = 11680. 전체 출력하면 너무 조밀해서 안보임.
            x_train, y_train = x_train[-11680:], y_train[-11680:]
            x_train = (np.array(x_train) - self.min_list) / (self.max_list - self.min_list)
            y_train = (np.array(y_train) - self.min_list) / (self.max_list - self.min_list)

            y_train_gt_arr = []
            y_train_prediction_arr = []

            for batch_idx in tqdm(range(len(x_train) // batch_size)):
                x_batch_test = x_train[batch_idx*batch_size:(batch_idx + 1)*batch_size]
                y_train_gt_arr.append(y_train[batch_idx*batch_size:(batch_idx + 1)*batch_size])
                y_train_prediction_arr.append(self.model.predict_on_batch(x_batch_test))

            y_train_gt_arr = np.array(y_train_gt_arr).reshape((-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])) * (self.max_list - self.min_list) + self.min_list
            y_train_prediction_arr = np.array(y_train_prediction_arr).reshape((-1, self.image_shape[0], self.image_shape[1], self.image_shape[2])) * (self.max_list - self.min_list) + self.min_list

            for idx in tqdm(range(y_train_gt_arr.shape[-1])):
                for y in range(0, self.image_shape[0]):
                    for x in range(0, self.image_shape[1]):
                        self.plot_one_point(y_train_gt_arr[..., idx], y_train_prediction_arr[..., idx], y, x, '{}_{}_{}_train'.format(self.label[idx],  y, x), idx)

            self.calculate_mae_percent(y_prediction_arr, y_gt_arr, 'train')

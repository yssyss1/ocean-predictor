from ocean_predictor import OceanPredictor
from keras.optimizers import Adam
from util import load_data
import numpy as np

if __name__ == '__main__':
    np.random.seed(5)
    dataset = load_data('./data_all.h5')
    label = ['sst', 'swh', 'mwd', 'mwp', 'u10', 'v10']

    """
    만약 특정 속성만 학습을 할 경우 이렇게 slicing하면 됨
    dataset = dataset[..., :3]
    label = ['sst', 'swh', 'mwd']
    """

    sample_length, sample_height, sample_width, channel = dataset.shape
    look_back = 240  # ecmwf는 6시간 간격이므로 240 / 4 = 60일 데이터를 하나의 입력값으로 취급함
    look_forward = 60  # ecmwf는 6시간 간격이므로 240 / 4 = 15일 뒤의 데이터를 예측함
    batch_size = 256
    epoch = 1
    optimizer_lr = 1e-2
    optimizer_lr_decay = 1e-3
    loss = 'mse'
    model_type = 'ae_convlstm_model'

    predictor = OceanPredictor(model_type=model_type,
                               input_shape=(batch_size, look_back, sample_height, sample_width, channel),
                               look_back=look_back,
                               label=label,
                               look_forward=look_forward,
                               dataset=dataset
                               )
    predictor.compile(optimizer=Adam(lr=optimizer_lr, decay=optimizer_lr_decay), loss=loss)
    predictor.train(batch=batch_size, epochs=epoch)
    predictor.prediction(print_train_data=True, batch_size=batch_size)

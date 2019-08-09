from ocean_predictor import OceanPredictor
from keras.optimizers import Adam
from util import load_data, is_exist
import numpy as np
import json
import os
import time
import shutil


if __name__ == '__main__':
    hyperparameter_json_path = './hyperparameters.json'
    is_exist(hyperparameter_json_path, 'hyperparameter json file')

    with open(hyperparameter_json_path) as json_file:
        """
        주의 - json은 문자를 표현할 때 작은 따옴표(')를 못 씀, 큰 따옴표(")만 사용 가능함 
        
        Hyperparameter는 hyperparameter_json_path에 위치한 json파일 내부 변수에서 세팅하면 됨

        null로 주어져도 되는 것들 (python에서 로드될 때는 null이 None으로 변환됨)
            exp_name => null로 주어질 경우 현재 시간을 root 폴더 이름으로 사용함 
            weight_path => null로 주어질 경우 학습을 처음부터 수행함
            random_seed => null로 주어질 경우 weight initialize가 매 학습마다 다르게 됨
        
        학습 할 때마다 exp_name 내부에 다음과 같은 파일들이 만들어짐
            exp_name ---- | SST ---- | train, test 데이터셋에 대한 예측값, 실제값 비교 그래프 
                     ---- | SWH ---- | train, test 데이터셋에 대한 예측값, 실제값 비교 그래프 
                     ---- | ... ( 다른 label 데이터들 )
                     ---- | ... ( 다른 label 데이터들 )
                     ---- | ... ( 다른 label 데이터들 )
                     ---- | model.png ( 모델 구조 이미지 )
                     ---- | hyperparameters.json ( 해당 학습할 때 사용한 hyperparameter )
                     ---- | loss.png ( train, validation 데이터셋에 대한 손실 함수 그래프 )
                     ---- | result_mae.txt ( 학습 결과 train, test 데이터셋에 대한 mae, 정규화된 mae 값을 저장한 txt 파일 )
        
        학습이 완료될 경우 train, test 데이터셋의 원래 스케일 mae, 정규화된 mae가 console 창 및 result_mae.txt 파일에 출력됨
        
        만약 특정 속성만 학습을 할 경우 다음과 같이 slicing하면 됨. label은 slicing된 데이터에 맞춰서 넘겨줘야 함
            dataset = dataset[..., :3]
            label = ['sst', 'swh', 'mwd']

        Train, Validation, Test는 전체 데이터셋에서 93%, 4%, 3% 비율로 split해서 사용하고 있음
        각 비율을 바꾸려면 util.py의 dataset_split 함수 인자를 바꾸기 바람
        """
        json_data = json.load(json_file)  # 학습에 사용할 Hyperparameter를 담고있는 json파일을 로드
        exp_name = json_data['exp_name']  # experiment 이름; 학습시 weight, loss curve, prediction 결과를 저장하기 위한 root 폴더 이름
        data_path = json_data['data_path']  # 학습 데이터(h5 파일) 경로
        label = json_data['label']  # 데이터셋 라벨 ( sst, u10, v10 ... )
        # 시퀀스의 길이 ( 과거 몇 개의 데이터를 이용해 미래를 예측할 때 '과거 몇 개의 데이터'에 해당하는 길이 )
        # ecmwf는 6시간 간격이므로 look_back / 4 일 데이터를 하나의 입력값으로 취급함
        look_back = json_data['look_back']
        # 예측하고자하는 값의 범위 ( 과거 몇 개의 데이터를 이용해 미래를 예측할 때 '미래'에 해당하는 길이 )
        # ecmwf는 6시간 간격이므로 look_forward / 4 일 뒤의 데이터를 예측함
        look_forward = json_data['look_forward']
        batch_size = json_data['batch_size']  # 학습 배치 사이즈
        epoch = json_data['epoch']  # 학습 에폭 수
        optimizer_lr = json_data['optimizer_lr']  # learning rate
        optimizer_lr_decay = json_data['oprtimizer_lr_decay']  # 매 iteration마다 learning rate를 줄이는 비율
        loss = json_data['loss']  # 사용할 손실 함수
        model_type = json_data['model_type']  # 생성할 모델 종류
        weight_path = json_data['weight_path']  # 로드할 weight 경로
        random_seed = json_data['random_seed']  # weight initialize를 동일하게 하기 위한 seed 값
        # 학습은 수행하지 않고 이미 학습된 weight를 이용해 prediction만 할 경우에 대한 플래그; True일 경우 weight_path가 None이면 안됨
        only_prediction = json_data['only_prediction']

    if random_seed is not None and type(random_seed) is int:
        np.random.seed(random_seed)

    dataset = load_data(data_path)
    sample_length, sample_height, sample_width, channel = dataset.shape

    save_path = exp_name if exp_name is not None else str(time.time())
    if os.path.exists(save_path):
        raise FileExistsError('Check your exp_name. It\'s duplicated!')
    os.makedirs(save_path)
    shutil.copy2(hyperparameter_json_path, os.path.join(save_path, os.path.basename(hyperparameter_json_path)))

    if only_prediction and weight_path is None:
        raise ValueError('For prediction, pretrained weights need to be given!')

    predictor = OceanPredictor(model_type=model_type,
                               input_shape=(batch_size, look_back, sample_height, sample_width, channel),
                               look_back=look_back,
                               label=label,
                               look_forward=look_forward,
                               dataset=dataset,
                               save_path=save_path,
                               weight_path=weight_path,
                               )
    predictor.compile(optimizer=Adam(lr=optimizer_lr, decay=optimizer_lr_decay), loss=loss)
    if not only_prediction:
        predictor.train(batch=batch_size, epochs=epoch)
    predictor.prediction(print_train_data=True, batch_size=batch_size)

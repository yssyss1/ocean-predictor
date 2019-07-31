import os
import h5py
import numpy as np


def build_hdf5(arr, save_path, data_label='data'):
    """
    array를 h5 파일 형태로 저장
    :param arr: 데이터셋 array
    :param save_path: h5 저장 경로
    :param data_label: h5 파일로 저장할 때 label ( 저장하고 불러올 때 label로 동일한 이름을 사용해주면 됨 )
    """
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    with h5py.File(save_path, 'w') as f:
        f.create_dataset(data_label, data=arr)


def load_data(path, data_label='data'):
    """
    h5 파일에서 데이터셋을 불러옴
    :param path: 불러올 h5 파일 경로
    :param data_label: h5 파일에서 불러오고자 하는 데이터 라벨
    :return:
    """
    if not os.path.exists(path):
        raise FileNotFoundError('Check your data path!')

    with h5py.File(path, 'r') as f:
        return f[data_label][:].astype(np.float32)


def dataset_split(dataset,
                  look_back,
                  look_forward,
                  train_dataset_range=(0, 0.93),
                  validation_dataset_range=(0.93, 0.97),
                  test_dataset_range=(0.97, 1.0)):
    """
    전체 데이터셋을 train, validation, test 데이터셋으로 분리함
    """
    dataset_len = len(dataset)
    dataset_arr = []

    for ratio in [train_dataset_range, validation_dataset_range, test_dataset_range]:
        split_dataset = dataset[int(dataset_len*ratio[0]):int(dataset_len*ratio[1])]
        dataset_x, dataset_y = [], []

        for idx in range(len(split_dataset) - look_back - look_forward):
            dataset_x.append(split_dataset[idx:(idx+look_back)])
            dataset_y.append(split_dataset[idx+look_back+look_forward-1])

        dataset_arr.append(dataset_x)
        dataset_arr.append(dataset_y)

    return dataset_arr
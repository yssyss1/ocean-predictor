import netCDF4
import numpy as np
import csv
from glob import glob
import os
from tqdm import tqdm
import h5py
import datetime


class ECMWFParser:
    @staticmethod
    def ecmwf_parsing_to_csv(save_path, nc_foler, longitude_range, latitude_range, label):
        """
        ecmwf를 읽어 csv파일로 저장하는 함수
        :param save_path: csv 파일 저장 경로
        :param nc_folder: nc file들이 저장된 폴더 경로
        :param longitude_range: 추출하고자 하는 경도 범위
        :param latitude_range: 추출하고자 하는 위도 범위
        :param label: nc file에 저장된 데이터 라벨 ex) ['time', 'longitude', 'latitude', 'sst', 'swh', 'mwd', 'mwp', 'u10', 'v10']
        """
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        csv_file = open(save_path, 'w')
        csv_writer = csv.writer(csv_file)
        # ecmwf label들 확인하고 저장할 데이터 label 형식 바꿀 것
        csv_writer.writerow(label)
        file_names = glob(os.path.join(nc_foler, '*.nc'))
        # nc file 이름 기준으로 sorting - auto_downloader_ecmwf.py를 통해 다운받은 경우 년_월.nc로 다운받아짐. 이를 기준으로 년 * 12 + 월로 오름차순 정렬
        file_names.sort(key=lambda x: int(x.split('/')[-1].strip('.nc').split('_')[0]) * 12 + int(x.split('/')[-1].strip('.nc').split('_')[1]))

        dataset_file = []
        for file in tqdm(file_names, desc='Load dataset'):
            dataset_file.append(netCDF4.Dataset(file))

        start_time = datetime.datetime(1900, 1, 1, 0, 0, 0)
        for lon_idx in np.arange(longitude_range[0], longitude_range[1]+1):
            for lat_idx in np.arange(latitude_range[0], latitude_range[1]+1):
                rows = []
                for f in dataset_file:
                    longitude = f.variables['longitude']
                    latitude = f.variables['latitude']
                    sst = f.variables['sst']
                    swh = f.variables['swh']
                    mwd = f.variables['mwd']
                    mwp = f.variables['mwp']
                    u10 = f.variables['u10']
                    v10 = f.variables['v10']
                    dataset_length = sst.shape[0]

                    for time_step in range(dataset_length):
                        time_data = start_time + datetime.timedelta(hours=int(f.variables['time'][time_step]))
                        single_row = ['{}_{}_{}_{}'.format(time_data.year, time_data.month, time_data.day, time_data.time().hour),
                                     longitude[lon_idx].data,
                                     latitude[lat_idx].data,
                                     sst[time_step, lat_idx, lon_idx],
                                     swh[time_step, lat_idx, lon_idx],
                                     mwd[time_step, lat_idx, lon_idx],
                                     mwp[time_step, lat_idx, lon_idx],
                                     u10[time_step, lat_idx, lon_idx],
                                     v10[time_step, lat_idx, lon_idx],
                                     ]
                        rows.append(single_row)
                csv_writer.writerows(rows)
        csv_file.close()

    @staticmethod
    def csv_to_h5(csv_path, save_path, longitude_range, latitude_range, label):
        """
        csv 파일에서 데이터를 추출해 학습을 위한 h5파일을 생성하는 함수
        :param csv_path: csv 파일 경로
        :param save_path: h5 파일 저장 경로
        :param longitude_range: 추출하고자 하는 경도 범위
        :param latitude_range: 추출하고자 하는 위도 범위
        :param label: nc file에 저장된 데이터 라벨 ex) ['time', 'longitude', 'latitude', 'sst', 'swh', 'mwd', 'mwp', 'u10', 'v10']
        """
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))

        csv_file = open(csv_path, 'r')
        csv_reader = csv.reader(csv_file)
        initial_check = True # csv 라벨 정보 제외하기 위함
        arr = []
        longitude_length = longitude_range[1] - longitude_range[0] + 1
        latitude_length = latitude_range[1] - latitude_range[0] + 1
        label_length = len(label) - 3
        for line in tqdm(csv_reader, desc='Read CSV...'):
            if initial_check:
                initial_check = False
                continue
            single_arr = np.array(line[3:])
            single_arr[single_arr == '--'] = '0' # 결손치, 육지는 '--'로 표시됨. 향후 0 보다 학습에 영향을 안주는 값으로 변경해야 함
            arr.append(single_arr.astype('float32'))
        # 시간, 경도, 위도, 데이터 순으로 데이터셋을 정렬함
        arr = np.array(arr).reshape((longitude_length, latitude_length, -1, label_length))
        arr = arr.transpose((2, 0, 1, 3))

        with h5py.File(save_path, 'w') as h5_f:
            h5_f.create_dataset('data', data=arr)
        csv_file.close()


if __name__ == '__main__':
    ECMWFParser.ecmwf_parsing_to_csv('/home/seok/abcd/ecmwf_all.csv',
                                     '/home/seok/ncnc',
                                     (160, 162),
                                     (71, 73),
                                     ['time', 'longitude', 'latitude', 'sst', 'swh', 'mwd', 'mwp', 'u10', 'v10']
                                     )

    ECMWFParser.csv_to_h5('/home/seok/abcd/ecmwf_all.csv',
                          './ecmwf.h5',
                          (160, 162),
                          (71, 73),
                          ['time', 'longitude', 'latitude', 'sst', 'swh', 'mwd', 'mwp', 'u10', 'v10']
                          )
import cv2
import os
import numpy as np
import h5py
from util.Deprecated.WaveSimulation import Wave2d


def build_hdf5(arr, save_path, data_label='data'):
    os.makedirs(save_path, exist_ok=True)
    hdf5_file = os.path.join(save_path, 'data.h5')

    with h5py.File(hdf5_file, 'w') as f:
        f.create_dataset(data_label, data=arr)


def make_video_from_images(image_path, vidoe_path='output.avi', fps=60):
    images = [img for img in os.listdir(image_path) if img.endswith(".jpg")]
    images.sort(key=lambda x: int(x.split('.')[0]))
    frame = cv2.imread(os.path.join(image_path, images[0]))
    height, width, channel = frame.shape
    video = cv2.VideoWriter(vidoe_path, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_path, image)))

    cv2.destroyAllWindows()
    video.release()


def make_simulation_wave_dataset(save_path, data_label):
    wave_eq = Wave2d(4, 4, 6, 300, 300, 3000, 1)

    # Initial value functions
    f = lambda x, y: np.exp(-10 * (x ** 2 + y ** 2))
    g = lambda x, y: 0

    # Solve
    u = np.array(wave_eq.solve(f, g))
    u = np.expand_dims(np.transpose(u, (2, 0, 1)), axis=-1)
    build_hdf5(u, save_path, data_label)

from mayavi import mlab
import os


class AnimationWave:
    def __init__(self, wave_1, wave_2, vis_type=None):
        if wave_1.shape != wave_2.shape:
            raise ValueError('wave_1 and wave_2 must have same shapes!')

        self.wave_1 = wave_1
        self.wave_2 = wave_2
        self.vis_type = vis_type
        if self.vis_type is None:
            self.wave1_plt = mlab.surf(wave_1[0, :, :], color=(0, 0, 0.5), opacity=0.3)
            self.wave2_plt = mlab.surf(wave_2[0, :, :], color=(0.5, 0, 0), opacity=0.3)
        elif self.vis_type is 'diff':
            self.wave_plt = mlab.surf(wave_1[0, :, :]-wave_2[0, :, :], color=(0, 0, 0.5), opacity=1)
        self.sample_length = wave_1.shape[0]
        self.save_path = './result/animation'
        os.makedirs(self.save_path, exist_ok=True)

    def start(self):
        self.__animation()
        mlab.show()

    @mlab.animate(delay=10)
    def __animation(self):
        while True:
            for i in range(self.sample_length):
                print('Updating scene...')
                if self.vis_type is None:
                    self.wave1_plt.mlab_source.scalars = self.wave_1[i, :, :]
                    self.wave2_plt.mlab_source.scalars = self.wave_2[i, :, :]
                elif self.vis_type is 'diff':
                    self.wave_plt.mlab_source.scalars = self.wave_1[i, :, :] - self.wave_2[i, :, :]
                mlab.savefig(os.path.join(self.save_path, '{}.jpg'.format(i)))
                yield

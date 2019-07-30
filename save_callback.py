import matplotlib.pyplot as plt
import os
from keras.callbacks import Callback


class CustomCallback(Callback):
    """
    epoch이 끝날 때마다 loss 그래프와 weight를 저장하기 위한 콜백
    save_path 내부에 저장이 됨
    """
    def __init__(self, model, save_path, val_gen):
        super(CustomCallback, self).__init__()
        self.model = model
        self.val_gen = val_gen
        self.save_path = save_path
        self.val_loss = []
        self.train_loss = []

    def on_epoch_end(self, epoch, logs=None):
        self.train_loss.append(logs['loss'])
        self.val_loss.append(self.model.evaluate_generator(self.val_gen, len(self.val_gen)))

        plt.clf()
        plt.plot(list(range(epoch+1)), self.val_loss, label='val')
        plt.plot(list(range(epoch+1)), self.train_loss, label='train')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'loss.png'))

        self.model.save_weights(os.path.join(self.save_path, 'model_weights_{}_epoch.h5'.format(epoch)))
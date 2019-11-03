from mnist import MNIST
import numpy as np

class MnistLoader:
    def __init__(self, path):
        self.data = MNIST(path)

    def get_training_data(self):
        images, labels = self.data.load_training()
        return self.prepare_data(images, labels)

    def get_testing_data(self):
        images, labels = self.data.load_testing()
        return self.prepare_data(images, labels)


    def prepare_data(self, images, labels):
        return [(np.array(image) / 256, self._prepare_label(label)) for image, label in zip(images, labels)]

    def _prepare_label(self, label):
        output = np.zeros(10)
        output[label] = 1.0
        return output
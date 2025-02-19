import os
import struct
import kagglehub
import numpy as np
from array import array
import torch


class MnistDataLoader:

    def __init__(self, train_image, train_label, test_image, test_label):
        self.train_image = train_image
        self.train_label = train_label
        self.test_image = test_image
        self.test_label = test_label

    def read_images_labels(self, img_path, label_path):

        # read images
        with open(label_path, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f"Magic number mismatch, expected 2049, got {magic}")
            labels = np.array(array("B", file.read()))

        # read labels
        with open(img_path, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f"Magic number mismatch, expected 2051, got {magic}")
            image_data = np.array(array("B", file.read())).reshape(size, rows, cols)

        return image_data, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.train_image, self.train_label)
        x_test, y_test = self.read_images_labels(self.test_image, self.test_label)

        return (x_train, y_train), (x_test, y_test)


def load_dataset():
    path = kagglehub.dataset_download("hojjatk/mnist-dataset")

    train_image = os.path.join(path, "train-images.idx3-ubyte")
    train_label = os.path.join(path, "train-labels.idx1-ubyte")
    test_image = os.path.join(path, "t10k-images.idx3-ubyte")
    test_label = os.path.join(path, "t10k-labels.idx1-ubyte")

    dataloader = MnistDataLoader(
        train_image,
        train_label,
        test_image,
        test_label,
    )
    return dataloader.load_data()  # (x_train, y_train), (x_test, y_test)


def preprocess_data(x, y, num_classes=10):
    x = x.reshape(-1, 784)
    x = x / 255
    x = torch.tensor(x, dtype=torch.float32)
    y = np.eye(num_classes)[y]
    y = torch.tensor(y, dtype=torch.float32)
    return x, y

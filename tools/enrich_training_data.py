import numpy as np
from scipy.ndimage import shift
from skimage.transform import rotate
import random


def shuffle_arrays(train_array, labels_array):
    indices = np.arange(train_array.shape[0])
    np.random.shuffle(indices)
    train_array = train_array[indices]
    labels_array = labels_array[indices]
    return train_array, labels_array


def enrich_training_data(training_data):
    enriched_training_data = []
    for data in training_data:
        random_rotation = random.randint(0, 360)
        x_shifted = random.randint(-10, 10)
        y_shifted = random.randint(-10, 10)
        data = shift(data, shift=[x_shifted, y_shifted])
        data = rotate(data, angle=random_rotation)
        enriched_training_data.append(data)

    training_data = np.append(training_data, enriched_training_data, axis=0)
    shuffle_arrays(training_data, training_data)
    return training_data


def enrich_labels(labels):
    labels = np.append(labels, labels, axis=0)
    return labels

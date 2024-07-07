import numpy as np
from skimage.transform import resize, rotate
import random


def enrich_training_data(training_data):
    enriched_training_data = []
    for data in training_data:
        random_rotation = random.randint(0, 360)
        data = rotate(data, angle=random_rotation)
        enriched_training_data.append(data)

    training_data = np.append(training_data, enriched_training_data, axis=0)
    return training_data


def enrich_labels(labels):
    labels = np.append(labels, labels, axis=0)
    return labels

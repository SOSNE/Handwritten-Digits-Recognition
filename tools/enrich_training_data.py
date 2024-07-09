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


def is_scientific_notation(number):
    # Convert number to string and check if it contains 'e' or 'E'
    number_str = f"{number:.16e}"  # Format to scientific notation with 16 decimal places
    return 'e' in number_str or 'E' in number_str


def round_array(array):
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            number_str = f"{array[i, j]}"
            if 'e' in number_str or 'E' in number_str:
                array[i, j] = 0
    return array


def enrich_training_data(training_data):
    enriched_training_data = []
    for data in training_data:
        random_rotation = random.randint(-30, 30)

        data = rotate(data, angle=random_rotation)

        x_shifted = random.randint(-10, 10)
        y_shifted = random.randint(-10, 10)
        x_iterations = x_shifted
        for _ in range(abs(x_iterations)):
            data_test = shift(data, shift=[x_shifted, y_shifted])
            data_test = round_array(data_test)
            for index in range(data_test.shape[0] - 1):
                if data_test[0][index] != 0:
                    x_shifted += 1
                    break
                elif data_test[-1][index] != 0:
                    x_shifted -= 1
                    break

        y_iterations = y_shifted
        for _ in range(abs(y_iterations)):
            data_test = shift(data, shift=[x_shifted, y_shifted])
            data_test = round_array(data_test)
            first_column = [row[0] for row in data_test]
            last_column = [row[-1] for row in data_test]
            for index in range(data_test.shape[1] - 1):
                if first_column[index] != 0:
                    y_shifted += 1
                    break
                elif last_column[index] != 0:
                    y_shifted -= 1
                    break
        data = shift(data, shift=[x_shifted, y_shifted])
        data = round_array(data)

        enriched_training_data.append(data)

    training_data = np.append(training_data, enriched_training_data, axis=0)
    shuffle_arrays(training_data, training_data)
    return training_data


def enrich_labels(labels):
    labels = np.append(labels, labels, axis=0)
    return labels

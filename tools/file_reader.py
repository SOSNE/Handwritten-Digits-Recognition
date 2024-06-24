import numpy as np
import struct


def read_file_image(filename):
    with open(filename, 'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2051:
            print('Magic number mismatch!')

        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]

        image_data = f.read(num_images * num_rows * num_cols)
        images = np.frombuffer(image_data, dtype=np.uint8).reshape(num_images, num_rows, num_cols)

        return images


def read_file_label(filename):
    with open(filename, 'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        if magic_number != 2049:
            print('Magic number mismatch!')

        num_labels = struct.unpack('>I', f.read(4))[0]

        labels_data = f.read(num_labels)
        labels_data = np.frombuffer(labels_data, dtype=np.uint8).reshape(num_labels, 1)

        return labels_data

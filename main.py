import matplotlib.pyplot as plt
from tools.file_reader import read_file_image, read_file_label
import random

images = read_file_image("data/t10k-images-idx3-ubyte")
labels = read_file_label("data/t10k-labels-idx1-ubyte")

image_index = random.randint(0, len(images) - 1)

print("Label:", labels[image_index])
plt.imshow(images[image_index], cmap='gray')
plt.show()


import matplotlib.pyplot as plt
from tools.file_reader import read_file_image, read_file_label
from tools.model import predict_output
from tools.utils import load_model
from tools.draw import start_drawing
import random

# images = read_file_image("data/t10k-images-idx3-ubyte")
labels = read_file_label("data/t10k-labels-idx1-ubyte")

model = load_model("store/model.json")

img = start_drawing()

predicted_value, _ = predict_output(img, model[0], model[1], model[2], labels[0])

print("Predicted label:", predicted_value)

plt.imshow(img, cmap='gray')
plt.show()

# image_index = random.randint(0, len(images) - 1)

# print("Label:", labels[image_index])
# predicted_value, _ = predict_output(images[image_index], model[0], model[1], model[2], labels[image_index])
# print("Predicted label:", predicted_value)

# plt.imshow(images[image_index], cmap='gray')
# plt.show()

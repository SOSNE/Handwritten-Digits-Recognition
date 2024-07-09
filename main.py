import matplotlib.pyplot as plt
from tools.file_reader import read_file_image, read_file_label
from tools.model import predict_output
from tools.utils import load_model
from tools.enrich_training_data import shuffle_arrays
# from tools.draw import start_drawing
import random

images = read_file_image("data/t10k-images-idx3-ubyte")
labels = read_file_label("data/t10k-labels-idx1-ubyte")

# images, labels = shuffle_arrays(images, labels)

model = load_model("store/model.json")

image_index = random.randint(0, len(images) - 1)


print("Label:", labels[image_index])
predicted_value, _ = predict_output(images[image_index], model[0], model[1], model[2], labels[image_index])
print("Predicted label:", predicted_value)

# bad_images = []

# img = start_drawing()
#
# predicted_value, _ = predict_output(img, model[0], model[1], [10, 10], labels[0])
#
# print("Predicted label:", predicted_value)

# for index in range(len(images)-1):
#     # image_index = random.randint(0, len(images) - 1)
#
#     predicted_value, _ = predict_output(images[index], model[0], model[1], model[2], labels[index])
#
#     if predicted_value != labels[index]:
#         print("Label:", labels[index])
#         print("Predicted label:", predicted_value)
#         bad_images.append({f"correct: {labels[index]} not: ": predicted_value})
#
# print(len(images))
# print(len(bad_images))
print("Label:", labels[image_index])
plt.imshow(images[image_index], cmap='gray')
plt.show()


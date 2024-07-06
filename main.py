import matplotlib.pyplot as plt
from tools.file_reader import read_file_image, read_file_label
from tools.model import predict_output
from tools.utils import load_model
import random

images = read_file_image("data/t10k-images-idx3-ubyte")
labels = read_file_label("data/t10k-labels-idx1-ubyte")

model = load_model("store/weights.json")
bad_images = []

for index in range(len(images)-1):
    # image_index = random.randint(0, len(images) - 1)

    predicted_value, _ = predict_output(images[index], model[0], model[1], [10, 10], labels[index])

    if predicted_value != labels[index]:
        print("Label:", labels[index])
        print("Predicted label:", predicted_value)
        bad_images.append({f"correct: {labels[index]} not: ": predicted_value})

print(len(images))
print(len(bad_images))
# plt.imshow(images[image_index], cmap='gray')
# plt.show()


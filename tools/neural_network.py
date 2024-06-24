import numpy as np
from tools.file_reader import read_file_image, read_file_label

hidden_layers = 1
input_nodes = 784
hidden_nodes = 785
output_nodes = 10
input_data = read_file_image("../data/t10k-images-idx3-ubyte")
labels = read_file_label("../data/t10k-labels-idx1-ubyte")
input_data = input_data.reshape(input_data.shape[0], input_data.shape[1] * input_data.shape[2])
# convert data to range form 0 to 1
input_data = input_data / 255.0
# first weight that is connected to 785 hidden layers.
weight_first = np.random.randn(input_nodes, hidden_nodes)
# rest of weights that are connected to 784 hidden layers.
# weights_rest = np.random.rand(hidden_layers, input_nodes, input_nodes)
# last weight that is connected to a 10 output nodes.
weight_last = np.random.randn(hidden_nodes, output_nodes)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


hidden_layer_values = input_data.dot(weight_first)
sigmoid_hidden_layer_values = sigmoid(hidden_layer_values)

output_values = sigmoid_hidden_layer_values.dot(weight_last)
sigmoid_output_values = sigmoid(output_values)

output_values_true = np.zeros((labels.size, 10))
output_values_true[np.arange(labels.size), labels.flatten()] = 1

loss = np.square(sigmoid_output_values - output_values_true).sum()

print(sigmoid_output_values)
print(type(output_values_true))

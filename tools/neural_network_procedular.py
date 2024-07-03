import numpy as np
from tools.file_reader import read_file_image, read_file_label


input_nodes = 784
hidden_nodes = 10
output_nodes = 10
input_data = read_file_image("data/t10k-images-idx3-ubyte")
labels = read_file_label("data/t10k-labels-idx1-ubyte")
input_data = input_data.reshape(input_data.shape[0], input_data.shape[1] * input_data.shape[2])
# convert data to range form 0 to 1
input_data = input_data / 255.0


def generate_weights_and_bias(hidden_layers_diagram):
    all_weights = []
    all_bias = []
    for index, hidden_layer_nodes in enumerate(hidden_layers_diagram):
        if index == 0:
            weight = np.random.randn(input_nodes, hidden_layer_nodes)
            all_weights.append(weight)
            weight = np.random.randn(hidden_layer_nodes, hidden_layers_diagram[index + 1])
            bias = np.full((hidden_layer_nodes,), 0.1)
        elif index != len(hidden_layers_diagram) - 1:
            weight = np.random.randn(hidden_layer_nodes, hidden_layers_diagram[index + 1])
            bias = np.full((hidden_layer_nodes,), 0.1)
        else:
            weight = np.random.randn(hidden_layer_nodes, output_nodes)
            bias = np.full((hidden_layer_nodes,), 0.1)
        all_weights.append(weight)
        all_bias.append(bias)
    return all_weights, all_bias

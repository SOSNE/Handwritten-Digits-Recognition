import numpy as np
from tools.file_reader import read_file_image, read_file_label


input_nodes = 784
output_nodes = 10
input_data = read_file_image("../data/t10k-images-idx3-ubyte")
labels = read_file_label("../data/t10k-labels-idx1-ubyte")
input_data = input_data.reshape(input_data.shape[0], input_data.shape[1] * input_data.shape[2])
# convert data to range form 0 to 1
input_data = input_data / 255.0

output_values_true = np.zeros((labels.size, 10))
output_values_true[np.arange(labels.size), labels.flatten()] = 1


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def generate_weights_and_bias(hidden_layers_diagram):
    all_weights = []
    all_biases = []
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
            print("last weight shape: ", weight.shape)
            bias = np.full((hidden_layer_nodes,), 0.1)
            all_biases.append(bias)
            bias = np.full((output_nodes,), 0.1)
        all_weights.append(weight)
        all_biases.append(bias)
    return all_weights, all_biases

# [2, 3, 1]


def forward_propagation(weights, biases, hidden_layers_diagram):
    previous_dot_product = 0
    hidden_layer_values_all = []
    hidden_layer_values_sigmoid_all = []
    for index in range(len(hidden_layers_diagram)+1):
        print(index)
        if index == 0:
            hidden_layer_values = input_data.dot(weights[index]) + biases[index]
            hidden_layer_values_sigmoid = sigmoid(hidden_layer_values)
            previous_dot_product = hidden_layer_values_sigmoid

        else:
            print(index, "shape is: ", previous_dot_product.shape)
            hidden_layer_values = previous_dot_product.dot(weights[index]) + biases[index]
            hidden_layer_values_sigmoid = sigmoid(hidden_layer_values)
            previous_dot_product = hidden_layer_values_sigmoid
        hidden_layer_values_all.append(hidden_layer_values)
        hidden_layer_values_sigmoid_all.append(hidden_layer_values_sigmoid)
    return previous_dot_product, hidden_layer_values_all, hidden_layer_values_sigmoid_all


def backward_propagation(weights, biases, hidden_layers_diagram):
    _, hidden_layer_values_all, hidden_layer_values_sigmoid_all = forward_propagation(
        weights, biases, hidden_layers_diagram)
    hidden_layers_number = len(hidden_layers_diagram)
    gradient_weights_all = []
    gradient_bias_all = []
    previous_gradient_derivative_values = 0
    for index in range(hidden_layers_number, -1, -1):

        if index == hidden_layers_number:
            gradient_sigmoid_output_values_loss = 2 * (hidden_layer_values_sigmoid_all[index] - output_values_true)
            gradient_derivative_values = (gradient_sigmoid_output_values_loss *
                                      sigmoid_derivative(hidden_layer_values_all[index]))

            gradient_weight = np.dot(gradient_derivative_values.T, hidden_layer_values_sigmoid_all[index-1]).T
            gradient_bias = np.sum(gradient_derivative_values, 0)
            previous_gradient_derivative_values = gradient_derivative_values
        elif index != 0:
            gradient_hidden_loss = np.dot(previous_gradient_derivative_values, weights[index+1].T)
            gradient_derivative_values = (gradient_hidden_loss * sigmoid_derivative(hidden_layer_values_all[index]))

            gradient_weight = np.dot(gradient_derivative_values.T, hidden_layer_values_sigmoid_all[index-1]).T
            gradient_bias = np.sum(gradient_derivative_values, 0)
            previous_gradient_derivative_values = gradient_derivative_values
        else:
            gradient_hidden_loss = np.dot(previous_gradient_derivative_values, weights[index+1].T)
            gradient_derivative_values = (gradient_hidden_loss * sigmoid_derivative(hidden_layer_values_all[index]))

            gradient_weight = np.dot(input_data.T, hidden_layer_values_sigmoid_all[index - 1]).T
            gradient_bias = np.sum(gradient_derivative_values, 0)

        gradient_weights_all.append(gradient_weight)
        gradient_bias_all.append(gradient_bias)


diagram = [22, 32, 11]
weights, bias = generate_weights_and_bias(diagram)
backward_propagation(weights, bias, diagram)

# data = forward_propagation(weights, bias, diagram)
# print(data)

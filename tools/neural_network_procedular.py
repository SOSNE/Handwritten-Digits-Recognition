import numpy as np
from tools.file_reader import read_file_image, read_file_label
from tools.utils import save_model
import time

input_nodes = 784
output_nodes = 10
input_data = read_file_image("../data/train-images-idx3-ubyte")
labels = read_file_label("../data/train-labels-idx1-ubyte")
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


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def generate_weights_and_bias(hidden_layers_diagram):
    all_weights = []
    all_biases = []
    for index, hidden_layer_nodes in enumerate(hidden_layers_diagram):
        if len(hidden_layers_diagram) == 1:
            weight = np.random.randn(input_nodes, hidden_layer_nodes)
            all_weights.append(weight)
            bias = np.full((hidden_layer_nodes,), 0.1)
            all_biases.append(bias)
            weight = np.random.randn(hidden_layer_nodes, output_nodes)
            all_weights.append(weight)
            bias = np.full((output_nodes,), 0.1)
            all_biases.append(bias)
        elif index == 0:
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
            all_biases.append(bias)
            bias = np.full((output_nodes,), 0.1)
        all_weights.append(weight)
        all_biases.append(bias)
    return all_weights, all_biases


def forward_propagation(weights, biases, hidden_layers_diagram,
                        activation_function):
    start_time = time.time()
    previous_dot_product = 0
    hidden_layer_values_all = []
    hidden_layer_values_activation_all = []
    for index in range(len(hidden_layers_diagram) + 1):
        if index == 0:
            hidden_layer_values = input_data.dot(weights[index]) + biases[index]
            hidden_layer_values_sigmoid = activation_function(hidden_layer_values)
            previous_dot_product = hidden_layer_values_sigmoid
        elif index != len(hidden_layers_diagram) + 1:
            hidden_layer_values = previous_dot_product.dot(weights[index]) + biases[index]
            hidden_layer_values_sigmoid = activation_function(hidden_layer_values)
            previous_dot_product = hidden_layer_values_sigmoid
        else:
            hidden_layer_values = previous_dot_product.dot(weights[index]) + biases[index]
            hidden_layer_values_sigmoid = sigmoid(hidden_layer_values)
            previous_dot_product = hidden_layer_values_sigmoid
        hidden_layer_values_all.append(hidden_layer_values)
        hidden_layer_values_activation_all.append(hidden_layer_values_sigmoid)
    loss = np.square(previous_dot_product - output_values_true).sum()
    return loss, previous_dot_product, hidden_layer_values_all, hidden_layer_values_activation_all, start_time


def backward_propagation(weights, biases, hidden_layers_diagram, activation_function, activation_function_derivative):
    loss, previous_dot_product, hidden_layer_values_all, hidden_layer_values_activation_all, start_time = forward_propagation(
        weights, biases, hidden_layers_diagram, activation_function)
    hidden_layers_number = len(hidden_layers_diagram)
    gradient_weights_all = []
    gradient_bias_all = []
    previous_gradient_derivative_values = 0
    for index in range(hidden_layers_number, -1, -1):

        if index == hidden_layers_number:

            gradient_sigmoid_output_values_loss = 2 * (previous_dot_product - output_values_true)
            gradient_derivative_values = (gradient_sigmoid_output_values_loss *
                                          sigmoid_derivative(hidden_layer_values_all[index]))

            gradient_weight = np.dot(gradient_derivative_values.T, hidden_layer_values_activation_all[index - 1]).T
            gradient_bias = np.sum(gradient_derivative_values, 0)
            previous_gradient_derivative_values = gradient_derivative_values
        elif index != 0:
            gradient_hidden_loss = np.dot(previous_gradient_derivative_values, weights[index + 1].T)
            gradient_derivative_values = (
                    gradient_hidden_loss * activation_function_derivative(hidden_layer_values_all[index]))

            gradient_weight = np.dot(gradient_derivative_values.T, hidden_layer_values_activation_all[index - 1]).T
            gradient_bias = np.sum(gradient_derivative_values, 0)
            previous_gradient_derivative_values = gradient_derivative_values
        else:
            gradient_hidden_loss = np.dot(previous_gradient_derivative_values, weights[index + 1].T)
            gradient_derivative_values = (
                    gradient_hidden_loss * activation_function_derivative(hidden_layer_values_all[index]))

            gradient_weight = np.dot(input_data.T, gradient_derivative_values)
            gradient_bias = np.sum(gradient_derivative_values, 0)

        gradient_weights_all.insert(0, gradient_weight)
        gradient_bias_all.insert(0, gradient_bias)

    learning_rate = 0.00001

    for i in range(len(weights)):
        weights[i] = weights[i] - gradient_weights_all[i] * learning_rate
        bias[i] = bias[i] - gradient_bias_all[i] * learning_rate
    end_time = time.time()
    return weights, bias, start_time, end_time, loss


diagram = [20, 10, 10]
training_iterations = 4000
weights, bias = generate_weights_and_bias(diagram)

weights, bias, start_time, end_time, _ = backward_propagation(weights, bias, diagram, relu, relu_derivative)
elapsed_time = end_time - start_time
print("Estimated completion time min:", (elapsed_time * training_iterations) / 60)

for i in range(training_iterations):
    weights, bias, start_time, end_time, loss = backward_propagation(weights, bias, diagram, relu, relu_derivative)
    if loss < 100000:
        learning_rate = 0.000001
    print("Loss: ", loss, " iteration: ", i)


def convert_array_of_ndarray_to_list(arr):
    _list = []
    for ndarray in arr:
        _list.append(ndarray.tolist())
    return _list


data = [convert_array_of_ndarray_to_list(weights),
        convert_array_of_ndarray_to_list(bias), diagram, "relu"]
save_model(data)

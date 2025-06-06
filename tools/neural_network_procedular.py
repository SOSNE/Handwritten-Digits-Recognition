import numpy as np
from tools.file_reader import read_file_image, read_file_label
from tools.utils import save_model
import time

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


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def generate_weights_and_bias(hidden_layers_diagram):
    all_weights = []
    all_biases = []
    layer_sizes = [input_nodes] + hidden_layers_diagram + [output_nodes]
    for i in range(len(layer_sizes) - 1):
        weight = np.random.randn(layer_sizes[i], layer_sizes[i + 1])
        bias = np.full((layer_sizes[i + 1],), 0.1)
        all_weights.append(weight)
        all_biases.append(bias)
    return all_weights, all_biases


def forward_propagation(weights, biases, hidden_layers_diagram,
                        activation_function):
    start_time = time.time()
    previous_dot_product = input_data
    hidden_layer_values_all = []
    hidden_layer_values_activation_all = []

    for i in range(len(weights)):
        hidden_layer_values = previous_dot_product.dot(weights[i]) + biases[i]
        hidden_layer_values_activation = activation_function(hidden_layer_values)

        previous_dot_product = hidden_layer_values_activation
        hidden_layer_values_all.append(hidden_layer_values)
        hidden_layer_values_activation_all.append(hidden_layer_values_activation)

    loss = np.square(previous_dot_product - output_values_true).sum()
    # print(loss)
    return loss, previous_dot_product, hidden_layer_values_all, hidden_layer_values_activation_all, start_time


def backward_propagation(weights, biases, hidden_layers_diagram, activation_function, activation_function_derivative,
                         learning_rate):
    loss, previous_dot_product, hidden_layer_values_all, hidden_layer_values_activation_all, start_time = forward_propagation(
        weights, biases, hidden_layers_diagram, activation_function)
    gradient_weights_all = []
    gradient_bias_all = []
    gradient_sigmoid_output_values_loss = 2 * (previous_dot_product - output_values_true)
    gradient_derivative_values = gradient_sigmoid_output_values_loss * sigmoid_derivative(hidden_layer_values_all[-1])

    gradient_weight = np.dot(hidden_layer_values_activation_all[-2].T, gradient_derivative_values)
    gradient_bias = np.sum(gradient_derivative_values, axis=0)
    gradient_weights_all.insert(0, gradient_weight)
    gradient_bias_all.insert(0, gradient_bias)

    previous_gradient_derivative_values = gradient_derivative_values

    for i in range(len(weights) - 2, -1, -1):
        if i == 0:
            gradient_hidden_loss = np.dot(previous_gradient_derivative_values, weights[i + 1].T)
            gradient_derivative_values = (
                    gradient_hidden_loss * activation_function_derivative(hidden_layer_values_all[i]))

            gradient_weight = np.dot(input_data.T, gradient_derivative_values)
            gradient_bias = np.sum(gradient_derivative_values, axis=0)
            previous_gradient_derivative_values = gradient_derivative_values

        else:
            gradient_hidden_loss = np.dot(previous_gradient_derivative_values, weights[i + 1].T)
            gradient_derivative_values = (
                    gradient_hidden_loss * activation_function_derivative(hidden_layer_values_all[i]))

            gradient_weight = np.dot(hidden_layer_values_activation_all[i - 1].T, gradient_derivative_values)
            gradient_bias = np.sum(gradient_derivative_values, axis=0)
            previous_gradient_derivative_values = gradient_derivative_values
        gradient_weights_all.insert(0, gradient_weight)
        gradient_bias_all.insert(0, gradient_bias)

    for i in range(len(weights)):
        weights[i] = weights[i] + learning_rate * -gradient_weights_all[i]
        biases[i] = biases[i] + learning_rate * -gradient_bias_all[i]
    end_time = time.time()
    return weights, biases, start_time, end_time, loss


diagram = [10]
training_iterations = 3000
learning_rate = 0.001

weights, bias = generate_weights_and_bias(diagram)
weights, bias, start_time, end_time, _ = backward_propagation(weights, bias, diagram, sigmoid, sigmoid_derivative,
                                                              learning_rate)
elapsed_time = end_time - start_time
print("Estimated completion time min:", (elapsed_time * training_iterations) / 60)

for i in range(training_iterations):
    weights, bias, start_time, end_time, loss = backward_propagation(weights, bias, diagram, sigmoid,
                                                                     sigmoid_derivative,
                                                                     learning_rate)
    # if loss < 1000:
    #     learning_rate = 0.0001
    print("Loss: ", loss, " iteration: ", i)


def convert_array_of_ndarray_to_list(arr):
    _list = []
    for ndarray in arr:
        _list.append(ndarray.tolist())
    return _list


data = [convert_array_of_ndarray_to_list(weights),
        convert_array_of_ndarray_to_list(bias), diagram, "sigmoid"]
save_model(data, name="model2")

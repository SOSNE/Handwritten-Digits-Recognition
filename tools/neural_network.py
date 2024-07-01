import numpy as np
from tools.file_reader import read_file_image, read_file_label

hidden_layers = 1
input_nodes = 784
hidden_nodes = 10
output_nodes = 10
input_data = read_file_image("../data/t10k-images-idx3-ubyte")
labels = read_file_label("../data/t10k-labels-idx1-ubyte")
input_data = input_data.reshape(input_data.shape[0], input_data.shape[1] * input_data.shape[2])
# convert data to range form 0 to 1
input_data = input_data / 255.0
# first weight that is connected to 785 hidden layers.
weight_first = np.random.randn(input_nodes, hidden_nodes)
bias_first = np.full((hidden_nodes,), 0.1)

# rest of weights that are connected to 784 hidden layers.
# weights_rest = np.random.rand(hidden_layers, input_nodes, input_nodes)
# last weight that is connected to a 10 output nodes.

weight_last = np.random.randn(hidden_nodes, output_nodes)
bias_last = np.full((output_nodes,), 0.1)


def calculate_end_lose_probability_value(max_iterations, current_iteration, lose, previous_lose, first_lose, previous_quotient_summ):
    if previous_lose is not None:
        sequence_quotient = np.sqrt(lose/previous_lose)
        #average_quotient = (previous_quotient_summ + sequence_quotient) / current_iteration
        end_lose_probability_value = first_lose * np.power(sequence_quotient, (max_iterations - 1))
        return end_lose_probability_value, sequence_quotient
    return None, 0


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def forward_propagation():
    hidden_layer_values = input_data.dot(weight_first) + bias_first
    sigmoid_hidden_layer_values = sigmoid(hidden_layer_values)

    output_values = sigmoid_hidden_layer_values.dot(weight_last) + bias_last
    sigmoid_output_values = sigmoid(output_values)

    output_values_true = np.zeros((labels.size, 10))
    output_values_true[np.arange(labels.size), labels.flatten()] = 1

    loss = np.square(sigmoid_output_values - output_values_true).sum()
    print(loss)
    return hidden_layer_values, output_values, sigmoid_hidden_layer_values, sigmoid_output_values, output_values_true, loss


def back_propagation():
    global weight_last, weight_first, bias_last, bias_first

    hidden_layer_values, output_values, sigmoid_hidden_layer_values, sigmoid_output_values, output_values_true, loss = forward_propagation()

    gradient_sigmoid_output_values_loss = 2 * (sigmoid_output_values - output_values_true)

    gradient_output_values = (gradient_sigmoid_output_values_loss * sigmoid_derivative(output_values))

    gradient_weight_output = np.dot(gradient_output_values.T, sigmoid_hidden_layer_values).T
    gradient_bias_output = np.sum(gradient_output_values, 0)

    gradient_hidden_loss = np.dot(gradient_output_values, weight_last.T)

    gradient_hidden_values = (gradient_hidden_loss * sigmoid_derivative(hidden_layer_values))

    gradient_weight_first = np.dot(input_data.T, gradient_hidden_values)
    gradient_bias_first = np.sum(gradient_hidden_values, 0)

    weight_last = weight_last - gradient_weight_output * 0.001
    weight_first = weight_first - gradient_weight_first * 0.001

    bias_last = bias_last - gradient_bias_output * 0.0001
    bias_first = bias_first - gradient_bias_first * 0.0001

    return loss


lose = back_propagation()
first_lose = 9371.429013677895
previous_lose = None
previous_quotient_summ = 0

for i in range(1000):
    lose = back_propagation()
    # max_iterations, current_iteration, lose, previous_lose, first_lose, previous_average_quotient
    end_lose, sequence_quotient = calculate_end_lose_probability_value(1000, i, lose, previous_lose, first_lose, previous_quotient_summ)
    previous_lose = lose
    previous_quotient_summ += sequence_quotient
    print(end_lose)

import numpy as np


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def forward_propagation(input_data, weight_first, weight_last, bias_first, bias_last, labels):
    hidden_layer_values = input_data.dot(weight_first) + bias_first
    sigmoid_hidden_layer_values = sigmoid(hidden_layer_values)

    output_values = sigmoid_hidden_layer_values.dot(weight_last) + bias_last
    sigmoid_output_values = sigmoid(output_values)

    output_values_true = np.zeros((labels.size, 10))
    output_values_true[np.arange(labels.size), labels.flatten()] = 1

    loss = np.square(sigmoid_output_values - output_values_true).sum()
    return loss, sigmoid_output_values, output_values_true


def predict_output(input_data, weight_first, weight_last, bias_first, bias_last, labels):
    loss, sigmoid_output_values, _ = forward_propagation(input_data, weight_first, weight_last, bias_first, bias_last, labels)
    print(loss)
    index = 0
    for row in sigmoid_output_values:
        position = 0
        biggest_value = 0
        biggest_value_position = 0
        for value in row:
            if value > biggest_value:
                biggest_value = value
                biggest_value_position = position
        position += 1
        return biggest_value_position, index
    index += 1

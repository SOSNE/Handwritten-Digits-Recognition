import numpy as np


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def forward_propagation(input_data, weights, biases, hidden_layers_diagram, labels):
    if input_data.ndim == 3:
        input_data = input_data.reshape(input_data.shape[0], input_data.shape[1] * input_data.shape[2])
    if input_data.ndim == 2:
        input_data = input_data.reshape(1, -1)

    input_data = input_data / 255.0
    previous_dot_product = 0
    hidden_layer_values_all = []
    hidden_layer_values_sigmoid_all = []
    for index in range(len(hidden_layers_diagram) + 1):
        if index == 0:
            hidden_layer_values = input_data.dot(weights[index]) + biases[index]
            hidden_layer_values_sigmoid = sigmoid(hidden_layer_values)
            previous_dot_product = hidden_layer_values_sigmoid

        else:
            hidden_layer_values = previous_dot_product.dot(weights[index]) + biases[index]
            hidden_layer_values_sigmoid = sigmoid(hidden_layer_values)
            previous_dot_product = hidden_layer_values_sigmoid
        hidden_layer_values_all.append(hidden_layer_values)
        hidden_layer_values_sigmoid_all.append(hidden_layer_values_sigmoid)
    output_values_true = np.zeros((labels.size, 10))
    output_values_true[np.arange(labels.size), labels.flatten()] = 1
    loss = np.square(previous_dot_product - output_values_true).sum()
    return loss, previous_dot_product, output_values_true


def predict_output(input_data, weights, biases, hidden_layers_diagram, labels):
    loss, sigmoid_output_values, _ = forward_propagation(input_data, weights, biases, hidden_layers_diagram, labels)
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

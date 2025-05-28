# Handwritten Digits Recognition

This project is a tool written in Python that allows you to train a neural network to recognize handwritten digits. It uses **NumPy** for all numerical operations. By default, it is configured to be trained on the popular MNIST dataset.

---

## Requirements

Make sure you have Python installed. This project also uses the **NumPy** library, you can install it with:

```
pip install numpy
```

---

## How to Use

### 1. Train the Model

Open the `neural_network_procedural.py` file. At the top of the file, connect your training data to the input variables. By default, it uses the MNIST dataset.

Then scroll down to these parameters and adjust them as needed:

```
diagram = [10]              # Neural network architecture (e.g., one hidden layer with 10 neurons)
training_iterations = 3000  # Number of training iterations
learning_rate = 0.001       # Learning rate
```

Next, set the name of the model and where to save it in the line:

```
save_model(data, path="../store", name="model")
```

When ready, run the training script:

```
python neural_network_procedural.py
```

After training, the model will be saved as a file (e.g., `model.json`).

---

### 2. Test the Model

Now open the `main.py` file. This script allows you to:

- Test the trained model on random digits from the MNIST test dataset.
- Draw your own digit using a built-in drawing feature and let the model try to guess it.

Run the script:

```
python main.py
```

---

## Dataset

The MNIST dataset is used by default. You can download it from:  
- http://yann.lecun.com/exdb/mnist/

- https://github.com/cvdfoundation/mnist?tab=readme-ov-file

---

## License

This project is open-source. Use it however you like!

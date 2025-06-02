# import numpy as np
# from sklearn.datasets import fetch_openml
# from sklearn.model_selection import train_test_split
# from keras.utils import np_utils

# from dense import Dense
# from activation_func import Tanh
# from loss import mse, mse_derivative
# from network import train, predict


# def preprocess_data(x, y, limit):
#     # reshape and normalize input data
#     x = x.reshape(x.shape[0], 28 * 28, 1)
#     x = x.astype("float32") / 255
#     # one-hot encode the labels
#     y = np_utils.to_categorical(y.astype(int), 10)
#     y = y.reshape(y.shape[0], 10, 1)
#     return x[:limit], y[:limit]


# # Load MNIST dataset from sklearn
# mnist = fetch_openml("mnist_784", version=1)
# X = mnist.data
# y = mnist.target

# # Split into training and test datasets
# x_train_full, x_test_full, y_train_full, y_test_full = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Preprocess data
# x_train, y_train = preprocess_data(x_train_full.values, y_train_full.values, 1000)
# x_test, y_test = preprocess_data(x_test_full.values, y_test_full.values, 20)

# # Neural network definition
# network = [Dense(28 * 28, 40), Tanh(), Dense(40, 10), Tanh()]

# # Train the network
# train(network, mse, mse_derivative, x_train, y_train, epochs=100, learning_rate=0.1)

# # Test the network
# for x, y in zip(x_test, y_test):
#     output = predict(network, x)
#     print("pred:", np.argmax(output), "\ttrue:", np.argmax(y))


import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from dense import Dense
from activation_func import Tanh
from loss import mse, mse_derivative
from network import train, predict

# Load MNIST dataset
print("Loading MNIST...")
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = X.astype(np.float32) / 255.0  # normalize
y = y.astype(np.int32)

# Encode labels
encoder = OneHotEncoder(sparse_output=False, categories="auto")
y_onehot = encoder.fit_transform(y.reshape(-1, 1))

# Reshape input and output for the network
X = X.reshape(-1, 784, 1)
y_onehot = y_onehot.reshape(-1, 10, 1)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    X, y_onehot, test_size=0.02, random_state=42
)

# Limit for faster training
x_train = x_train[:1000]
y_train = y_train[:1000]
x_test = x_test[:20]
y_test = y_test[:20]

# Create neural network
network = [Dense(784, 40), Tanh(), Dense(40, 10), Tanh()]

# Train
train(network, mse, mse_derivative, x_train, y_train, epochs=100, learning_rate=0.1)

# Test
print("\nTesting:")
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print("Pred:", np.argmax(output), "\tTrue:", np.argmax(y))

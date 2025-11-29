import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def swish(x):
    return x * sigmoid(x)


# Combined function
def combined(x):
    return (
        0.2483 * relu(x) +
        0.1401 * np.tanh(x) +
        0.2604 * sigmoid(x) +
        0.3509 * swish(x)
    )

# Generate x range
x = np.linspace(-5, 5, 500)
y = combined(x)

# Plot
plt.figure(figsize=(7, 4))
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("y = 0.8733·ReLU(x) + 0.0658·tanh(x) + 0.0609·sigmoid(x)")
plt.grid(True)
plt.tight_layout()
plt.show()

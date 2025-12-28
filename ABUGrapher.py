import numpy as np
import matplotlib.pyplot as plt

# Activation functions
def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def swish(x):
    return x * sigmoid(x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# Combined function
def combined(x):
    return (
        0.9042 * relu(x) +
        0.0420 * tanh(x) +
        0.0312 * swish(x) +
        0.0203 * elu(x) +
        0.0002 * x
    )

# Generate x range
x = np.linspace(-5, 5, 500)
y = combined(x)

# Plot
plt.figure(figsize=(7, 4))
plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.title("y = 0.9042*ReLU(x) + 0.0420*tanh(x) + 0.0312*Swish(x) + 0.0203*ELU(x) + 0.0002 * identity(x)")
plt.grid(True)
plt.tight_layout()
plt.show()
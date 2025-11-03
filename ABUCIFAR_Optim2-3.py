# CIFAR without identity function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt


# 1. Load and preprocess CIFAR-10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize images to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode labels
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


# Adaptive Blending Units
# 2. Define the ABU

class ABU(layers.Layer):

    def __init__(self, activation_fns=None, **kwargs):
        super().__init__(**kwargs)
        self.activation_fns = [
            tf.nn.relu,
            tf.nn.tanh,
            tf.nn.sigmoid,
            #tf.identity
            ]

    def build(self, input_shape):
        num_funcs = len(self.activation_fns)
        self.alpha = self.add_weight(
            shape=(num_funcs,),
            initializer='uniform',
            trainable=True,
            name='alpha'
        )
        super().build(input_shape)

    def call(self, inputs):
        activations = [fn(inputs) for fn in self.activation_fns]
        stacked = tf.stack(activations, axis=-1)
        weights = tf.nn.softmax(self.alpha)
        return tf.reduce_sum(stacked * weights, axis=-1)


# Callback to freeze ABUs once their alphas converge

class FreezeABUCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, tol=1e-4, patience=2):
        super().__init__()
        self.model = model
        self.tol = tol
        self.patience = patience
        self.prev_alphas = None
        self.stable_epochs = 0
        self.frozen = False

    def get_abu_alphas(self):
        return [layer.alpha.numpy() for layer in self.model.layers if isinstance(layer, ABU)]

    def on_epoch_end(self, epoch, logs=None):
        current_alphas = self.get_abu_alphas()

        if self.prev_alphas is not None:
            deltas = [np.abs(curr - prev).max() for curr, prev in zip(current_alphas, self.prev_alphas)]
            if max(deltas) < self.tol:
                self.stable_epochs += 1
            else:
                self.stable_epochs = 0

            if self.stable_epochs >= self.patience and not self.frozen:
                print(f"\n✅ ABU alphas converged and frozen at epoch {epoch}")
                for layer in self.model.layers:
                    if isinstance(layer, ABU):
                        layer.alpha._trainable = False
                self.frozen = True

        self.prev_alphas = current_alphas


# 3. Define Model

class CIFARABUModel(Model):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = layers.Conv2D(32, (3, 3), padding='same')
        self.bn1 = layers.BatchNormalization()
        self.abu1 = ABU()

        self.conv2 = layers.Conv2D(64, (3, 3), padding='same')
        self.bn2 = layers.BatchNormalization()
        self.abu2 = ABU()
        self.pool2 = layers.MaxPooling2D((2, 2))

        self.conv3 = layers.Conv2D(128, (3, 3), padding='same')
        self.bn3 = layers.BatchNormalization()
        self.abu3 = ABU()
        self.pool3 = layers.MaxPooling2D((2, 2))

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(256)
        self.abu_fc = ABU()
        self.dropout = layers.Dropout(0.4)
        self.out = layers.Dense(num_classes, activation='softmax')

    def call(self, x, training=False):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.abu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.abu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = self.abu3(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.abu_fc(x)
        x = self.dropout(x, training=training)
        return self.out(x)


# 4. Compile and Train

model = CIFARABUModel(num_classes=num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

callback = FreezeABUCallback(model, tol=1e-4, patience=2)

history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    callbacks=[callback]
)


# 5. Evaluate the model
print(model.summary())
print("")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")
print(model.abu1.alpha)
print(model.abu2.alpha)
print(model.abu3.alpha)
print(model.summary())


# 6. Plot the final blended activation function

def plot_activation(abu_layer, name="ABU Activation"):
    # Create values to test over inputs from -3 to 3
    x = np.linspace(-3, 3, 400)
    activations = [fn(x).numpy() for fn in abu_layer.activation_fns]
    weights = tf.nn.softmax(abu_layer.alpha).numpy()

    blended = np.sum(np.stack(activations, axis=-1) * weights, axis=-1)

    plt.figure(figsize=(8, 5))
    for i, act in enumerate(activations):
        plt.plot(x, act, linestyle='--', label=f'Base fn {i}')
    plt.plot(x, blended, linewidth=3, label='Blended Activation')
    plt.title(f"{name} (final alphas: {weights.round(3)})")
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot the final activation for abu1
plot_activation(model.abu1, name="Final ABU1 Activation")

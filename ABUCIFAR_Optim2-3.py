import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# 1. Load and preprocess CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)


# 2. Define the ABU Layer
class ABU(layers.Layer):
    def __init__(self, activation_fns=None, **kwargs):
        super().__init__(**kwargs)
        self.activation_fns = [
            tf.nn.relu,
            tf.nn.tanh,
            tf.nn.sigmoid,
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


# 3. Define the Model
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


# 4. Callback to detect slow-moving ABU weights
class ABUCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold=0.01, patience=3):
        super().__init__()
        self.threshold = threshold
        self.patience = patience
        self.prev_alphas = {}
        self.slow_counter = {}

    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            if isinstance(layer, ABU):
                if layer.name not in self.prev_alphas:
                    self.prev_alphas[layer.name] = tf.nn.softmax(layer.alpha).numpy()
                    self.slow_counter[layer.name] = 0
                    continue

                current_alpha = tf.nn.softmax(layer.alpha).numpy()
                prev_alpha = self.prev_alphas[layer.name]
                delta = np.linalg.norm(current_alpha - prev_alpha, ord=2)

                self.prev_alphas[layer.name] = current_alpha

                if delta < self.threshold:
                    self.slow_counter[layer.name] += 1
                    if self.slow_counter[layer.name] >= self.patience:
                        print(f"Epoch {epoch+1}: ABU layer '{layer.name}' weights are moving slowly (delta={delta:.6f})")
                else:
                    self.slow_counter[layer.name] = 0


# 5. Compile and Train
model = CIFARABUModel(num_classes=num_classes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.1,
    callbacks=[ABUCallback(threshold=0.2, patience=2)]
)


# 6. Evaluate the model
print(model.summary())
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest accuracy: {test_acc:.4f}")

# Print learned ABU alpha weights
print("ABU1 alpha:", model.abu1.alpha.numpy())
print("ABU2 alpha:", model.abu2.alpha.numpy())
print("ABU3 alpha:", model.abu3.alpha.numpy())
print("ABU_fc alpha:", model.abu_fc.alpha.numpy())


# 7. Plot final learned ABU weights
def plot_abu_weights(model):
    for i, layer in enumerate(model.layers):
        if isinstance(layer, ABU):
            weights = tf.nn.softmax(layer.alpha).numpy()
            plt.figure()
            plt.bar(range(len(weights)), weights)
            plt.title(f"ABU Layer {i+1} ({layer.name}) Softmax Weights")
            plt.xlabel("Activation function")
            plt.ylabel("Weight")
            plt.show()

plot_abu_weights(model)

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('./Heart.csv').dropna()
cols = ["Age", "Sex", "ChestPain", "RestBP", "Chol", "Fbs",
        "RestECG", "MaxHR", "ExAng", "Oldpeak", "Slope", "Ca", "Thal"]
X = df[cols]
y = df['AHD'].map({'Yes': 1, 'No': 0}).astype(np.float32)

categorical_cols = ["Sex", "ChestPain", "Fbs", "RestECG", "ExAng", "Slope", "Ca", "Thal"]
X = pd.get_dummies(X, columns=categorical_cols)

numerical_cols = ["Age", "RestBP", "Chol", "MaxHR", "Oldpeak"]
X[numerical_cols] = (X[numerical_cols] - X[numerical_cols].mean()) / X[numerical_cols].std()

X = X.to_numpy().astype(np.float32)
y = y.to_numpy()

split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

class ABU(tf.keras.layers.Layer):
    def __init__(self):
        super(ABU, self).__init__()
        self.activations = [
            tf.nn.relu,
            tf.math.tanh,
            tf.identity,
            tf.nn.sigmoid
        ]

    def build(self, input_shape):
        initial_alpha = tf.constant([0.05, 0.85, 0.05, 0.05], dtype=tf.float32)
        self.alpha = self.add_weight(
            shape=(len(self.activations),),
            initializer=tf.keras.initializers.Constant(initial_alpha),
            trainable=True,
            name="abu_weights"
        )

    def call(self, inputs):
        alpha_softmax = tf.nn.softmax(self.alpha)
        entropy = -tf.reduce_sum(alpha_softmax * tf.math.log(alpha_softmax + 1e-8))
        self.add_loss(0.01 * entropy)
        output = sum(alpha_softmax[i] * act(inputs) for i, act in enumerate(self.activations))
        return output

def build_model(activation_layer):
    return models.Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(64, kernel_regularizer=regularizers.l2(0.001)),
        activation_layer,
        layers.Dropout(0.3),
        layers.Dense(32, kernel_regularizer=regularizers.l2(0.001)),
        activation_layer.__class__() if isinstance(activation_layer, ABU) else activation_layer,
        layers.Dense(1, activation='sigmoid')
    ])

def train_model(model, label):
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train,
                        validation_split=0.2,
                        epochs=60,
                        batch_size=16,
                        verbose=1)
    return history

abu_model = build_model(ABU())
sigmoid_model = build_model(layers.Activation('sigmoid'))
tanh_model = build_model(layers.Activation('tanh'))

abu_hist = train_model(abu_model, "ABU")
sigmoid_hist = train_model(sigmoid_model, "Sigmoid")
tanh_hist = train_model(tanh_model, "Tanh")

plt.plot(abu_hist.history['val_loss'], label='ABU Val Loss')
plt.plot(sigmoid_hist.history['val_loss'], label='Sigmoid Val Loss')
plt.plot(tanh_hist.history['val_loss'], label='Tanh Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Validation Loss')
plt.title('Validation Loss Comparison')
plt.legend()
plt.grid(True)
plt.show()

abu_layers = [layer for layer in abu_model.layers if isinstance(layer, ABU)]
for i, abu in enumerate(abu_layers):
    print(f"Final ABU weights (layer {i+1}):", tf.nn.softmax(abu.alpha).numpy())

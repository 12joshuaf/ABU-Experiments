import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import cifar10


# 1. Load and preprocess CIFAR-10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize images to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encode labels
num_classes = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)



# 2. Define the ABU (Adaptive Blending Unit)

class ABU(layers.Layer):

    def __init__(self, activation_fns=None, **kwargs):
        super().__init__(**kwargs)
        if activation_fns is None:
            self.activation_fns = [
                tf.nn.relu,
                tf.nn.tanh,
                tf.nn.sigmoid,
                #remove tf.identity, more efficient
            ]
        else:
            self.activation_fns = activation_fns

    def build(self, input_shape):
        num_funcs = len(self.activation_fns)
        # Single trainable alpha per layer (global blend)
        self.alpha = self.add_weight(
            shape=(num_funcs,),
            initializer='uniform',
            trainable=True,
            name='alpha'
        )
        super().build(input_shape)

    def call(self, inputs):
        # Apply each base activation
        activations = [fn(inputs) for fn in self.activation_fns]
        stacked = tf.stack(activations, axis=-1)  # [..., num_funcs]
        weights = tf.nn.softmax(self.alpha)       # normalized blending
        return tf.reduce_sum(stacked * weights, axis=-1)



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
    metrics=['accuracy']
)

print(model.summary())

history = model.fit(
    x_train, y_train,
    epochs=1,
    batch_size=64,
    validation_split=0.1
)



# 5. Evaluate the model

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\n✅ Test accuracy: {test_acc:.4f}")

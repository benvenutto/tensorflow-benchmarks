import tensorflow as tf
from tensorflow import keras

device_name = 'GPU' # 'GPU'
batch_size=64
num_epochs=10


class ResidualBlock(keras.layers.Layer):
    """A residual block (ResNet) for image processing.
    """


def __init__(self):
    super().__init__()

    self.conv1 = tf.keras.layers.Conv2D(activation='relu')
    self.relu1 = tf.keras.layers.ReLU()
    self.conv2 = tf.keras.layers.Conv2D(activation='relu')
    self.add = tf.keras.layers.Add()
    self.relu2 = tf.keras.layers.ReLU()


def call(self, inputs):
    x = self.conv1(inputs)
    x = self.relu1(x)
    x = self.conv2(x)
    x = self.add(x, inputs)
    x = self.relu2(x)
    return x


device_id = f'/device:{device_name}:0'
with tf.device(device_id):
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    print(train_images.shape, len(train_labels), test_images.shape, len(test_labels))

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = tf.keras.Sequential([
        ResidualBlock(input_shape=(28, 28)),
        ResidualBlock(input_shape=(28, 28)),
        ResidualBlock(input_shape=(28, 28)),
        ResidualBlock(input_shape=(28, 28)),
        ResidualBlock(input_shape=(28, 28)),
        ResidualBlock(input_shape=(28, 28)),
        ResidualBlock(input_shape=(28, 28)),
        ResidualBlock(input_shape=(28, 28)),
        ResidualBlock(input_shape=(28, 28)),
        ResidualBlock(input_shape=(28, 28)),
        ResidualBlock(input_shape=(28, 28)),
        ResidualBlock(input_shape=(28, 28)),
        ResidualBlock(input_shape=(28, 28)),
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, batch_size=batch_size, epochs=num_epochs)

    test_loss, test_acc = model.evaluate(test_images,  test_labels, batch_size=batch_size, verbose=2)

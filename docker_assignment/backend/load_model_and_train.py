import tensorflow as tf
from keras.src.utils import to_categorical

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Reshape the data to include the channel dimension
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Load the saved model
model = tf.keras.models.load_model('mnist_cnn_model_retrained.keras')
print("Model loaded successfully")

# Compile the model again (if necessary)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Continue training the model
model.fit(x_train, y_train, epochs=50, batch_size=64, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy after additional training: {test_acc}')

model.save('mnist_cnn_model_retrained_2.keras')
print("Model saved successfully")

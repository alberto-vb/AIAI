import tensorflow as tf
from keras import Sequential
from keras.src.layers import Dense, Flatten, MaxPooling2D, Conv2D, BatchNormalization, Dropout
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import Adam
from keras.src.utils import to_categorical
from tensorflow.keras import layers, models

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
# Normalize the images to values between 0 and 1
x_train = x_train / 255.0
x_test = x_test / 255.0
print(x_train.shape)
print(x_test.shape)

# Before one hot encoding
print("ytrain Shape: %s and value: %s" % (y_train.shape, y_train))
print("ytest Shape: %s and value: %s" % (y_test.shape, y_test))

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# After one hot encoding
print("ytrain Shape: %s and value: %s" % (y_train.shape, y_train[0]))
print("ytest Shape: %s and value: %s" % (y_test.shape, y_test[1]))

# Reshape the data to include the channel dimension (28x28x1 for grayscale images)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Build a simple CNN model
model = Sequential()
# First convolutional block
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

# Second convolutional block
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

# Third convolutional block
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))

# Flattening and fully connected layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Add dropout to reduce overfitting
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(
    optimizer=Adam(),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# Train the model
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

model.save('mnist_cnn_model.keras')
print("Model saved as mnist_cnn_model.keras")

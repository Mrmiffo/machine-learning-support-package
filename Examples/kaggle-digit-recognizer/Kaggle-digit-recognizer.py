from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Input paremeters
image_width = 28
image_height = 28
channels = 1
classes = 10

# Set hyperparameters
epochs = 20
validation_split = 0.2
batch_size = 256

# Load the training data

train_data = np.genfromtxt('Examples/kaggle-digit-recognizer/train.csv', delimiter=',')     # There are quicker ways to load data, but this is simple :)

train_data = np.delete(train_data, 0, axis = 0)                                             # Delete the header row
train_y = to_categorical(train_data[:,0], classes)                                          # Create a onehot encoding of the possible outputs. For example the value 4 becomes [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
train_x = np.delete(train_data, 0, axis = 1)                                                # Remove the first column containing the lable
train_x = np.reshape(train_x, (train_x.shape[0], image_width, image_height, channels))      # Reshape the 1D vector image to a 2D vector with 1 channel (greyscale). Shape is now (#ofSamples, width, heigt, #ofChannels)
print("Found %d training samples" % (train_x.shape[0]))

# Load the test data

test_x = np.genfromtxt('Examples/kaggle-digit-recognizer/test.csv', delimiter=',') 
test_x = np.delete(test_x, 0, axis = 0)                                                     # Delete the header row
test_x = np.reshape(test_x, (test_x.shape[0], image_width, image_height, channels))         # Reshape the 1D vector image to a 2D vector.
print("Found %d test samples" % (test_x.shape[0]))

# Create the model
model = Sequential([
    Conv2D(16, 3, padding='same', input_shape=(image_width,image_height,channels), activation='relu'), 
    Conv2D(16, 3, padding='same', activation='relu'), 
    MaxPooling2D(),
    Conv2D(16, 3, padding='same', activation='relu'), 
    Conv2D(16, 3, padding='same', activation='relu'), 
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Train the model
model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])                            # Adam is generally a reliable optimizer. As this is a classification task, categorical_crossentropy is used as the loss. Accuracy is a good metric for a classification task.
model.fit(train_x, train_y, batch_size=batch_size, validation_split=validation_split, epochs=epochs)    # Train the network with the given hyperparameters (see top)

# Predict results
pred = model.predict(test_x)                                                        # Use the model to predict the output of all the test samples (test samples has no "true" label so we don't know what they are)
pred_to_print = np.reshape(test_x, (test_x.shape[0], image_width, image_height))    # imshow does not want the channel value for greyscale images, so rescale it to shape (28,28) instead of (28, 28, 1)
for i in range(20):
    print("Prediction: %d" %(np.argmax(pred[i])))   # This is what our network predicted for this image
    plt.imshow(pred_to_print[i],cmap='gray')        # Display the image so we can see what it looks like
    plt.show()

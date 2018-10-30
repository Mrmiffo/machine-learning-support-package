# Kaggle digit reconizer
Source: https://www.kaggle.com/c/digit-recognizer
This example use the famous MNIST data set and applies a simple CNN (convolutional neural network) with:
- Four convolutional layers:
    Conv2D(16, 3, padding='same', activation='relu')
    Each with 16 channels, and a 3x3 filter. The padding keeps the output size equal to the input size and the activation is the Rectified Linear Unit. 
    input_shape=(28,28,1) declares that the images that are sent into the network is of the size 28x28 and there is only 1 channel (greyscale). 
        For an RBG image the 1 would be 3 for the three color channels.
        The input shape must always be declared on the first Keras layer, but is implicit in the rest of the network.
- MaxPooling after each set of two convolutional layers.
    MaxPooling2D()
    Maxpooling is an efficient way to downsample the output of convolutional layers.
- Flatten after the convolutional layers.
    Flatten turns the 3D output of the convolutional filters into a 1D vector to be used by the fully connected Dense layers. 
- Three fully connected layers.
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
    In a CNN these layers are called the "top", and this is quite a typical setup. First there are two hidden layers using the typical 'relu' activation function, followed by the output layer having the same number of neurons as there are output classes. 
        The 'softmax' activation of the output layer turns each output value into a percentage value. 
        Using two hidden layers is the minimum for a neural network to be able to linearly seperate the XOR function, and one want to reduce the number of parameters in the nework (that is one of the reasons why you use convolutional layers to begin with). Also this setup has proven itself in systems such as VGG and ResNet (although with more neurons in those setups).

## Setup
Simply run "python Examples/kaggle-digit-recognizer/Kaggle-digit-recognizer.py" from the root folder of the project. 
Will run for 20 epochs and should reach a val_acc of about 98.5%. Will take about 0.5-1min on a GPU. Training on a CPU may take alot longer.
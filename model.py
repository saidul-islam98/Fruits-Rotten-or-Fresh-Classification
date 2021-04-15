import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout 
from tensorflow.keras.utils import plot_model


def model_definition():
    model = Sequential()
    model.add(Conv2D(64, (3,3), activation='relu', input_shape=(228, 228, 3)))
    model.add(MaxPooling2D(2, 2))
    # The second convolution
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    # The third convolution
    model.add(Conv2D(256, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    # The fourth convolution
    model.add(Conv2D(512, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    # Flatten the results to feed into a DNN
    model.add(Conv2D(512, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dropout(0.5))

    model.add(Dense(512, activation='relu'))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(6, activation='softmax'))

    return model


def model_summary(model):
    model.summary()


# def model_plot(model):
#     plot_model(model)
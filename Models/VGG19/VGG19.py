import tensorflow as tf, numpy
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten, Dropout

feature = Sequential()

feature.add(Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
feature.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
feature.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

feature.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
feature.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
feature.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

feature.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
feature.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
feature.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
feature.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
feature.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

feature.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
feature.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
feature.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
feature.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
feature.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

feature.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
feature.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
feature.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
feature.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
feature.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

# freeze feature extraction
for layer in feature.layers:
    layer.trainable = False









# flatten feature vector & pass through final dense layers

x = Flatten()(feature.output)
x = Dense(units=4096, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(units=4096, activation="relu")(x)
x = Dropout(0.5)(x)
prediction = Dense(units=2, activation="softmax")(x)


# Create final model

model = Model(inputs=feature.input, outputs=prediction)

feature.summary()
model.summary()

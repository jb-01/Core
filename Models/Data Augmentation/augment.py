import tensorflow as tf
from keras import layers

img_size = 256
batch_size = 32


""" flowers_photos/
      daisy/
      dandelion/
      roses/
      sunflowers/
      tulips/ 
"""
# Create training dataset from flowers_photos/ folder
train_set = tf.keras.utils.image_dataset_from_directory(
        '/data',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size)

# Create validation training data set from flowers_photos/ folder
validation_set = tf.keras.utils.image_dataset_from_directory(
        '/data',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_size, img_size),
        batch_size=batch_size)

# Resize & rescale images
resize_and_rescale = tf.keras.Sequential([
  layers.Resizing(img_size, img_size),
  layers.Rescaling(1./255)
])

# Augment data
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])

# Create model
num_classes = 5

model = tf.keras.Sequential([
  resize_and_rescale,
  data_augmentation,
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
        train_set,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=validation_set,
        validation_steps=800)


import os
import numpy as np
import tensorflow as tf
import scipy
from keras.preprocessing.image import ImageDataGenerator

# Set the path to the dataset
dataset_path = 'PetImages'

# Create an ImageDataGenerator object
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load the training and validation data
train_generator = datagen.flow_from_directory(dataset_path,
                                                target_size=(150, 150),
                                                color_mode="grayscale",
                                                class_mode='binary',
                                                subset='training')

validation_generator = datagen.flow_from_directory(dataset_path,
                                                target_size=(150, 150),
                                                color_mode="grayscale",
                                                class_mode='binary',
                                                subset='validation')

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(),
                metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    steps_per_epoch=100,
                    epochs=10,
                    validation_data=validation_generator,
                    validation_steps=50)

model.save('models')


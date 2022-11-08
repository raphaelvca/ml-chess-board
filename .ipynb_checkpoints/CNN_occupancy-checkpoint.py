# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 20:56:45 2022

@author: PC
"""
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from Creator_dataset_occupancy import Creator_datset
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import regularizers
# from keras_visualizer import visualizer 

BATCH_SIZE = 32
IMG_SIZE = (100, 100)


AUTOTUNE = tf.data.AUTOTUNE

creator_dataset = Creator_datset(BATCH_SIZE) 
# dataset_train = creator_dataset.create_dataset("train")
# dataset_test = creator_dataset.create_dataset("test")
# dataset_val = creator_dataset.create_dataset("val")

dataset_train, dataset_val, dataset_test = creator_dataset.create_dataset("occupancy")

train_dataset = dataset_train.prefetch(buffer_size=AUTOTUNE)
validation_dataset = dataset_val.prefetch(buffer_size=AUTOTUNE)
test_dataset = dataset_test.prefetch(buffer_size=AUTOTUNE)

image_batch = np.concatenate([x for x, y in dataset_train], axis=0)
label_batch = np.concatenate([y for x, y in dataset_train], axis=0)
image_batch_test = np.concatenate([x for x, y in dataset_test], axis=0)
label_batch_test = np.concatenate([y for x, y in dataset_test], axis=0)

#image_batch, label_batch = dataset_train.as_numpy_iterator().next()
#image_batch_test, label_batch_test = test_dataset.as_numpy_iterator().next()

#first try
#model = models.Sequential()
#model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#model.add(layers.Flatten())
#model.add(layers.Dense(64, activation='sigmoid'))
#model.add(layers.Dense(1))

#second try
model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(100, 100, 3)),
tf.keras.layers.MaxPooling2D(2, 2),
tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
tf.keras.layers.MaxPooling2D(2,2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')])


model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
#tf.keras.utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# visualizer(model, format='png', view=True)

history = model.fit(image_batch, label_batch, epochs=10, 
                    validation_data=(image_batch_test, label_batch_test))


plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(image_batch_test,  label_batch_test, verbose=2)
model.save("models/CNN_occupancy")


image_batch_val = np.concatenate([x for x, y in dataset_val], axis=0)
label_batch_val = np.concatenate([y for x, y in dataset_val], axis=0)

val_loss, val_acc = model.evaluate(image_batch_val,  label_batch_val, verbose=2)

predictions = model.predict(image_batch_val)


predlist = predictions.tolist()
predlist = [x[0] for x in predlist]

pred = pd.DataFrame({"pred": predlist,
              "lab": label_batch_val})

pred["pred"] = pred["pred"] > 0.5
pred["pred"] = pred["pred"].astype(int)

pd.crosstab(index=pred["lab"], columns=pred["pred"])


plt.figure(figsize=(10, 10))
for i in range(16):
  ax = plt.subplot(4, 4, i + 1)
  plt.imshow(image_batch_val[i])
  plt.title(str(predictions[i] > 0.5))
  plt.axis("off")

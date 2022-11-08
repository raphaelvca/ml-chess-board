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
from tensorflow.keras.utils import to_categorical
# from keras_visualizer import visualizer 

AUTOTUNE = tf.data.AUTOTUNE
creator_dataset = Creator_datset(32) 
train_ds, val_ds, test_ds = creator_dataset.create_dataset("piece")

image_batch_train, label_batch_train = next(iter(train_ds))
image_batch_test, label_batch_test = next(iter(test_ds))
image_batch_val, label_batch_val = next(iter(val_ds))

image_train = np.concatenate([x1 for x1, y1 in train_ds], axis=0)
label_train = np.concatenate([y2 for x2, y2 in train_ds], axis=0)
image_val = np.concatenate([x for x, y in val_ds], axis=0)
label_val = np.concatenate([y for x, y in val_ds], axis=0)
image_test = np.concatenate([x for x, y in test_ds], axis=0)
label_test = np.concatenate([y for x, y in test_ds], axis=0)

## Transforming labels to correct format
num_classes = 12
label_train = to_categorical(label_train, num_classes=num_classes)
label_test = to_categorical(label_test, num_classes=num_classes)
label_val = to_categorical(label_val, num_classes=num_classes)
label_batch_train = to_categorical(label_batch_train, num_classes=num_classes)
label_batch_test = to_categorical(label_batch_test, num_classes=num_classes)
label_batch_val = to_categorical(label_batch_val, num_classes=num_classes)

# Achtung mit der Batchsize
# print(image_train.shape)
# print(label_train.shape)
# print(image_val.shape)
# print(label_val.shape)
# print(image_test.shape)
# print(label_test.shape)

train_dataset = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_dataset = val_ds.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_ds.prefetch(buffer_size=AUTOTUNE)


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
tf.keras.layers.Dense(512, activation='relu',
                 kernel_regularizer=regularizers.l2(0.001)),
tf.keras.layers.Dropout(0.3),
tf.keras.layers.Dense(12, activation='softmax')])


model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])

# model.summary()

history = model.fit(image_train, label_train, epochs=50, validation_data=(image_val, label_val))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(image_batch_test,  label_batch_test, verbose=2)

val_loss, val_acc = model.evaluate(image_test,  label_test, verbose=2)

predictions = model.predict(image_test)


predlist = predictions.tolist()
predlist2 = [x[0] for x in predlist]

# pred = pd.DataFrame({"pred": predlist2, "lab": label_test})

# pred["pred"] = pred["pred"] > 0.5
# pred["pred"] = pred["pred"].astype(int)

# pd.crosstab(index=pred["lab"], columns=pred["pred"])

def pred_to_piece(prediction):
    fen_posibilities = 'RNBQKPrnbqkp'
    pred_bool = list(prediction)
    return fen_posibilities[pred_bool.index(max(pred_bool))]


plt.figure(figsize=(10, 10))
for i in range(16):
  ax = plt.subplot(4, 4, i + 1)
  plt.imshow(image_val[i])
  plt.title(pred_to_piece(label_val[i]))
  plt.axis("off")

plt.figure(figsize=(10, 10))
for i in range(25):
  ax = plt.subplot(5, 5, i + 1)
  plt.imshow(image_val[i])
  plt.title(pred_to_piece(predictions[i]))
  plt.axis("off")


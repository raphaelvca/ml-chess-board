# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 09:59:15 2022

@author: PC
"""


import os
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from Creator_dataset import Creator_dataset
from tensorflow.keras import datasets, layers, models
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
import cv2
from os.path import isfile, join
from os import listdir

# Step 1: load model
reconstructed_model = keras.models.load_model("models/CNN_binary_stable")

# Step 2: get img
img_path = "02_created_data/piece/"
img_files = [f for f in listdir(img_path) if isfile(join(img_path, f))][0:64]

imgs = [cv2.cvtColor(cv2.imread(img_path + img), cv2.COLOR_BGR2RGB) for img in img_files]

imgs = np.array(imgs)

# Step 3: predict
pred = reconstructed_model.predict(imgs)


def pred_to_piece(prediction):
    fen_posibilities = 'RNBQKPrnbqkp'
    pred_bool = list(prediction)
    return fen_posibilities[pred_bool.index(max(pred_bool))]

pred_lit = [pred_to_piece(p) for p in pred]

fen = "".join(pred_lit)

plt.figure(figsize=(10, 10))
for i in range(64):
  ax = plt.subplot(8, 8, i + 1)
  plt.imshow(imgs[i])
  plt.title(pred_lit[i])
  plt.axis("off")



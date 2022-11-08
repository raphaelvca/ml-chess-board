import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import glob  

# Source
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/TensorFlow/Basics/tutorial18-customdata-images/2_csv_file.py
# https://www.youtube.com/watch?v=q7ZuZ8ZOErE

class Creator_dataset():

  def __init__(self, batchsize):
    self.batchsize = batchsize

  def create_dataset(self, dir_name, file_name):
    directory = "02_created_data/" + dir_name + "/"
    path_csv = directory + file_name + ".csv"
    df = pd.read_csv(path_csv)

    file_paths = df["file_name"].values

    labels = df["label"].values
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    def read_image(image_file, label):
        image = tf.io.read_file(directory + image_file)
        image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
        return image, label

    dataset = dataset.map(read_image).batch(self.batchsize)

    # shuffle_size=500000
    # ds = dataset.shuffle(shuffle_size, seed=12)
    
    # train_size = int(0.8 * len(dataset))
    # val_size = int(0.1 * len(dataset))

    # train_ds = ds.take(train_size)    
    # val_ds = ds.skip(train_size).take(val_size)
    # test_ds = ds.skip(train_size).skip(val_size)

    print('-'*20)
    print("Dataset " + str(dir))
    print(file_name + "\t" + str(len(file_paths)) + "\t" + str(len(dataset)) + " bt. x " + str(self.batchsize) + " bt.size" +  ": done")
    # print("Train\t\t" + str(len(list(train_ds))) + " bt. x " + str(self.batchsize) + " bt.size" +  ": done")
    # print("Val\t\t" + str(len(list(val_ds))) + " bt. x " + str(self.batchsize) + " bt.size" +  ": done")
    # print("Test\t\t" + str(len(list(test_ds))) + " bt. x " + str(self.batchsize) + " bt.size" +  ": done")
    print('-'*20)

    return dataset


if __name__ == "__main__":

    creator_dataset = Creator_dataset(32) 
    # ds_train = creator_dataset.create_dataset("occupancy", "occupancy_test")
    ds_train = creator_dataset.create_dataset("piece", "piece_test")


# Notes

# def process_path(file_path):
#     label = get_label(file_path)
#     img = tf.io.read_file(file_path)
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.image.convert_image_dtype(img, tf.float32)
#     img = tf.image.resize(img, size=(180, 180))
#     return img, label

# ds_train = ds_train.map(read_image)#.batch(self.batchsize)

    # directory = "02_created_data/MH/"
    # ds_train = tf.keras.utils.image_dataset_from_directory(
    # directory,
    # validation_split=0.2,
    # subset="training",
    # seed=123,
    # image_size=(150, 150),
    # batch_size=32)
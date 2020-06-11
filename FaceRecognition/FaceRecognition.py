#!/usr/bin/env python
import time
import os
import data  # get custom dataset
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Softmax
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
project_dir_name = os.getcwd()
data_path = os.path.join(project_dir_name, 'Datasets')
training_dir = os.path.join(data_path, 'att_faces', 'Training')
testing_dir = os.path.join(data_path, 'att_faces', 'Testing')

DATASET = 'ATT'  # Please Change ATT to YALE if you want to train Yale Dataset

dropout = 0.75

imageSize = 92*112
image_width = 92
image_height = 112
NChannels = 1

classes = 40
batch_size = 5

epochs = 20
learning_rate = 1

# Change some parameters if we use Yale dataset
if(DATASET == 'YALE'):
    training_dir = os.path.join(data_path, 'yalefaces', 'Training')
    testing_dir = os.path.join(data_path, 'yalefaces', 'Testing')

    imageSize = 320 * 243
    image_width = 320
    image_height = 243
    classes = 15
    epochs = 15

train_faces, train_labels = data.LoadTrainingData(
    training_dir, (image_width, image_height))


test_faces, test_labels = data.LoadTestingData(
    testing_dir, (image_width, image_height))

model = Sequential()
model.add(Flatten(input_shape=(image_width, image_height)))
model.add(Dense(128, activation='relu'))
model.add(Dense(classes))

model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
model.fit(train_faces, train_labels, epochs=epochs)

test_loss, test_acc = model.evaluate(test_faces, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

probability_model = Sequential([model, Softmax()])
predictions = probability_model.predict(test_faces)
print(predictions)

idx = 0
img = test_faces[1]
img = np.expand_dims(img, 0)
predictions_single = probability_model.predict(img)
print(predictions_single)

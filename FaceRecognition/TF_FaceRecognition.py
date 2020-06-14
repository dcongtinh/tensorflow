# Import modules
import os
import data  # get custom dataset
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Softmax
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

dataset_name = 'att_faces'  # or 'yalefaces'
project_dir_name = os.getcwd()
data_path = os.path.join(project_dir_name, 'Datasets')
training_dir = os.path.join(data_path, dataset_name, 'Training')
testing_dir = os.path.join(data_path, dataset_name, 'Testing')

image_width = 92
image_height = 112
classes = 40
epochs = 100

# Change some parameters if we use Yale dataset
if dataset_name == 'yalefaces':
    image_width = 320
    image_height = 243
    classes = 15
    epochs = 100

# Import dataset
train_faces, train_labels = data.LoadTrainingData(
    training_dir, (image_width, image_height))

test_faces, test_labels = data.LoadTestingData(
    testing_dir, (image_width, image_height))
print(train_faces[0], train_faces[0].shape)
# NN model
model = Sequential()
model.add(Flatten(input_shape=(image_width, image_height)))
model.add(Dense(128, activation='relu'))
model.add(Dense(classes))
model.compile(optimizer='adam',
              loss=SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(train_faces, train_labels, epochs=epochs)
model.add(Softmax())

predictions = model.predict(test_faces)
print(predictions[0])
pred_labels = np.array([np.argmax(pred) for pred in predictions])

# Calculate metrics
average = 'weighted'
accuracy = accuracy_score(test_labels, pred_labels)*100
precision = precision_score(
    test_labels, pred_labels, average=average)*100
f1 = f1_score(test_labels, pred_labels, average=average)*100
recall = recall_score(test_labels, pred_labels, average=average)*100

print('\n')
print('Accuracy  = {}%'.format(accuracy))
print('Precision = {}%'.format(precision))
print('Recall    = {}%'.format(recall))
print('F1_Score  = {}%'.format(f1))
print('\n')

for i in range(classes):
    res = 'Correct' if test_labels[i] == pred_labels[i] else 'Incorrect'
    print('True: %2d   Pred: %2d   %s' % (test_labels[i], pred_labels[i], res))

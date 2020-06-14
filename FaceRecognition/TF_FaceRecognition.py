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


def load_data():
    # Import dataset
    train_faces, train_labels = data.LoadTrainingData(
        training_dir, (image_width, image_height))

    test_faces, test_labels = data.LoadTestingData(
        testing_dir, (image_width, image_height))
    return train_faces, train_labels, test_faces, test_labels


def build_model(X_train, y_train):
    # NN model
    model = Sequential()
    model.add(Flatten(input_shape=(image_width, image_height)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(classes))
    model.compile(optimizer='adam',
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=epochs)
    model.add(Softmax())
    return model


def predict(model, X_Pred):
    preds = model.predict(X_Pred)
    # print(preds[0])
    y_pred = np.array([np.argmax(pred) for pred in preds])
    return y_pred


def cal_metrics(y_true, y_pred, average='weighted'):
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)*100
    precision = precision_score(
        y_true, y_pred, average=average, zero_division=1)*100
    f1 = f1_score(y_true, y_pred, average=average)*100
    recall = recall_score(y_true, y_pred, average=average)*100

    print('\n')
    print('Accuracy  = {:.3f}%'.format(accuracy))
    print('Precision = {:.3f}%'.format(precision))
    print('Recall    = {:.3f}%'.format(recall))
    print('F1_Score  = {:.3f}%'.format(f1))
    print('\n')


def plot_image(idx, images):
    img = images[idx].reshape(image_height, image_width)
    plt.imshow(img, cmap='gray')
    plt.show()


def main():
    X_train, y_train, X_test, y_test = load_data()
    model = build_model(X_train, y_train)
    y_pred = predict(model, X_test)
    cal_metrics(y_test, y_pred)
    for i in range(classes):
        res = 'Correct' if y_test[i] == y_pred[i] else 'Incorrect'
        print('True: %-2d   Pred: %-2d   %s' % (y_test[i], y_pred[i], res))


if __name__ == '__main__':
    main()

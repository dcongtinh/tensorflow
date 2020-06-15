# import logging as logger
# logger.basicConfig(format='%(asctime)-15s %(clientip)s %(user)-8s %(message)s')
# logger.setLevel(logger.DEBUG)

from data import *
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tensorflow.keras as keras
from sklearn.metrics import confusion_matrix

import seaborn as sns

# constants for training
EPOCHS = 100

def trainModel(data, label, classCount):
    embedding = 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1'

    hub_layer = hub.KerasLayer(
        embedding,
        output_shape=[20],
        input_shape=[],
        dtype=tf.string
    )
    model = keras.Sequential()
    model.add(hub_layer)
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(classCount))

    optimizer = keras.optimizers.Adam(learning_rate=0.02)
    model.compile(
        optimizer = optimizer,
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(data, label, batch_size=12, epochs=EPOCHS)

    return model

def validateModel(model, data, label):
    results = model.evaluate(data, label, verbose=2)
    for name, value in zip(model.metrics_names, results):
        print("%s: %.3f" % (name, value))
    return None

def main():
    data = loadData()
    train, test = data['train'], data['test']

    print(data['labels'])
    print('Sample: Label %s' % data['labels'][0])
    print('==== A sample in train data ====')
    print(train[0])
    print('==== Equivalent in test data ====')
    print(test[0])

    # shuffle train data
    np.random.shuffle(train)

    # get data and label splitted
    train_data, train_label = train[:,0], train[:,1].reshape((train.shape[0],-1)).astype(int)
    test_data, test_label = test[:,0], test[:,1].reshape((test.shape[0],-1)).astype(int)

    print('=== Training  ===')
    model = trainModel(train_data, train_label, len(data['labels']))

    print('=== Validation ===')
    result = validateModel(model, test_data, test_label)


    print('=== Prediction ===')
    pred   = [np.argmax(x) for x in model.predict(test_data)]
    actual = test_label.reshape(-1)

    cm = confusion_matrix(actual, pred)
    print(cm)

    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap="Blues", xticklabels=data['labels'][::-1], yticklabels=data['labels'][::-1]); #annot=True to annotate cells
    plt.show()

if __name__=='__main__':
    main()

# import logging as logger
# logger.basicConfig(format='%(asctime)-15s %(clientip)s %(user)-8s %(message)s')
# logger.setLevel(logger.DEBUG)

from data import *
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import SparseCategoricalCrossentropy

epochs = 100
batch_size = 12


def trainModel(data, label, classes):
    embedding = 'https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1'

    hub_layer = hub.KerasLayer(
        embedding,
        output_shape=[20],
        input_shape=[],
        dtype=tf.string
    )
    model = Sequential()
    model.add(hub_layer)
    model.add(Dense(16, activation='relu'))
    model.add(Dense(classes))

    model.compile(
        optimizer='adam',
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    model.fit(data, label, batch_size=batch_size, epochs=epochs)

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
    train_data, train_label = train[:, 0], train[:, 1].reshape(
        (train.shape[0], -1)).astype(int)
    test_data, test_label = test[:, 0], test[:, 1].reshape(
        (test.shape[0], -1)).astype(int)

    print('=== Training  ===')
    model = trainModel(train_data, train_label, len(data['labels']))

    print('=== Validation ===')
    result = validateModel(model, test_data, test_label)

    # print('=== Prediction ===')
    # pred = model.predict([['']])
    # print(np.argmax(pred))


if __name__ == '__main__':
    main()

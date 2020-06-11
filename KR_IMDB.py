import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy

train_data, validation_data, test_data = tfds.load(name="imdb_reviews",
                                                   split=('train[:60%]', 'train[60%:]', 'test'), as_supervised=True)
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch[0])

embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
# hub_layer(train_examples_batch[:3])
model = Sequential()
model.add(hub_layer)
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.summary()

model.compile(optimizer='adam', loss=BinaryCrossentropy(
    from_logits=True), metrics=['accuracy'])

history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20, validation_data=validation_data.batch(512),
                    verbose=1)
results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))

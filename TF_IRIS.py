from sklearn import datasets
import numpy as np
import tensorflow as tf


class Model(object):
    def __init__(self):
        dimension = 4
        num_classes = 3
        self.W = tf.Variable(random.uniform((dimension, num_classes), -1, 1))
        self.b = tf.Variable(tf.zeros((num_classes,)))
        self.losses = []

    def train(self, X, y, learning_rate):
        with tf.GradientTape() as tape:
            current_loss = loss(y, self.predict(X))


data = datasets.load_iris()
X = data.data
y = data.target

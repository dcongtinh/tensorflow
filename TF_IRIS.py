import numpy as np
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


class Model(object):
    def __init__(self):
        dimension = 4
        num_classes = 3
        self.W = tf.Variable(tf.random.uniform(
            (dimension, num_classes), -1, 1))
        self.b = tf.Variable(tf.zeros((num_classes,)))
        self.losses = []

    def g(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return tf.matmul(x, self.W) + self.b

    def predict(self, x):
        return tf.nn.softmax(self.g(x))

    def loss(self, y_true, y_pred):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_true *
                                                      tf.math.log(y_pred), 1))
        return cross_entropy

    def train(self, X, y, learning_rate):
        with tf.GradientTape() as tape:
            current_loss = self.loss(y, self.predict(X))
        dW, db = tape.gradient(current_loss, [self.W, self.b])
        self.W.assign_sub(learning_rate * dW)
        self.b.assign_sub(learning_rate * db)
        self.losses.append(current_loss)
        return current_loss

    def plot_loss(self):
        plt.plot(self.losses, label='train_loss')
        plt.xlabel('Iter #')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.show()


data = datasets.load_iris()
X = data.data
y = data.target
y = y.reshape((y.shape[0], 1))
print(X)
enc = OneHotEncoder(handle_unknown='ignore')
y_trans = enc.fit_transform(y).toarray()

model = Model()
print("\nTraining ...")
Ws, bs = [], []
epochs = 1000
lr = 0.2

for epoch in range(epochs):
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    loss = model.train(X, y_trans, learning_rate=lr)
# model.plot_loss()
print(model.predict([[5.9, 3., 5.1, 1.8]]))

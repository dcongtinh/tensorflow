import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from sklearn.preprocessing import OneHotEncoder

print('Tensorflow Version:', tf.__version__)


class Model(object):
    def __init__(self):
        dimension = 2

        self.W1 = tf.Variable(tf.random.uniform((dimension, 2), -1, 1))
        self.b1 = tf.Variable(tf.random.uniform((2,), -1, 1))
        self.W2 = tf.Variable(tf.random.uniform((2, 1), -1, 1))
        self.b2 = tf.Variable(tf.random.uniform((1,), -1, 1))
        self.losses = []

    def predict(self, x):
        hidden = tf.nn.sigmoid(tf.matmul(x, self.W1) + self.b1)
        return tf.nn.sigmoid(tf.matmul(hidden, self.W2) + self.b2)

    def loss(self, y, y_pred):
        # loss_value = tf.reduce_mean(-tf.reduce_sum(y*tf.math.log(y_pred)+(tf.constant(1.0)-y)*tf.math.log(tf.constant(1.0) - y_pred), 1))
        # print('square:',tf.square(y-y_pred))
        loss_value = tf.reduce_mean(tf.square(y-y_pred))
        return loss_value

    def train(self, X, y, learning_rate = 0.05):
        with tf.GradientTape() as tape:
            current_loss = self.loss(y, self.predict(X))
        dW1, db1, dW2, db2 = tape.gradient(current_loss, [self.W1, self.b1, self.W2, self.b2])
        self.W1.assign_sub(learning_rate * dW1)
        self.b1.assign_sub(learning_rate * db1)
        self.W2.assign_sub(learning_rate * dW2)
        self.b2.assign_sub(learning_rate * db2)
        self.losses.append(current_loss)
        return current_loss

    def plot_loss(self):
        plt.plot(self.losses, label='train_loss')
        plt.xlabel('Iter #')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.show()

X = [[0., 0],
      [0, 1],
      [1, 0],
      [1, 1]]
Y = [[0.],
      [1],
      [1],
      [0]]
# enc = OneHotEncoder(handle_unknown='ignore')
# y_trans = enc.fit_transform(y).toarray()
model = Model()
print("\nTraining ...")

epochs = 5000
lr = 1

print(model.predict(X))
for epoch in range(epochs):
    loss = model.train(X, Y, learning_rate=lr)
    print(loss)

# model.plot_loss()
print(model.predict(X))

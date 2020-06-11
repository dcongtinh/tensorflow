import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print('Tensorflow Version:', tf.__version__)


class Model(object):
    def __init__(self):
        dimension = 2
        self.W1 = tf.Variable(tf.random.uniform((dimension, 2), -1, 1))
        self.b1 = tf.Variable(tf.zeros((2,)))
        self.W2 = tf.Variable(tf.random.uniform((2, 1), -1, 1))
        self.b2 = tf.Variable(tf.zeros((1,)))
        self.losses = []

    def predict(self, x):
        hidden = tf.nn.sigmoid(tf.matmul(x, self.W1) + self.b1)
        return tf.nn.sigmoid(tf.matmul(hidden, self.W2) + self.b2)

    def loss(self, y_true, y_pred):
        loss_value = tf.reduce_mean(-tf.reduce_sum(y_true*tf.math.log(y_pred)+(
            tf.constant(1.0)-y_true)*tf.math.log(tf.constant(1.0) - y_pred), 1))
        # print('square:',tf.square(y-y_pred))
        # loss_value = tf.reduce_mean(tf.square(y_true-y_pred))
        return loss_value

    def train(self, X, y, learning_rate=0.05):
        with tf.GradientTape() as tape:
            current_loss = self.loss(y, self.predict(X))
        dW1, db1, dW2, db2 = tape.gradient(
            current_loss, [self.W1, self.b1, self.W2, self.b2])
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


X = [[0.0, 0.0],
     [0.0, 1.0],
     [1.0, 0.0],
     [1.0, 1.0]]
y = [[0],
     [1],
     [1],
     [0]]

model = Model()
print("\nTraining ...")

epochs = 3000
lr = 0.3

for epoch in range(epochs):
    loss = model.train(X, y, learning_rate=lr)
    # print("Epoch %d: Loss=%.6f" % (epoch+1, loss))

print('W1 =', model.W1)
print('b1 =', model.b1)
print('W2 =', model.W2)
print('b2 =', model.b2)
print(model.predict(X))
model.plot_loss()

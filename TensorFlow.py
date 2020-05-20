# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

print('Tensorflow Version:', tf.__version__)


class Model(object):
    def __init__(self):
        # w1 w2 with 1 output layer
        self.W = tf.Variable(tf.random.uniform((2, 1), -1, 1))
        self.b = tf.Variable(tf.zeros((1,)))
        self.losses = []

    def train(self, X, y, learning_rate):
        with tf.GradientTape() as tape:
            current_loss = self.loss(y, self.predict(X))
        dW, db = tape.gradient(current_loss, [self.W, self.b])
        self.W.assign_sub(learning_rate * dW)
        self.b.assign_sub(learning_rate * db)
        self.losses.append(current_loss)
        return current_loss

    def g(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return tf.matmul(x, self.W) + self.b

    def predict(self, x):
        return tf.sigmoid(self.g(x))

    def loss(self, y_true, y_pred):
        # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def plot_loss(self):
        plt.plot(self.losses, label='train_loss')
        plt.xlabel('Iter #')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.show()

    def plot_2d_data(self, X, y):
        plt.plot(X[pos_idx][:, 0], X[pos_idx][:, 1], 'bo')
        plt.plot(X[neg_idx][:, 0], X[neg_idx][:, 1], 'rx')
        W = self.W.numpy()
        b = self.b.numpy()
        # w1x1 + w2x2 + b = 0
        x1 = np.array([0, -b/W[1]])
        x2 = (-b - W[0]*x1)/W[1]
        plt.plot(x1, x2)
        plt.show()

    def plot_data(self, X, y, y_pred):
        fig = plt.figure(figsize=(10, 5))
        fig.suptitle('Display data 2D & 3D', fontweight='bold')
        # plot 2d
        plt2d = fig.add_subplot(121)
        # plt2d.set_title('2D')
        plt2d.plot(X[pos_idx][:, 0], X[pos_idx][:, 1], 'bo')
        plt2d.plot(X[neg_idx][:, 0], X[neg_idx][:, 1], 'rx')
        W = self.W.numpy()
        b = self.b.numpy()
        # w1x1 + w2x2 + b = 0
        x1 = np.array([0, -b/W[1]])
        x2 = (-b - W[0]*x1)/W[1]
        plt2d.plot(x1, x2)

        # plot 3d
        plt3d = fig.add_subplot(122, projection='3d')
        # plt3d.set_title('3D')
        data = [('o', 'b', X[pos_idx], y[pos_idx]),  # Positive class
                ('x', 'r', X[neg_idx], y[neg_idx]),  # Negative class
                ('*', 'g', X, y_pred)]               # Predicted class
        for m, c, X, y in data:
            xs = X[:, 0]
            ys = X[:, 1]
            zs = y[:, 0]
            plt3d.scatter(xs, ys, zs, marker=m, color=c)
        plt3d.set_xlabel('x1')
        plt3d.set_ylabel('x2')
        plt3d.set_zlabel('y')
        plt.show()


X = np.array([[0.0, 0.0],
              [0.0, 1.0],
              [1.0, 0.0],
              [1.0, 1.0]])
y = np.array([[0],
              [0],
              [0],
              [1]])

pos_idx = np.array([i for i in range(len(y)) if y[i][0] == 1])
neg_idx = np.array([i for i in range(len(y)) if y[i][0] == 0])

model = Model()
print("\nTraining ...")
Ws, bs = [], []
epochs, lr = 1000, 0.2
for epoch in range(epochs):
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    loss = model.train(X, y, learning_rate=lr)
    print("Epoch %d: W1=%.6f, W2=%.6f, b=%.6f, Loss=%.6f" %
          (epoch+1, Ws[-1][0], Ws[-1][1], bs[-1], loss))

print(model.W)
print(model.b)
print('W1 =', model.W.numpy()[0][0], 'W2 =', model.W.numpy()[1][0])
print('b  =', model.b.numpy())
y_pred = model.predict(X)
print(y_pred)

model.plot_loss()
# model.plot_2d_data(X, y)
model.plot_data(X, y, y_pred.numpy())

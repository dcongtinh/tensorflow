import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder

print('Tensorflow Version:', tf.__version__)


class Model(object):
    def __init__(self):
        # w1 w2 with 3 output layers
        dimension = 2
        num_classes = 3  # or output layer
        self.W = tf.Variable(tf.random.uniform(
            (dimension, num_classes), -1, 1))
        self.b = tf.Variable(tf.zeros((num_classes,)))
        self.losses = []

    def g(self, x):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        return tf.matmul(x, self.W) + self.b

    def predict(self, x):
        return tf.nn.softmax(self.g(x))

    def loss(self, y, y_pred):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y *
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


X = np.array([[0., 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0.],
              [1],
              [1],
              [2]])

enc = OneHotEncoder(handle_unknown='ignore')
y_trans = enc.fit_transform(y).toarray()

model = Model()
print("\nTraining ...")
epochs, lr = 1000, 0.2


for epoch in range(epochs):
    loss = model.train(X, y_trans, learning_rate=lr)
    print("Epoch %d: Loss=%.6f" % (epoch+1, loss))

model.plot_loss()
print(model.predict([[-1., -1]]))

import time
import numpy as np
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score


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

    def metrics(self, y_true, y_pred, average='weighted'):
        for y in y_pred:
            idx_max = np.argmax(y)
            y[:] = 0.
            y[idx_max] = 1.
        self.accuracy = accuracy_score(y_true, y_pred)*100
        self.precision = precision_score(
            y_true, y_pred, average=average, zero_division=1)*100
        self.f1 = f1_score(y_true, y_pred, average=average)*100
        self.recall = recall_score(y_true, y_pred, average=average)*100
        print('Accuracy  = {}%'.format(model.accuracy))
        print('Precision = {}%'.format(model.precision))
        print('F1_Score  = {}%'.format(model.f1))
        print('Recall    = {}%'.format(model.recall))

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
# print(X)
enc = OneHotEncoder(handle_unknown='ignore')
y_trans = enc.fit_transform(y).toarray()

model = Model()
print("\nTraining ...")
Ws, bs = [], []
epochs = 10000
lr = 0.2

start_time = time.time()
for epoch in range(epochs):
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    loss = model.train(X, y_trans, learning_rate=lr)
end_time = time.time()
print("Training time: %fs\n" % (end_time-start_time))
model.plot_loss()
model.metrics(y_trans, model.predict(X).numpy())
print(model.predict([[5.9, 3., 5.1, 1.8]]))  # Lớp 2

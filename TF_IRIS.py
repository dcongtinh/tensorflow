import time
import numpy as np
import tensorflow as tf
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split as splitter
from sklearn.metrics import confusion_matrix

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
        print('Recall    = {}%'.format(model.recall))
        print('F1_Score  = {}%'.format(model.f1))
        matrix = confusion_matrix([np.argmax(y_ele) for y_ele in y_true], [np.argmax(y_ele) for y_ele in y_pred])
        return matrix

    def plot_loss(self):
        plt.plot(self.losses, label='train_loss')
        plt.xlabel('Iter #')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')


data = datasets.load_iris()
X = data.data
y = data.target
y = y.reshape((y.shape[0], 1))
# print(X)
enc = OneHotEncoder(handle_unknown='ignore')
y_trans = enc.fit_transform(y).toarray()

X_train, X_test, Y_train, Y_test = splitter(X, y_trans, test_size=1.0/3, shuffle=True, stratify=y_trans)
model = Model()
print("\nTraining ...")
epochs, lr = 1000, 0.2

start_time = time.time()
for epoch in range(epochs):
    loss = model.train(X_train, Y_train, learning_rate=lr)
    print("Epoch %d: Loss=%.6f" % (epoch+1, loss))
end_time = time.time()

print("Training time: %fs\n" % (end_time-start_time))
print('W =', model.W.numpy())
print('b =', model.b.numpy())

print('===== Train metrics ======')
matrix = model.metrics(Y_train, model.predict(X_train).numpy())
print(matrix)
print('====== Test metrics ======')
matrix = model.metrics(Y_test, model.predict(X_test).numpy())
print(matrix)

model.plot_loss()
plt.show()

import seaborn as sns
ax = plt.subplot()
sns.heatmap(matrix, annot=True, ax = ax, cmap="Blues"); #annot=True to annotate cells

# labels, title and ticks
ax.set_xlabel('True labels');
ax.set_ylabel('Predicted labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['0', '1', '2']); 
ax.yaxis.set_ticklabels(['0', '1', '2']);
plt.show()
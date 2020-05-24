import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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

    def loss(self, y_true, y_pred):
        # loss_value = tf.reduce_mean(-tf.reduce_sum(y_true*tf.math.log(y_pred)+(
        #     tf.constant(1.0)-y_true)*tf.math.log(tf.constant(1.0) - y_pred), 1))
        # print('square:',tf.square(y-y_pred))
        loss_value = tf.reduce_mean(tf.square(y_true-y_pred))
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

def doRender(axLoss, lossValues, axData, X, Y, pred):

    axData.cla()  # clear axes
    axLoss.cla()
    axData.set_title('Data presentation')
    axLoss.set_title('Loss through epoches')
    axLoss.set_xlabel('Epoch')
    axLoss.set_ylabel('Loss')
    for _ in range(len(Y)):
        axData.scatter(X[_][0], X[_][1], Y[_][0], c='b', marker='o')
    for _ in range(len(Y)):
        axData.scatter(X[_][0], X[_][1], pred[_][0], c='g', marker='x')

    axLoss.set_ylim(0, 1)
    axLoss.set_xlim(0, epochs)
    axLoss.plot(lossValues)
    plt.pause(0.01)

'''
    init axes
'''
(axData, axLoss) = plt.subplot(121, projection='3d'), plt.subplot(122)
'''
    data 
'''
X = [[0., 0],
     [0, 1],
     [1, 0],
     [1, 1]]
Y = [[0.],
     [1],
     [1],
     [0]]

model = Model()
print("\nTraining ...")

epochs = 5000
lr = 0.5
'''
Visualization config
'''
renderEvery = 100
lossValues = []

# Now render initial values
for _ in range(len(Y)):
    axData.scatter(X[_][0], X[_][1], Y[_][0], marker='o')
plt.show(block=False)

print(model.predict(X))
for ep in range(epochs):
    loss = model.train(X, Y, learning_rate=lr)
    print("Epoch %d: Loss=%.6f" % (ep+1, loss))
    lossValues.append(loss)
    if ep % renderEvery == 0:
        pred = model.predict(X).numpy()
        doRender(axLoss, lossValues, axData, X, Y, pred)
print('prediction:',pred)

pred = model.predict(X).numpy()
doRender(axLoss, lossValues, axData, X, Y, pred)
plt.ioff()  # interaction mode
plt.show()
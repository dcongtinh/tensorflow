import tensorflow.compat.v1 as tf
import numpy as np

num_input = 4
num_classes = 2  # 0 1

# Create model
x = tf.placeholder(tf.float32, [num_input, num_classes], name="x")
W = tf.Variable(tf.zeros([num_classes, num_input]), name="W")
b = tf.Variable(0, name="b")

g = tf.matmul(x, W) + b

# Construct model
prediction = tf.sigmoid(g)  # y
label = tf.placeholder(tf.float32, [None, 1], name="y_")  # y_

# Define loss and optimizer
cross_entropy = tf.reduce_mean(-tf.reduce_sum(label *
                                              tf.log(prediction), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.5)  # learning_rate 0.5
train_step = optimizer.minimize(cross_entropy)

X = np.array([[0., 0.],
              [0., 1.],
              [1., 0.],
              [1., 1.]])
y = np.array([[0.], [0.], [0.], [1.]])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# writer = tf.summary.FileWriter('./graphs', sess.graph)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: X, label: y}) * 100)

# res = sess.run(train_step, feed_dict={x: X, label: y})
# print(res)
# writer.close()

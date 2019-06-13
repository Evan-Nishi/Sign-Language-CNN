'''
https://www.kaggle.com/soumikrakshit/sign-language-translation-cnn was used for reference
'''

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

tf.reset_default_graph()

data = pd.read_csv('sign_mnist_train.csv')
test_data = pd.read_csv('sign_mnist_test.csv')

final_test_data = test_data.iloc[:, 1:].values
final_test_labels = test_data.iloc[:, :1].values.flatten()
final_data = data.iloc[:, 1:].values
final_labels = data.iloc[:, :1].values.flatten()

final_data1 = data.iloc[:, :].values
print(final_data1[1:2, :])
print(final_data[1:2, :])

#image height/width and other vars
h = 28
w = 28
steps = 20000
batch_size = 32
num_classes = 26
color_channels = 1
final_data = final_data.reshape(-1, h, w, 1)
final_test_data = (-1, h, w, 1)
test_img = final_data[1:2, :]
epochs = 70


class TheActualNetwork:
    def __init__(self, h, w, color_channels, num_classes):
        self.image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, h, w, color_channels], name="final_data")
        self.labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[None], name="final_labels")
        conv_layer1 = tf.layers.conv2d(self.image_placeholder, filters=64, kernel_size=[2,2], padding='SAME', activation=tf.nn.relu)
        pool_layer1 = tf.layers.max_pooling2d(conv_layer1, pool_size=[2,2], strides=2)
        one_flat_boi = tf.layers.flatten(pool_layer1)
        dense_layer = tf.layers.dense(one_flat_boi, 1024, activation=tf.nn.relu)
        dropout_layer = tf.layers.dropout(dense_layer, rate=0.4, training=True)
        outputs = tf.layers.dense(dropout_layer, num_classes)

        self.choice = tf.argmax(outputs, axis=1)
        self.probability = tf.nn.softmax(outputs)
        self.accuracy, self.accuracy_op = tf.metrics.accuracy(self.labels_placeholder, self.choice)
        one_hot_labels = tf.one_hot(indices=tf.cast(self.labels_placeholder, dtype=tf.int32), depth=num_classes)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=outputs)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
        self.train_operation = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())


cnn = TheActualNetwork(28, 28, 1, 26)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    done_epochs = 0
    for epoch in range(epochs):
        position = 0
        done_epochs += 1
        done_batches = 0
        print("Epoch #", done_epochs)
        while position < steps:
            done_batches += 1
            position += batch_size
            stuff = sess.run((cnn.train_operation, cnn.accuracy_op), feed_dict={cnn.image_placeholder: final_data[position:position + batch_size],cnn.labels_placeholder: final_labels[position:position + batch_size]})
            if done_batches % 200 == 0:
                print(stuff, 'on epoch:', done_epochs, "accuracy: ",stuff )
        print(sess.run(cnn.choice, feed_dict={cnn.image_placeholder: test_img}))

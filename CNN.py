import numpy as np
import tensorflow as tf
import os
import pandas as pd
import cv2
import matplotlib.pyplot as plt

data = pd.read_csv("sign_mnist_train.csv", skiprows=1)
test_data = pd.read_csv("sign_mnist_test.csv",skiprows=1)

final_test_data = test_data.iloc[:, 1:].values
final_test_labels = test_data.iloc[:, :1].values.flatten

final_data = data.iloc[:, 1:].values
final_labels = data.iloc[:, :1].values.flatten()
'''
step1 = np.asarray(data.head(7))
print(step1.shape)
step2 = step1[6:7, 1:]
print(step2)
done = step2.reshape(28,28)
plt.imshow(done)
plt.show()
'''


#from the kaggle demo==================================================================
def change_batch (batch_size, data, labels):
    index = np.arange(0, len(data))
    np.random.shuffle(index)
    array_index = np.array(index[:batch_size])
    data_shuffle = [data[num] for num in array_index]
    label_shuffle = [labels[num] for num in array_index]
    print(np.asarray(data_shuffle))
    print(np.asarray(label_shuffle))
#======================================================================================


change_batch(100,final_data,final_labels)
h = 28
w = 28
steps = 1000
batch_size = 32
num_classes = 26
color_channels = 3



class TheActualNetwork:
    def __init__(self, h, w, color_channels, numclass):
        image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, h, w, color_channels], name="inputs")
        conv_layer1 = tf.layers.conv2d(image_placeholder, filters=64, kernel_size = [2,2], padding='same',activation=tf.nn.relu)
        pool_layer1 = tf.layers.max_pooling2d(conv_layer1, pool_size=[2,2], strides=2)
        one_flat_boi = tf.layers.flatten(pool_layer1)
        dense_layer = tf.layers.dense(one_flat_boi, 1024, activation=tf.nn.relu)
        dropout_layer = tf.layers.dropout(dense_layer, rate=0.4, training=True)
        outputs = tf.layers.dense(dropout_layer, num_class)

        self.choice = tf.argmax(outputs, axis=1)
        self.probability = tf.nn.softmax(outputs)
        self.label_placeholder = tf.placeholder(dtype=tf.float32, shape=[None], name="input labels")
        self.accuracy, self.accuracy_op = tf.metrics.accuracy(self.labels_placeholder, self.choice)
        one_hot_labels = tf.one_hot(indices=tf.cast(self.labels_placehoder, dtype=tf.int32), depth=num_classes)
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=outputs)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
        self.train_operation = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())

data_train =

with tf.Session() as sess:


cnn = TheActualNetwork(h, w, color_channels,  num_classes)

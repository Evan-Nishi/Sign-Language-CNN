'''
https://www.kaggle.com/soumikrakshit/sign-language-translation-cnn was used for reference
'''

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

tf.reset_default_graph()

data = pd.read_csv('sign_mnist_train.csv', skiprows=1)
test_data = pd.read_csv('sign_mnist_test.csv',skiprows=1)

final_test_data = test_data.iloc[:, 1:].values
final_test_labels = test_data.iloc[:, :1].values.flatten()
final_data = data.iloc[:, 1:].values
final_labels = data.iloc[:, :1].values.flatten()
print(final_labels)
print(final_data)
print(final_test_labels)
print(final_test_data)

'''
step1 = np.asarray(data.head(7))
print(step1.shape)
import matplotlib.pyplot as plt
step2 = step1[6:7, 1:]
print(step2)
done = step2.reshape(28,28)
plt.imshow(done)
plt.show()

#from the kaggle demo===================================================================================================
def change_batch (batch_size, data, final_labels):
    index = np.arange(0, len(data))
    np.random.shuffle(index)
    array_index = np.array(index[:batch_size])
    data_shuffle = [data[num] for num in array_index]
    label_shuffle = [final_labels[num] for num in array_index]
    print(np.asarray(data_shuffle))
    print(np.asarray(label_shuffle))
    
change_batch(100,final_data,final_labels)

sess.run((TheActualNetwork.train_operation, TheActualNetwork.accuracy_op), feed_dict={TheActualNetwork.input_layer: final_data[position:position + batch_size],TheActualNetwork.label_placeholder: final_test_labels[position:position + batch_size]})
#=======================================================================================================================
'''

h = 28
w = 28
epochs = 5000
batch_size = 32
num_classes = 26
color_channels = 1
final_data = final_data.reshape(-1, h, w, 1)
final_test_data = (-1, h, w, 1)
test_img = final_data[1:2, :]



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
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-2)
        self.train_operation = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())


cnn = TheActualNetwork(28, 28, 1, 26)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    position = 0
    while position < epochs:
        print(sess.run((cnn.train_operation, cnn.accuracy_op), feed_dict={cnn.image_placeholder: final_data[position:position + batch_size], cnn.labels_placeholder: final_labels[position:position + batch_size]}))
        position += batch_size
    print(sess.run(cnn.choice, feed_dict={cnn.image_placeholder: test_img}))

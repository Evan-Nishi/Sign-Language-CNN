import tensorflow as tf
import vidcap
import os
import cv2

img_done = cv2.imread('new_img.jpg')
px = img_done[:]

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('my_test_model-2000.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    graph = tf.get_default_graph()

    input_x = graph.get_tensor_by_name("final_data:0")
    result = graph.get_tensor_by_name("result:0")

    feed_dict = {input_x:  }

    predictions = result.eval(feed_dict=feed_dict)
os.remove('new_img.jpg')

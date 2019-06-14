import tensorflow as tf
import os
import cv2


img_done = cv2.imread('new_img.jpg')
px = img_done[:]
px_done1 = px[:, :, :1]
px_done = px_done1.reshape(1, 28, 28, 1)

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('my_test_model-2000.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    graph = tf.get_default_graph()
    graph = tf.get_default_graph()

    input_x = graph.get_tensor_by_name("final_data:0")
    result = graph.get_tensor_by_name("final_answer:0")

    feed_dict = {input_x: px_done}

    predictions = result.eval(feed_dict=feed_dict)
    print(predictions)
delete_input = input('do you want to delete the jpg file? ')

if 'y' in delete_input:
    os.remove('new_img.jpg')
    print('file deleted')
else:
    print('file kept')

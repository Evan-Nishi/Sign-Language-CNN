import tensorflow as tf
import os
import cv2
import time


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

ans_to_text = {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}
nums_keys = ans_to_text.keys()
ans1 = str(predictions)
ans2 = ans1.strip('[')
ans3 = ans2.strip(']')

index_num = int(ans3)
print(ans_to_text.get(index_num))

time.sleep(3)
delete_input = input('do you want to delete the jpg file? ')

if 'y' in delete_input:
    os.remove('new_img.jpg')
    print('file deleted')
else:
    print('file kept')

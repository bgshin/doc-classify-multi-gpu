import tensorflow as tf
import os

from datasets import tweets
from tensorflow.contrib.slim.nets import inception
from preprocessing import inception_preprocessing
import time
import numpy as np
from datetime import datetime

slim = tf.contrib.slim
BATCH_SIZE = 158  # How many images can pass through *a single GPU*
# (if they are different specs you'll have to adapt the script)
MAX_STEPS = 1000000
NUM_GPUS = 2

tweets_data_dir = '/tmp/bgshin'

def load_batch(data_provider, batch_size=4):
    label = data_provider.get(['label'])[0]
    # txt, label = data_provider.get(['txt', 'label'])
    # print 'txt.shape', txt.shape
    print 'label.shape', label.shape

    # Batch it up.
    # txts, labels = tf.train.batch(
    #     [txt, label],
    #     batch_size=batch_size,
    #     num_threads=1,
    #     capacity=2 * batch_size)

    labels = tf.train.batch(
        [label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2 * batch_size)

    # return txts, labels
    return None, labels


    # features = tf.parse_single_example(
    #     data_provider,
    #     features={
    #         'txt': tf.FixedLenFeature([], tf.string),
    #         'label': tf.FixedLenFeature([], tf.string)
    #     })
    #
    # txt = tf.decode_raw(features['txt'], tf.float32)
    # label = tf.decode_raw(features['label'], tf.uint8)
    #
    # resized_x = tf.reshape(txt, [seqlen, w2vdim])
    # resized_label = tf.reshape(label, [nclass])
    #
    #
    # txts, labels = tf.train.shuffle_batch([resized_x, resized_label],
    #                                       batch_size=batch_size,
    #                                       capacity=30,
    #                                       num_threads=2,
    #                                       min_after_dequeue=10)
    #
    # return txts, labels

dataset = tweets.get_split('trn', tweets_data_dir)
print ("number of classes: ", dataset.num_classes)
print ("number of samples: ", dataset.num_samples)
data_provider = slim.dataset_data_provider.DatasetDataProvider(
    dataset, common_queue_capacity=32,
    common_queue_min=8, shuffle=True)

txts, labels = load_batch(data_provider)

print 'd'


# import numpy as np
# import tensorflow as tf
# from tensorflow.python.ops import math_ops
# from slimcnn import slimcnn
# import time, os
# from datetime import datetime
#
# slim = tf.contrib.slim
#
#
#
# datasets = []
# tfrecords_filename = '/tmp/bgshin/tw.trn.tfrecords'
#
# filename_queue = tf.train.string_input_producer(
#     [tfrecords_filename], num_epochs=10)
#
# reader = tf.TFRecordReader()
# _, serialized_example = reader.read(filename_queue)
#
# features = tf.parse_single_example(
#     serialized_example,
#     # Defaults are not specified since both keys are required.
#     features={
#         'txt': tf.FixedLenFeature([], tf.string),
#         'label': tf.FixedLenFeature([], tf.string)
#     })
#
# txt = features['txt']
# label = features['label']
#
# txts, labels = tf.train.shuffle_batch([txt, label],
#                                       batch_size=4,
#                                       capacity=30,
#                                       num_threads=2,
#                                       min_after_dequeue=10)
#
# # Batch the variable length tensor with dynamic padding
# txts, labels = tf.train.batch(
#     tensors=[txt, label],
#     batch_size=5,
#     dynamic_pad=True,
#     name="y_batch"
# )
#
#
print 'a'
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
print 'b'
with tf.Session() as sess:
    sess.run(init_op)
    print 'c'
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print 'd'
    # tt, ll = sess.run([txts, labels])
    ll = sess.run([labels])
    print 'e'
    coord.request_stop()
    coord.join(threads)

for l in ll:
    print l
# for t in tt:
#     print t

print 'f'

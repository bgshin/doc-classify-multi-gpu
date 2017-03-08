import numpy as np
import tensorflow as tf
from utils.word2vecReader import Word2Vec
from utils import cnn_data_helpers

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def load_w2v(w2vdim):
    model_path = '../data/emory_w2v/w2v-%d.bin' % w2vdim
    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    print("The vocabulary size is: " + str(len(model.vocab)))

    return model

w2vdim = 400
max_len = 60
w2vmodel = load_w2v(w2vdim)

trnpath = '/home/bgshin/works/cnntweets/data/tweets/txt/trn'

tfrecords_filename = '/tmp/bgshin/tw.trn.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

x_train, y_train = cnn_data_helpers.load_data_new('trn', w2vmodel, max_len)

for idx in range(len(x_train)):
    x = x_train[idx]
    y = y_train[idx]
    example = tf.train.Example(
        features=tf.train.Features(feature={
            'txt': _bytes_feature(x),
            'label': _bytes_feature(y)}))

    writer.write(example.SerializeToString())

with open(trnpath, 'rt') as trntxt:
    for line in trntxt:
        line_list = line.split('\t')
        label = line_list[1]
        tw = line_list[2].replace('\n', '')
        # print label, tw, len(tw.split(' '))

        example = tf.train.Example(
            features=tf.train.Features(feature={
            'txt': _bytes_feature(tw),
            'label': _bytes_feature(label)}))

        writer.write(example.SerializeToString())



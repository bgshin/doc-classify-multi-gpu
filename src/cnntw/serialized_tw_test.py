"""Tests for cnnt input."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from src.utils.butils import Timer
import cnnt_input
import numpy as np
from src.utils.word2vecReader import Word2Vec
from src.utils import cnn_data_helpers
from tqdm import tqdm

def load_w2v(w2vdim):
    model_path = '../../data/emory_w2v/w2v-%d.bin' % w2vdim
    model = Word2Vec.load_word2vec_format(model_path, binary=True)
    print("The vocabulary size is: " + str(len(model.vocab)))

    return model


class CNNTInputTest(tf.test.TestCase):
    def _test(self, target):
        with Timer('w2v..'):
            w2vmodel = load_w2v(400)

        x_train, y_train = cnn_data_helpers.load_data_new(target, w2vmodel, 60)


        filename = '../../data/tw.%s.tfrecords' % target

        gpu_options = tf.GPUOptions(visible_device_list=str('2,3'), allow_growth=True)
        config=tf.ConfigProto(gpu_options=gpu_options)

        with self.test_session(config = config) as sess:
            q = tf.train.string_input_producer([filename])
            q.enqueue([filename]).run()
            q.close().run()
            result = cnnt_input.read_cnnt(q)

            with Timer('run...'):
                for i in tqdm(range(1588)):
                    key, label, features = sess.run([
                        result.key, result.label, result.features])
                    self.assertEqual("%s:%d" % (filename, i), tf.compat.as_text(key))
                    self.assertAllEqual(x_train[i], features)
                    self.assertAllEqual(y_train[i], label)



                # print (tf.compat.as_text(key), label, features)
                # self.assertEqual(labels[i], label)
                # self.assertAllEqual(expected[i], features)


    def testSimple(self):
        for t in ['dev']:
            self._test(t)

if __name__ == "__main__":
    tf.test.main()




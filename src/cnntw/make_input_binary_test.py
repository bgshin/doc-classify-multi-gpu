"""Tests for cnnt input."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from src.utils.butils import Timer
import cnnt_input
from tqdm import tqdm
import numpy as np


class CNNTInputTest(tf.test.TestCase):
    def _test(self, target):
        filename = '../../data/tw.%s.tfrecords' % target

        with self.test_session() as sess:
            q = tf.train.string_input_producer([filename])
            q.enqueue([filename]).run()
            q.close().run()
            result = cnnt_input.read_cnnt(q)




            batch_size = 4
            # Ensure that the random shuffling has good mixing properties.
            min_fraction_of_examples_in_queue = 0.4
            NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1588
            min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                     min_fraction_of_examples_in_queue)

            features = tf.expand_dims(result.features, 2)
            result.label.set_shape([3])

            ff, ll = tf.train.batch(
                [features, result.label],
                batch_size=batch_size,
                num_threads=2,
                capacity=min_queue_examples + 3 * batch_size)

            # Start the queue runners.
            tf.train.start_queue_runners(sess=sess)
            lls, ffs = sess.run([ll, ff])

            print (lls.shape)
            print (ffs.shape)




            # ff, ll = tf.train.shuffle_batch(
            #     [result.features, result.label],
            #     batch_size=batch_size,
            #     num_threads=2,
            #     capacity=min_queue_examples + 3 * batch_size,
            #     min_after_dequeue=min_queue_examples)

            # with Timer('run...'):
            #     ll, ff = sess.run([features, labels])
            #
            for idx, f in enumerate(ffs):
                print(lls[idx], f)
            #
            #
        # with Timer('run...'):
        #     for i in tqdm(range(1588)):
        #     # for i in range(1588):
        #         key, label, features = sess.run([
        #             result.key, result.label, result.features])
        #         self.assertEqual("%s:%d" % (filename, i), tf.compat.as_text(key))


                # print (tf.compat.as_text(key), label, features)
                # self.assertEqual(labels[i], label)
                # self.assertAllEqual(expected[i], features)


    def testSimple(self):
        for t in ['dev']:
            self._test(t)

if __name__ == "__main__":
    tf.test.main()




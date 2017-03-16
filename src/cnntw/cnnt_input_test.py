"""Tests for cnnt input."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

import cnnt_input
import numpy as np


class CNNTInputTest(tf.test.TestCase):
    def _record(self, label, num):
        input_size = 60 * 400
        x = np.ones([60,400]) * num
        x = x.reshape([input_size])
        x = np.append(float(label), x)
        # x = np.array([label] + [num] * input_size)
        record = x.tostring()
        expected = np.array([[[num]] * 400] * 60)
        return record, expected[:,:,0]

    def testSimple(self):
        labels = [1, 0, 2]
        records = [self._record(labels[0], 0.1),
                   self._record(labels[1], 0.2),
                   self._record(labels[2], 0.3)]
        contents = b"".join([record for record, _ in records])
        expected = [expected for _, expected in records]
        filename = os.path.join(self.get_temp_dir(), "cnnt")
        open(filename, "wb").write(contents)

        with self.test_session() as sess:
            q = tf.train.string_input_producer([filename])
            q.enqueue([filename]).run()
            q.close().run()
            result = cnnt_input.read_cnnt(q)

            for i in range(3):
                key, label, features = sess.run([
                    result.key, result.label, result.features])
                self.assertEqual("%s:%d" % (filename, i), tf.compat.as_text(key))
                self.assertEqual(labels[i], label)
                self.assertAllEqual(expected[i], features)

            with self.assertRaises(tf.errors.OutOfRangeError):
                sess.run([result.key, result.features])

if __name__ == "__main__":
    tf.test.main()




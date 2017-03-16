"""Routine for decoding the cnnt binary file format."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Process images of this size. Note that this differs from the original CIFAR
# image size of 32 x 32. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_cnnt(filename_queue):
    class CNNTRecord(object):
        pass
    result = CNNTRecord()

    label_bytes = 1*8
    result.maxlen = 60
    result.w2vdim = 400
    input_bytes = (result.maxlen * result.w2vdim)*8
    record_bytes = label_bytes + input_bytes

    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(filename_queue)

    value_as_floats = tf.decode_raw(value, tf.float64)
    result.label = tf.cast(value_as_floats[0], tf.int32)

    features = value_as_floats[1:1 + record_bytes]
    result.features = tf.reshape(features, [result.maxlen, result.w2vdim])

    return result



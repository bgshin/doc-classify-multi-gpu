# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Routine for decoding the CIFAR-10 binary file format."""

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
  """Reads and parses examples from CIFAR10 data files.

  Recommendation: if you want N-way read parallelism, call this function
  N times.  This will give you N independent Readers reading different
  files & positions within those files, which will give better mixing of
  examples.

  Args:
    filename_queue: A queue of strings with the filenames to read from.

  Returns:
    An object representing a single example, with the following fields:
      height: number of rows in the result (32)
      width: number of columns in the result (32)
      depth: number of color channels in the result (3)
      key: a scalar string Tensor describing the filename & record number
        for this example.
      label: an int32 Tensor with the label in the range 0..9.
      uint8image: a [height, width, depth] uint8 Tensor with the image data
  """

  class CNNTRecord(object):
    pass
  result = CNNTRecord()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1*8  # 2 for CIFAR-100
  result.maxlen = 60
  result.w2vdim = 400
  input_bytes = (result.maxlen * result.w2vdim)*8
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  record_bytes = label_bytes + input_bytes

  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  result.key, value = reader.read(filename_queue)

  # Convert from a string to a vector of uint8 that is record_bytes long.
  # record_bytes = tf.decode_raw(value, tf.float32)

  # Decode the result from bytes to int32
  # record_bytes = tf.decode_raw(value, tf.int32, little_endian=True)
  value_as_floats = tf.decode_raw(value, tf.float64)
  result.label = value_as_floats[0]

  # Decode the result from bytes to float32
  # value_as_floats = tf.decode_raw(value, tf.float32)
  features = value_as_floats[1:1 + record_bytes]
  result.features = tf.reshape(features, [result.maxlen, result.w2vdim])


  #
  # # The first bytes represent the label, which we convert from uint8->int32.
  # result.label = tf.cast(
  #     tf.strided_slice(record_bytes, [0], [label_bytes]), tf.float32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  # result.depth_major = tf.reshape(
  #     tf.strided_slice(record_bytes, [label_bytes],
  #                      [label_bytes + input_bytes], tf.float32),
  #     [result.maxlen, result.w2vdim])
  # Convert from [depth, height, width] to [height, width, depth].
  # result.uint8image = tf.transpose(depth_major, [1, 2, 0])

  return result

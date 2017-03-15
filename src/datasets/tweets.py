import tensorflow as tf
from datasets import dataset_utils
import os
slim = tf.contrib.slim

SPLITS_TO_SIZES = {'trn': 15385, 'dev': 1588, 'tst': 20632}
_FILE_PATTERN = 'tw.%s.tfrecord'
_NUM_CLASSES = 3

_ITEMS_TO_DESCRIPTIONS = {
    'txt': 'a tweet.',
    'label': 'A single integer between 0 and 3',
}


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    # keys_to_features = {
    #     'txt': tf.FixedLenFeature([], tf.string),
    #     'label': tf.FixedLenFeature([], tf.string)
    # }

    keys_to_features = {
        'txt': tf.FixedLenFeature([], tf.float32, default_value=tf.zeros([], dtype=tf.float32)),
        'label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }


    items_to_handlers = {
        'txt': slim.tfexample_decoder.Tensor('txt'),
        'label': slim.tfexample_decoder.Tensor('label')
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

        # txt = tf.decode_raw(features['txt'], tf.float32)
        # label = tf.decode_raw(features['label'], tf.uint8)
        #
        # resized_x = tf.reshape(txt, [seqlen, w2vdim])
        # resized_label = tf.reshape(label, [nclass])


    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES,
        labels_to_names=labels_to_names)
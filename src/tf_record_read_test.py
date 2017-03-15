import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
from slimcnn import slimcnn
import time, os
from datetime import datetime
slim = tf.contrib.slim

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
# batched_data = tf.train.batch(
#     tensors=[y],
#     batch_size=5,
#     dynamic_pad=True,
#     name="y_batch"
# )
#
#
# print 'a'
# init_op = tf.group(tf.global_variables_initializer(),
#                    tf.local_variables_initializer())
# print 'b'
# with tf.Session() as sess:
#     sess.run(init_op)
#     print 'c'
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     print 'd'
#     tt, ll = sess.run([txts, labels])
#     print 'e'
#     coord.request_stop()
#     coord.join(threads)
#
# for t in tt:
#     print t
#
# print 'f'





def load_batch(data_provider, seqlen=60, w2vdim=400, nclass=3, batch_size=4):
    features = tf.parse_single_example(
        data_provider,
        features={
            'txt': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        })

    txt = tf.decode_raw(features['txt'], tf.float32)
    label = tf.decode_raw(features['label'], tf.uint8)

    resized_x = tf.reshape(txt, [seqlen, w2vdim])
    resized_label = tf.reshape(label, [nclass])


    txts, labels = tf.train.shuffle_batch([resized_x, resized_label],
                                          batch_size=batch_size,
                                          capacity=30,
                                          num_threads=2,
                                          min_after_dequeue=10)

    return txts, labels


# x,y = load_batch(serialized_example, 10)

# init_op = tf.group(tf.global_variables_initializer(),
#                    tf.local_variables_initializer())
#
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     print 'c'
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     print 'd'
#     tt, ll = sess.run([x, y])
#     print 'e'
#     coord.request_stop()
#     coord.join(threads)
#
# for t in tt:
#     print t
#
# print 'f'

BATCH_SIZE = 10

def get_total_loss(scope, name="total_loss"):
    losses = slim.losses.get_losses(scope=scope)
    losses += slim.losses.get_regularization_losses(scope=scope)
    return math_ops.add_n(losses, name=name)

def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def tower_loss(scope, data_provider):
    txts, labels = load_batch(data_provider, batch_size=BATCH_SIZE)

    # Create the model, use the default arg scope to configure the batch norm parameters.
    num_class = 3
    seq_len = 60
    w2v_dim = 400
    logits, _ = slimcnn(txts, num_class, seq_len, w2v_dim, [2,3,4,5], 32,
                        is_training=True)
    # one_hot_labels = slim.one_hot_encoding(labels, num_class, scope=scope)
    slim.losses.softmax_cross_entropy(logits, labels, scope=scope)
    total_loss = get_total_loss(scope)

    return total_loss

MAX_STEPS = 1500
NUM_GPUS = 1
train_dir = '/tmp/bgshin/slim_test/'

tfrecords_filename = '/tmp/bgshin/tw.trn.tfrecords'
filename_queue = tf.train.string_input_producer(
    [tfrecords_filename], num_epochs=10)

reader = tf.TFRecordReader()
_, data_provider = reader.read(filename_queue)

with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0),
                                  trainable=False)

    # Specify the optimizer and create the train op:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

    # init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())

    # init_op = tf.global_variables_initializer()
    # init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    # sess = tf.Session(config=tf.ConfigProto(
    #     allow_soft_placement=True,
    #     log_device_placement=True))
    # sess.run(init_op)

    # Here you can substitute the flowers dataset for your own dataset.



    tower_grads = []
    losses = []
    for i in range(NUM_GPUS):
        with tf.device("/gpu:" + str(i)):
            with tf.name_scope("tower_" + str(i)) as scope:
                loss = tower_loss(scope, data_provider)
                losses.append(loss)

                tf.get_variable_scope().reuse_variables()

                grads = optimizer.compute_gradients(loss)

                tower_grads.append(grads)

    grads = average_gradients(tower_grads)

    apply_gradient_op = optimizer.apply_gradients(grads, global_step=global_step)

    train_op = apply_gradient_op

    saver = tf.train.Saver(tf.all_variables())

    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False))
    sess.run(init)

    # init_fn = get_init_fn()
    # init_fn(sess)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

    for step in range(MAX_STEPS):
        start_time = time.time()

        # This code gets the average loss, and the losses on GPUs 1 and 2, to print.
        # If you have more GPUs then you will need to adapt it.
        # _, loss_value, losses_value_0, losses_value_1 = sess.run([train_op, loss, losses[0], losses[1]])
        _, loss_value = sess.run([train_op, loss])
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
            num_examples_per_step = BATCH_SIZE * NUM_GPUS
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration) / NUM_GPUS

            # format_str = ('%s: step %d, loss = %.2f (0: %.2f, 1: %.2f) (%.1f examples/sec; %.3f '
            #               'sec/batch)')
            # print (format_str % (datetime.now(), step, loss_value, losses_value_0, losses_value_1,
            #                      examples_per_sec, sec_per_batch))
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print (format_str % (datetime.now(), step, loss_value,
                                 examples_per_sec, sec_per_batch))

        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == MAX_STEPS:
            checkpoint_path = os.path.join(train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)


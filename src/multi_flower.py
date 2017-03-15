from preprocessing import inception_preprocessing
import tensorflow as tf
from tensorflow.python.ops import math_ops
import os

from datasets import flowers
# from nets import inception
from tensorflow.contrib.slim.nets import inception
from preprocessing import inception_preprocessing
import time
import numpy as np
from datetime import datetime

checkpoints_dir = '/tmp/checkpoints'
train_dir = '/tmp/inception_finetuned/'
flowers_data_dir = '/tmp/flowers'
slim = tf.contrib.slim
image_size = inception.inception_v1.default_image_size

BATCH_SIZE = 158  # How many images can pass through *a single GPU*
# (if they are different specs you'll have to adapt the script)
MAX_STEPS = 1000000
NUM_GPUS = 2


def load_batch(data_provider, batch_size=32, height=299, width=299, is_training=False):
    """Loads a single batch of data.

    Args:
      data_provider: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.

    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """

    image_raw, label = data_provider.get(['image', 'label'])

    # Preprocess image for usage by Inception.
    image = inception_preprocessing.preprocess_image(image_raw, height, width, is_training=is_training, fast_mode=True)

    # Preprocess the image for display purposes.
    image_raw = tf.expand_dims(image_raw, 0)
    image_raw = tf.image.resize_images(image_raw, [height, width])
    image_raw = tf.squeeze(image_raw)

    # Batch it up.
    images, images_raw, labels = tf.train.batch(
        [image, image_raw, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=2 * batch_size)

    return images, images_raw, labels




def get_total_loss(scope, name="total_loss"):
    losses = slim.losses.get_losses(scope=scope)
    losses += slim.losses.get_regularization_losses(scope=scope)
    return math_ops.add_n(losses, name=name)


def tower_loss(scope, data_provider):
    """Calculate the total loss on a single tower running the model.

    Args:
    scope: unique prefix string identifying the tower, e.g. 'tower_0'

    Returns:
    Tensor of shape [] containing the total loss for a batch of data
    """

    images, _, labels = load_batch(data_provider, batch_size=BATCH_SIZE, height=image_size, width=image_size,
                                   is_training=True)

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception.inception_v1_arg_scope()):
        logits, _ = inception.inception_v1(images, num_classes=dataset.num_classes, is_training=True)

    one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes, scope=scope)

    slim.losses.softmax_cross_entropy(logits, one_hot_labels, scope=scope)
    # tf.losses.softmax_cross_entropy(logits, one_hot_labels, scope=scope)

    total_loss = get_total_loss(scope)

    return total_loss

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
    is over individual gradients. The inner list is over the gradient
    calculation for each tower.
    Returns:
    List of pairs of (gradient, variable) where the gradient has been averaged
    across all towers.
    """
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


def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes = ["InceptionV1/Logits", "InceptionV1/AuxLogits"]

    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_v1.ckpt'),
        variables_to_restore)


with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)

    global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0),
                                  trainable=False)

    # Specify the optimizer and create the train op:
    # optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.001)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)

    # Here you can substitute the flowers dataset for your own dataset.
    dataset = flowers.get_split('train', flowers_data_dir)
    print ("number of classes: ", dataset.num_classes)
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8, shuffle=True)

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

    saver = tf.train.Saver(tf.global_variables())

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False))
    sess.run(init)

    init_fn = get_init_fn()
    init_fn(sess)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    # summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
    summary_writer = tf.summary.FileWriter(train_dir, sess.graph)

    for step in range(MAX_STEPS):
        start_time = time.time()

        # This code gets the average loss, and the losses on GPUs 1 and 2, to print.
        # If you have more GPUs then you will need to adapt it.
        _, loss_value, losses_value_0, losses_value_1 = sess.run([train_op, loss, losses[0], losses[1]])
        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 10 == 0:
            num_examples_per_step = BATCH_SIZE * NUM_GPUS
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration) / NUM_GPUS

            format_str = ('%s: step %d, loss = %.2f (0: %.2f, 1: %.2f) (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print (format_str % (datetime.now(), step, loss_value, losses_value_0, losses_value_1,
                                 examples_per_sec, sec_per_batch))

        # Save the model checkpoint periodically.
        if step % 1000 == 0 or (step + 1) == MAX_STEPS:
            checkpoint_path = os.path.join(train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
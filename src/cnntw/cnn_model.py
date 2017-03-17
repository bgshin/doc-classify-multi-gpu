
import tensorflow as tf
import re


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_boolean('use_fp16', False,
                            """Train the model using fp16.""")
TOWER_NAME = 'tower'
NUM_CLASSES = 3

def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor
    """

    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def _variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
        name: name of the variable
        shape: list of ints
        stddev: standard deviation of a truncated Gaussian
        wd: add L2Loss weight decay multiplied by this float. If None, weight
            decay is not added for this Variable.

    Returns:
        Variable Tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var = _variable_on_cpu(
        name,
        shape,
        tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)

    return var


def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
    x: Tensor
    Returns:
    nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity',
                                       tf.nn.zero_fraction(x))



def inference(txts):
    """Build the CIFAR-10 model.

    Args:
    txts: Images returned from distorted_inputs() or inputs().

    Returns:
    Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1 [filter_size, embedding_size, 1, num_filters]
    sequence_length = 60
    filter_size = 2
    embedding_size = 400
    num_filters = 64
    cnn_shape = [filter_size, embedding_size, 1, num_filters]
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights',
                                                shape=cnn_shape,
                                                stddev=5e-2,
                                                wd=0.0)
        conv = tf.nn.conv2d(txts, kernel, [1, 1, 1, 1], padding='VALID')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)
        _activation_summary(conv1)

    # pool1 [1, sequence_length - filter_size + 1, 1, 1],

    ksize = [1, sequence_length - filter_size + 1, 1, 1]
    pool1 = tf.nn.max_pool(conv1, ksize=ksize, strides=[1, 1, 1, 1],
                         padding='VALID', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

    print 'norm1', norm1
    num_filters_total = num_filters * 1
    norm_flat = tf.reshape(norm1, [-1, num_filters_total])


    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [64, NUM_CLASSES],
                                              stddev=1/64.0, wd=0.0)
        biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                  tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(norm_flat, weights), biases, name=scope.name)
        _activation_summary(softmax_linear)

    return softmax_linear




def loss(logits, labels):
    """Add L2Loss to all the trainable variables.

    Add summary for "Loss" and "Loss/avg".
    Args:
    logits: Logits from inference().
    labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]

    Returns:
    Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    # labels = tf.cast(labels, tf.int64)
    labels = tf.cast(tf.argmax(labels, 1), tf.int64)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

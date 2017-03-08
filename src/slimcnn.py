import tensorflow as tf

slim = tf.contrib.slim

trunc_normal = lambda stddev: tf.truncated_normal_initializer(stddev=stddev)



def slimcnn(txts, num_classes, sequence_length, embedding_size, filter_sizes, num_filters, is_training=False,
             dropout_keep_prob=0.5,
             prediction_fn=slim.softmax,
             scope='SlimCNN'):

    end_points = {}

    with tf.variable_scope(scope, 'SlimCNN', [txts, num_classes]):
        pooled_outputs = []
        for filter_size in filter_sizes:
            net = slim.conv2d(txts, num_filters, [filter_size, embedding_size], scope='conv_%d' % filter_size)
            net = slim.max_pool2d(net, [1, sequence_length - filter_size + 1], stride=1, scope='pool_%d' % filter_size)
            pooled_outputs.append(net)

        print 'pooled_outputs', pooled_outputs

        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        h_drop = slim.dropout(h_pool_flat, dropout_keep_prob)
        logits = slim.fully_connected(h_drop, num_classes, activation_fn=None,
                                      scope='fc')

    end_points['Logits'] = logits
    end_points['Predictions'] = slim.softmax(logits, scope='Predictions')

    return logits, end_points

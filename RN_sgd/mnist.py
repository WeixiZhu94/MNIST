import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import mnist

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'train.tfrecords'
TEST_FILE = 'test.tfrecords'


regularizer = slim.l2_regularizer(0.0005)

def _si_conv(net, in_filter, out_filter, prefix):
   net = slim.layers.conv2d(net, out_filter, [3,3], scope=prefix + 'conv_1', activation_fn=None, biases_regularizer=regularizer, weights_regularizer=regularizer)
   return net

def _residual(net, in_filter, out_filter, prefix):
   # ori_net : not activated; net -> BN -> RELU
   with tf.variable_scope(prefix+'_pre_act'):
      ori_net = net
      net = slim.layers.batch_norm(net)
      net = tf.nn.relu(net)
   with tf.variable_scope(prefix+'_residual'):
      # net -> Weight -> BN -> RELU
      net = slim.layers.conv2d(net, out_filter, [3,3], scope='conv_1', normalizer_fn=slim.layers.batch_norm, biases_regularizer=regularizer, weights_regularizer=regularizer)
      # net -> Weight
      net = slim.layers.conv2d(net, out_filter, [3,3], scope='conv_2', activation_fn=None, biases_regularizer=regularizer, weights_regularizer=regularizer)
   with tf.variable_scope(prefix+'_res_add'):
      if in_filter != out_filter:
         ori_net = tf.nn.avg_pool(ori_net, [1,1,1,1], [1,1,1,1], 'VALID')
         ori_net = tf.pad(ori_net, [[0,0],[0,0],[0,0],[(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      net += ori_net
   return net

def network(net, labels, mode):

   net = _si_conv(net, 8, 8, 'res_init')

   net = _residual(net, 8, 8, 'unit_8_1')
   net = _residual(net, 8, 8, 'unit_8_2')
   net = _residual(net, 8, 8, 'unit_8_3')
   net = _residual(net, 8, 8, 'unit_8_4')
   net = _residual(net, 8, 8, 'unit_8_5')
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_1')
   net = _residual(net, 8, 16, 'unit_16_1')
   net = _residual(net, 16, 16, 'unit_16_2')
   net = _residual(net, 16, 16, 'unit_16_3')
   net = _residual(net, 16, 16, 'unit_16_4')
   net = _residual(net, 16, 16, 'unit_16_5')
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_2')
   net = _residual(net, 16, 32, 'unit_32_1')
   net = _residual(net, 32, 32, 'unit_32_2')
   net = _residual(net, 32, 32, 'unit_32_3')
   net = _residual(net, 32, 32, 'unit_32_4')
   net = _residual(net, 32, 32, 'unit_32_5')
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_3')

   with tf.variable_scope('res_last'):
      net = slim.layers.batch_norm(net)
      net = tf.nn.relu(net)
      net = tf.reduce_mean(net, [1,2])

   logits = slim.layers.fully_connected(net, 10, activation_fn=None, scope='FC_10',biases_regularizer=regularizer, weights_regularizer=regularizer)
 
   if mode:
     labels = tf.expand_dims(logits, 0)
   loss = tf.losses.sparse_softmax_cross_entropy(labels,logits)
   total_loss = loss
   return logits, total_loss


def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })

    # Convert from a scalar string tensor (whose single string has
    # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
    # [mnist.IMAGE_PIXELS].
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([mnist.IMAGE_PIXELS])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    image = tf.reshape(image, [mnist.IMAGE_SIZE, mnist.IMAGE_SIZE, 1])

    # OPTIONAL: Could reshape into a 28x28 image and apply distortions
    # here.  Since we are not applying any distortions in this
    # example, and the next step expects the image to be flattened
    # into a vector, we don't bother.

    # Convert label from a scalar uint8 tensor to an int32 scalar.
    label = tf.cast(features['label'], tf.int32)

    return image, label


def inputs(train_dir, train, batch_size, num_epochs, one_hot_labels=False):
    """Reads input data num_epochs times.
    Args:
        train: Selects between the training (True) and validation (False) data.
        batch_size: Number of examples per returned batch.
        num_epochs: Number of times to read the input data, or 0/None to
        train forever.
    Returns:
        A tuple (images, labels), where:
        * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
        in the range [-0.5, 0.5].
        * labels is an int32 tensor with shape [batch_size] with the true label,
        a number in the range [0, mnist.NUM_CLASSES).
        Note that an tf.train.QueueRunner is added to the graph, which
        must be run using e.g. tf.train.start_queue_runners().
    """
    if not num_epochs: num_epochs = None
    filename = os.path.join(train_dir,
                            TRAIN_FILE if train else TEST_FILE)

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
            [filename], num_epochs=num_epochs)

        # Even when reading in multiple threads, share the filename
        # queue.
        image, label = read_and_decode(filename_queue)

        if one_hot_labels:
            label = tf.one_hot(label, mnist.NUM_CLASSES, dtype=tf.int32)

        # Shuffle the examples and collect them into batch_size batches.
        # (Internally uses a RandomShuffleQueue.)
        # We run this in two threads to avoid being a bottleneck.

        if train:
          example_queue = tf.RandomShuffleQueue(
              capacity=16 * batch_size,
              min_after_dequeue=8 * batch_size,
              dtypes=[tf.float32, tf.int32],
              shapes=[[mnist.IMAGE_SIZE, mnist.IMAGE_SIZE, 1], []],
              seed=7)
          num_threads = 16
        else:
          example_queue = tf.FIFOQueue(
              3 * batch_size,
              dtypes=[tf.float32, tf.int32],
              shapes=[[mnist.IMAGE_SIZE, mnist.IMAGE_SIZE, 1], []])
          num_threads = 1

        example_enqueue_op = example_queue.enqueue([image, label])
        tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
            example_queue, [example_enqueue_op] * num_threads))

        # Read 'batch' labels + images from the example queue.
        images, sparse_labels = example_queue.dequeue_many(batch_size)
        sparse_labels = tf.squeeze(sparse_labels)

    return images, sparse_labels
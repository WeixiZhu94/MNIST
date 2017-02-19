import os

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import mnist

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'train.tfrecords'
TEST_FILE = 'test.tfrecords'

def network(images, labels):
   input = slim.layers.flatten(images)
   hidden_1 = slim.layers.fully_connected(input, 128, scope='hidden_1', normalizer_fn=slim.layers.batch_norm)
   hidden_2 = slim.layers.fully_connected(hidden_1, 128, scope='hidden_2', normalizer_fn=slim.layers.batch_norm)
   hidden_3 = slim.layers.fully_connected(hidden_2, 128, scope='hidden_3', normalizer_fn=slim.layers.batch_norm, activation_fn=None)
   res = tf.nn.relu(hidden_1+hidden_3)
   logits = slim.layers.fully_connected(res, 10, scope='softmax', activation_fn=None)
   
   slim.losses.sparse_softmax_cross_entropy(logits, labels)
   total_loss = slim.losses.get_total_loss()
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
        images, sparse_labels = tf.train.shuffle_batch(
            [image, label], batch_size=batch_size, num_threads=2,
            capacity=1000 + 3 * batch_size,
            # Ensures a minimum amount of shuffling of examples.
            min_after_dequeue=1000)

    return images, sparse_labels
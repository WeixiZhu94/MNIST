import os

import tensorflow as tf
import tensorflow.contrib.slim as slim\

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'train.tfrecords'
TEST_FILE = 'test.tfrecords'

def network(images, labels):

   net = slim.layers.conv2d(images, 16, [3,3], scope='conv_0_0', normalizer_fn=slim.layers.batch_norm)
   
   net = slim.layers.conv2d(net, 16, [3,3], scope='conv_0_1', normalizer_fn=slim.layers.batch_norm)
   net = slim.layers.conv2d(net, 16, [5,5], scope='conv_1_1', normalizer_fn=slim.layers.batch_norm)
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_1')

   net = slim.layers.conv2d(net, 64, [3,3], scope='conv_2_1', normalizer_fn=slim.layers.batch_norm)
   net = slim.layers.conv2d(net, 64, [5,5], scope='conv_2_2', normalizer_fn=slim.layers.batch_norm)
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_2')

   net = slim.layers.conv2d(net, 256, [3,3], scope='conv_3_1', normalizer_fn=slim.layers.batch_norm)
   net = slim.layers.conv2d(net, 256, [5,5], scope='conv_3_2', normalizer_fn=slim.layers.batch_norm)
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_3')

   net = slim.layers.flatten(net, scope='flatten3')
   net = slim.layers.fully_connected(net, 512, scope='fully_connected_1')
   net = slim.layers.fully_connected(net, 512, scope='fully_connected_2')
   logits = slim.layers.fully_connected(net, 10, activation_fn=None, scope='logits')
   
   total_loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
   return logits, total_loss

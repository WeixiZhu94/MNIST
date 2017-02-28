import os

import tensorflow as tf
import tensorflow.contrib.slim as slim\

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'train.tfrecords'
TEST_FILE = 'test.tfrecords'

def network(net, labels):

   net_1 = slim.layers.conv2d(net, 8, [3,3], scope='init_conv_1', normalizer_fn=slim.layers.batch_norm)
   net_2 = slim.layers.conv2d(net, 4, [5,5], scope='init_conv_2', normalizer_fn=slim.layers.batch_norm)
   net_3 = slim.layers.conv2d(net, 2, [7,7], scope='init_conv_3', normalizer_fn=slim.layers.batch_norm)
   net_4 = slim.layers.conv2d(net, 2, [9,9], scope='init_conv_4', normalizer_fn=slim.layers.batch_norm)
   net = tf.concat([net_1, net_2, net_3, net_4], 3)

   net_1 = slim.layers.conv2d(net, 8, [3,3], scope='conv_1_1_1', normalizer_fn=slim.layers.batch_norm)
   net_2 = slim.layers.conv2d(net, 4, [5,5], scope='conv_1_1_2', normalizer_fn=slim.layers.batch_norm)
   net_3 = slim.layers.conv2d(net, 2, [7,7], scope='conv_1_1_3', normalizer_fn=slim.layers.batch_norm)
   net_4 = slim.layers.conv2d(net, 2, [9,9], scope='conv_1_1_4', normalizer_fn=slim.layers.batch_norm)
   net = tf.concat([net_1, net_2, net_3, net_4], 3)
   net_1 = slim.layers.conv2d(net, 8, [3,3], scope='conv_1_2_1', normalizer_fn=slim.layers.batch_norm)
   net_2 = slim.layers.conv2d(net, 4, [5,5], scope='conv_1_2_2', normalizer_fn=slim.layers.batch_norm)
   net_3 = slim.layers.conv2d(net, 2, [7,7], scope='conv_1_2_3', normalizer_fn=slim.layers.batch_norm)
   net_4 = slim.layers.conv2d(net, 2, [9,9], scope='conv_1_2_4', normalizer_fn=slim.layers.batch_norm)
   net = tf.concat([net_1, net_2, net_3, net_4], 3)
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_1')

   net_1 = slim.layers.conv2d(net, 16, [3,3], scope='conv_2_1_1', normalizer_fn=slim.layers.batch_norm)
   net_2 = slim.layers.conv2d(net, 9, [5,5], scope='conv_2_1_2', normalizer_fn=slim.layers.batch_norm)
   net_3 = slim.layers.conv2d(net, 5, [7,7], scope='conv_2_1_3', normalizer_fn=slim.layers.batch_norm)
   net_4 = slim.layers.conv2d(net, 2, [9,9], scope='conv_2_1_4', normalizer_fn=slim.layers.batch_norm)
   net = tf.concat([net_1, net_2, net_3, net_4], 3)
   net_1 = slim.layers.conv2d(net, 16, [3,3], scope='conv_2_2_1', normalizer_fn=slim.layers.batch_norm)
   net_2 = slim.layers.conv2d(net, 9, [5,5], scope='conv_2_2_2', normalizer_fn=slim.layers.batch_norm)
   net_3 = slim.layers.conv2d(net, 5, [7,7], scope='conv_2_2_3', normalizer_fn=slim.layers.batch_norm)
   net_4 = slim.layers.conv2d(net, 2, [9,9], scope='conv_2_2_4', normalizer_fn=slim.layers.batch_norm)
   net = tf.concat([net_1, net_2, net_3, net_4], 3)
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_2')

   net_1 = slim.layers.conv2d(net, 33, [3,3], scope='conv_3_1_1', normalizer_fn=slim.layers.batch_norm)
   net_2 = slim.layers.conv2d(net, 17, [5,5], scope='conv_3_1_2', normalizer_fn=slim.layers.batch_norm)
   net_3 = slim.layers.conv2d(net, 9, [7,7], scope='conv_3_1_3', normalizer_fn=slim.layers.batch_norm)
   net_4 = slim.layers.conv2d(net, 5, [9,9], scope='conv_3_1_4', normalizer_fn=slim.layers.batch_norm)
   net = tf.concat([net_1, net_2, net_3, net_4], 3)
   net_1 = slim.layers.conv2d(net, 33, [3,3], scope='conv_3_2_1', normalizer_fn=slim.layers.batch_norm)
   net_2 = slim.layers.conv2d(net, 17, [5,5], scope='conv_3_2_2', normalizer_fn=slim.layers.batch_norm)
   net_3 = slim.layers.conv2d(net, 9, [7,7], scope='conv_3_2_3', normalizer_fn=slim.layers.batch_norm)
   net_4 = slim.layers.conv2d(net, 5, [9,9], scope='conv_3_2_4', normalizer_fn=slim.layers.batch_norm)
   net = tf.concat([net_1, net_2, net_3, net_4], 3)
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_3')

   net = tf.reduce_mean(net, [1,2])
   logits = slim.layers.fully_connected(net, 10, activation_fn=None, scope='logits', normalizer_fn=slim.layers.batch_norm)
   
   slim.losses.sparse_softmax_cross_entropy(logits, labels)
   total_loss = slim.losses.get_total_loss()
   return logits, total_loss
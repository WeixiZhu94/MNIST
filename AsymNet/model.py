import os

import tensorflow as tf
import tensorflow.contrib.slim as slim\

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'train.tfrecords'
TEST_FILE = 'test.tfrecords'

def _AsymConv_0(net, name):
   net_1 = slim.layers.conv2d(net, 10, [3,3], scope=name+'_1', normalizer_fn=slim.layers.batch_norm)
   net_2 = slim.layers.conv2d(net, 4, [5,5], scope=name+'_2', normalizer_fn=slim.layers.batch_norm)
   net_3 = slim.layers.conv2d(net, 2, [7,7], scope=name+'_3', normalizer_fn=slim.layers.batch_norm)
   return tf.concat([net_1, net_2, net_3], 3)

def _AsymConv_1(net, name):
   net_1 = slim.layers.conv2d(net, 20, [3,3], scope=name+'_1', normalizer_fn=slim.layers.batch_norm)
   net_2 = slim.layers.conv2d(net, 8, [5,5], scope=name+'_2', normalizer_fn=slim.layers.batch_norm)
   net_3 = slim.layers.conv2d(net, 4, [7,7], scope=name+'_3', normalizer_fn=slim.layers.batch_norm)
   return tf.concat([net_1, net_2, net_3], 3)

def _AsymConv(net, name):
   net_1 = slim.layers.conv2d(net, 65, [3,3], scope=name+'_1', normalizer_fn=slim.layers.batch_norm)
   net_2 = slim.layers.conv2d(net, 23, [5,5], scope=name+'_2', normalizer_fn=slim.layers.batch_norm)
   net_3 = slim.layers.conv2d(net, 12, [7,7], scope=name+'_3', normalizer_fn=slim.layers.batch_norm)
   return tf.concat([net_1, net_2, net_3], 3)
   
def network(net, labels):

   net = _AsymConv_0(net, 'conv_1_0')
   net = _AsymConv_0(net, 'conv_1_1')
   net = _AsymConv_0(net, 'conv_1_2')
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_1')

   net = _AsymConv_1(net, 'conv_2_0')
   net = _AsymConv_1(net, 'conv_2_1')
   net = _AsymConv_1(net, 'conv_2_2')
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_2')

   net = _AsymConv(net, 'conv_3_1')
   net = _AsymConv(net, 'conv_3_2')
   net = _AsymConv(net, 'conv_3_3')
   net = _AsymConv(net, 'conv_3_4')
   net = _AsymConv(net, 'conv_3_5')
   net = _AsymConv(net, 'conv_3_6')

   net = tf.reduce_mean(net, [1,2])
   logits = slim.layers.fully_connected(net, 10, activation_fn=None, scope='logits', normalizer_fn=slim.layers.batch_norm)
   
   slim.losses.sparse_softmax_cross_entropy(logits, labels)
   total_loss = slim.losses.get_total_loss()
   return logits, total_loss

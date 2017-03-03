import os

import tensorflow as tf
import tensorflow.contrib.slim as slim\

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'train.tfrecords'
TEST_FILE = 'test.tfrecords'

def _cat2(labels):
   table1 = tf.constant([1,1,0,0,0,0,0,0,1,1])
   table2 = tf.constant([0,0,1,1,1,1,1,1,0,0])
   A = tf.transpose(tf.stack([table1, table2], axis=0))
   one_hot = tf.one_hot(labels, 10, 1, 0, axis=-1)
   return tf.argmax(tf.matmul(one_hot, A), axis=1)

def _residual(net, name, in_filter, out_filter):
   # ori_net : not activated; net -> BN -> RELU
   with tf.variable_scope(name + 'pre_act'):
      ori_net = net
      net = slim.layers.batch_norm(net)
      net = tf.nn.relu(net)
   with tf.variable_scope(name + 'residual'):
      # net -> Weight -> BN -> RELU
      net = slim.layers.conv2d(net, out_filter, [3,3], scope=name+'conv_1', normalizer_fn=slim.layers.batch_norm)
      # net -> Weight
      net = slim.layers.conv2d(net, out_filter, [3,3], scope=name+'conv_2', activation_fn=None)
   with tf.variable_scope(name + 'res_add'):
      if in_filter != out_filter:
         ori_net = tf.nn.avg_pool(ori_net, [1,1,1,1], [1,1,1,1], 'VALID')
         ori_net = tf.pad(ori_net, [[0,0],[0,0],[0,0],[(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      net += ori_net
   return net


def network(images, labels):
   prefix = ''
   net = slim.layers.conv2d(images, 16, [3,3], scope=prefix+'res_init', normalizer_fn=slim.layers.batch_norm)
   
   net = _residual(net, prefix+'res_1_', 16, 16)
   net = _residual(net, prefix+'res_2_', 16, 16)
   net = _residual(net, prefix+'res_3_', 16, 16)
   net = slim.layers.max_pool2d(net, [2,2], scope=prefix+'pool_1')

   net = _residual(net, prefix+'res_4_', 16, 32)
   net = _residual(net, prefix+'res_5_', 32, 32)
   net = _residual(net, prefix+'res_6_', 32, 32)
   net = slim.layers.max_pool2d(net, [2,2], scope=prefix+'pool_2')

   net = _residual(net, prefix+'res_7_', 32, 64)
   net = _residual(net, prefix+'res_8_', 64, 64)
   net = _residual(net, prefix+'res_9_', 64, 64)
   
   with tf.variable_scope(prefix+'res_last'):
      net = slim.layers.batch_norm(net)
      net = tf.nn.relu(net)
      net = tf.reduce_mean(net, [1,2])

   #net = slim.layers.fully_connected(net, 1024, scope='fully_connected', normalizer_fn=slim.layers.batch_norm)
   logits = slim.layers.fully_connected(net, 10, activation_fn=None, scope=prefix+'logits', normalizer_fn=slim.layers.batch_norm)

   labels_cat2 = _cat2(labels)
   total_loss = tf.losses.sparse_softmax_cross_entropy(labels,logits)
   return logits, total_loss, labels_cat2


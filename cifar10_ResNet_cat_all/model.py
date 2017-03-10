import os

import tensorflow as tf
import tensorflow.contrib.slim as slim\

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'train.tfrecords'
TEST_FILE = 'test.tfrecords'

""" activate x before using any _higher """
def _cat1(labels):
   """FullyConnected layer for Lv_1 category."""
   cat_1_sparse = tf.SparseTensor(indices = [[0,0], [1,1], [2,4], [3,3], [4,3], [5,3], [6,5], [7,3], [8,2], [9,1]], values = [1,1,1,1,1,1,1,1,1,1], shape = [10, 6])
   value = tf.sparse_tensor_to_dense(cat_1_sparse)
   value = tf.cast(value, tf.float32)
   w = tf.get_variable('L1', initializer=value, trainable=False)
   one_hot = tf.one_hot(labels, 10, 1, 0, axis=-1)
   return tf.argmax(tf.matmul(x, w), axis=1)

def _cat2(labels):
   table1 = tf.constant([1,1,0,0,0,0,0,0,1,1])
   table2 = tf.constant([0,0,1,1,1,1,1,1,0,0])
   A = tf.transpose(tf.stack([table1, table2], axis=0))
   one_hot = tf.one_hot(labels, 10, 1, 0, axis=-1)
   return tf.argmax(tf.matmul(one_hot, A), axis=1)

""" activate x before using any _higher """
def _cat1_logits(logits):
   """FullyConnected layer for Lv_1 category."""
   cat_1_sparse = tf.SparseTensor(indices = [[0,0], [1,1], [2,4], [3,3], [4,3], [5,3], [6,5], [7,3], [8,2], [9,1]], values = [1,1,1,1,1,1,1,1,1,1], shape = [10, 6])
   value = tf.sparse_tensor_to_dense(cat_1_sparse)
   value = tf.cast(value, tf.float32)
   w = tf.get_variable('L2', initializer=value, trainable=False)
   return tf.matmul(logits, tf.to_float(w))

def _cat2_logits(logits):
   table1 = tf.constant([1,1,0,0,0,0,0,0,1,1])
   table2 = tf.constant([0,0,1,1,1,1,1,1,0,0])
   A = tf.transpose(tf.stack([table1, table2], axis=0))
   return tf.matmul(logits, tf.to_float(A))

def _residual(net, in_filter, out_filter, prefix):
   # ori_net : not activated; net -> BN -> RELU
   with tf.variable_scope(prefix+'_pre_act'):
      ori_net = net
      net = slim.layers.batch_norm(net)
      net = tf.nn.relu(net)
   with tf.variable_scope(prefix+'_residual'):
      # net -> Weight -> BN -> RELU
      net = slim.layers.conv2d(net, out_filter, [3,3], scope='conv_1', normalizer_fn=slim.layers.batch_norm)
      # net -> Weight
      net = slim.layers.conv2d(net, out_filter, [3,3], scope='conv_2', activation_fn=None)
   with tf.variable_scope(prefix+'_res_add'):
      if in_filter != out_filter:
         ori_net = tf.nn.avg_pool(ori_net, [1,1,1,1], [1,1,1,1], 'VALID')
         ori_net = tf.pad(ori_net, [[0,0],[0,0],[0,0],[(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      net += ori_net
   return net

def network(images, labels):

   net = slim.layers.conv2d(images, 16, [3,3], scope='res_init', normalizer_fn=slim.layers.batch_norm)
   
   net = _residual(net, 16, 16, 'unit_16_1')
   net = _residual(net, 16, 16, 'unit_16_2')
   net = _residual(net, 16, 16, 'unit_16_3')
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_1')

   net = _residual(net, 16, 32, 'unit_32_1')
   net = _residual(net, 32, 32, 'unit_32_2')
   net = _residual(net, 32, 32, 'unit_32_3')
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_2')

   net = _residual(net, 32, 64, 'unit_64_1')
   net = _residual(net, 64, 64, 'unit_64_2')
   net = _residual(net, 64, 64, 'unit_64_3')
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_3')

   with tf.variable_scope('res_last'):
      net = slim.layers.batch_norm(net)
      net = tf.nn.relu(net)
      net = tf.reduce_mean(net, [1,2])

   #net = slim.layers.fully_connected(net, 1024, scope='fully_connected', normalizer_fn=slim.layers.batch_norm)
   logits = slim.layers.fully_connected(net, 10, activation_fn=None, scope='logits', normalizer_fn=slim.layers.batch_norm)
   logits_cat1 = _cat1_logits(logits)
   logits_cat2 = _cat2_logits(logits)
   
   labels_cat1 = _cat1(labels)
   labels_cat2 = _cat2(labels)

   loss = tf.losses.sparse_softmax_cross_entropy(labels,logits)
   loss_cat1 = tf.losses.sparse_softmax_cross_entropy(labels_cat1, logits_cat1)
   loss_cat2 = tf.losses.sparse_softmax_cross_entropy(labels_cat2, logits_cat2)
   return logits, logits_cat1, logits_cat2, loss, loss_cat1, loss_cat2, labels_cat1, labels_cat2


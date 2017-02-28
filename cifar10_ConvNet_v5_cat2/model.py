import os

import tensorflow as tf
import tensorflow.contrib.slim as slim\

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'train.tfrecords'
TEST_FILE = 'test.tfrecords'

def _cat2(labels):
   #table1 = tf.constant([1,1,0,0,0,0,0,0,1,1])
   #table2 = tf.constant([0,0,1,1,1,1,1,1,0,0])
   #A = tf.transpose(tf.stack([table1, table2], axis=0))
   one_hot = tf.one_hot(labels, 10, 1.0, 0.0, axis=-1)
   cat_1_sparse = tf.SparseTensor(indices = [[0,1], [1,1], [2,0], [3,0], [4,0], [5,0], [6,0], [7,0], [8,1], [9,1]], values = [1,1,1,1,1,1,1,1,1,1], shape = [10, 2])
   value = tf.sparse_tensor_to_dense(cat_1_sparse)
   value = tf.cast(value, tf.float32)
   w = tf.get_variable('L1', initializer=value, trainable=False)
   labels_cat2 = tf.matmul(one_hot, w)

def network(images, labels):

   net = slim.layers.conv2d(images, 16, [3,3], scope='conv_0_0', normalizer_fn=slim.layers.batch_norm)
   
   net = slim.layers.conv2d(net, 64, [3,3], scope='conv_0_1', normalizer_fn=slim.layers.batch_norm)
   net = slim.layers.conv2d(net, 64, [3,3], scope='conv_1_1', normalizer_fn=slim.layers.batch_norm)
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_1')

   net = slim.layers.conv2d(net, 256, [3,3], scope='conv_2_1', normalizer_fn=slim.layers.batch_norm)
   net = slim.layers.conv2d(net, 256, [3,3], scope='conv_2_2', normalizer_fn=slim.layers.batch_norm)
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_2')

   net = slim.layers.conv2d(net, 1024, [3,3], scope='conv_3_1', normalizer_fn=slim.layers.batch_norm)
   net = slim.layers.conv2d(net, 1024, [3,3], scope='conv_3_2', normalizer_fn=slim.layers.batch_norm)
   net = slim.layers.max_pool2d(net, [2,2], scope='pool_3')

   net = slim.layers.flatten(net, scope='flatten')
   net = slim.layers.fully_connected(net, 1024, scope='fully_connected_1', normalizer_fn=slim.layers.batch_norm)
   #net = slim.layers.fully_connected(net, 1024, scope='fully_connected_2', normalizer_fn=slim.layers.batch_norm)
   logits = slim.layers.fully_connected(net, 2, activation_fn=None, scope='logits')
   
   labels_cat2 = _cat2(labels)
   total_loss = tf.losses.softmax_cross_entropy(logits, labels_cat2)
   return logits, total_loss, labels_cat2

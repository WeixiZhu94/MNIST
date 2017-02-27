import os

import tensorflow as tf
import tensorflow.contrib.slim as slim\

TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'train.tfrecords'
TEST_FILE = 'test.tfrecords'

def attention(net):
   net = slim.layers.conv2d(net, 64, [7,7], scope='att_conv_0_1', normalizer_fn=slim.layers.batch_norm)
   net = slim.layers.conv2d(net, 64, [5,5], scope='att_conv_1_1', normalizer_fn=slim.layers.batch_norm)
   net = slim.layers.max_pool2d(net, [2,2], scope='att_pool_1')

   net = slim.layers.conv2d(net, 256, [3,3], scope='att_conv_2_1', normalizer_fn=slim.layers.batch_norm)
   net = slim.layers.conv2d(net, 256, [3,3], scope='att_conv_2_2', normalizer_fn=slim.layers.batch_norm)
   net = slim.layers.max_pool2d(net, [2,2], scope='att_pool_2')

   net = slim.layers.conv2d(net, 1024, [3,3], scope='att_conv_3_1', normalizer_fn=slim.layers.batch_norm)
   net = slim.layers.conv2d(net, 1024, [3,3], scope='att_conv_3_2', normalizer_fn=slim.layers.batch_norm)
   net = slim.layers.max_pool2d(net, [2,2], scope='att_pool_3')

   net = slim.layers.flatten(net, scope='att_flatten')
   net = slim.layers.fully_connected(net, 1024, activation_fn=None, scope='mask', normalizer_fn=slim.layers.batch_norm)
   
   return tf.sign(tf.nn.relu(net))

def network(images, labels):
   images = images + 6
   mask = attention(images)
   mask = tf.reshape(mask, [100,32,32])
   tf.summary.histogram('images_his', images)
   tf.summary.histogram('mask', mask)
   mask = tf.stack([mask, mask, mask], axis=3)

   tf.summary.image('image', images)
   net = tf.multiply(images, mask) - 6
   tf.summary.image('attention', net)
   net = slim.layers.conv2d(net, 16, [3,3], scope='conv_0_0', normalizer_fn=slim.layers.batch_norm)
   
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
   net = slim.layers.fully_connected(net, 1024, scope='fully_connected', normalizer_fn=slim.layers.batch_norm)
   logits = slim.layers.fully_connected(net, 10, activation_fn=None, scope='logits')
   
   
   slim.losses.sparse_softmax_cross_entropy(logits, labels)
   total_loss = slim.losses.get_total_loss()
   return logits, total_loss

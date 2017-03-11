import tensorflow as tf

import tensorflow.contrib.slim as slim
from model import network
from cifar_input import build_input


flags = tf.app.flags
flags.DEFINE_string('train_dir', '../data',
                    'Directory with the training data.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('num_batches', 100, 'Num of batches to evaluate.')
flags.DEFINE_string('log_dir', '../log_cifar10_cat/conv_sum_v2/eval',
                    'Directory where to log data.')
flags.DEFINE_string('checkpoint_dir', '../log_cifar10_cat/conv_sum_v2/train',
                    'Directory with the model checkpoint data.')
FLAGS = flags.FLAGS


def main(train_dir, batch_size, num_batches, log_dir, checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = log_dir
    with tf.device('/cpu:0'):
      images, labels = build_input('cifar10', 100, 'test')
      logits, logits_cat1, logits_cat2, loss, loss_cat1, loss_cat2, labels_cat1, labels_cat2 = network(images, labels)
    
      tf.summary.scalar('losses/loss', loss)
      tf.summary.scalar('losses/loss_cat1', loss_cat1)
      tf.summary.scalar('losses/loss_cat2', loss_cat2)

      logits = tf.argmax(logits, axis=1)
      logits_cat1 = tf.argmax(logits_cat1, axis=1)
      logits_cat2 = tf.argmax(logits_cat2, axis=1)

      tf.summary.scalar('accuracy', slim.metrics.accuracy(logits, tf.to_int64(labels)))
      tf.summary.scalar('accuracy_cat_1', slim.metrics.accuracy(logits_cat1, tf.to_int64(labels_cat1)))
      tf.summary.scalar('accuracy_cat_2', slim.metrics.accuracy(logits_cat2, tf.to_int64(labels_cat2)))

      # These are streaming metrics which compute the "running" metric,
      # e.g running accuracy
      metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
          'accuracies/accuracy': slim.metrics.streaming_accuracy(logits, labels),
          'accuracies/accuracy_cat_1': slim.metrics.streaming_accuracy(logits_cat1, labels_cat1),
          'accuracies/accuracy_cat_2': slim.metrics.streaming_accuracy(logits_cat2, labels_cat2),
      })

      # Define the streaming summaries to write:
      for metric_name, metric_value in metrics_to_values.items():
          tf.summary.scalar(metric_name, metric_value)

      # Evaluate every 30 seconds
      slim.evaluation.evaluation_loop(
          '',
          checkpoint_dir,
          log_dir,
          num_evals=num_batches,
          eval_op=list(metrics_to_updates.values()),
          summary_op=tf.summary.merge_all(),
          eval_interval_secs=60,
          max_number_of_evaluations = 100000000)


if __name__=='__main__':
    main(FLAGS.train_dir,
         FLAGS.batch_size,
         FLAGS.num_batches,
         FLAGS.log_dir,
         FLAGS.checkpoint_dir)
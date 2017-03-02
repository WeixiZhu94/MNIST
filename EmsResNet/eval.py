import tensorflow as tf

import tensorflow.contrib.slim as slim
from model import network
from cifar_input import build_input


flags = tf.app.flags
flags.DEFINE_string('train_dir', '../data',
                    'Directory with the training data.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('num_batches', 100, 'Num of batches to evaluate.')
flags.DEFINE_string('log_dir', '../log/EmsResNet/eval',
                    'Directory where to log data.')
flags.DEFINE_string('checkpoint_dir', '../log/EmsResNet/train',
                    'Directory with the model checkpoint data.')
FLAGS = flags.FLAGS


def main(train_dir, batch_size, num_batches, log_dir, checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = log_dir
    with tf.device('/cpu:0'):
      images, labels, images_cat_0, labels_cat_0, images_cat_1, labels_cat_1 = build_input('cifar10', 100, 'test')
      predictions, total_loss, labels_cat2 = network(images, labels)
      predictions_cat_0, loss_0 = cat_0_network(images_cat_0, labels_cat_0)
      predictions_cat_1, loss_1 = cat_0_network(images_cat_1, labels_cat_1)
    
      tf.summary.scalar('loss', total_loss)
      tf.summary.scalar('loss_0', loss_0)
      tf.summary.scalar('loss_1', loss_1)
    

      predictions_cat_0 = tf.argmax(predictions_cat_0, axis=1)
      predictions_cat_1 = tf.argmax(predictions_cat_1, axis=1)
      predictions = tf.argmax(predictions, 1)
      tf.summary.scalar('accuracy_cat_0', slim.metrics.accuracy(predictions_cat_0, tf.to_int64(labels_cat_0)))
      tf.summary.scalar('accuracy_cat_1', slim.metrics.accuracy(predictions_cat_1, tf.to_int64(labels_cat_1)))
      tf.summary.scalar('accuracy_cat2', slim.metrics.accuracy(predictions, labels_cat2))

      # These are streaming metrics which compute the "running" metric,
      # e.g running accuracy
      metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
          'accuracy_cat2': slim.metrics.streaming_accuracy(predictions, labels_cat2),
          'accuracy_cat_0': slim.metrics.streaming_accuracy(predictions_cat_0, labels_cat2),
          'accuracy_cat_1': slim.metrics.streaming_accuracy(predictions_cat_1, labels_cat2),
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
import tensorflow as tf

import tensorflow.contrib.slim as slim
from mnist import inputs, network

flags = tf.app.flags
flags.DEFINE_string('train_dir', '../data',
                    'Directory with the training data.')
flags.DEFINE_integer('batch_size', 10000, 'Batch size.')
flags.DEFINE_integer('num_batches', 1, 'Num of batches to evaluate.')
flags.DEFINE_string('log_dir', '../log_ass2/RN_mom/eval',
                    'Directory where to log data.')
flags.DEFINE_string('checkpoint_dir', '../log_ass2/RN_mom/train',
                    'Directory with the model checkpoint data.')
FLAGS = flags.FLAGS


def main(train_dir, batch_size, num_batches, log_dir, checkpoint_dir=None):
    if checkpoint_dir is None:
        checkpoint_dir = log_dir

    images, labels = inputs(train_dir, False, batch_size, num_batches)
    predictions, total_loss = network(images, labels)
    
    tf.summary.scalar('loss', total_loss)
    predictions = tf.to_int32(tf.argmax(predictions, 1))
    
    tf.summary.scalar('accuracy', slim.metrics.accuracy(predictions, labels))

    # These are streaming metrics which compute the "running" metric,
    # e.g running accuracy
    metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
        'accuracy': slim.metrics.streaming_accuracy(predictions, labels),
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
        eval_interval_secs=30,
        max_number_of_evaluations = 100000000)


if __name__=='__main__':
    main(FLAGS.train_dir,
         FLAGS.batch_size,
         FLAGS.num_batches,
         FLAGS.log_dir,
         FLAGS.checkpoint_dir)
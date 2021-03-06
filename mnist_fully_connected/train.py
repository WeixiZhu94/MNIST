import tensorflow as tf
import tensorflow.contrib.slim as slim
from mnist import inputs, network

flags = tf.app.flags
flags.DEFINE_string('train_dir', '../data',
                    'Directory with the training data.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('num_batches', None, 'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir', '../log/mnist_fully_connected/train',
                    'Directory with the log data.')
FLAGS = flags.FLAGS


def main(train_dir, batch_size, num_batches, log_dir):
    images, labels = inputs(train_dir,
                            True,
                            batch_size,
                            num_batches)
    predictions, total_loss = network(images, labels)

    tf.summary.scalar('loss', total_loss)

    optimizer = tf.train.GradientDescentOptimizer(0.001)
    train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)

    slim.learning.train(train_op, log_dir, save_summaries_secs=10, save_interval_secs=10)


if __name__ == '__main__':
    main(FLAGS.train_dir,
         FLAGS.batch_size,
         FLAGS.num_batches,
         FLAGS.log_dir)
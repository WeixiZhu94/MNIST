import tensorflow as tf
import tensorflow.contrib.slim as slim
from model import network
from cifar_input import build_input

flags = tf.app.flags
flags.DEFINE_string('train_dir', '../data',
                    'Directory with the training data.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('num_batches', None, 'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir', '../log/cifar10_AttNet_v55/train',
                    'Directory with the log data.')
FLAGS = flags.FLAGS


def main(train_dir, batch_size, num_batches, log_dir):

    images, labels = build_input('cifar10', 100, 'train')
    predictions, total_loss, att_loss = network(images, labels)

    tf.summary.scalar('loss', total_loss)
    tf.summary.scalar('att_loss', att_loss)
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train_op = slim.learning.create_train_op(total_loss + att_loss, optimizer, summarize_gradients=True)

    slim.learning.train(train_op, log_dir, save_summaries_secs=20, save_interval_secs=20)


if __name__ == '__main__':
    main(FLAGS.train_dir,
         FLAGS.batch_size,
         FLAGS.num_batches,
         FLAGS.log_dir)
import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
from model import network, cat_0_network, cat_1_network
from cifar_input import build_input, build_input_cat_0, build_input_cat_1

flags = tf.app.flags
flags.DEFINE_string('train_dir', '../data',
                    'Directory with the training data.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('num_batches', None, 'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir', '../log/EmsResNet_cat_0/train',
                    'Directory with the log data.')
FLAGS = flags.FLAGS

def report():
    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.
            TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

    tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

def main(train_dir, batch_size, num_batches, log_dir):

    images_cat_0, labels_cat_0 = build_input_cat_0('cifar10', 100, 'train')

    predictions_cat_0, loss_0, labels_cat_0_ = cat_0_network(images_cat_0, labels_cat_0)
    
    report()

    tf.summary.scalar('loss_0', loss_0)
    predictions_cat_0 = tf.argmax(predictions_cat_0, axis=1)

    tf.summary.scalar('accuracy_cat_0', slim.metrics.accuracy(predictions_cat_0, tf.to_int64(labels_cat_0)))


    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train_op = slim.learning.create_train_op(loss_0, optimizer, summarize_gradients=True)

    slim.learning.train(train_op, log_dir, save_summaries_secs=20, save_interval_secs=20)


if __name__ == '__main__':
    main(FLAGS.train_dir,
         FLAGS.batch_size,
         FLAGS.num_batches,
         FLAGS.log_dir)
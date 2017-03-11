import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
from model import network
from cifar_input import build_input

flags = tf.app.flags
flags.DEFINE_string('train_dir', '../data',
                    'Directory with the training data.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.')
flags.DEFINE_integer('num_batches', None, 'Num of batches to train (epochs).')
flags.DEFINE_string('log_dir', '../log_cifar10_cat/conv_origin_v2/train',
                    'Directory with the log data.')
flags.DEFINE_float('lrn', 0.1, 'learning_rate')
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

def main(train_dir, batch_size, num_batches, log_dir, lrn):

    images, labels = build_input('cifar10', 100, 'train')
    logits, logits_cat1, logits_cat2, loss, loss_cat1, loss_cat2, labels_cat1, labels_cat2 = network(images, labels)
    
    report()

    tf.summary.scalar('losses/loss', loss)
    tf.summary.scalar('losses/loss_cat1', loss_cat1)
    tf.summary.scalar('losses/loss_cat2', loss_cat2)

    logits = tf.argmax(logits, axis=1)
    logits_cat1 = tf.argmax(logits_cat1, axis=1)
    logits_cat2 = tf.argmax(logits_cat2, axis=1)

    tf.summary.scalar('accuracy', slim.metrics.accuracy(logits, tf.to_int64(labels)))
    tf.summary.scalar('accuracy_cat_1', slim.metrics.accuracy(logits_cat1, tf.to_int64(labels_cat1)))
    tf.summary.scalar('accuracy_cat_2', slim.metrics.accuracy(logits_cat2, tf.to_int64(labels_cat2)))

    optimizer = tf.train.MomentumOptimizer(lrn, 0.9)
    tf.summary.scalar('learning_rate', lrn)
    total_loss = loss
    #train_op = slim.learning.create_train_op(total_loss, optimizer, summarize_gradients=True)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(total_loss, global_step=global_step)
    slim.learning.train(train_op, log_dir, save_summaries_secs=20, save_interval_secs=20)


if __name__ == '__main__':
    main(FLAGS.train_dir,
         FLAGS.batch_size,
         FLAGS.num_batches,
         FLAGS.log_dir,
         FLAGS.lrn)
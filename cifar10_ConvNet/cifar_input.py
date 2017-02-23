import tensorflow as tf


def build_input(dataset, batch_size, mode):
  if mode == 'train':
    data_path = '../cifar10/data_batch*'
  elif mode == 'test':
    data_path = '../cifar10/test_batch.bin'

  tf.set_random_seed(2)
  image_size = 32
  if dataset == 'cifar10':
    label_bytes = 1
    label_offset = 0
    num_classes = 10
  elif dataset == 'cifar100':
    label_bytes = 1
    label_offset = 1
    num_classes = 100
  else:
    raise ValueError('Not supported dataset %s', dataset)

  depth = 3
  image_bytes = image_size * image_size * depth
  record_bytes = label_bytes + label_offset + image_bytes

  data_files = tf.gfile.Glob(data_path)
  file_queue = tf.train.string_input_producer(data_files, shuffle=True)
  # Read examples from files in the filename queue.
  reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
  _, value = reader.read(file_queue)

  # Convert these examples to dense labels and processed images.
  record = tf.reshape(tf.decode_raw(value, tf.uint8), [record_bytes])
  label = tf.cast(tf.slice(record, [label_offset], [label_bytes]), tf.int32)
  # Convert from string to [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(tf.slice(record, [label_offset + label_bytes], [image_bytes]),
                           [depth, image_size, image_size])
  # Convert from [depth, height, width] to [height, width, depth].
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

  if mode == 'train':
    image = tf.image.resize_image_with_crop_or_pad(
        image, image_size+4, image_size+4)
    image = tf.random_crop(image, [image_size, image_size, 3], seed=3)
    image = tf.image.random_flip_left_right(image, seed=5)
    # Brightness/saturation/constrast provides small gains .2%~.5% on cifar.
    # image = tf.image.random_brightness(image, max_delta=63. / 255.)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    image = tf.image.per_image_standardization(image)

    example_queue = tf.RandomShuffleQueue(
        capacity=16 * batch_size,
        min_after_dequeue=8 * batch_size,
        dtypes=[tf.float32, tf.int32],
        shapes=[[image_size, image_size, depth], [1]],
        seed=7)
    num_threads = 16
  else:
    image = tf.image.resize_image_with_crop_or_pad(
        image, image_size, image_size)
    image = tf.image.per_image_standardization(image)

    example_queue = tf.FIFOQueue(
        3 * batch_size,
        dtypes=[tf.float32, tf.int32],
        shapes=[[image_size, image_size, depth], [1]])
    num_threads = 1

  example_enqueue_op = example_queue.enqueue([image, label])
  tf.train.add_queue_runner(tf.train.queue_runner.QueueRunner(
      example_queue, [example_enqueue_op] * num_threads))

  # Read 'batch' labels + images from the example queue.
  images, labels = example_queue.dequeue_many(batch_size)
  labels = tf.squeeze(labels)
  assert len(images.get_shape()) == 4
  assert images.get_shape()[0] == batch_size
  assert images.get_shape()[-1] == 3
  assert len(labels.get_shape()) == 1

  # Display the training images in the visualizer.
  tf.summary.image('images', images)
  return images, labels

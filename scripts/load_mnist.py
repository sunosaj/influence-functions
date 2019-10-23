# Adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/learn/python/learn/datasets/mnist.py

import gzip
import numpy as np

from tensorflow.contrib.learn.python.learn.datasets import base
from influence.dataset import DataSet

from sklearn.utils import shuffle


def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
  """Extract the images into a 4D uint8 np array [index, y, x, depth].
  Args:
    f: A file object that can be passed into a gzip reader.
  Returns:
    data: A 4D unit8 np array [index, y, x, depth].
  Raises:
    ValueError: If the bytestream does not start with 2051.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, f.name))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def extract_labels(f, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 np array [index].
  Args:
    f: A file object that can be passed into a gzip reader.
    one_hot: Does one hot encoding for the result.
    num_classes: Number of classes for the one hot encoding.
  Returns:
    labels: a 1D unit8 np array.
  Raises:
    ValueError: If the bystream doesn't start with 2049.
  """
  print('Extracting', f.name)
  with gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, f.name))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels


def load_mnist(train_dir, validation_size=5000):

  SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
 
  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                   SOURCE_URL + TRAIN_IMAGES)
  with open(local_file, 'rb') as f:
    train_images = extract_images(f)

  local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                   SOURCE_URL + TRAIN_LABELS)
  with open(local_file, 'rb') as f:
    train_labels = extract_labels(f)

  local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                   SOURCE_URL + TEST_IMAGES)
  with open(local_file, 'rb') as f:
    test_images = extract_images(f)

  local_file = base.maybe_download(TEST_LABELS, train_dir,
                                   SOURCE_URL + TEST_LABELS)
  with open(local_file, 'rb') as f:
    test_labels = extract_labels(f)

  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

  validation_images = train_images[:validation_size]
  validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]

  train_images = train_images.astype(np.float32) / 255
  validation_images = validation_images.astype(np.float32) / 255
  test_images = test_images.astype(np.float32) / 255
  print('train_images.shape', train_images.shape)
  print('validation_images.shape', validation_images.shape)
  print('test_images.shape', test_images.shape)

  train = DataSet(train_images, train_labels)
  validation = DataSet(validation_images, validation_labels)
  test = DataSet(test_images, test_labels)

  return base.Datasets(train=train, validation=validation, test=test)


def load_small_mnist(train_dir, validation_size=5000, random_seed=0):
  np.random.seed(random_seed)
  data_sets = load_mnist(train_dir, validation_size)

  train_images = data_sets.train.x
  train_labels = data_sets.train.labels
  perm = np.arange(len(train_labels))
  np.random.shuffle(perm)
  num_to_keep = int(len(train_labels) / 10)
  perm = perm[:num_to_keep]
  train_images = train_images[perm, :]
  train_labels = train_labels[perm]

  validation_images = data_sets.validation.x
  validation_labels = data_sets.validation.labels
  # perm = np.arange(len(validation_labels))
  # np.random.shuffle(perm)
  # num_to_keep = int(len(validation_labels) / 10)
  # perm = perm[:num_to_keep]  
  # validation_images = validation_images[perm, :]
  # validation_labels = validation_labels[perm]

  test_images = data_sets.test.x
  test_labels = data_sets.test.labels
  # perm = np.arange(len(test_labels))
  # np.random.shuffle(perm)
  # num_to_keep = int(len(test_labels) / 10)
  # perm = perm[:num_to_keep]
  # test_images = test_images[perm, :]
  # test_labels = test_labels[perm]

  train = DataSet(train_images, train_labels)
  validation = DataSet(validation_images, validation_labels)
  test = DataSet(test_images, test_labels)

  return base.Datasets(train=train, validation=validation, test=test)


def load_6_class_mnist_small(train_dir, validation_size=3000, fraction_size=0.5):
  SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                   SOURCE_URL + TRAIN_IMAGES)
  with open(local_file, 'rb') as f:
    train_images = extract_images(f)

  local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                   SOURCE_URL + TRAIN_LABELS)
  with open(local_file, 'rb') as f:
    train_labels = extract_labels(f)

  local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                   SOURCE_URL + TEST_IMAGES)
  with open(local_file, 'rb') as f:
    test_images = extract_images(f)

  local_file = base.maybe_download(TEST_LABELS, train_dir,
                                   SOURCE_URL + TEST_LABELS)
  with open(local_file, 'rb') as f:
    test_labels = extract_labels(f)

  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
      'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

  print(type(train_images))
  print(train_images.shape)
  print(test_images.shape)

  images = np.concatenate((train_images, test_images), axis=0)
  labels = np.concatenate((train_labels, test_labels), axis=0)

  # print('images.shape', images.shape)
  # print('labels.shape', labels.shape)

  class_0_indices = np.argwhere(labels == 0)
  class_2_indices = np.argwhere(labels == 2)
  class_3_indices = np.argwhere(labels == 3)
  class_6_indices = np.argwhere(labels == 6)
  class_7_indices = np.argwhere(labels == 7)
  class_9_indices = np.argwhere(labels == 9)

  # print('class_0_indices.shape', class_0_indices.shape)
  class_0_indices = np.reshape(class_0_indices, (class_0_indices.shape[0],))[:int(class_0_indices.shape[0] * fraction_size)]
  class_2_indices = np.reshape(class_2_indices, (class_2_indices.shape[0],))[:int(class_2_indices.shape[0] * fraction_size)]
  class_3_indices = np.reshape(class_3_indices, (class_3_indices.shape[0],))[:int(class_3_indices.shape[0] * fraction_size)]
  class_6_indices = np.reshape(class_6_indices, (class_6_indices.shape[0],))[:int(class_6_indices.shape[0] * fraction_size)]
  class_7_indices = np.reshape(class_7_indices, (class_7_indices.shape[0],))[:3567]# int(class_7_indices.shape[0] * fraction_size)]
  class_9_indices = np.reshape(class_9_indices, (class_9_indices.shape[0],))[:int(class_9_indices.shape[0] * fraction_size)]

  print('class_0_indices.shape', class_0_indices.shape)
  print('class_2_indices.shape', class_2_indices.shape)
  print('class_3_indices.shape', class_3_indices.shape)
  print('class_6_indices.shape', class_6_indices.shape)
  print('class_7_indices.shape', class_7_indices.shape)
  print('class_9_indices.shape', class_9_indices.shape)

  reduced_class_indices = np.concatenate(
    (class_0_indices, class_2_indices, class_3_indices, class_6_indices, class_7_indices, class_9_indices))

  # print('reduced_class_indices.shape', reduced_class_indices.shape)

  images = images[reduced_class_indices]
  labels = labels[reduced_class_indices]

  total_num_samples = images.shape[0]

  # Have to replace labels with 0, 1, 2, 3, 4, 5 or training won't work
  labels = np.where(labels == 2, 1, labels)
  labels = np.where(labels == 3, 2, labels)
  labels = np.where(labels == 6, 3, labels)
  labels = np.where(labels == 7, 4, labels)
  labels = np.where(labels == 9, 5, labels)

  images, labels = shuffle(images, labels, random_state=0)

  # print('images.shape', images.shape)
  # print('labels.shape', labels.shape)
  # print('6', np.unique(labels))
  # print('images[0]')
  # print(images[0])
  # print('6 labels')
  # print(labels)

  train_size = int(36000 * fraction_size)
  validation_size = int(validation_size * fraction_size)
  total_num_samples = int(total_num_samples * fraction_size)

  # print('validation_size', validation_size)

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  train_images = images[validation_size:train_size]
  train_labels = labels[validation_size:train_size]
  test_images = images[train_size:]
  test_labels = labels[train_size:]

  # print(len(validation_images))
  # print(len(validation_labels))
  # print(len(train_labels))
  # print(len(train_images))
  # print(len(test_images))
  # print(len(test_labels))

  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
      'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

  # print('train_images.shape', train_images.shape)
  # print('train_labels.shape', train_labels.shape)
  # print('test_images.shape', test_images.shape)
  # print('test_images.shape', test_labels.shape)
  # print('validation_images.shape', validation_images.shape)
  # print('validation_images.shape', validation_labels.shape)

  train_images = train_images.astype(np.float64) / 255
  validation_images = validation_images.astype(np.float64) / 255
  test_images = test_images.astype(np.float64) / 255

  print('train_images.shape', train_images.shape)
  print('validation_images.shape', validation_images.shape)
  print('test_images.shape', test_images.shape)

  train = DataSet(train_images, train_labels)
  validation = DataSet(validation_images, validation_labels)
  test = DataSet(test_images, test_labels)

  return base.Datasets(train=train, validation=validation, test=test)


def load_2_class_mnist(train_dir, digit1, digit2, size):
  SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

  TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
  TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
  TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
  TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

  local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
                                   SOURCE_URL + TRAIN_IMAGES)
  with open(local_file, 'rb') as f:
    train_images = extract_images(f)

  local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                   SOURCE_URL + TRAIN_LABELS)
  with open(local_file, 'rb') as f:
    train_labels = extract_labels(f)

  local_file = base.maybe_download(TEST_IMAGES, train_dir,
                                   SOURCE_URL + TEST_IMAGES)
  with open(local_file, 'rb') as f:
    test_images = extract_images(f)

  local_file = base.maybe_download(TEST_LABELS, train_dir,
                                   SOURCE_URL + TEST_LABELS)
  with open(local_file, 'rb') as f:
    test_labels = extract_labels(f)

  if not 0 <= 1000 <= len(train_images):
    raise ValueError(
      'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), 1000))

  print(type(train_images))
  print(train_images.shape)
  print(test_images.shape)

  images = np.concatenate((train_images, test_images), axis=0)
  labels = np.concatenate((train_labels, test_labels), axis=0)

  # print('images.shape', images.shape)
  # print('labels.shape', labels.shape)

  class_0_indices = np.argwhere(labels == 0)
  class_1_indices = np.argwhere(labels == 1)
  class_2_indices = np.argwhere(labels == 2)
  class_3_indices = np.argwhere(labels == 3)
  class_4_indices = np.argwhere(labels == 4)
  class_5_indices = np.argwhere(labels == 5)
  class_6_indices = np.argwhere(labels == 6)
  class_7_indices = np.argwhere(labels == 7)
  class_8_indices = np.argwhere(labels == 8)
  class_9_indices = np.argwhere(labels == 9)

  print('class_0_indices.shape', class_0_indices.shape)  # (6903, 1)
  print('class_1_indices.shape', class_1_indices.shape)  # (7877, 1)
  print('class_2_indices.shape', class_2_indices.shape)  # (6990, 1)
  print('class_3_indices.shape', class_3_indices.shape)  # (7141, 1)
  print('class_4_indices.shape', class_4_indices.shape)  # (6824, 1)
  print('class_5_indices.shape', class_5_indices.shape)  # (6313, 1)
  print('class_6_indices.shape', class_6_indices.shape)  # (6876, 1)
  print('class_7_indices.shape', class_7_indices.shape)  # (7293, 1)
  print('class_8_indices.shape', class_8_indices.shape)  # (6825, 1)
  print('class_9_indices.shape', class_9_indices.shape)  # (6958, 1)

  # Regular Size (7, 9), (4, 9), (1, 7), (4, 6), (5, 6)
  if size == 'regular':
    if digit1 == 7 and digit2 == 9:
      digit1_indices = np.reshape(class_7_indices, (class_7_indices.shape[0],))[:7042]
      digit2_indices = np.reshape(class_9_indices, (class_9_indices.shape[0],))
    elif digit1 == 4 and digit2 == 9:
      digit1_indices = np.reshape(class_4_indices, (class_4_indices.shape[0],))[:6800]
      digit2_indices = np.reshape(class_9_indices, (class_9_indices.shape[0],))[:6900]
    elif digit1 == 1 and digit2 == 7:
      digit1_indices = np.reshape(class_1_indices, (class_1_indices.shape[0],))[:7000]
      digit2_indices = np.reshape(class_7_indices, (class_7_indices.shape[0],))[:7000]
    elif digit1 == 4 and digit2 == 6:
      digit1_indices = np.reshape(class_4_indices, (class_4_indices.shape[0],))[:6800]
      digit2_indices = np.reshape(class_6_indices, (class_6_indices.shape[0],))[:6800]
    elif digit1 == 5 and digit2 == 6:
      digit1_indices = np.reshape(class_5_indices, (class_5_indices.shape[0],))[:6300]
      digit2_indices = np.reshape(class_6_indices, (class_6_indices.shape[0],))[:6800]

  # Small (7, 9), (4, 9), (1, 7), (4, 6), (5, 6)
  elif size == 'small':
    if digit1 == 7 and digit2 == 9:
      digit1_indices = np.reshape(class_7_indices, (class_7_indices.shape[0],))[:3000]
      digit2_indices = np.reshape(class_9_indices, (class_9_indices.shape[0],))[:3000]
    elif digit1 == 4 and digit2 == 9:
      digit1_indices = np.reshape(class_4_indices, (class_4_indices.shape[0],))[:3000]
      digit2_indices = np.reshape(class_9_indices, (class_9_indices.shape[0],))[:3000]
    elif digit1 == 1 and digit2 == 7:
      digit1_indices = np.reshape(class_1_indices, (class_1_indices.shape[0],))[:3000]
      digit2_indices = np.reshape(class_7_indices, (class_7_indices.shape[0],))[:3000]
    elif digit1 == 4 and digit2 == 6:
      digit1_indices = np.reshape(class_4_indices, (class_4_indices.shape[0],))[:3000]
      digit2_indices = np.reshape(class_6_indices, (class_6_indices.shape[0],))[:3000]
    elif digit1 == 5 and digit2 == 6:
      digit1_indices = np.reshape(class_5_indices, (class_5_indices.shape[0],))[:3000]
      digit2_indices = np.reshape(class_6_indices, (class_6_indices.shape[0],))[:3000]

  # print('class_1_indices.shape', class_1_indices.shape)
  # class_0_indices = np.reshape(class_0_indices, (class_0_indices.shape[0],))[:int(class_0_indices.shape[0] * fraction_size)]
  # class_1_indices = np.reshape(class_1_indices, (class_1_indices.shape[0],))[:7000]
  # class_2_indices = np.reshape(class_2_indices, (class_2_indices.shape[0],))[:int(class_2_indices.shape[0] * fraction_size)]
  # class_3_indices = np.reshape(class_3_indices, (class_3_indices.shape[0],))[:7000] # int(class_3_indices.shape[0] * fraction_size)]
  # class_6_indices = np.reshape(class_6_indices, (class_6_indices.shape[0],))[:int(class_6_indices.shape[0] * fraction_size)]
  # class_7_indices = np.reshape(class_7_indices, (class_7_indices.shape[0],))[:3567]# int(class_7_indices.shape[0] * fraction_size)]
  # class_9_indices = np.reshape(class_9_indices, (class_9_indices.shape[0],))[:int(class_9_indices.shape[0] * fraction_size)]

  # print('class_0_indices.shape', class_0_indices.shape)
  # print('class_1_indices.shape', class_1_indices.shape)
  # print('class_2_indices.shape', class_2_indices.shape)
  # print('class_3_indices.shape', class_3_indices.shape)
  # print('class_6_indices.shape', class_6_indices.shape)
  # print('class_7_indices.shape', class_7_indices.shape)
  # print('class_9_indices.shape', class_9_indices.shape)

  class_indices = np.concatenate((digit1_indices, digit2_indices))

  # print('reduced_class_indices.shape', reduced_class_indices.shape)

  images = images[class_indices]
  labels = labels[class_indices]

  total_num_samples = images.shape[0]

  print('total_num_samples', total_num_samples)

  # Have to replace labels with 0, 1 or training won't work
  labels = np.where(labels == digit1, 0, labels)
  labels = np.where(labels == digit2, 1, labels)
  # labels = np.where(labels == 3, 2, labels)
  # labels = np.where(labels == 6, 3, labels)
  # labels = np.where(labels == 7, 4, labels)
  # labels = np.where(labels == 9, 5, labels)

  images, labels = shuffle(images, labels, random_state=0)

  # print('images.shape', images.shape)
  # print('labels.shape', labels.shape)
  # print('6', np.unique(labels))
  # print('images[0]')
  # print(images[0])
  # print('6 labels')
  # print(labels)

  if size == 'regular':
    train_size = 11000
    validation_size = 1000
    total_num_samples = total_num_samples
  elif size == 'small':
    train_size = 4000
    validation_size = 0

  # print('validation_size', validation_size)

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  train_images = images[validation_size:validation_size + train_size]
  train_labels = labels[validation_size:validation_size + train_size]
  test_images = images[validation_size + train_size:]
  test_labels = labels[validation_size + train_size:]

  print(len(validation_images))
  print(len(validation_labels))
  print(len(train_labels))
  print(len(train_images))
  print(len(test_images))
  print(len(test_labels))

  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
      'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))

  print('train_images.shape', train_images.shape)
  print('train_labels.shape', train_labels.shape)
  print('test_images.shape', test_images.shape)
  print('test_images.shape', test_labels.shape)
  print('validation_images.shape', validation_images.shape)
  print('validation_images.shape', validation_labels.shape)

  train_images = train_images.astype(np.float64) / 255
  validation_images = validation_images.astype(np.float64) / 255
  test_images = test_images.astype(np.float64) / 255

  print('train_images.shape', train_images.shape)
  print('validation_images.shape', validation_images.shape)
  print('test_images.shape', test_images.shape)

  train = DataSet(train_images, train_labels)
  validation = DataSet(validation_images, validation_labels)
  test = DataSet(test_images, test_labels)

  return base.Datasets(train=train, validation=validation, test=test)


def get_feature_vectors(model):
  train_feature_vectors = np.concatenate((model.sess.run(model.feature_vector, feed_dict=model.all_train_feed_dict),
                                          model.sess.run(model.feature_vector, feed_dict=model.all_validation_feed_dict)))
  validation_feature_vectors = np.empty([0, 32])
  test_feature_vectors = model.sess.run(model.feature_vector, feed_dict=model.all_test_feed_dict)
  # validation_feature_vectors = model.sess.run(model.feature_vector, feed_dict=model.all_validation_feed_dict)

  train_labels = np.concatenate((model.data_sets.train.labels, model.data_sets.validation.labels))
  validation_labels = np.empty([0])
  test_labels = model.data_sets.test.labels

  # print('train_feature_vectors.shape', type(train_feature_vectors))
  # print('train_feature_vectors.shape', type(train_feature_vectors))
  #
  # print('train_feature_vectors.shape', train_feature_vectors.shape)
  # print('test_feature_vectors.shape', test_feature_vectors.shape)
  # print('validation_feature_vectors.shape', validation_feature_vectors.shape)
  #
  # print('train_labels.shape', train_labels.shape)
  # print('test_labels.shape', test_labels.shape)
  # print('validation_labels.shape', validation_labels.shape)

  train = DataSet(train_feature_vectors, train_labels)
  validation = DataSet(validation_feature_vectors, validation_labels)
  test = DataSet(test_feature_vectors, test_labels)

  return base.Datasets(train=train, validation=validation, test=test)


# data_sets_2_small = load_2_class_mnist('data', 1, 2, validation_size=1000)
# data_sets = load_mnist('data')

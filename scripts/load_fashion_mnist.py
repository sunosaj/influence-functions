# https://medium.com/tensorflow/hello-deep-learning-fashion-mnist-with-keras-50fcff8cd74a

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

import random
from sklearn.utils import shuffle

from tensorflow.contrib.learn.python.learn.datasets import base

from influence.dataset import DataSet


# create minority classes and be able to load them from here
# pick a random class from training set (incorporate np.random.seed?)
# reduce the number of them to 10% of original amount?
# how do I want to modify them? just change label? just perturb image? or both? something about adding a dot?
# label 2(pullover) and label 4(coat) seem similar as they are both long sleeve shirts sometimes both with hoods
# 0 and 6, 2 and 6, 7 and 9

# Basic: Randomly select class and reduce to only 10%?
# Basic: Select specific class and reduce to only 10% as well as distort (what kind of distortions)

# count number of instances of each class in training? do i wanna combine given training and test sets an randomize?

# # Basic minority class: randomly select a class and reduce its number of occurrences in the training set to 10%
# def randomly_minoritize_fashion_mnist():
#     (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
#
#     rand_class = np.random.rand(0, 9)
#     count_rand_class_samples = np.count_nonzero(train_labels == rand_class)
#
#     rand_class_indices = np.argwhere(train_labels == rand_class)
#     np.random.shuffle(rand_class_indices)
#
#     count_minority_deletions = count_rand_class_samples * 0.9 #make sure this is an int
#     train_images = np.delete(train_images, rand_class_indices[:count_minority_deletions])
#     train_labels = np.delete(train_labels, rand_class_indices[:count_minority_deletions])
#
#     return (train_images, train_labels), (test_images, test_labels)
#
#
# def randomly_mislabel_fashion_mnist(portion_mislabeled):
#     (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
#     images = np.concatenate((train_images, test_images), axis=0)
#     labels = np.concatenate((train_labels, test_labels), axis=0)
#
#     # https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
#     images, labels = shuffle(images, labels, random_state=0)
#
#     # Split train and test to 80% to 20%
#     train_images = images[:int(labels.shape[0] * 0.8)]
#     train_labels = labels[:int(labels.shape[0] * 0.8)]
#
#     test_images = images[int(labels.shape[0] * 0.8):]
#     test_labels = labels[int(labels.shape[0] * 0.8):]
#
#     # Mislabel portion_mislabeled of training data
#     num_samples = labels.shape[0]
#     num_mislabels = int(num_samples * portion_mislabeled)
#
#
#     for i in range(num_mislabels):
#         # https://stackoverflow.com/questions/42999093/generate-random-number-in-range-excluding-some-numbers/42999212
#         train_labels[i] = random.choice([j for j in range(0, 9) if j != labels[i]])
#
#     return (train_images, train_labels), (test_images, test_labels)


def load_fashion_mnist(validation_size=5000):


    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data() #keras only added to tensorflow
    # since version 1.4, currently using 1.1, fashion_mnist only added to keras.datasets in even later versions than 1.4
    # so apparently tensorflow==1.13 (latest) works even when i run run_spam_experiment. so I guess keep tf at this version
    # and all I had to do was downgrade spacy?

    plt.imshow(train_images[0])
    plt.show()
    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
                .format(len(train_images), validation_size))

    # print('uniques', np.unique(train_labels))
    # print('uniques', np.unique(test_labels))
    #
    # print('train_images.shape', train_images.shape)
    # print('train_labels.shape', train_labels.shape)
    #
    # print('train_images.shape[0], train_images.shape[1], train_images.shape[2], 1', train_images.shape[0], train_images.shape[1], train_images.shape[2], 1)
    train_images = np.reshape(train_images, (train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
    test_images = np.reshape(test_images, (test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))


    # # testing if error when using 6 classes is due to dataset too small (answer is no) ###################
    # validation_size = int(validation_size * 0.6)
    # train_images = train_images[:int(train_labels.shape[0] * 0.6)]
    # train_labels = train_labels[:int(train_labels.shape[0] * 0.6)]
    # test_images = test_images[:int(test_labels.shape[0] * 0.6)]
    # test_labels = test_labels[:int(test_labels.shape[0] * 0.6)]
    # # testing if error when using 6 classes is due to dataset too small (answer is no) ###################

    # print('train_images.shape', train_images.shape)
    # print('train_labels.shape', train_labels.shape)

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    # print(len(validation_images))
    # print(len(validation_labels))
    # print(len(train_labels))
    # print(len(train_images))
    # print(len(test_images))
    # print(len(test_labels))

    train_images = train_images.astype(np.float64) / 255
    validation_images = validation_images.astype(np.float64) / 255
    test_images = test_images.astype(np.float64) / 255

    # print('train_images.shape', train_images.shape)
    # print('validation_images.shape', validation_images.shape)
    # print('test_images.shape', test_images.shape)

    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)

    return base.Datasets(train=train, validation=validation, test=test)


def load_6_class_fashion_mnist(validation_size=3000):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data() #keras only added to tensorflow
    # since version 1.4, currently using 1.1, fashion_mnist only added to keras.datasets in even later versions than 1.4
    # so apparently tensorflow==1.13 (latest) works even when i run run_spam_experiment. so I guess keep tf at this version
    # and all I had to do was downgrade spacy?

    train_images = np.reshape(train_images, (train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
    test_images = np.reshape(test_images, (test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))

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
    class_0_indices = np.reshape(class_0_indices, (class_0_indices.shape[0],))
    class_2_indices = np.reshape(class_2_indices, (class_2_indices.shape[0],))
    class_3_indices = np.reshape(class_3_indices, (class_3_indices.shape[0],))
    class_6_indices = np.reshape(class_6_indices, (class_6_indices.shape[0],))
    class_7_indices = np.reshape(class_7_indices, (class_7_indices.shape[0],))
    class_9_indices = np.reshape(class_9_indices, (class_9_indices.shape[0],))

    # print('class_0_indices.reshape', class_0_indices.shape)

    reduced_class_indices = np.concatenate(
        (class_0_indices, class_2_indices, class_3_indices, class_6_indices, class_7_indices, class_9_indices))

    # print('reduced_class_indices.shape', reduced_class_indices.shape)

    images = images[reduced_class_indices]
    labels = labels[reduced_class_indices]

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

    train_size = 36000

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
    # print('train_images.shape', test_images.shape)
    # print('train_images.shape', test_labels.shape)
    # print('train_images.shape', validation_images.shape)
    # print('train_images.shape', validation_labels.shape)


    train_images = train_images.astype(np.float64) / 255
    validation_images = validation_images.astype(np.float64) / 255
    test_images = test_images.astype(np.float64) / 255

    # print('train_images.shape', train_images.shape)
    # print('validation_images.shape', validation_images.shape)
    # print('test_images.shape', test_images.shape)

    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)

    return base.Datasets(train=train, validation=validation, test=test)


def load_6_class_fashion_mnist_small(validation_size=3000, fraction_size=0.5):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data() #keras only added to tensorflow
    # since version 1.4, currently using 1.1, fashion_mnist only added to keras.datasets in even later versions than 1.4
    # so apparently tensorflow==1.13 (latest) works even when i run run_spam_experiment. so I guess keep tf at this version
    # and all I had to do was downgrade spacy?

    train_images = np.reshape(train_images, (train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
    test_images = np.reshape(test_images, (test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))

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
    class_7_indices = np.reshape(class_7_indices, (class_7_indices.shape[0],))[:int(class_7_indices.shape[0] * fraction_size)]
    class_9_indices = np.reshape(class_9_indices, (class_9_indices.shape[0],))[:int(class_9_indices.shape[0] * fraction_size)]

    # print('class_0_indices.shape', class_0_indices.shape)
    # print('class_2_indices.shape', class_2_indices.shape)
    # print('class_3_indices.shape', class_3_indices.shape)
    # print('class_6_indices.shape', class_6_indices.shape)
    # print('class_7_indices.shape', class_7_indices.shape)
    # print('class_9_indices.shape', class_9_indices.shape)


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


    # images, labels = shuffle(images, labels, random_state=0)

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

    # print('train_images.shape', train_images.shape)
    # print('validation_images.shape', validation_images.shape)
    # print('test_images.shape', test_images.shape)

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
    print('train_feature_vectors.shape', train_feature_vectors.shape)
    print('test_feature_vectors.shape', test_feature_vectors.shape)
    print('validation_feature_vectors.shape', validation_feature_vectors.shape)

    print('train_labels.shape', train_labels.shape)
    print('test_labels.shape', test_labels.shape)
    print('validation_labels.shape', validation_labels.shape)

    train = DataSet(train_feature_vectors, train_labels)
    validation = DataSet(validation_feature_vectors, validation_labels)
    test = DataSet(test_feature_vectors, test_labels)

    return base.Datasets(train=train, validation=validation, test=test)


def load_2_class_fashion_mnist(validation_size=1000):
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data() #keras only added to tensorflow
    # since version 1.4, currently using 1.1, fashion_mnist only added to keras.datasets in even later versions than 1.4
    # so apparently tensorflow==1.13 (latest) works even when i run run_spam_experiment. so I guess keep tf at this version
    # and all I had to do was downgrade spacy?

    images = np.concatenate((train_images, test_images), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)

    print('1 images.shape', images.shape)
    print('1 labels.shape', labels.shape)

    class_0_indices = np.argwhere(labels == 0)
    class_1_indices = np.argwhere(labels == 1)

    print('class_0_indices.shape', class_0_indices.shape)
    class_0_indices = np.reshape(class_0_indices, (class_0_indices.shape[0],))
    class_1_indices = np.reshape(class_1_indices, (class_1_indices.shape[0],))

    print('class_0_indices.reshape', class_0_indices.shape)

    reduced_class_indices = np.concatenate(
        (class_0_indices, class_1_indices))

    print('reduced_class_indices.shape', reduced_class_indices.shape)

    images = images[reduced_class_indices]
    labels = labels[reduced_class_indices]

    images, labels = shuffle(images, labels, random_state=0)

    print('images.shape', images.shape)
    print('labels.shape', labels.shape)
    print('np.unique(labels)', np.unique(labels))
    # print('images[0]')
    # print(images[0])
    # print('6 labels')
    # print(labels)

    train_size = 12000

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    train_images = images[validation_size:train_size]
    train_labels = labels[validation_size:train_size]
    test_images = images[train_size:]
    test_labels = labels[train_size:]

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
    print('train_images.shape', test_images.shape)
    print('train_images.shape', test_labels.shape)
    print('train_images.shape', validation_images.shape)
    print('train_images.shape', validation_labels.shape)


    train_images = train_images.astype(np.float32) / 255
    validation_images = validation_images.astype(np.float32) / 255
    test_images = test_images.astype(np.float32) / 255

    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)

    return base.Datasets(train=train, validation=validation, test=test)


def load_small_fashion_mnist(validation_size=5000, random_seed=0):
    np.random.seed(random_seed)
    data_sets = load_fashion_mnist(validation_size)

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


def load_fashion_mnist_A(validation_size=5000):


    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data() #keras only added to tensorflow
    # since version 1.4, currently using 1.1, fashion_mnist only added to keras.datasets in even later versions than 1.4
    # so apparently tensorflow==1.13 (latest) works even when i run run_spam_experiment. so I guess keep tf at this version
    # and all I had to do was downgrade spacy?

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

    train = DataSet(train_images, train_labels)
    validation = DataSet(validation_images, validation_labels)
    test = DataSet(test_images, test_labels)

    return base.Datasets(train=train, validation=validation, test=test)


# # def randomly_mislabel_fashion_mnist(portion_mislabeled): #############################################################
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
# images = np.concatenate((train_images, test_images), axis=0)
# labels = np.concatenate((train_labels, test_labels), axis=0)
#
# print('images.shape', images.shape[0])
# print('labels.shape', labels.shape[0])
#
# # # https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
# images, labels = shuffle(images, labels, random_state=0)
#
# print(type(images))
# print(type(labels))
# print(int(labels.shape[0] * 0.8))
#
# # Split train and test to 80% to 20%
# train_images = images[:int(labels.shape[0] * 0.8)]
# train_labels = labels[:int(labels.shape[0] * 0.8)]
#
# test_images = images[int(labels.shape[0] * 0.8):]
# test_labels = labels[int(labels.shape[0] * 0.8):]
#
# print('train_images.shape', train_images.shape)
# print('train_labels.shape', train_labels.shape)
# print('test_images.shape', test_images.shape)
# print('test_labels.shape', test_labels.shape)
#
# # Mislabel portion_mislabeled of training data
# portion_mislabeled = 0.1
# num_samples = labels.shape[0]
# print('num_samples', num_samples)
# num_mislabels = int(num_samples * portion_mislabeled)
# print('num_mislabels', num_mislabels)
#
# print(train_labels[:7002])
#
# for i in range(num_mislabels):
#     # https://stackoverflow.com/questions/42999093/generate-random-number-in-range-excluding-some-numbers/42999212
#     # print('train_labels[i] before', train_labels[i])
#     train_labels[i] = random.choice([j for j in range(0, 9) if j != labels[i]])
#     # print('train_labels[i] after', train_labels[i])
#
# # print(original_train_labels[:7000])
# print(train_labels[:7002])
# #######################################################################################################################

# # Check size of load_fashion_mnist and load_small_fashion_mnist
# data_sets = load_fashion_mnist()
# print(data_sets.train.x.shape[0])
# print(data_sets.train.labels.shape[0])
# print(data_sets.test.x.shape[0])
# print(data_sets.test.labels.shape[0])
#
# data_sets = load_small_fashion_mnist()
# print(data_sets.train.x.shape[0])
# print(data_sets.train.labels.shape[0])
# print(data_sets.test.x.shape[0])
# print(data_sets.test.labels.shape[0])

########################################################################################################################

# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data() #keras only added to tensorflow
# images = np.concatenate((train_images, test_images), axis=0)
# labels = np.concatenate((train_labels, test_labels), axis=0)
# print(len(images))
# print(len(labels))


# sorted_indices = np.argsort(labels)
# images = images[sorted_indices]
# labels = labels[sorted_indices]
# print(labels[0])
# print(labels[6999])
# print(labels[7000])
# print(labels[13999])
# print(labels[14000])
# print(labels[20999])
# print(labels[21000])
# print(labels[27999])
# print(labels[28000])
# print(labels[34999])
# print(labels[35000])
# print(labels[69999])
# # print(labels[70000])
#
# reduced_classes_images = np.concatenate((images[0:6999], images[49000:55999], images[21000:27999],
#                                          images[42000:48999], images[63000:69999], images[14000:20999]), axis=0)
# reduced_classes_labels = np.concatenate((labels[0:6999], labels[49000:55999], labels[21000:27999],
#                                          labels[42000:48999], labels[63000:69999], labels[14000:20999]), axis=0)
#
# print("len(reduced_classes_images)", len(reduced_classes_images))
# print("len(reduced_classes_labels)", len(reduced_classes_labels))

# class_0_indices = np.argwhere(labels == 0)
# class_7_indices = np.argwhere(labels == 7)
# class_3_indices = np.argwhere(labels == 3)
# class_6_indices = np.argwhere(labels == 6)
# class_9_indices = np.argwhere(labels == 9)
# class_2_indices = np.argwhere(labels == 2)
#
# print(len(class_0_indices))
# print(len(class_7_indices))
# print(len(class_3_indices))
# print(len(class_6_indices))
# print(len(class_9_indices))
# print(len(class_2_indices))
#
# reduced_class_indices = np.concatenate((class_0_indices, class_2_indices, class_3_indices, class_6_indices, class_7_indices, class_9_indices), axis=0)
# print(reduced_class_indices)
# print(len(reduced_class_indices))
#
# images = images[reduced_class_indices]
# labels = labels[reduced_class_indices]
#
# print(len(images))
# print(len(labels))
#
# print(images.shape)
#
# # reduced_indices = np.where(((labels == 0) or (labels == 7) or (labels == 3) or (labels == 6) or (labels == 9) or (labels == 2)))
# # print(len(reduced_indices))
#
# test = np.array([1, 4, 0, 5])
# print(test)
# test_indices = np.argsort(test)
# print(test_indices)
# test_sorted = test[test_indices]
# print(test_sorted)

#######################################################

# data_sets_6 = load_6_class_fashion_mnist()
# print('------')
# data_sets = load_fashion_mnist()
# print('------')
# data_sets_2 = load_2_class_fashion_mnist()

##############################################################################

# for j in range(20):
#     new_rand = random.choice([i for i in range(3) if i not in [2]])
#     print(new_rand)

####################################################################################

# data_sets_6 = load_6_class_fashion_mnist()
# print('-----')
data_sets_6_small = load_6_class_fashion_mnist_small()

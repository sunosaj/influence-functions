from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

# import os
# import math
import numpy as np
# import pandas as pd
# import sklearn.linear_model as linear_model
import tensorflow as tf

# import scipy
# import sklearn

import random
import pickle
import sys

import time
import os.path

from influence.all_CNN_c import All_CNN_C
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
import influence.experiments as experiments

import multiprocessing as mp

from load_mnist import get_feature_vectors, load_2_class_mnist


# # data_sets = load_fashion_mnist()
# # data_sets = load_6_class_fashion_mnist_small()
# # data_sets = load_6_class_fashion_mnist()
# # data_sets = load_2_class_fashion_mnist()
# data_sets = load_2_class_mnist_small('data', fraction_size=1)
#
# num_classes = 2
# input_side = 28
# input_channels = 1
# input_dim = input_side * input_side * input_channels
# weight_decay = 0.001
# batch_size = 500
#
# initial_learning_rate = 0.0001
# decay_epochs = [10000, 20000]
# hidden1_units = 8
# hidden2_units = 8
# hidden3_units = 8
# conv_patch_size = 3
# keep_probs = [1.0, 1.0]
#
#
# model = All_CNN_C(
#     input_side=input_side,
#     input_channels=input_channels,
#     conv_patch_size=conv_patch_size,
#     hidden1_units=hidden1_units,
#     hidden2_units=hidden2_units,
#     hidden3_units=hidden3_units,
#     weight_decay=weight_decay,
#     num_classes=num_classes,
#     batch_size=batch_size,
#     data_sets=data_sets,
#     initial_learning_rate=initial_learning_rate,
#     damping=1e-2,  # try changing? 80
#     decay_epochs=decay_epochs,
#     mini_batch=True,
#     train_dir='output',
#     log_dir='log',
#     model_name='mnist_small_2_classes')
#
# num_steps = 5000
# model.train(
#     num_steps=num_steps,
#     iter_to_switch_to_batch=1000000,
#     iter_to_switch_to_sgd=100000)
#
# # feature_vectors = model.sess.run(model.feature_vector, feed_dict=model.all_train_feed_dict)
# data_sets = get_feature_vectors(model)
# # print('feature_vectors.shape', feature_vectors.shape)
#
# # feature_vectors_labels = data_sets.train.labels
# # print('feature_vectors_labels.shape', feature_vectors_labels.shape)
# # np.random.seed(42)
#
# # data_sets = load_small_fashion_mnist()
# # data_sets = load_fashion_mnist()

np.random.seed(42)

# digit1, digit2 = 7, 9
digit1, digit2 = 4, 9
# digit1, digit2 = 1, 7
# digit1, digit2 = 4, 6
# digit1, digit2 = 5, 6

data_sets = load_2_class_mnist('data', digit1, digit2, 'small')
# data_sets = load_2_class_mnist('data', 4, 9)
# data_sets = load_2_class_mnist('data', 1, 7)
# data_sets = load_2_class_mnist('data', 4, 6)
# data_sets = load_2_class_mnist('data', 5, 6)

num_classes = 2

input_dim = data_sets.train.x.shape[1]
weight_decay = 0.0001
# weight_decay = 1000 / len(data_sets.train.labels) #0.09090909091
batch_size = 100
initial_learning_rate = 0.001
keep_probs = None
max_lbfgs_iter = 1000 #try adjusting this to get lowest validation error
decay_epochs = [1000, 10000]

tf.reset_default_graph()

tf_model = BinaryLogisticRegressionWithLBFGS(
    input_dim=input_dim,
    weight_decay=weight_decay,
    max_lbfgs_iter=max_lbfgs_iter,
    num_classes=num_classes,
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    # damping=1e-2,  # try changing? 80
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output',
    log_dir='log',
    model_name='mnist_small_binlogreg_lbfgs_2_classes_' + str(digit1) + '_' + str(digit2))

tf_model.train()

X_train = np.copy(tf_model.data_sets.train.x)
print('X_train.shape', X_train.shape)
Y_train = np.copy(tf_model.data_sets.train.labels)
X_test = np.copy(tf_model.data_sets.test.x)
Y_test = np.copy(tf_model.data_sets.test.labels)

# for fashion mnist, it's no longer binary classes like spam vs ham. It's 10 classes so can't just flip classes around anymore
# num_train_examples = Y_train.shape[0] #70000
# num_flip_vals = 700 #0.1% = 70, 1% = 700, 2% = 1400, 5% = 3500, 10% = 7000
# num_check_vals = 700
# num_random_seeds = 40
########################
num_train_examples = Y_train.shape[0] #11000
num_flip_vals = 6
num_check_vals = 6
num_random_seeds = 1

dims = (num_flip_vals, num_check_vals, num_random_seeds, 3)
fixed_influence_loo_results = np.zeros(dims)
fixed_loss_results = np.zeros(dims)
fixed_random_results = np.zeros(dims)

flipped_results = np.zeros((num_flip_vals, num_random_seeds, 3))
########################################################################################################################

orig_results = tf_model.sess.run(
    [tf_model.loss_no_reg, tf_model.accuracy_op],
    feed_dict=tf_model.all_test_feed_dict)

print('Orig loss: %.5f. Accuracy: %.3f' % (orig_results[0], orig_results[1]))

for flips_idx in range(num_flip_vals):
    for random_seed_idx in range(num_random_seeds):

        # random_seed_idx = 0
        random_seed = flips_idx * (num_random_seeds * 3) + (random_seed_idx * 2)
        np.random.seed(random_seed)

        num_flips = int(num_train_examples / 20) * (flips_idx + 1)

        idx_to_flip = np.random.choice(num_train_examples, size=num_flips, replace=False)
        Y_train_flipped = np.copy(Y_train)
        Y_train_flipped[idx_to_flip] = 1 - Y_train[idx_to_flip]

        # # MY CODE TO MISLABEL
        # Y_train_flipped = np.copy(Y_train)
        #
        # # # Just checking if 2 class fashion mnist works ###########
        # # idx_to_flip = np.random.choice(num_train_examples, size=num_flips,
        # #                                replace=False)  # Generates random indices from num_train_examples
        # # Y_train_flipped[idx_to_flip] = 1 - Y_train[idx_to_flip]
        # # # Just checking if 2 class fashion mnist works ###########
        #
        # # 1: Randomly change labels to another class ---------------------------------------------------------------
        # for i in range(num_flips):
        #     # https://stackoverflow.com/questions/42999093/generate-random-number-in-range-excluding-some-numbers/42999212
        #     # Y_train_flipped[i] = random.choice([j for j in range(10) if j != Y_train[i]])
        #     Y_train_flipped[i] = random.choice([j for j in range(num_classes) if j != Y_train[i]])

        # 1: Randomly change labels to another class ---------------------------------------------------------------

        tf_model.update_train_x_y(X_train, Y_train_flipped)
        # 2: Include parameters in model.train() to be for ALL_CNN_C for now ---------------------------------------
        tf_model.train()
        # 2: Include parameters in model.train() to be for ALL_CNN_C for now  --------------------------------------
        flipped_results[flips_idx, random_seed_idx, 1:] = tf_model.sess.run(
            [tf_model.loss_no_reg, tf_model.accuracy_op],
            feed_dict=tf_model.all_test_feed_dict)
        print('Flipped loss: %.5f. Accuracy: %.3f' % (
            flipped_results[flips_idx, random_seed_idx, 1], flipped_results[flips_idx, random_seed_idx, 2]))

        train_losses = tf_model.sess.run(tf_model.indiv_loss_no_reg, feed_dict=tf_model.all_train_feed_dict)
        train_loo_influences = tf_model.get_loo_influences()

        # 3 -------------------------------------------------------------------------------------------------------------------#

        # # simple test for parallelization (works)
        # np.random.RandomState(100)
        # arr = np.random.randint(0, 10, size=[200000, 5])
        # data = arr.tolist()
        # data[:5]
        # def howmany_within_range(row, minimum, maximum):
        #     """Returns how many numbers lie within `maximum` and `minimum` in a given `row`"""
        #     count = 0
        #     for n in row:
        #         if minimum <= n <= maximum:
        #             count = count + 1
        #     return count
        #
        # pool = mp.Pool(mp.cpu_count())
        # results = [pool.apply(howmany_within_range, args=(row, 4, 8)) for row in data]
        # print(results)
        # pool.close()
        # # simple test for parallelization (works)

        # # calculate influence in parallel
        # # train_loo_influences = np.zeros(len(tf_model.data_sets.train.labels))
        # pool = mp.Pool(mp.cpu_count())
        # # print(mp.cpu_count())
        # # for i in range(len(tf_model.data_sets.train.labels)):
        # #     train_loo_influences[i] = pool.apply(get_generic_loo_influences, args=(tf_model, [i], 'lissa', {'scale': 25, 'recursion_depth': 5000, 'damping': 0, 'batch_size': 1, 'num_samples': 10}, True))
        #
        # train_loo_influences = tf_model.go()
        # print(type(train_loo_influences))
        # pool.close()
        # # calculate influence in parallel

        # if len(sys.argv) == 1:
        #     print('sys.argv == 0\n\n\n\n')
        #     # calculate influence in parallel using ThreadPool as Pool
        #     cpu_count = mp.cpu_count()
        #     print(cpu_count)
        #     print('\n')
        #     print(cpu_count)
        #     print('\n')
        #     print(cpu_count)
        #     print('\n')
        #     print(cpu_count)
        #     print('\n')
        #     print(cpu_count)
        #     print('\n')
        #     print(cpu_count)
        #     print('\n')
        #     print(cpu_count)
        #     print('\n')
        #     train_loo_influences = tf_model.go(cpu_count)
        #     print(type(train_loo_influences))
        #     train_loo_influences = np.asarray(train_loo_influences)
        #
        # elif len(sys.argv) == 3:
        #     print('sys.argv == 2\n\n\n\n')
        #     # calculate influence in parallel using ThreadPool as Pool and partition
        #     # Here, there are 16500 + 1500 = 18000 train
        #     start_idx = int(sys.argv[1])
        #     end_idx = int(sys.argv[2])
        #     cpu_count = mp.cpu_count()
        #     print(cpu_count)
        #     print('\n')
        #     print(cpu_count)
        #     print('\n')
        #     print(cpu_count)
        #     print('\n')
        #     print(cpu_count)
        #     print('\n')
        #     print(cpu_count)
        #     print('\n')
        #     print(cpu_count)
        #     print('\n')
        #     print(cpu_count)
        #     print('\n')
        #     train_loo_influences = tf_model.go_partitions(cpu_count, start_idx, end_idx)
        #     print(type(train_loo_influences))
        #     # with open(str(flips_idx) + '_' + str(start_idx) + '_' + str(end_idx), 'wb') as f:
        #     with open('{}_{}_{}'.format(flips_idx, start_idx, end_idx), 'wb') as f:
        #         pickle.dump(train_loo_influences, f)

        # # calculate influence sequentially, normally
        # train_loo_influences = np.zeros(len(tf_model.data_sets.train.labels))
        # for i in range(len(tf_model.data_sets.train.labels)):
        #     train_loo_influences[i] = tf_model.get_generic_loo_influences(
        #         train_index=[i], #this is z_i from paper the I_(up, loss)(z_i, z_i) try i=1000?
        #
        #         # only for lissa
        #         approx_type='lissa',
        #         approx_params={'scale': 25, 'recursion_depth': 5000, 'damping': 0, 'batch_size': 1, 'num_samples': 10},
        #         # only for lissa
        #
        #         # # for CG
        #         # approx_type='cg',
        #         # # for CG
        #
        #         force_refresh=True)
        # # calculate influence sequentially, normally

        # 3 ------------------------------------------------------------------------------------------------------ #

        for checks_idx in range(num_check_vals):
            np.random.seed(random_seed + 1)
            num_checks = int(num_train_examples / 20) * (checks_idx + 1)

            print('### Flips: %s, rs: %s, checks: %s' % (num_flips, random_seed_idx, num_checks))

            fixed_influence_loo_results[flips_idx, checks_idx, random_seed_idx, :], \
            fixed_loss_results[flips_idx, checks_idx, random_seed_idx, :], \
            fixed_random_results[flips_idx, checks_idx, random_seed_idx, :] \
                = experiments.test_mislabeled_detection_batch(
                tf_model,
                X_train, Y_train,
                Y_train_flipped,
                X_test, Y_test,
                train_losses, train_loo_influences,
                num_flips, num_checks)

np.savez(
    'output/randomly_mislabel_mnist_small_results_2_classes_' + str(digit1) + '_' + str(digit2),
    orig_results=orig_results,
    flipped_results=flipped_results,
    fixed_influence_loo_results=fixed_influence_loo_results,
    fixed_loss_results=fixed_loss_results,
    fixed_random_results=fixed_random_results
)

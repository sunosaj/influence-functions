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
from influence.logisticRegressionWithLBFGS import LogisticRegressionWithLBFGS
import influence.experiments as experiments

import multiprocessing as mp

from load_fashion_mnist import get_feature_vectors, load_6_class_fashion_mnist_small, load_6_class_fashion_mnist, load_small_fashion_mnist

num_flip_vals = 6
for flips_idx in range(num_flip_vals):

    f = open("run_fashion_mnist_save_flipped_values_" + str(flips_idx), "r")
    tf_model = pickle.load(f)
    X_train = pickle.load(f)
    Y_train = pickle.load(f)
    X_test = pickle.load(f)
    Y_test = pickle.load(f)
    Y_train_flipped = pickle.load(f)
    num_train_examples = pickle.load(f)
    num_flip_vals = pickle.load(f)
    num_check_vals = pickle.load(f)
    num_random_seeds = pickle.load(f)
    train_losses = pickle.load(f)
    orig_results = pickle.load(f)
    f.close()
#
    random_seed_idx = 0
    random_seed = flips_idx * (num_random_seeds * 3) + (random_seed_idx * 2)
#     np.random.seed(random_seed)
#
    num_flips = int(num_train_examples / 20) * (flips_idx + 1)
#
#     Y_train_flipped = np.copy(Y_train)
#
#     # # Just checking if 2 class fashion mnist works ###########
#     # idx_to_flip = np.random.choice(num_train_examples, size=num_flips,
#     #                                replace=False)  # Generates random indices from num_train_examples
#     # Y_train_flipped[idx_to_flip] = 1 - Y_train[idx_to_flip]
#     # # Just checking if 2 class fashion mnist works ###########
#
#     # 1: Randomly change labels to another class ---------------------------------------------------------------
#     for i in range(num_flips):
#         # https://stackoverflow.com/questions/42999093/generate-random-number-in-range-excluding-some-numbers/42999212
#         # Y_train_flipped[i] = random.choice([j for j in range(10) if j != Y_train[i]])
#         Y_train_flipped[i] = random.choice([j for j in range(6) if j != Y_train[i]])

    # f = open("Y_trained_flipped", "w")
    # pickle.dump(Y_train_flipped, f)
    # f.close()
    #
    # f = open("example", "r")
    # value1 = pickle.load(f)
    # value2 = pickle.load(f)
    # f.close()

    # 1: Randomly change labels to another class ---------------------------------------------------------------

    # tf_model.update_train_x_y(X_train, Y_train_flipped)
    # # 2: Include parameters in model.train() to be for ALL_CNN_C for now ---------------------------------------
    # tf_model.train()
    # # 2: Include parameters in model.train() to be for ALL_CNN_C for now  --------------------------------------
    # flipped_results[flips_idx, random_seed_idx, 1:] = tf_model.sess.run(
    #     [tf_model.loss_no_reg, tf_model.accuracy_op],
    #     feed_dict=tf_model.all_test_feed_dict)
    # print('Flipped loss: %.5f. Accuracy: %.3f' % (
    #     flipped_results[flips_idx, random_seed_idx, 1], flipped_results[flips_idx, random_seed_idx, 2]))
    #
    # train_losses = tf_model.sess.run(tf_model.indiv_loss_no_reg, feed_dict=tf_model.all_train_feed_dict)
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

    # # calculate influence in parallel using ThreadPool as Pool
    # cpu_count = mp.cpu_count()
    # train_loo_influences = tf_model.go(cpu_count)
    # print(type(train_loo_influences))
    # train_loo_influences = np.asarray(train_loo_influences)

    # calculate influence in parallel using ThreadPool as Pool and partition
    # Here, there are 16500 + 1500 = 18000 train
    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    cpu_count = mp.cpu_count()
    print(cpu_count)
    print('\n')
    print(cpu_count)
    print('\n')
    print(cpu_count)
    print('\n')
    print(cpu_count)
    print('\n')
    train_loo_influences = tf_model.go_partitions(cpu_count, start_idx, end_idx)
    print(type(train_loo_influences))
    # with open(str(flips_idx) + '_' + str(start_idx) + '_' + str(end_idx), 'wb') as f:
    with open('{}_{}_{}'.format(flips_idx, start_idx, end_idx), 'wb') as f:
        pickle.dump(train_loo_influences, f)

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

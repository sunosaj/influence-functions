from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

# import os
# import math
import numpy as np
# import pandas as pd
# import sklearn.linear_model as linear_model

# import scipy
# import sklearn

import random

import influence.experiments as experiments
from influence.all_CNN_c import All_CNN_C

from load_mnist import load_small_mnist, load_mnist

import tensorflow as tf

# np.random.seed(42)

data_sets = load_small_mnist('data')

num_classes = 10
input_side = 28
input_channels = 1
input_dim = input_side * input_side * input_channels
weight_decay = 0.001
batch_size = 500

initial_learning_rate = 0.0001
decay_epochs = [10000, 20000]
hidden1_units = 8
hidden2_units = 8
hidden3_units = 8
conv_patch_size = 3
keep_probs = [1.0, 1.0]


model = All_CNN_C(
    input_side=input_side,
    input_channels=input_channels,
    conv_patch_size=conv_patch_size,
    hidden1_units=hidden1_units,
    hidden2_units=hidden2_units,
    hidden3_units=hidden3_units,
    weight_decay=weight_decay,
    num_classes=num_classes,
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    damping=1e-2,
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir='output',
    log_dir='log',
    model_name='mnist')

num_steps = 5000
model.train(
    num_steps=num_steps,
    iter_to_switch_to_batch=100000,
    iter_to_switch_to_sgd=100000)
iter_to_load = num_steps - 1

# test_idx = 6558

# orig_results, flipped_results, fixed_influence_loo_results, fixed_loss_results, fixed_random_results = \
#     experiments.test_mislabeled_training(
#         model,
#         iter_to_load=iter_to_load,
#         num_steps=10000,
#         force_refresh=True)

####################################################################################################################
# the fix mislabel portion of run_spam_experiment but fix it
X_train = np.copy(model.data_sets.train.x)
Y_train = np.copy(model.data_sets.train.labels)
X_test = np.copy(model.data_sets.test.x)
Y_test = np.copy(model.data_sets.test.labels)

num_train_examples = Y_train.shape[0]  # 70000
num_flip_vals = 6  # why is this only 6?
num_check_vals = 6  # why is this only 6?
num_random_seeds = 40  # why do we need this?

dims = (num_flip_vals, num_check_vals, num_random_seeds, 3)
fixed_influence_loo_results = np.zeros(dims)
fixed_loss_results = np.zeros(dims)
fixed_random_results = np.zeros(dims)

flipped_results = np.zeros((num_flip_vals, num_random_seeds, 3))

orig_results = model.sess.run(
    [model.loss_no_reg, model.accuracy_op],
    feed_dict=model.all_test_feed_dict)

print('Orig loss: %.5f. Accuracy: %.3f' % (orig_results[0], orig_results[1]))

for flips_idx in range(num_flip_vals):
    for random_seed_idx in range(num_random_seeds):

        random_seed = flips_idx * (num_random_seeds * 3) + (random_seed_idx * 2)
        np.random.seed(random_seed)

        num_flips = int(num_train_examples / 20) * (flips_idx + 1)

        Y_train_flipped = np.copy(Y_train)

        # # Just checking if 2 class fashion mnist works ###########
        # idx_to_flip = np.random.choice(num_train_examples, size=num_flips,
        #                                replace=False)  # Generates random indices from num_train_examples
        # Y_train_flipped[idx_to_flip] = 1 - Y_train[idx_to_flip]
        # # Just checking if 2 class fashion mnist works ###########

        # 1: Randomly change labels to another class ---------------------------------------------------------------
        for i in range(num_flips):
            # https://stackoverflow.com/questions/42999093/generate-random-number-in-range-excluding-some-numbers/42999212
            Y_train_flipped[i] = random.choice([j for j in range(10) if j != Y_train[i]])
            # Y_train_flipped[i] = random.choice([j for j in range(6) if j != Y_train[i]])

        # 1: Randomly change labels to another class ---------------------------------------------------------------

        model.update_train_x_y(X_train, Y_train_flipped)
        # 2: Include parameters in model.train() to be for ALL_CNN_C for now ---------------------------------------
        model.train(
            num_steps=num_steps,
            iter_to_switch_to_batch=100000,
            iter_to_switch_to_sgd=100000)
        # 2: Include parameters in model.train() to be for ALL_CNN_C for now  --------------------------------------
        flipped_results[flips_idx, random_seed_idx, 1:] = model.sess.run(
            [model.loss_no_reg, model.accuracy_op],
            feed_dict=model.all_test_feed_dict)
        print('Flipped loss: %.5f. Accuracy: %.3f' % (
            flipped_results[flips_idx, random_seed_idx, 1], flipped_results[flips_idx, random_seed_idx, 2]))

        train_losses = model.sess.run(model.indiv_loss_no_reg, feed_dict=model.all_train_feed_dict)
        # 3 -------------------------------------------------------------------------------------------------------------------#
        # train_loo_influences = model.get_loo_influences()

        train_loo_influences = np.zeros(len(model.data_sets.train.labels))
        for i in range(len(model.data_sets.train.labels)):
            train_loo_influences[i] = model.get_generic_loo_influences(
                train_indices=[i],
                approx_type='cg',
                force_refresh=True)

        # 3 ------------------------------------------------------------------------------------------------------ #

        for checks_idx in range(num_check_vals):
            np.random.seed(random_seed + 1)
            num_checks = int(num_train_examples / 20) * (checks_idx + 1)

            print('### Flips: %s, rs: %s, checks: %s' % (num_flips, random_seed_idx, num_checks))

            fixed_influence_loo_results[flips_idx, checks_idx, random_seed_idx, :], \
            fixed_loss_results[flips_idx, checks_idx, random_seed_idx, :], \
            fixed_random_results[flips_idx, checks_idx, random_seed_idx, :] \
                = experiments.test_mislabeled_detection_batch(
                model,
                X_train, Y_train,
                Y_train_flipped,
                X_test, Y_test,
                train_losses, train_loo_influences,
                num_flips, num_checks)

np.savez(
    'output/randomly_mislabel_mnist_results',
    orig_results=orig_results,
    flipped_results=flipped_results,
    fixed_influence_loo_results=fixed_influence_loo_results,
    fixed_loss_results=fixed_loss_results,
    fixed_random_results=fixed_random_results
)

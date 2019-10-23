from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import os
import math
import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model

import scipy
import sklearn

import influence.experiments as experiments
from influence.nlprocessor import NLProcessor
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS
from load_heart import load_heart

import tensorflow as tf

from sklearn.metrics import roc_auc_score

np.random.seed(42)

data_sets = load_heart()
minority_group = ''
# minority_group = 'Female'

num_classes = 2

input_dim = data_sets.train.x.shape[1]
weight_decay = 0.0001
# weight_decay = 1000 / len(lr_data_sets.train.labels)
batch_size = 100
initial_learning_rate = 0.001
keep_probs = None
decay_epochs = [1000, 10000]
max_lbfgs_iter = 1000

tf.reset_default_graph()

tf_model = BinaryLogisticRegressionWithLBFGS(
    input_dim=input_dim,
    weight_decay=weight_decay,
    max_lbfgs_iter=max_lbfgs_iter,
    num_classes=num_classes,
    batch_size=batch_size,
    data_sets=data_sets,
    initial_learning_rate=initial_learning_rate,
    keep_probs=keep_probs,
    decay_epochs=decay_epochs,
    mini_batch=False,
    train_dir='output',
    log_dir='log',
    model_name='heart_logreg')

tf_model.train()

X_train = np.copy(tf_model.data_sets.train.x)
Y_train = np.copy(tf_model.data_sets.train.labels)
X_test = np.copy(tf_model.data_sets.test.x)
Y_test = np.copy(tf_model.data_sets.test.labels)

# Model Performance
preds = tf_model.sess.run(
    [tf_model.preds],
    feed_dict=tf_model.all_test_feed_dict)
preds = np.asarray(preds)
preds = preds[0]
roc_auc_score = roc_auc_score(tf_model.data_sets.test.labels, preds[:, 1])
print('roc_auc_score', roc_auc_score)
# Model Performance

female_indices = np.argwhere(X_train[:, 5] == 1)
male_indices = np.argwhere(X_train[:, 6] == 1)

print(female_indices.shape)
print(male_indices.shape)

if minority_group != '':
    print('There is a minority group!\n\n\n')

    if minority_group == 'Female':
        minority_group_indices = female_indices
    num_train_examples = minority_group_indices.shape[0]
elif minority_group == '':
    print('There is no minority group!\n\n\n')
    num_train_examples = Y_train.shape[0]

print(num_train_examples)

num_flip_vals = 1 # 6  # how much % mislabeled (0.05, 0.1, 0.15, 0.20, 0.25, 0.30)
num_check_vals = 6  # how much % checked (0.05, 0.1, 0.15, 0.20, 0.25, 0.30)
num_random_seeds = 40

# 3 refers to check_num, check_loss, check_acc from experiments.try_check
dims = (num_flip_vals, num_check_vals, num_random_seeds, 3)

fixed_influence_loo_results = np.zeros(dims)  # influence curve
fixed_loss_results = np.zeros(dims)
fixed_random_results = np.zeros(dims)

flipped_results = np.zeros((num_flip_vals, num_random_seeds, 3))

proportion_counts_influence = np.zeros((num_flip_vals, num_random_seeds, num_check_vals, 2))
proportion_counts_loss = np.zeros((num_flip_vals, num_random_seeds, num_check_vals, 2))
proportion_counts_random = np.zeros((num_flip_vals, num_random_seeds, num_check_vals, 2))

# for the dotted line on first graph (test accuracy for no mislabeled data)
orig_results = tf_model.sess.run(
    [tf_model.loss_no_reg, tf_model.accuracy_op],
    feed_dict=tf_model.all_test_feed_dict)

print('Orig loss: %.5f. Accuracy: %.3f' % (orig_results[0], orig_results[1]))

# Each flip_idx is a different set of graphs (each pair of graphs is for 1 flips_idx, which is how much % is mislabeled)
for flips_idx in range(num_flip_vals):
    for random_seed_idx in range(num_random_seeds):

        random_seed = flips_idx * (num_random_seeds * 3) + (random_seed_idx * 2)
        np.random.seed(random_seed)

        # num_flips = how many mislabels
        num_flips = int(num_train_examples / 20) * (flips_idx + 1)  # (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
        print('num_flips', num_flips, '\n\n\n\n\n\n\n')
        num_flips = 0

        # Minority Group - race (index 8): White(38903), Asian-Pac-Islander(1303), Amer-Indian-Eskimo(435), Other(353), Black(4228)
        # Minority Group - sex (index 9): Female(14695), Male(30527)
        if minority_group != '':
            idx_to_flip = np.random.choice(num_train_examples, size=num_flips, replace=False)
            minority_idx_to_flip = minority_group_indices[idx_to_flip]
            Y_train_flipped = np.copy(Y_train)
            Y_train_flipped[minority_idx_to_flip] = 1 - Y_train[minority_idx_to_flip]
        # Minority Group - race (index 8): White(38903), Asian-Pac-Islander(1303), Amer-Indian-Eskimo(435), Other(353), Black(4228)
        # Minority Group - sex (index 9): Female(14695), Male(30527)

        # No Minority Group
        elif minority_group == '':
            idx_to_flip = np.random.choice(num_train_examples, size=num_flips, replace=False)
            print('\n\n\n\n\n\n\n\nidx_to_flip', idx_to_flip)
            Y_train_flipped = np.copy(Y_train)
            Y_train_flipped[idx_to_flip] = 1 - Y_train[idx_to_flip]
            print(Y_train_flipped)
            print(Y_train)
        # No Minority Group

        tf_model.update_train_x_y(X_train, Y_train_flipped)
        tf_model.train()

        flipped_results[flips_idx, random_seed_idx, 1:] = tf_model.sess.run(
            [tf_model.loss_no_reg, tf_model.accuracy_op],
            feed_dict=tf_model.all_test_feed_dict)

        print('Flipped loss: %.5f. Accuracy: %.3f' % (
                flipped_results[flips_idx, random_seed_idx, 1], flipped_results[flips_idx, random_seed_idx, 2]))

        train_losses = tf_model.sess.run(tf_model.indiv_loss_no_reg, feed_dict=tf_model.all_train_feed_dict)

        # gets array of influence values for each training point
        train_loo_influences = tf_model.get_loo_influences()

        # num_check_vals is (0.05, 0.10, 0.15, 0.20, ...) in x-axis for both graphs
        for checks_idx in range(num_check_vals):
            np.random.seed(random_seed + 1)
            num_checks = int(num_train_examples / 20) * (checks_idx + 1)  # (0.05. 0.10, 0.15, 0.20, ...)

            # look at proportion of protected groups in most influential samples #######################################
            # influence
            idx_to_check = np.argsort(train_loo_influences)[-num_checks:]
            samples_to_check = X_train[idx_to_check]

            female_count = np.argwhere(samples_to_check[:, 5] == 1).shape[0]
            male_count = np.argwhere(samples_to_check[:, 6] == 1).shape[0]

            proportion_counts_influence[flips_idx, random_seed_idx, checks_idx, 0] = female_count
            proportion_counts_influence[flips_idx, random_seed_idx, checks_idx, 1] = male_count

            # loss
            idx_to_check = np.argsort(np.abs(train_losses))[-num_checks:]
            samples_to_check = X_train[idx_to_check]

            female_count = np.argwhere(samples_to_check[:, 5] == 1).shape[0]
            male_count = np.argwhere(samples_to_check[:, 6] == 1).shape[0]

            proportion_counts_loss[flips_idx, random_seed_idx, checks_idx, 0] = female_count
            proportion_counts_loss[flips_idx, random_seed_idx, checks_idx, 1] = male_count

            # random
            idx_to_check = np.random.choice(num_train_examples, size=num_checks, replace=False)

            samples_to_check = X_train[idx_to_check]

            female_count = np.argwhere(samples_to_check[:, 5] == 1).shape[0]
            male_count = np.argwhere(samples_to_check[:, 6] == 1).shape[0]

            proportion_counts_random[flips_idx, random_seed_idx, checks_idx, 0] = female_count
            proportion_counts_random[flips_idx, random_seed_idx, checks_idx, 1] = male_count
            # look at proportion of protected groups in most influential samples #######################################

            print('### Flips: %s, rs: %s, checks: %s' % (num_flips, random_seed_idx, num_checks))

            # for each flips_idx and checks_idx and random_seed_idx, save each check_num, check_loss, check_acc to
            # fixed_loo_influence_results, fixed_loss_results, and fixed_random_results to 4th dimension of
            if minority_group != '':
                fixed_influence_loo_results[flips_idx, checks_idx, random_seed_idx, :], \
                    fixed_loss_results[flips_idx, checks_idx, random_seed_idx, :], \
                    fixed_random_results[flips_idx, checks_idx, random_seed_idx, :] \
                    = experiments.test_mislabeled_detection_batch_minority(
                        tf_model,
                        X_train, Y_train,
                        Y_train_flipped,
                        X_test, Y_test,
                        train_losses, train_loo_influences,
                        num_flips, num_checks, minority_group_indices, minority_idx_to_flip)
            elif minority_group == '':
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

if minority_group != '':
    if minority_group == 'Female':
        save_file_name = 'output/heart_results_minority_female'
elif minority_group == '':
    save_file_name = 'output/heart_results'

np.savez(
    save_file_name,
    orig_results=orig_results,
    flipped_results=flipped_results,
    fixed_influence_loo_results=fixed_influence_loo_results,
    fixed_loss_results=fixed_loss_results,
    fixed_random_results=fixed_random_results,
    proportion_counts_influence=proportion_counts_influence,
    proportion_counts_loss=proportion_counts_loss,
    proportion_counts_random=proportion_counts_random
)

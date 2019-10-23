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
from load_spam import load_spam

import tensorflow as tf

# Load spam data, train BinaryLogisticRegressionWithLBFGS(), retrieve training and test sets ###########################
np.random.seed(42)

data_sets = load_spam()

# print(data_sets.train.x.shape)
# print(data_sets.train.labels.shape)

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
    model_name='spam_logreg')

tf_model.train()

X_train = np.copy(tf_model.data_sets.train.x)
Y_train = np.copy(tf_model.data_sets.train.labels)
print("len(Y_train)", len(Y_train)) #4137
X_test = np.copy(tf_model.data_sets.test.x)
Y_test = np.copy(tf_model.data_sets.test.labels)
print("len(Y_test)", len(Y_test)) #1035

num_train_examples = Y_train.shape[0]
num_flip_vals = 6 #([0.00,] 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
num_check_vals = 6
num_random_seeds = 40

# what does this 3 mean? fixed_influence_loo_results, fixed_loss_results, fixed_random_results from experiments.test_mis
# labeled_detection_batch
dims = (num_flip_vals, num_check_vals, num_random_seeds, 3)  # (6, 6, 40, 3)
fixed_influence_loo_results = np.zeros(dims)  # influence curve
fixed_loss_results = np.zeros(dims)  # loss curve
fixed_random_results = np.zeros(dims)  # random curve

flipped_results = np.zeros((num_flip_vals, num_random_seeds, 3))  # (6, 40, 3)

orig_results = tf_model.sess.run(
    [tf_model.loss_no_reg, tf_model.accuracy_op],
    feed_dict=tf_model.all_test_feed_dict)

print('Orig loss: %.5f. Accuracy: %.3f' % (orig_results[0], orig_results[1]))

for flips_idx in range(num_flip_vals): # num_flip_vals = 6
    for random_seed_idx in range(num_random_seeds): # num_random_seeds = 40 # see fig 6 description

        # What is this for?
        random_seed = flips_idx * (num_random_seeds * 3) + (random_seed_idx * 2) # flips_idx * (40 * 3) + (random_seed_idx * 2)
        np.random.seed(random_seed)

        num_flips = int(num_train_examples / 20) * (flips_idx + 1) # x-axis labels of figure 6
        idx_to_flip = np.random.choice(num_train_examples, size=num_flips, replace=False) # Generates random indices from num_train_examples

        # Flip training labels at indices idx_to_flip by doing 1 - Y_train[idx_to_flip]
        Y_train_flipped = np.copy(Y_train)
        Y_train_flipped[idx_to_flip] = 1 - Y_train[idx_to_flip]

        # update model's training set w/mislabeled labels and re-train
        tf_model.update_train_x_y(X_train, Y_train_flipped)
        tf_model.train()
        # (what does this part do?) this works the same as orig_results
        flipped_results[flips_idx, random_seed_idx, 1:] = tf_model.sess.run(
            [tf_model.loss_no_reg, tf_model.accuracy_op],
            feed_dict=tf_model.all_test_feed_dict)
        print('Flipped loss: %.5f. Accuracy: %.3f' % (
                flipped_results[flips_idx, random_seed_idx, 1], flipped_results[flips_idx, random_seed_idx, 2]))

        #
        train_losses = tf_model.sess.run(tf_model.indiv_loss_no_reg, feed_dict=tf_model.all_train_feed_dict) #tf_model.indiv_loss_no_reg from genericNeuralNet.py. Calculates loss for each train data?
        train_loo_influences = tf_model.get_loo_influences() # tf_model.get_loo_influences from binaryLogisticRegressioWithLBFGS.py this calculates loo influence for each train data?
        print(train_loo_influences.shape)
        print('\n')
        print('\n')
        print(train_loo_influences)
        print('\n')
        print('\n')
        #
        for checks_idx in range(num_check_vals): #num_check_vals = 6 # ACTUAL x-axis of figure 6
            np.random.seed(random_seed + 1)
            num_checks = int(num_train_examples / 20) * (checks_idx + 1) #

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

# numpy.savez: Save several arrays into a single file in uncompressed .npz format. (Saves above results)
np.savez(
    'output/spam_results',
    orig_results=orig_results,
    flipped_results=flipped_results,
    fixed_influence_loo_results=fixed_influence_loo_results,
    fixed_loss_results=fixed_loss_results,
    fixed_random_results=fixed_random_results
)

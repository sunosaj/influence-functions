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


f = open("run_fashion_mnist_train_saved_values", "r")
# tf_model = pickle.load(f)
data_sets = pickle.load(f)
X_train = pickle.load(f)
Y_train = pickle.load(f)
X_test = pickle.load(f)
Y_test = pickle.load(f)
num_train_examples = pickle.load(f)
num_flip_vals = pickle.load(f)
num_check_vals = pickle.load(f)
num_random_seeds = pickle.load(f)
orig_results = pickle.load(f)
f.close()

num_classes = 6

input_dim = data_sets.train.x.shape[1]
weight_decay = 0.01
batch_size = 1400
initial_learning_rate = 0.001
keep_probs = None
max_lbfgs_iter = 1000
decay_epochs = [1000, 10000]

tf.reset_default_graph()

tf_model = LogisticRegressionWithLBFGS(
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
    model_name='fashion_mnist_small_logreg_lbfgs')

tf_model.saver.restore(tf_model.sess, "logreg_model")

flipped_results = np.zeros((num_flip_vals, num_random_seeds, 3))

for flips_idx in range(num_flip_vals):

    random_seed_idx = 0
    random_seed = flips_idx * (num_random_seeds * 3) + (random_seed_idx * 2)
    np.random.seed(random_seed)

    num_flips = int(num_train_examples / 20) * (flips_idx + 1)

    Y_train_flipped = np.copy(Y_train)

    # 1: Randomly change labels to another class ---------------------------------------------------------------
    for i in range(num_flips):
        # https://stackoverflow.com/questions/42999093/generate-random-number-in-range-excluding-some-numbers/42999212
        # Y_train_flipped[i] = random.choice([j for j in range(10) if j != Y_train[i]])
        Y_train_flipped[i] = random.choice([j for j in range(6) if j != Y_train[i]])

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

    f = open("run_fashion_mnist_save_flipped_values_" + str(flips_idx), "w")
    # pickle.dump(tf_model, f)
    pickle.dump(X_train, f)
    pickle.dump(Y_train, f)
    pickle.dump(X_test, f)
    pickle.dump(Y_test, f)
    pickle.dump(Y_train_flipped, f)
    pickle.dump(num_train_examples, f)
    pickle.dump(num_flip_vals, f)
    pickle.dump(num_check_vals, f)
    pickle.dump(num_random_seeds, f)
    pickle.dump(train_losses)
    pickle.dump(orig_results, f)
    f.close()

    # f = open("run_fashion_mnist_save_flipped_values_" + str(flips_idx), "r")
    # # tf_model = pickle.load(f)
    # X_train = pickle.load(f)
    # Y_train = pickle.load(f)
    # X_test = pickle.load(f)
    # Y_test = pickle.load(f)
    # Y_train_flipped = pickle.load(f)
    # num_train_examples = pickle.load(f)
    # num_flip_vals = pickle.load(f)
    # num_check_vals = pickle.load(f)
    # num_random_seeds = pickle.load(f)
    # train_losses = pickle.load(f)
    # orig_results = pickle.load(f)
    # f.close()
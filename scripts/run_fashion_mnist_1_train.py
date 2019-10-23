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


# data_sets = load_fashion_mnist()
data_sets = load_6_class_fashion_mnist_small()
# data_sets = load_6_class_fashion_mnist()
# data_sets = load_2_class_fashion_mnist()

num_classes = 6
input_side = 28
input_channels = 1
input_dim = input_side * input_side * input_channels
weight_decay = 0.001
batch_size = 500

initial_learning_rate = 0.0001
decay_epochs = [10000, 20000]
hidden1_units = 32
hidden2_units = 32
hidden3_units = 32
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
    damping=1e-1,  # try changing? 80
    decay_epochs=decay_epochs,
    mini_batch=True,
    train_dir='output',
    log_dir='log',
    model_name='fashion_mnist')

num_steps = 5000
model.train(
    num_steps=num_steps,
    iter_to_switch_to_batch=1000000,
    iter_to_switch_to_sgd=100000)

# feature_vectors = model.sess.run(model.feature_vector, feed_dict=model.all_train_feed_dict)
data_sets = get_feature_vectors(model)
# print('feature_vectors.shape', feature_vectors.shape)

# feature_vectors_labels = data_sets.train.labels
# print('feature_vectors_labels.shape', feature_vectors_labels.shape)
# np.random.seed(42)

# data_sets = load_small_fashion_mnist()
# data_sets = load_fashion_mnist()

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

tf_model.train()

X_train = np.copy(tf_model.data_sets.train.x)
print('X_train.shape', X_train.shape)
Y_train = np.copy(tf_model.data_sets.train.labels)
X_test = np.copy(tf_model.data_sets.test.x)
Y_test = np.copy(tf_model.data_sets.test.labels)

num_train_examples = Y_train.shape[0] #70000
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

tf_model.saver.save(tf_model.sess, "logreg_model")

f = open("run_fashion_mnist_train_saved_values", "w")
# pickle.dump(tf_model, f)
pickle.dump(data_sets, f)
pickle.dump(X_train, f)
pickle.dump(Y_train, f)
pickle.dump(X_test, f)
pickle.dump(Y_test, f)
pickle.dump(num_train_examples, f)
pickle.dump(num_flip_vals, f)
pickle.dump(num_check_vals, f)
pickle.dump(num_random_seeds, f)
pickle.dump(orig_results, f)
f.close()

# f = open("run_fashion_mnist_train_saved_values", "r")
# # tf_model = pickle.load(f)
# data_sets = pickle.load(f)
# X_train = pickle.load(f)
# Y_train = pickle.load(f)
# X_test = pickle.load(f)
# Y_test = pickle.load(f)
# num_train_examples = pickle.load(f)
# num_flip_vals = pickle.load(f)
# num_check_vals = pickle.load(f)
# num_random_seeds = pickle.load(f)
# orig_results = pickle.load(f)
# f.close()

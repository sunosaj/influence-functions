from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import abc
import sys

import numpy as np
import pandas as pd
from sklearn import linear_model, preprocessing, cluster
import scipy.linalg as slin
import scipy.sparse.linalg as sparselin
import scipy.sparse as sparse

import os.path
import time
import tensorflow as tf
import math

from tensorflow.python.ops import array_ops

from influence.hessians import hessians
from influence.genericNeuralNet import GenericNeuralNet, variable, variable_with_weight_decay


class LogisticRegressionWithLBFGS(GenericNeuralNet):

    def __init__(self, input_dim, weight_decay, max_lbfgs_iter, **kwargs):
        self.weight_decay = weight_decay
        self.input_dim = input_dim
        self.max_lbfgs_iter = max_lbfgs_iter

        super(LogisticRegressionWithLBFGS, self).__init__(**kwargs)

        self.set_params_op = self.set_params()
        # self.hessians_op = hessians(self.total_loss, self.params)

        # Multinomial has weird behavior when it's binary
        C = 1.0 / (self.num_train_examples * self.weight_decay)
        self.sklearn_model = linear_model.LogisticRegression(
            C=C,
            tol=1e-8,
            fit_intercept=False,
            solver='lbfgs',
            multi_class='multinomial',
            warm_start=True,  # True
            max_iter=max_lbfgs_iter)

        C_minus_one = 1.0 / ((self.num_train_examples - 1) * self.weight_decay)
        self.sklearn_model_minus_one = linear_model.LogisticRegression(
            C=C_minus_one,
            tol=1e-8,
            fit_intercept=False,
            solver='lbfgs',
            multi_class='multinomial',
            warm_start=True,  # True
            max_iter=max_lbfgs_iter)

    def get_all_params(self):
        all_params = []
        for layer in ['softmax_linear']:
            # for var_name in ['weights', 'biases']:
            for var_name in ['weights']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))
                all_params.append(temp_tensor)
        return all_params

    def placeholder_inputs(self):
        input_placeholder = tf.placeholder(
            tf.float32,
            shape=(None, self.input_dim),
            name='input_placeholder')
        labels_placeholder = tf.placeholder(
            tf.int32,
            shape=(None),
            name='labels_placeholder')
        return input_placeholder, labels_placeholder

    def inference(self, input):
        with tf.variable_scope('softmax_linear'):
            weights = variable_with_weight_decay(
                'weights',
                [self.input_dim * self.num_classes],
                stddev=1.0 / math.sqrt(float(self.input_dim)),
                wd=self.weight_decay)
            logits = tf.matmul(input, tf.reshape(weights, [self.input_dim, self.num_classes]))
            # biases = variable(
            #     'biases',
            #     [self.num_classes],
            #     tf.constant_initializer(0.0))
            # logits = tf.matmul(input, tf.reshape(weights, [self.input_dim, self.num_classes])) + biases

        self.weights = weights
        # self.biases = biases

        return logits

    def predictions(self, logits):
        preds = tf.nn.softmax(logits, name='preds')
        return preds

    def set_params(self):
        # See if we can automatically infer weight shape
        self.W_placeholder = tf.placeholder(
            tf.float32,
            shape=[self.input_dim * self.num_classes],
            name='W_placeholder')
        # self.b_placeholder = tf.placeholder(
        #     tf.float32,
        #     shape=[self.num_classes],
        #     name='b_placeholder')
        set_weights = tf.assign(self.weights, self.W_placeholder, validate_shape=True)
        return [set_weights]
        # set_biases = tf.assign(self.biases, self.b_placeholder, validate_shape=True)
        # return [set_weights, set_biases]

    def retrain(self, num_steps, feed_dict):

        self.train_with_LBFGS(
            feed_dict=feed_dict,
            save_checkpoints=False,
            verbose=False)

        # super(LogisticRegressionWithLBFGS, self).train(
        #     num_steps, 
        #     iter_to_switch_to_batch=0,
        #     iter_to_switch_to_sgd=1000000,
        #     save_checkpoints=False, verbose=False)

    def train(self, num_steps=None,
              iter_to_switch_to_batch=None,
              iter_to_switch_to_sgd=None,
              save_checkpoints=True, verbose=True):

        self.train_with_LBFGS(
            feed_dict=self.all_train_feed_dict,
            save_checkpoints=save_checkpoints,
            verbose=verbose)

        # super(LogisticRegressionWithLBFGS, self).train(
        #     num_steps=500, 
        #     iter_to_switch_to_batch=0,
        #     iter_to_switch_to_sgd=100000,
        #     save_checkpoints=True, verbose=True)

    def train_with_SGD(self, **kwargs):
        super(LogisticRegressionWithLBFGS, self).train(**kwargs)

    def train_with_LBFGS(self, feed_dict, save_checkpoints=True, verbose=True):
        # More sanity checks to see if predictions are the same?        

        X_train = feed_dict[self.input_placeholder]
        Y_train = feed_dict[self.labels_placeholder]
        num_train_examples = len(Y_train)
        assert len(Y_train.shape) == 1
        assert X_train.shape[0] == Y_train.shape[0]

        if num_train_examples == self.num_train_examples:
            if verbose: print('Using normal model')
            model = self.sklearn_model
        elif num_train_examples == self.num_train_examples - 1:
            if verbose: print('Using model minus one')
            model = self.sklearn_model_minus_one
        else:
            raise ValueError, "feed_dict has incorrect number of training examples"

        # print(X_train)
        # print(Y_train)
        model.fit(X_train, Y_train)
        # sklearn returns coefficients in shape num_classes x num_features
        # whereas our weights are defined as num_features x num_classes
        # so we have to tranpose them first.
        W = np.reshape(model.coef_.T, -1)
        # b = model.intercept_

        params_feed_dict = {}
        params_feed_dict[self.W_placeholder] = W
        # params_feed_dict[self.b_placeholder] = b
        self.sess.run(self.set_params_op, feed_dict=params_feed_dict)
        if save_checkpoints: self.saver.save(self.sess, self.checkpoint_file, global_step=0)

        if verbose:
            print('LBFGS training took %s iter.' % model.n_iter_)
            print('After training with LBFGS: ')
            self.print_model_eval()

    #
    # # my new functions based on genericNeuralNet.get_influence_on_test_loss (based on fig 2) (cant write this function
    # # similar to binaryLogisticRegressionWithLBFGS.get_loo_influences because that one is simplified and cannot be used for more than 2 classes
    # def get_generic_loo_influences(self, train_index, approx_type='cg', approx_params=None, force_refresh=True,
    #                                test_description=None, loss_type='normal_loss'):
    #
    #     # 1: Unmodified from genericNeuralNet.get_influence_on_test_loss ####################################
    #     test_grad_loss_no_reg_val = self.get_train_grad_loss_no_reg_val(train_index, batch_size=1, loss_type=loss_type)
    #
    #     print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))
    #
    #     start_time = time.time()
    #
    #     if test_description is None:
    #         test_description = train_index
    #
    #     approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (
    #         self.model_name, approx_type, loss_type, test_description))
    #
    #     if os.path.exists(approx_filename) and force_refresh == False:
    #         inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
    #         print('Loaded inverse HVP from %s' % approx_filename)
    #     else:
    #         inverse_hvp = self.get_inverse_hvp(
    #             test_grad_loss_no_reg_val,
    #             approx_type,
    #             approx_params)
    #         np.savez(approx_filename, inverse_hvp=inverse_hvp)
    #         print('Saved inverse HVP to %s' % approx_filename)
    #
    #     duration = time.time() - start_time
    #     print('Inverse HVP took %s sec' % duration)
    #
    #     start_time = time.time()
    #     # 1: Unmodified from genericNeuralNet.get_influence_on_test_loss ####################################
    #
    #     # 2: else statement from genericNeuralNet.get_influence_on_test_loss ###########################################
    #
    #     # 3: changed from num_to_remove = len(train_idx) because
    #     num_to_remove = 1
    #     # 3: changed from num_to_remove = len(train_idx) because
    #
    #     predicted_loss_diffs = np.zeros([num_to_remove])
    #     for counter, idx_to_remove in enumerate(train_index):
    #         single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)
    #         train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
    #         predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp),
    #                                                np.concatenate(train_grad_loss_val)) / self.num_train_examples
    #     # 2: else statement from genericNeuralNet.get_influence_on_test_loss ###########################################
    #
    #     # 4: Unmodified from genericNeuralNet.get_influence_on_test_loss ####################################
    #     duration = time.time() - start_time
    #     print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))
    #     # 4: Unmodified from genericNeuralNet.get_influence_on_test_loss ####################################
    #
    #     print("predicted_loss_diffs", predicted_loss_diffs)
    #
    #     return predicted_loss_diffs

    # # USING multiprocess as mp
    # def go(self):
    #     pool = mp.Pool(mp.cpu_count())
    #     # train_loo_influences = np.zeros(len(self.data_sets.train.labels))
    #     print(mp.cpu_count())
    #     print(len(self.data_sets.train.labels))
    #     something = pool.map(self, range(len(self.data_sets.train.labels)))
    #     pool.close()
    #     return something
    #     # return train_loo_influences
    #     # train_loo_influences = np.asarray(train_loo_influences)
    #     # pool.apply(self, args=(train_index, approx_type, approx_params, force_refresh, test_description, loss_type))
    #
    # def __call__(self, train_index):
    #     return self.get_generic_loo_influences([train_index], approx_type='lissa',
    #                                            approx_params={'scale': 25, 'recursion_depth': 5000, 'damping': 0,
    #                                                           'batch_size': 1, 'num_samples': 10},
    #                                            force_refresh=True,
    #                                            test_description=None, loss_type='normal_loss')
    # # USING multiprocessing as mp

    # Trying various multiprocessing methods
    def go(self, cpu_count):
        pool = Pool(cpu_count) # is there a pp.cpu_count()?
        something = pool.map(self.get_generic_loo_influences_parallel, range(len(self.data_sets.train.labels)))
        # pool.close()
        return something

    def go_partitions(self, cpu_count, start_idx, end_idx):
        pool = Pool(cpu_count) # is there a pp.cpu_count()?
        something = pool.map(self.get_generic_loo_influences_parallel, range(start_idx, end_idx))
        # pool.close()
        return something

    def get_generic_loo_influences_parallel(self, train_index, approx_type='lissa', approx_params=None, force_refresh=True,
                                   test_description=None, loss_type='normal_loss'):

        approx_params = {'scale': 25, 'recursion_depth': 5000, 'damping': 0, 'batch_size': 1, 'num_samples': 10}

        # 1: Unmodified from genericNeuralNet.get_influence_on_test_loss ####################################
        test_grad_loss_no_reg_val = self.get_train_grad_loss_no_reg_val([train_index], batch_size=1, loss_type=loss_type)

        print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

        start_time = time.time()

        if test_description is None:
            test_description = [train_index]

        approx_filename = os.path.join(self.train_dir, '%s-%s-%s-test-%s.npz' % (
            self.model_name, approx_type, loss_type, test_description))

        if os.path.exists(approx_filename) and force_refresh == False:
            inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
            print('Loaded inverse HVP from %s' % approx_filename)
        else:
            inverse_hvp = self.get_inverse_hvp(
                test_grad_loss_no_reg_val,
                approx_type,
                approx_params)
            np.savez(approx_filename, inverse_hvp=inverse_hvp)
            print('Saved inverse HVP to %s' % approx_filename)

        duration = time.time() - start_time
        print('Inverse HVP took %s sec' % duration)

        start_time = time.time()
        # 1: Unmodified from genericNeuralNet.get_influence_on_test_loss ####################################

        # 2: else statement from genericNeuralNet.get_influence_on_test_loss ###########################################

        # 3: changed from num_to_remove = len(train_idx) because
        num_to_remove = 1
        # 3: changed from num_to_remove = len(train_idx) because

        predicted_loss_diffs = np.zeros([num_to_remove])
        for counter, idx_to_remove in enumerate([train_index]):
            single_train_feed_dict = self.fill_feed_dict_with_one_ex(self.data_sets.train, idx_to_remove)
            train_grad_loss_val = self.sess.run(self.grad_total_loss_op, feed_dict=single_train_feed_dict)
            predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp),
                                                   np.concatenate(train_grad_loss_val)) / self.num_train_examples
        # 2: else statement from genericNeuralNet.get_influence_on_test_loss ###########################################

        # 4: Unmodified from genericNeuralNet.get_influence_on_test_loss ####################################
        duration = time.time() - start_time
        print('Multiplying by %s train examples took %s sec' % (num_to_remove, duration))
        # 4: Unmodified from genericNeuralNet.get_influence_on_test_loss ####################################

        print("predicted_loss_diffs", predicted_loss_diffs)

        return predicted_loss_diffs

    # USING pathos from https://stackoverflow.com/questions/1816958/cant-pickle-type-instancemethod-when-using-multiprocessing-pool-map

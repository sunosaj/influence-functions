import numpy as np
import os
import time

import IPython
from scipy.stats import pearsonr

import random

from sklearn.metrics import roc_auc_score


def get_try_check(model, X_train, Y_train, Y_train_flipped, X_test, Y_test):
    def try_check(idx_to_check, label):
        Y_train_fixed = np.copy(Y_train_flipped)
        Y_train_fixed[idx_to_check] = Y_train[idx_to_check]
        model.update_train_x_y(X_train, Y_train_fixed)
        model.train()
        check_num = np.sum(Y_train_fixed != Y_train_flipped)
        check_loss, check_acc = model.sess.run(
            [model.loss_no_reg, model.accuracy_op], 
            feed_dict=model.all_test_feed_dict)

        print('%20s: fixed %3s labels. Loss %.5f. Accuracy %.3f.' % (
            label, check_num, check_loss, check_acc))
        return check_num, check_loss, check_acc
    return try_check


def test_mislabeled_detection_batch(
    model, 
    X_train, Y_train,
    Y_train_flipped,
    X_test, Y_test, 
    train_losses, train_loo_influences,
    num_flips, num_checks):    

    assert num_checks > 0

    num_train_examples = Y_train.shape[0] 
    
    try_check = get_try_check(model, X_train, Y_train, Y_train_flipped, X_test, Y_test)

    # Pick by LOO influence    
    idx_to_check = np.argsort(train_loo_influences)[-num_checks:]
    fixed_influence_loo_results = try_check(idx_to_check, 'Influence (LOO)')

    # Pick by top loss to fix
    idx_to_check = np.argsort(np.abs(train_losses))[-num_checks:]    
    fixed_loss_results = try_check(idx_to_check, 'Loss')

    # Randomly pick stuff to fix
    idx_to_check = np.random.choice(num_train_examples, size=num_checks, replace=False)    
    fixed_random_results = try_check(idx_to_check, 'Random')
    
    return fixed_influence_loo_results, fixed_loss_results, fixed_random_results


def get_try_check_minority(model, X_train, Y_train, Y_train_flipped, X_test, Y_test):
    def try_check_minority(idx_to_check, label, minority_idx_to_flip):
        Y_train_fixed = np.copy(Y_train_flipped)
        Y_train_fixed[idx_to_check] = Y_train[idx_to_check]
        model.update_train_x_y(X_train, Y_train_fixed)
        model.train()

        # # Model Performance for when labels are fixed
        # preds = model.sess.run(
        #     [model.preds],
        #     feed_dict=model.all_test_feed_dict)
        # preds = np.asarray(preds)
        # preds = preds[0]
        # rocauc_score = roc_auc_score(model.data_sets.test.labels, preds[:, 1])
        # print('roc_auc_score', rocauc_score)
        # # Model Performance for when labels are fixed

        # check_num = np.sum(Y_train_fixed != Y_train_flipped) # number of fixed points
        check_num = np.intersect1d(idx_to_check, minority_idx_to_flip).shape[0]
        check_loss, check_acc = model.sess.run(
            [model.loss_no_reg, model.accuracy_op],
            feed_dict=model.all_test_feed_dict)

        print('%20s: fixed %3s labels. Loss %.5f. Accuracy %.3f.' % (
            label, check_num, check_loss, check_acc))
        return check_num, check_loss, check_acc
    return try_check_minority


def test_mislabeled_detection_batch_minority(
        model,
        X_train, Y_train,
        Y_train_flipped,
        X_test, Y_test,
        train_losses, train_loo_influences,
        num_flips, num_checks, minority_group_indices, minority_idx_to_flip):

    assert num_checks > 0

    num_train_examples = minority_group_indices.shape[0]

    try_check_minority = get_try_check_minority(model, X_train, Y_train, Y_train_flipped, X_test, Y_test)

    # Pick by LOO influence
    idx_to_check = np.argsort(train_loo_influences)[-num_checks:]
    fixed_influence_loo_results = try_check_minority(idx_to_check, 'Influence (LOO)', minority_idx_to_flip)

    # Pick by top loss to fix
    idx_to_check = np.argsort(np.abs(train_losses))[-num_checks:]
    fixed_loss_results = try_check_minority(idx_to_check, 'Loss', minority_idx_to_flip)

    # Randomly pick stuff to fix
    idx_to_check = np.random.choice(num_train_examples, size=num_checks, replace=False)
    fixed_random_results = try_check_minority(idx_to_check, 'Random', minority_idx_to_flip)

    return fixed_influence_loo_results, fixed_loss_results, fixed_random_results


def viz_top_influential_examples(model, test_idx):

    model.reset_datasets()
    print('Test point %s has label %s.' % (test_idx, model.data_sets.test.labels[test_idx]))

    num_to_remove = 10000
    indices_to_remove = np.arange(num_to_remove)
    
    predicted_loss_diffs = model.get_influence_on_test_loss(
        test_idx, 
        indices_to_remove,
        force_refresh=True)

    # If the predicted difference in loss is high (very positive) after removal,
    # that means that the point helped it to be correct.
    top_k = 10
    helpful_points = np.argsort(predicted_loss_diffs)[-top_k:][::-1]
    unhelpful_points = np.argsort(predicted_loss_diffs)[:top_k]

    for points, message in [
        (helpful_points, 'better'), (unhelpful_points, 'worse')]:
        print("Top %s training points making the loss on the test point %s:" % (top_k, message))
        for counter, idx in enumerate(points):
            print("#%s, class=%s, predicted_loss_diff=%.8f" % (
                idx, 
                model.data_sets.train.labels[idx], 
                predicted_loss_diffs[idx]))




def test_retraining(model, test_idx, iter_to_load, force_refresh=False, 
                    num_to_remove=50, num_steps=1000, random_seed=17,
                    remove_type='random'):

    np.random.seed(random_seed)

    model.load_checkpoint(iter_to_load)
    sess = model.sess

    # y_test label for 1 test point
    y_test = model.data_sets.test.labels[test_idx]
    print('Test label: %s' % y_test)

    ## Or, randomly remove training examples
    if remove_type == 'random':
        indices_to_remove = np.random.choice(model.num_train_examples, size=num_to_remove, replace=False)
        predicted_loss_diffs = model.get_influence_on_test_loss(
            [test_idx], 
            indices_to_remove,
            force_refresh=force_refresh)
    ## Or, remove the most influential training examples
    elif remove_type == 'maxinf':
        predicted_loss_diffs = model.get_influence_on_test_loss(
            [test_idx], 
            np.arange(len(model.data_sets.train.labels)),
            force_refresh=force_refresh)
        indices_to_remove = np.argsort(np.abs(predicted_loss_diffs))[-num_to_remove:]
        predicted_loss_diffs = predicted_loss_diffs[indices_to_remove]
    else:
        raise ValueError, 'remove_type not well specified'
    actual_loss_diffs = np.zeros([num_to_remove])

    # Sanity check
    test_feed_dict = model.fill_feed_dict_with_one_ex(
        model.data_sets.test,  
        test_idx)    
    test_loss_val, params_val = sess.run([model.loss_no_reg, model.params], feed_dict=test_feed_dict)
    train_loss_val = sess.run(model.total_loss, feed_dict=model.all_train_feed_dict)
    # train_loss_val = model.minibatch_mean_eval([model.total_loss], model.data_sets.train)[0]

    model.retrain(num_steps=num_steps, feed_dict=model.all_train_feed_dict)
    retrained_test_loss_val = sess.run(model.loss_no_reg, feed_dict=test_feed_dict)
    retrained_train_loss_val = sess.run(model.total_loss, feed_dict=model.all_train_feed_dict)
    # retrained_train_loss_val = model.minibatch_mean_eval([model.total_loss], model.data_sets.train)[0]

    model.load_checkpoint(iter_to_load, do_checks=False)

    print('Sanity check: what happens if you train the model a bit more?')
    print('Loss on test idx with original model    : %s' % test_loss_val)
    print('Loss on test idx with retrained model   : %s' % retrained_test_loss_val)
    print('Difference in test loss after retraining     : %s' % (retrained_test_loss_val - test_loss_val))
    print('===')
    print('Total loss on training set with original model    : %s' % train_loss_val)
    print('Total loss on training with retrained model   : %s' % retrained_train_loss_val)
    print('Difference in train loss after retraining     : %s' % (retrained_train_loss_val - train_loss_val))
    
    print('These differences should be close to 0.\n')

    # Retraining experiment
    for counter, idx_to_remove in enumerate(indices_to_remove):

        print("=== #%s ===" % counter)
        print('Retraining without train_idx %s (label %s):' % (idx_to_remove, model.data_sets.train.labels[idx_to_remove]))

        train_feed_dict = model.fill_feed_dict_with_all_but_one_ex(model.data_sets.train, idx_to_remove)
        model.retrain(num_steps=num_steps, feed_dict=train_feed_dict)
        retrained_test_loss_val, retrained_params_val = sess.run([model.loss_no_reg, model.params], feed_dict=test_feed_dict)
        actual_loss_diffs[counter] = retrained_test_loss_val - test_loss_val

        print('Diff in params: %s' % np.linalg.norm(np.concatenate(params_val) - np.concatenate(retrained_params_val)))      
        print('Loss on test idx with original model    : %s' % test_loss_val)
        print('Loss on test idx with retrained model   : %s' % retrained_test_loss_val)
        print('Difference in loss after retraining     : %s' % actual_loss_diffs[counter])
        print('Predicted difference in loss (influence): %s' % predicted_loss_diffs[counter])

        # Restore params
        model.load_checkpoint(iter_to_load, do_checks=False)
        

    np.savez(
        'output/%s_loss_diffs' % model.model_name, 
        actual_loss_diffs=actual_loss_diffs, 
        predicted_loss_diffs=predicted_loss_diffs)

    print('Correlation is %s' % pearsonr(actual_loss_diffs, predicted_loss_diffs)[0])
    return actual_loss_diffs, predicted_loss_diffs, indices_to_remove


# my new function based on run_spam_experiment
def test_mislabeled_training(model, iter_to_load, force_refresh=False,
                    num_steps=1000, random_seed=17):

    # from experiments.test_retraining above (necessary?) ###########################################################################
    np.random.seed(random_seed)

    model.load_checkpoint(iter_to_load)
    sess = model.sess
    # from experiments.test_retraining above (necessary?) ###########################################################################

    ####################################################################################################################
    # the fix mislabel portion of run_spam_experiment but fix it
    X_train = np.copy(model.data_sets.train.x)
    Y_train = np.copy(model.data_sets.train.labels)
    X_test = np.copy(model.data_sets.test.x)
    Y_test = np.copy(model.data_sets.test.labels)

    num_train_examples = Y_train.shape[0]  # 70000
    num_flip_vals = 6 # why is this only 6?
    num_check_vals = 6 # why is this only 6?
    num_random_seeds = 40 #why do we need this?

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
                # Y_train_flipped[i] = random.choice([j for j in range(10) if j != Y_train[i]])
                Y_train_flipped[i] = random.choice([j for j in range(6) if j != Y_train[i]])

            # 1: Randomly change labels to another class ---------------------------------------------------------------

            model.update_train_x_y(X_train, Y_train_flipped)
            # 2: Include parameters in model.train() to be for ALL_CNN_C for now ---------------------------------------
            model.train(
                num_steps=num_steps,
                iter_to_switch_to_batch=2000000,
                iter_to_switch_to_sgd=2000000)
            # 2: Include parameters in model.train() to be for ALL_CNN_C for now  --------------------------------------
            flipped_results[flips_idx, random_seed_idx, 1:] = model.sess.run(
                [model.loss_no_reg, model.accuracy_op],
                feed_dict=model.all_test_feed_dict)
            print('Flipped loss: %.5f. Accuracy: %.3f' % (
                flipped_results[flips_idx, random_seed_idx, 1], flipped_results[flips_idx, random_seed_idx, 2]))

            train_losses = model.sess.run(model.indiv_loss_no_reg, feed_dict=model.all_train_feed_dict)
            # 3 -------------------------------------------------------------------------------------------------------------------#
            train_loo_influences = np.zeros(len(model.data_sets.train.labels))
            for i in range(len(model.data_sets.train.labels)):
                train_loo_influences[i] = model.get_generic_loo_influences(
                    train_indices=[i],
                    force_refresh=force_refresh)

            # train_loo_influences = model.get_loo_influences()
            # 3 ------------------------------------------------------------------------------------------------------ #

            for checks_idx in range(num_check_vals):
                np.random.seed(random_seed + 1)
                num_checks = int(num_train_examples / 20) * (checks_idx + 1)

                print('### Flips: %s, rs: %s, checks: %s' % (num_flips, random_seed_idx, num_checks))

                fixed_influence_loo_results[flips_idx, checks_idx, random_seed_idx, :], \
                fixed_loss_results[flips_idx, checks_idx, random_seed_idx, :], \
                fixed_random_results[flips_idx, checks_idx, random_seed_idx, :] \
                    = test_mislabeled_detection_batch(
                    model,
                    X_train, Y_train,
                    Y_train_flipped,
                    X_test, Y_test,
                    train_losses, train_loo_influences,
                    num_flips, num_checks)
    ####################################################################################################################
    return orig_results, flipped_results, fixed_influence_loo_results, fixed_loss_results, fixed_random_results

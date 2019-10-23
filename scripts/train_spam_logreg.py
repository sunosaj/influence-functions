from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals  

import numpy as np
import IPython

import tensorflow as tf

import influence.experiments as experiments
from influence.binaryLogisticRegressionWithLBFGS import BinaryLogisticRegressionWithLBFGS


from load_spam import load_spam


data_sets = load_spam()

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
    model_name='spam_logreg_lbfgs')

tf_model.train()

test_idx = 8
actual_loss_diffs, predicted_loss_diffs_cg, indices_to_remove = experiments.test_retraining(
    tf_model,
    test_idx,
    iter_to_load=0,
    force_refresh=False,
    num_to_remove=500,
    remove_type='maxinf',
    random_seed=0)


# LiSSA
np.random.seed(17)
predicted_loss_diffs_lissa = tf_model.get_influence_on_test_loss(
    [test_idx],
    indices_to_remove,
    approx_type='cg',
    approx_params={'scale':25, 'recursion_depth':5000, 'damping':0, 'batch_size':1, 'num_samples':10},
    force_refresh=True
)



np.savez(
    'output/spam_logreg_lbfgs_retraining-500.npz',
    actual_loss_diffs=actual_loss_diffs, 
    predicted_loss_diffs_cg=predicted_loss_diffs_cg,
    predicted_loss_diffs_lissa=predicted_loss_diffs_lissa,
    indices_to_remove=indices_to_remove
    )
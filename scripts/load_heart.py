import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from tensorflow.contrib.learn.python.learn.datasets import base
from influence.dataset import DataSet
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# https://towardsdatascience.com/predicting-presence-of-heart-diseases-using-machine-learning-36f00f3edb2c
# https://www.kaggle.com/ronitf/heart-disease-uci

def load_heart():
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
               'thal', 'target']

    data = pd.read_csv('data/heart/heart.csv', names=columns, sep=' *, *', skiprows=1, na_values='?')
    # test_data = pd.read_csv('data/adult/adult.test', names=columns, sep=' *, *', skiprows=1, na_values='?')

    # data = pd.concat([train_data, test_data])

    print(data.info())
    print(data)
    print(data.shape)

    # # Before data.dropna() #############################################################################################
    # print('data[data.sex == \'Female\'].shape[0]', data[data.sex == 'Female'].shape[0])
    # print('data[data.sex == \'Male\'].shape[0]', data[data.sex == 'Male'].shape[0])
    # # Before data.dropna() #############################################################################################

    # drop rows with missing values (where there is '?')
    # data = data.dropna()
    #
    # print(data.info())
    # print(data)
    # print(data.shape)

    # # count number of occurrences of a specific value ##################################################################
    # # Minority Group - sex (index 9): Female=0(96), Male=1(207)
    print('data[data.sex == \'Female\'].shape[0]', data[data.sex == 0].shape[0])
    print('data[data.sex == \'Male\'].shape[0]', data[data.sex == 1].shape[0])
    # count number of occurrences of a specific value ##################################################################

    # data = data.replace({'<=50K.': '<=50K', '>50K.': '>50K'})
    # data = data.replace({'<=50K': 0, '>50K': 1})

    labels = data['target'].values
    data = data.drop(['target'], axis=1)

    data = data.to_numpy()

    print(data.shape) # 303 -> [0.7, 0.15, 0.15] = [213, 45, 45], [0.8, 0.1, 0.1] = [243, 30, 30]

    data, labels = shuffle(data, labels, random_state=0)

    # print('data[0]', data[0:10])  # [23, 'Private', 84726, 'Assoc-acdm', 12, 'Married-civ-spouse', 'Farming-fishing', 'Wife', 'White', 'Female', 0, 0, 45, 'Germany']
    # print('data[0].shape', data[0:10].shape)
    # # sex_column = data[:, 9]
    # print('np.argwhere(data[:37998, 9] == \'Female\')', np.argwhere(data[:37998, 9] == 'Female'))

    # Columns that are categorical: (sex, cp, fbs, restecg, exang, slope, ca, thal)
    data_categorical = data[:, [1, 2, 5, 6, 8, 10, 11, 12]]
    data_numerical = data[:, [0, 3, 4, 7, 9]]

    # print('np.asarray(data_categorical).shape', np.asarray(data_categorical).shape)

    # one-hot encoding
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data_categorical)
    data_categorical_onehot = enc.transform(data_categorical).toarray()

    print('enc.get_feature_names()', enc.get_feature_names())
    # print('enc.get_feature_names().shape', enc.get_feature_names().shape)
    print(np.where(enc.get_feature_names() == 'x0_0.0'))  # Female: 0 + 5
    # print(np.where(enc.get_feature_names() == 'x5_Amer-Indian-Eskimo'))  # Amer-Indian-Eskimo: 50 + 6
    # print(np.where(enc.get_feature_names() == 'x5_Asian-Pac-Islander'))  # Asian-Pac-Islander: 51 + 6
    # print(np.where(enc.get_feature_names() == 'x5_Black'))  # Black: 52 + 6
    # print(np.where(enc.get_feature_names() == 'x5_Other'))  # Other: 53 + 6

    data_num_and_onehot = np.concatenate((data_numerical, data_categorical_onehot), axis=1)

    # print('data_num_and_onehot[0]', data_num_and_onehot[0])
    # print('data_num_and_onehot[0].shape', data_num_and_onehot[0].shape)
    # print('np.argwhere(data_num_and_onehot[:37998, 61] == 1)', np.argwhere(data_num_and_onehot[:37998, 61] == 1))

    train_size = 240 # 240
    validation_size = 30 # 30 #data.shape[0] * 0.1  # fraction_size = 1, validation_size = 2000
    test_size = 33
    train_and_validation_data = data_num_and_onehot[:train_size + validation_size]
    test_data = data_num_and_onehot[train_size + validation_size:train_size + validation_size + test_size]

    # normalize
    scaler = MinMaxScaler()
    scaler.fit(train_and_validation_data)
    train_and_validation_data = scaler.transform(train_and_validation_data)
    test_data = scaler.transform(test_data)

    # print('np.argwhere(train_and_validation_data[:37998, 61] == 1)', np.argwhere(train_and_validation_data[:37998, 61] == 1))
    #
    # print('train_and_validation_data[0]', train_and_validation_data[0])
    # print('train_and_validation_data[0].shape', train_and_validation_data[0].shape)

    # X_train = train_and_validation_data[:train_size]
    # Y_train = labels[:train_size]
    # X_valid = train_and_validation_data[train_size:train_size + validation_size]
    # Y_valid = labels[train_size:train_size + validation_size]

    X_valid = train_and_validation_data[:validation_size]
    Y_valid = labels[:validation_size]
    X_train = train_and_validation_data[validation_size:validation_size + train_size]
    Y_train = labels[validation_size:validation_size + train_size]

    X_test = test_data
    Y_test = labels[train_size + validation_size:train_size + validation_size + test_size]

    train = DataSet(X_train, Y_train)
    validation = DataSet(X_valid, Y_valid)
    test = DataSet(X_test, Y_test)

    return base.Datasets(train=train, validation=validation, test=test)


# data_sets = load_heart()

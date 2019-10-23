import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from tensorflow.contrib.learn.python.learn.datasets import base
from influence.dataset import DataSet
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


def load_adult():
    columns = ["age", "workClass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
               "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
               "income"]

    train_data = pd.read_csv('data/adult/adult.data', names=columns, sep=' *, *', na_values='?')
    test_data = pd.read_csv('data/adult/adult.test', names=columns, sep=' *, *', skiprows=1, na_values='?')

    data = pd.concat([train_data, test_data])

    # # Before data.dropna() #############################################################################################
    # print('data[data.race == \'White\'].shape[0]', data[data.race == 'White'].shape[0])
    # print('data[data.race == \'Asian-Pac-Islander\'].shape[0]', data[data.race == 'Asian-Pac-Islander'].shape[0])
    # print('data[data.race == \'Amer-Indian-Eskimo\'].shape[0]', data[data.race == 'Amer-Indian-Eskimo'].shape[0])
    # print('data[data.race == \'Other\'].shape[0]', data[data.race == 'Other'].shape[0])
    # print('data[data.race == \'Black\'].shape[0]', data[data.race == 'Black'].shape[0])
    #
    # print('data[data.sex == \'Female\'].shape[0]', data[data.sex == 'Female'].shape[0])
    # print('data[data.sex == \'Male\'].shape[0]', data[data.sex == 'Male'].shape[0])
    # # Before data.dropna() #############################################################################################

    # drop rows with missing values (where there is '?')
    data = data.dropna()

    # # count number of occurrences of a specific value ##################################################################
    # # Minority Group - race (index 8): White(38903), Asian-Pac-Islander(1303), Amer-Indian-Eskimo(435), Other(353), Black(4228)
    # # Minority Group - sex (index 9): Female(14695), Male(30527)
    # print('data[data.race == \'White\'].shape[0]', data[data.race == 'White'].shape[0])
    # print('data[data.race == \'Asian-Pac-Islander\'].shape[0]', data[data.race == 'Asian-Pac-Islander'].shape[0])
    # print('data[data.race == \'Amer-Indian-Eskimo\'].shape[0]', data[data.race == 'Amer-Indian-Eskimo'].shape[0])
    # print('data[data.race == \'Other\'].shape[0]', data[data.race == 'Other'].shape[0])
    # print('data[data.race == \'Black\'].shape[0]', data[data.race == 'Black'].shape[0])
    #
    # print('data[data.sex == \'Female\'].shape[0]', data[data.sex == 'Female'].shape[0])
    # print('data[data.sex == \'Male\'].shape[0]', data[data.sex == 'Male'].shape[0])
    # count number of occurrences of a specific value ##################################################################

    data = data.replace({'<=50K.': '<=50K', '>50K.': '>50K'})
    data = data.replace({'<=50K': 0, '>50K': 1})

    labels = data['income'].values
    data = data.drop(['income'], axis=1)

    data = data.to_numpy()

    print(data.shape) # 45222 -> [0.7, 0.15, 0.15] = [31656, 6783, 6783]

    data, labels = shuffle(data, labels, random_state=0)

    # print('data[0]', data[0:10])  # [23, 'Private', 84726, 'Assoc-acdm', 12, 'Married-civ-spouse', 'Farming-fishing', 'Wife', 'White', 'Female', 0, 0, 45, 'Germany']
    # print('data[0].shape', data[0:10].shape)
    # # sex_column = data[:, 9]
    # print('np.argwhere(data[:37998, 9] == \'Female\')', np.argwhere(data[:37998, 9] == 'Female'))

    # Columns that are categorical: 1, 3, 5, 6, 7, 8, 9, 13
    data_categorical = data[:, [1, 3, 5, 6, 7, 8, 9, 13]]
    data_numerical = data[:, [0, 2, 4, 10, 11, 12]]

    # print('np.asarray(data_categorical).shape', np.asarray(data_categorical).shape)

    # one-hot encoding
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data_categorical)
    data_categorical_onehot = enc.transform(data_categorical).toarray()

    print('enc.get_feature_names()', enc.get_feature_names())
    # print('enc.get_feature_names().shape', enc.get_feature_names().shape)
    print(np.where(enc.get_feature_names() == 'x6_Female'))  # Female: 55 + 6, Male: 56 + 6
    print(np.where(enc.get_feature_names() == 'x5_Amer-Indian-Eskimo'))  # Amer-Indian-Eskimo: 50 + 6
    print(np.where(enc.get_feature_names() == 'x5_Asian-Pac-Islander'))  # Asian-Pac-Islander: 51 + 6
    print(np.where(enc.get_feature_names() == 'x5_Black'))  # Black: 52 + 6
    print(np.where(enc.get_feature_names() == 'x5_Other'))  # Other: 53 + 6
    print(np.where(enc.get_feature_names() == 'x5_White'))  # Other: 54 + 6

    data_num_and_onehot = np.concatenate((data_numerical, data_categorical_onehot), axis=1)

    # print('data_num_and_onehot[0]', data_num_and_onehot[0])
    # print('data_num_and_onehot[0].shape', data_num_and_onehot[0].shape)
    # print('np.argwhere(data_num_and_onehot[:37998, 61] == 1)', np.argwhere(data_num_and_onehot[:37998, 61] == 1))

    train_size = 38000
    validation_size = 3000 #data.shape[0] * 0.1  # fraction_size = 1, validation_size = 2000
    train_and_validation_data = data_num_and_onehot[:train_size + validation_size]
    test_data = data_num_and_onehot[train_size + validation_size:]

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
    Y_test = labels[train_size + validation_size:]

    train = DataSet(X_train, Y_train)
    validation = DataSet(X_valid, Y_valid)
    test = DataSet(X_test, Y_test)

    return base.Datasets(train=train, validation=validation, test=test)


data_sets = load_adult()

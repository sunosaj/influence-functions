import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from tensorflow.contrib.learn.python.learn.datasets import base
from influence.dataset import DataSet
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

# https://towardsdatascience.com/predicting-hospital-readmission-for-patients-with-diabetes-using-scikit-learn-a2e359b15f0

def load_diabetes():
    columns = ['encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'weight', 'admission_type_id',
               'discharge_disposition_id', 'admission_source_id', 'time_in_hospital', 'payer_code', 'medical_specialty',
               'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency',
               'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses', 'max_glu_serum', 'A1Cresult',
               'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
               'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
               'tolazamide', 'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
               'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone', 'change', 'diabetesMed',
               'readmitted']

    data = pd.read_csv('data/diabetes/diabetic_data.csv', names=columns, sep=' *, *', skiprows=1, na_values='?')
    # data = pd.read_csv('data/diabetes/diabetic_data.csv')
    # test_data = pd.read_csv('data/adult/adult.test', names=columns, sep=' *, *', skiprows=1, na_values='?')
    # print(data.info()) #race, weight, payer_code, medical speciality, diag_1, diag_2, diag_3 have missing values
    # print(data.shape)
    # print(data.iloc[0, :])
    # data = pd.concat([train_data, test_data])

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

    # drop rows with missing values (where there is '?') in these columns
    data = data.dropna(axis=0, subset=['race', 'diag_1', 'diag_2', 'diag_3'])
    data = data[data.gender != 'Unknown/Invalid']
    # print(data.info())
    # print(data)
    # print(data.shape)

    # drop columns weight, payer_code, medical_specialty because too many missing values and probably not useful info?
    data = data.drop(['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty'], axis=1)
    # print(data.info())
    # print(data)
    # print(data.shape)
    # drop the weight and payer-code columns because there are so many missing values?
    # drop rows where there are missing values for race or gender
    # drop rows where there are missing values for diag_1, diag_2, or diag_3

    # count number of occurrences of a specific value ##################################################################
    # Minority Group - race (index 8): Caucasian(75079), Asian(625), AfricanAmerican(18881), Hispanic(1984), Other(1483)
    # Minority Group - gender (index 9): Female(52833), Male(45219)
    print('data[data.race == \'Caucasian\'].shape[0]', data[data.race == 'Caucasian'].shape[0])
    print('data[data.race == \'Asian\'].shape[0]', data[data.race == 'Asian'].shape[0])
    print('data[data.race == \'AfricanAmerican\'].shape[0]', data[data.race == 'AfricanAmerican'].shape[0])
    print('data[data.race == \'Hispanic\'].shape[0]', data[data.race == 'Hispanic'].shape[0])
    print('data[data.race == \'Other\'].shape[0]', data[data.race == 'Other'].shape[0])

    print('data[data.gender == \'Female\'].shape[0]', data[data.gender == 'Female'].shape[0])
    print('data[data.gender == \'Male\'].shape[0]', data[data.gender == 'Male'].shape[0])
    # count number of occurrences of a specific value ##################################################################

    # convert to binary classes
    data[['readmitted']] = data[['readmitted']].replace({'<30': 1, '>30': 1, 'NO': 0})
    print(data.info())
    print(data)
    print(data.shape)
    # extract labels from data
    labels = data['readmitted'].values
    data = data.drop(['readmitted'], axis=1)

    print(data.info())
    print(data)
    print(data.shape)

    data = data.to_numpy()

    print(data.shape) # 98052 -> [0.7, 0.15, 0.15] = [68636, 14708, 14708]

    data, labels = shuffle(data, labels, random_state=0)

    # print('data[0]', data[0:10])  # [23, 'Private', 84726, 'Assoc-acdm', 12, 'Married-civ-spouse', 'Farming-fishing', 'Wife', 'White', 'Female', 0, 0, 45, 'Germany']
    # print('data[0].shape', data[0:10].shape)
    # # sex_column = data[:, 9]
    # print('np.argwhere(data[:37998, 9] == \'Female\')', np.argwhere(data[:37998, 9] == 'Female'))

    # Columns that are categorical: 1, 3, 5, 6, 7, 8, 9, 13
    data_categorical = data[:, [0, 1, 2, 3, 4, 5, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                                31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43]]
    data_numerical = data[:, [6, 7, 8, 9, 10, 11, 12, 16]]

    print('np.asarray(data_categorical).shape', np.asarray(data_categorical).shape)
    print('np.asarray(data_numerical).shape', np.asarray(data_numerical).shape)

    # one-hot encoding
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data_categorical)
    data_categorical_onehot = enc.transform(data_categorical).toarray()

    print('enc.get_feature_names()', enc.get_feature_names()[:15])
    print('enc.get_feature_names().shape', enc.get_feature_names().shape)
    print(np.where(enc.get_feature_names() == 'x0_AfricanAmerican'))  # AfricanAmerican: 0 + 8
    print(np.where(enc.get_feature_names() == 'x0_Asian'))  # Asian: 1 + 8
    print(np.where(enc.get_feature_names() == 'x0_Caucasian'))  # Caucasian: 2 + 8
    print(np.where(enc.get_feature_names() == 'x0_Hispanic'))  # Hispanic: 3 + 8
    print(np.where(enc.get_feature_names() == 'x0_Other'))  # Other: 4 + 8
    print(np.where(enc.get_feature_names() == 'x1_Female'))  # Female: 5 + 8
    print(np.where(enc.get_feature_names() == 'x1_Male'))  # Male: 6 + 8

    data_num_and_onehot = np.concatenate((data_numerical, data_categorical_onehot), axis=1)

    print('data_num_and_onehot[0]', data_num_and_onehot[0])
    print('data_num_and_onehot[0].shape', data_num_and_onehot[0].shape)
    print('data_num_and_onehot.shape', data_num_and_onehot.shape)
    # print('np.argwhere(data_num_and_onehot[:37998, 61] == 1)', np.argwhere(data_num_and_onehot[:37998, 61] == 1))

    train_size = 84000
    validation_size = 7000 #data.shape[0] * 0.1  # fraction_size = 1, validation_size = 2000
    test_size = 7052
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


# data_sets = load_diabetes()

"""
file: InterestRatePrediction.py
author: Subhash Bylaiah
description: Program to train a model for Interest rate prediction and make predictions
######################################################################################################################
#       USES PYTHON VERSION 3
######################################################################################################################

"""
#!/usr/local/bin/python3

from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cluster import  AffinityPropagation
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVR
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import xgboost as xgb
from sklearn.externals import joblib

from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import model_from_json
from keras.regularizers import l2


import numpy as np
import pandas as pd
import nltk
import sys
import time
import json

if sys.version_info.major==2:
	print("warning! use python3")


column_mapping = {
    'X1': 'interest_rate',
    'X2': 'loan_id',
    'X3': 'borrower_id',
    'X4': 'request_amt',
    'X5': 'funded_amt',
    'X6': 'investor_funded_amt',
    'X7': 'num_instalments',
    'X8': 'loan_grade',
    'X9': 'loan_subgrade',
    'X10': 'emp_or_jobtitle',
    'X11': 'years_employed', #Number of years employed (0 to 10; 10 = 10 or more)
    'X12': 'home_ownership', #"Home ownership status: RENT, OWN, MORTGAGE, OTHER."
    'X13': 'annual_income',
    'X14': 'income_verification_status',
    'X15': 'loan_issue_date',
    'X16': 'loan_reason',
    'X17': 'loan_category',
    'X18': 'loan_title',
    'X19': 'zip_code',
    'X20': 'state',
    'X21': 'debt_income_ratio',
    'X22': 'num_delinqs',
    'X23': 'earliest_credit_line',
    'X24': 'num_credit_inquiries',
    'X25': 'months_since_last_delinq',
    'X26': 'months_since_last_rec',
    'X27': 'open_credit_lines',
    'X28': 'num_derog_pub_recs',
    'X29': 'revolving_bal',
    'X30': 'revolving_line_util',
    'X31': 'tot_credit_lines',
    'X32': 'initial_list_status'
}



############################ EXLPORATORY DATA ANALYSIS #################################################
# Done in IPython notebook
########################################################################################################



# INPUTS FROM EXPLORATORY DATA ANALYSIS
# Drop Non informative features after EDA
features_to_drop = ['loan_id',
                    'borrower_id',
                    'funded_amt',
                    'investor_funded_amt',
                    'loan_grade',
                    'initial_list_status',
                    'zip_code',
                    'state',
                    'loan_reason',
                    'loan_title',
                    'emp_or_jobtitle',
                    'loan_title',
                    'loan_issue_date',
                    'issue_month',

                    # additional attribs
                    'gen_emp_title',
                    # 'issue_year',
                    # 'earliest_credit_line'

                    ]

# attributes with missing data for imputation
categorical_missing = ['loan_subgrade', 'gen_emp_title', 'home_ownership']
numeric_missing = ['annual_income', 'months_since_last_delinq', 'months_since_last_rec', 'revolving_line_util']


#
target_var = 'interest_rate'

# numeric_columns
numeric_cols = ['interest_rate', 'request_amt', 'funded_amt', 'investor_funded_amt',
                'revolving_line_util', 'num_instalments']

training_cols = []
val_to_num_mapping = {}

training_values_model_file = 'training_values_model.out'

scalingModelFile = 'scalingmodel.pkl'
xgbModelFile = 'xgbmodel.pkl'
xgbModelWithBestParsFile = 'xgbmodel-withbestpars.pkl'
nnModelFile = 'nnmodel.json'
nnModelWeightsFile = 'nnmodelweights.out'

# scalerModel = StandardScaler()
# xgbsavedmodel = xgb.XGBRegressor()
# xgbwithbestparssavedmodel = xgb.XGBRegressor()
#

def transform_emp_title(df):
    '''
    Implemented a clustering technique to normalize and cluster the emp_job_titles into a smaller set of exemplar representatives
        Uses sklearn's AffinityPropagation clustering technique and NLTK's editdistance as a distance measure
    :param df:
    :return:
    '''
    # convert everything into lower case
    df['emp_or_jobtitle'] = df['emp_or_jobtitle'].str.lower()

    emp_value_counts = df['emp_or_jobtitle'].value_counts(normalize=True).nlargest(100) * 100

    titles = emp_value_counts.keys()

    # Using NLTK's edit distance metric to find distances between words
    # we are using a clustering method

    word_similarity = -1 * np.array([[nltk.edit_distance(w1, w2) for w1 in titles] for w2 in titles])

    affprop = AffinityPropagation(affinity="precomputed", damping=0.5)
    affprop.fit(word_similarity)
    for cluster_id in np.unique(affprop.labels_):
        exemplar = titles[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(titles[np.nonzero(affprop.labels_ == cluster_id)])
        # df['gen_emp_title'] = df['emp_or_jobtitle'].apply(lambda old: exemplar if old in cluster else 'NA')
        df.ix[df['emp_or_jobtitle'].isin(cluster), 'gen_emp_title'] = exemplar
        cluster_str = ", ".join(cluster)
        print(" - *%s:* %s" % (exemplar, cluster_str))

    return df

def remove_invalid_training_data(df):
    '''
    removes identified invalid data in training set
    :param df:
    :return:
    '''
    # there seems to be one record which has everything null..lets remove that..
    df.drop(df[df['loan_id'].isnull()].index, axis=0, inplace=True)

    # There are many records for which interest rate is Null.
    # As interest_rate is our target variable and we are learning a function approximator to predict that,
    # it does not make sense to have any records that are null..
    # Lets investigate what these records are anyways..

    df.drop(df[df[target_var].isnull()].index, axis=0, inplace=True)

    # lets reset the index
    df = df.reset_index(drop=True)
    return df

def cleanup(df):
    '''
    data cleanup, using regex, to strip off invalid chars to before converting to numeric, for attrs representing numeric data
    :param df:
    :return:
    '''
    # looks like some cleanup is in order
    # % symbols in the percentages, $ symbols, "months" in the num_instalments
    df[numeric_cols] = df[numeric_cols].replace({'\$': '', ',': '', '\%': '', 'months': '', ' ': ''}, regex=True) \
        .astype(float)

    return df

def transform_train(df):
    '''
        This function will extract new features, transform existing ones to help make better predictions
         this is applied on training dataset

    :param df:
    :return:
    '''

    # we are not using Emp Title, as per EDA this is not helping much, so commenting out below line
    # transform_emp_title(df)

    # It will be good to have the years_employed as a real valued quantitative varaible
    # as it contains useful order information
    # Lets convert this to numeric, we can remove every non-digit characters using regex

    df['years_employed'] = df['years_employed'].replace({'[^\\d]': ''}, regex=True).replace('', 0).astype(float)

    # transforming earliest_credit_line
    # Its useful to have this variable as a quantitative information of the length of time since
    # a borrower has had the first credit line

    df['earliest_credit_line'] = pd.to_datetime(df.earliest_credit_line, format='%b-%y')

    # ALso, the years do not have centuries, so doing a little hack to correct the date
    # when it gets wrong.. ex. May-63 is interepreted as 01-May-2063, instead of 01-May-1963
    df.ix[df.earliest_credit_line > pd.datetime.now().date(), 'earliest_credit_line'] = \
        df.ix[df.earliest_credit_line > pd.datetime.now().date(), 'earliest_credit_line'] \
        - pd.Timedelta(days=365 * 100)

    # now convert then into length, in number of days
    df['earliest_credit_line'] = \
        (pd.to_timedelta(pd.datetime.now().date() - pd.to_datetime(df.earliest_credit_line, format='%b-%y'))).dt.days

    df['loan_issue_date'] = pd.to_datetime(df.loan_issue_date, format='%b-%y')

    # ALso, the years do not have centuries, so doing a little hack to correct the date
    # when it gets wrong.. ex. May-63 is interepreted as 01-May-2063, instead of 01-May-1963
    df.ix[df.loan_issue_date > pd.datetime.now().date(), 'loan_issue_date'] = \
        df.ix[df.loan_issue_date > pd.datetime.now().date(), 'loan_issue_date'] - pd.Timedelta(days=365 * 100)

    df['issue_year'] = df.loan_issue_date.dt.year
    df['issue_month'] = df.loan_issue_date.dt.month

    return df

def transform_test(df):
    '''
        This function will extract new features, transform existing ones to help make better predictions
         this is specific to test/holdout dataset, as the formats of some of the columns(date fields) are different in testset

    :param df:
    :return:
    '''
    # we are not using Emp Title, as per EDA this is not helping much, so commenting out below line
    # transform_emp_title(df)

    # It will be good to have the years_employed as a real valued quantitative varaible
    # as it contains useful order information
    # Lets convert this to numeric, we can remove every non-digit characters using regex

    df['years_employed'] = df['years_employed'].replace({'[^\\d]': ''}, regex=True).replace('', 0).astype(float)

    # transforming earliest_credit_line
    # Its useful to have this variable as a quantitative information of the length of time since
    # a borrower has had the first credit line

    # HOWEVER IN TEST DATESET THIS DATE VALUES ARE APPEARING IN 2 FORMATS like.. 2-Mar and Mar-68
    # Making an assumption that in the first case above it represents dd-Mon with year missing and assuming current year
    # Making an assumption that in the second case above it represents Mon-yy with date missing, which is not needed

    #     first create a temp variable to detect if the date is in first format above
    df['temp1'] = \
        df.earliest_credit_line.apply(lambda date_val: date_val.split('-')[0])

    #   try to convert to numeric, it will be Null where it couldnt be converted
    df['temp1'] = df['temp1'].apply(pd.to_numeric, errors='coerce')

    # whereever it is notnull, it is in first format and where it is null its in second format
    # When date is in the format like 2-Jan, assuming the format as dd-Mon,
    df.ix[df['temp1'].notnull(), 'earliest_credit_line1'] = \
        pd.to_datetime(df.ix[df['temp1'].notnull(), 'earliest_credit_line'], format='%d-%b')

    # also, the year got set to 0, i.e 1900, so making a informed best guess that it should be the current year
    # of the data set which appears to be the year 2015, so will also have to add back 115 years,
    df.ix[df['temp1'].notnull(), 'earliest_credit_line1'] = \
        df.ix[df['temp1'].notnull(), 'earliest_credit_line1'] \
        + pd.Timedelta(days=365 * 115)  # we are not taking care of the leap years which is fine

    # When date is in the format like Dec-68, assuming the format as Mon-yy,
    df.ix[df['temp1'].isnull(), 'earliest_credit_line1'] = \
        pd.to_datetime(df.ix[df['temp1'].isnull(), 'earliest_credit_line'], format='%b-%y')

    df.drop(['temp1', 'earliest_credit_line'], axis=1, inplace=True)
    df = df.rename(columns={'earliest_credit_line1': 'earliest_credit_line'})

    # ALso, the years do not have centuries, so doing a little hack to correct the date
    # when it gets wrong.. ex. May-63 is interepreted as 01-May-2063, instead of 01-May-1963
    df.ix[df.earliest_credit_line > pd.datetime.now().date(), 'earliest_credit_line'] = \
        df.ix[df.earliest_credit_line > pd.datetime.now().date(), 'earliest_credit_line'] \
        - pd.Timedelta(days=365 * 100)

    # now convert then into length, in number of days
    df['earliest_credit_line'] = \
        (pd.to_timedelta(pd.datetime.now().date() - pd.to_datetime(df.earliest_credit_line, format='%b-%y'))).dt.days

    # the dates are in the format like 15-Mar, making a best guess the first 2 digits are year and not a date
    df['loan_issue_date'] = pd.to_datetime(df.loan_issue_date, format='%y-%b')

    df['issue_year'] = df.loan_issue_date.dt.year
    df['issue_month'] = df.loan_issue_date.dt.month

    return df

def drop_non_informative_feats(df, cols):
    # just drop the given cols from the dataframe
    actual_cols = [col for col in cols if col in df.columns]
    df.drop(actual_cols, axis=1, inplace=True)
    return df


def impute_categorical_features(df, cols):
    # We will replace all categorical missing values with a new category - lets say NA
    for col in cols:
        if col in df.columns:
            df.loc[df[col].isnull(), col] = 'NA'

    return df

def impute_numeric_features(df, cols):
    # We will impute all real valued variables with 0 and add an indicator variable to indicate missingness
    # so the new indicator variable will have a value 1 when the data is missing, else 0
    # In a linear model this will have a nice effect of automatically learning an approriate co-efficient or a constant
    # value, when the data is missing for col in cols:
    col_suffix = '_is_missing'
    for col in cols:
        if col in df.columns:
            new_col = col + col_suffix
            df[new_col] = 0
            df.loc[df[col].isnull(), new_col] = 1
            df.loc[df[col].isnull(), col] = 0
    return df

def read_data(filename):
    return pd.read_csv(filename) \
        .rename(columns=column_mapping)


def prepare_training_data(df):
    return pd.get_dummies(df)


def prepare_test_data(df, train_cols):
    df = pd.get_dummies(df)

    # find columns that are not in testdata due to encoding
    # add these columns to test, setting them equal to zero
    for col in np.setdiff1d(train_cols, df.columns):
        df[col] = 0

    # select and reorder the test columns using the train columns
    df = df[train_cols]
    return df


def process_training_data(filename):
    # read data in
    training_data = read_data(filename)

    # remove invalid records
    training_data = remove_invalid_training_data(training_data)

    # cleanup data
    training_data = cleanup(training_data)

    # transform
    training_data = transform_train(training_data)

    training_data = drop_non_informative_feats(training_data, features_to_drop)

    # lets transform loan_subgrade to numeric
    # This will help with performance

    val_to_num_mapping = {value:idx for idx,value in
              enumerate(np.unique(training_data.loc[training_data['loan_subgrade'].notnull(),'loan_subgrade']))}

    training_data['loan_subgrade_numeric'] = training_data['loan_subgrade'].map(val_to_num_mapping)

    # lets also drop loan_subgrade which is categorical, as we have transformed to numeric
    training_data = drop_non_informative_feats(training_data, ['loan_subgrade'])

    # impute categorical features
    training_data = impute_categorical_features(training_data, categorical_missing)

    # impute numeric features
    training_data = impute_numeric_features(training_data, numeric_missing + ['loan_subgrade_numeric'])

    # prepare data
    training_data = prepare_training_data(training_data)

    y = training_data.interest_rate.values
    training_data.drop('interest_rate', axis=1, inplace=True)
    X = training_data.values

    with open(training_values_model_file, 'w+') as filehandle:
        json.dump({'loan_subgrade': val_to_num_mapping, 'training_cols': list(training_data.columns)},
                  filehandle, sort_keys=True, indent=4, ensure_ascii=False)

    return (X,y)


def process_test_data(filename):
    # read data in
    test_data = read_data(filename)

    # cleanup data
    test_data = cleanup(test_data)

    # transform
    test_data = transform_test(test_data)

    test_data = drop_non_informative_feats(test_data, features_to_drop)

    # lets transform loan_subgrade to numeric
    # This will help with performance


    test_data['loan_subgrade_numeric'] = test_data['loan_subgrade'].map(val_to_num_mapping)

    # lets also drop loan_subgrade which is categorical, as we have transformed to numeric
    test_data = drop_non_informative_feats(test_data, ['loan_subgrade'])

    # impute categorical features
    test_data = impute_categorical_features(test_data, categorical_missing)

    # impute numeric features
    test_data = impute_numeric_features(test_data, numeric_missing + ['loan_subgrade_numeric'])

    # prepare data
    test_data = prepare_test_data(test_data, training_cols)

    # test_data.drop('interest_rate', axis=1, inplace=True)
    X = test_data.values

    return (X)


def fitStandardScaler(X, X_train, X_test):
    sc = StandardScaler()

    # we will fit using all of the data
    sc.fit(X)

    # transform X_train and X_test separately
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # save the scaling model
    joblib.dump(sc, scalingModelFile)

    return (X_train_std, X_test_std)

def transformUsingScalerModel(X):
    scalerModel = joblib.load(scalingModelFile)
    return scalerModel.transform(X)


def train_linear_reg_model(X_train_std, y_train, X_test_std, y_test):
    '''
    Trains a simple linear regression model with L2 regularization, using a CV based searcch for hyperparameter tuning

    :param X_train_std:
    :param y_train:
    :param X_test_std:
    :param y_test:
    :return:
    '''

    reg_model = Ridge(normalize=True, fit_intercept=True)
    params = {
        'alpha': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    }
    model = GridSearchCV(reg_model, param_grid=params, cv=5, refit=True)
    model.fit(X_train, y_train)

    est = model.best_estimator_

    print('############ Model performance ###########################################')
    print('Training Score:', model.best_score_)
    print('Testing Score:', est.score(X_test_std, y_test))
    print('##########################################################################')
    pass


def train_XGBRegressor_withCV(X_train_std, y_train, X_test_std, y_test):
    start_time = time.time()
    print('Training XGB Regressor model...')
    xgb_model = xgb.XGBRegressor()
    params = {
              'max_depth': [4,6,8],
              'n_estimators': [100, 300, 500]
    }
    model = GridSearchCV(xgb_model, param_grid=params, verbose=1, n_jobs=-1, refit=True)
    model.fit(X_train_std, y_train)

    print('Model scores are:')
    print(model.grid_scores_)

    print('Model best params are:')
    print(model.best_params_)
    est = model.best_estimator_

    print('############ Model performance ###########################################')
    print('Training Score:', model.best_score_)
    print('Testing Score:', est.score(X_test_std, y_test))
    print('##########################################################################')

    # save the scaling model
    joblib.dump(est, xgbModelFile)

    end_time = time.time()
    print("Trained XGB Regressor model in ", end_time - start_time, " seconds")
    pass

def train_XGBRegressor_with_known_best_params(X_train_std, y_train, X_test_std, y_test):
    '''
    this function trains an XGBoost regression model on the training set,
        also the model is evaluated on the test set..
    :param X_train_std:
    :param y_train:
    :param X_test_std:
    :param y_test:
    :return:
    '''
    best_num_estimators = 500
    best_max_depth = 6
    start_time = time.time()
    print('Training XGB Regressor model with known best params of Max Depth:%d, and Num Estimators:%d' %(best_max_depth, best_num_estimators))
    xgb_model = xgb.XGBRegressor(n_estimators=best_num_estimators, max_depth=best_max_depth)
    xgb_model.fit(X_train_std, y_train)

    print('############ Model performance ###########################################')
    print('Training Score:', xgb_model.score(X_train_std, y_train))
    print('Testing Score:', xgb_model.score(X_test_std, y_test))
    print('##########################################################################')


    y_preds = xgb_model.predict(X_test_std)
    y_preds = pd.DataFrame(y_preds, columns=['xgbmodel_predictions'])
    # y_preds.to_csv('predictions-test.csv')

    # save the scaling model
    joblib.dump(xgb_model, xgbModelWithBestParsFile)
    end_time = time.time()
    print("Trained XGB Regressor model with known best params in ", end_time - start_time, " seconds")
    pass


def predict_using_XGB_known_best_pars_model(X_std):
    '''
    make predictions on the input data using the model that was trained in the training phase
    :param X_std:
    :return:
    '''
    start_time = time.time()
    print('Predicting using XGB Model')
    xgb_model = joblib.load(xgbModelWithBestParsFile)
    y_preds = np.round(xgb_model.predict(X_std),2)

    y_preds = pd.DataFrame(y_preds, columns=['xgbmodel_predictions'])
    end_time = time.time()
    print("Completed predictions in ", end_time - start_time, " seconds")
    return y_preds

def buildNeuralNetModel(X_train, y_train):
    '''
    Build a Neutal network model and compile it..
        The neuralnet has one hidden layer with 100 nodes, these parameters are determined on multiple runs on
            the training set and evaluations
        using L2 regularization

    :param X_train:
    :param y_train:
    :return:
    '''
    model = Sequential()
    # hidden_nodes = int(X_train.shape[1])
    # model.add(Dropout(0.2, input_shape=(X_train.shape[1],)))
    model.add(Dense(input_dim=X_train.shape[1], output_dim=100, init='uniform', activation='relu', W_regularizer = l2(.01)))
    # model.add(Dense(input_dim=50, output_dim=50, init='uniform', activation='tanh'))
    # model.add(Dropout(0.2))
    model.add(Dense(input_dim=100, output_dim=1, init='uniform', W_regularizer = l2(.01)))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_NN(X_train_std, y_train, X_test_std, y_test):
    '''
    Train the neuralnet model using the training set and evaluate performance on the test set
    :param X_train_std:
    :param y_train:
    :param X_test_std:
    :param y_test:
    :return:
    '''
    start_time = time.time()
    num_hidden_layers = 1
    num_hidden_nodes = 100

    print('Training Neural Network model with known best params of Hidden Layers:%d, and Nodes in Hidden Layers:%d' %(num_hidden_layers, num_hidden_nodes))
    nnModel = buildNeuralNetModel(X_train_std, y_train)
    nnModel.fit(X_train_std, y_train, nb_epoch=100, batch_size=500, verbose=0, validation_split=0.1)

    print('############ Model performance ###########################################')
    print('Training Score:', r2_score(y_train, nnModel.predict(X_train_std)))
    print('Testing Score:', r2_score(y_test, nnModel.predict(X_test_std)))
    print('##########################################################################')


    # save the scaling model
    model_json = nnModel.to_json()
    with open(nnModelFile, "w") as json_file:
        json_file.write(model_json)
    nnModel.save_weights(nnModelWeightsFile)
    print("Saved model files to: %s, %s" %(nnModelFile, nnModelWeightsFile))

    end_time = time.time()
    print("Trained Neural network model with known best params in ", end_time - start_time, " seconds")
    pass


def predict_using_NN_model(X_std):
    '''
    make predictions using the trained model that was saved
    :param X_std:
    :return:
    '''
    start_time = time.time()
    print('Predicting using NN Model')

    with open(nnModelFile, 'r') as filehandle:
        model_json = filehandle.read()

    nnModel = model_from_json(model_json)

    # load weights into new model
    nnModel.load_weights(nnModelWeightsFile)
    print("Loaded model from disk")

    y_preds = np.round(nnModel.predict(X_std),2)

    y_preds = pd.DataFrame(y_preds, columns=['neuralnet_predictions'])
    end_time = time.time()
    print("Completed predictions in ", end_time - start_time, " seconds")
    return y_preds


def read_models():
    # read the additional parameters from the trained model,
    #   specific to data transformation
    global val_to_num_mapping
    global training_cols
    with open(training_values_model_file, 'r') as filehandle:
        training_values_model = json.load(filehandle, encoding="ISO-8859-1")
        val_to_num_mapping = training_values_model['loan_subgrade']
        training_cols = training_values_model['training_cols']


if __name__ == "__main__":

    (mode, datafile) = sys.argv[1:3]
    # (mode, datafile) = ('train', 'Data for Cleaning & Modeling.csv')
    # (mode, datafile) = ('predict', 'Holdout for Testing.csv')
    if mode == 'train':
        print('Running in train mode...')
        X,y = process_training_data(datafile)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        X_train_std, X_test_std = fitStandardScaler(X, X_train, X_test)

        # train the XGBoost model
        train_XGBRegressor_with_known_best_params(X_train_std, y_train, X_test_std, y_test)

        # train neuralnetwork model
        train_NN(X_train_std, y_train, X_test_std, y_test)

    if mode == 'predict':
        print('Running in predict mode...')
        read_models()
        X = process_test_data(datafile)
        X_std = transformUsingScalerModel(X)

        # make predictions with XGBoost model
        y_preds_xgb = predict_using_XGB_known_best_pars_model(X_std)

        # make predictions using NN model
        y_preds_nn = predict_using_NN_model(X_std)

        y_preds = y_preds_xgb.join(y_preds_nn)

        # save predictions from the two models into a csv file
        y_preds.to_csv('Results from SubhashBylaiah.csv', index=False, float_format='%.2f')




    #
    # nnModel.summary()
    pass
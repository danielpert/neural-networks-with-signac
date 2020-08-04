import pandas as pd
import numpy as np
import json
import pickle
import signac
from flow import FlowProject
import os
from subprocess import Popen, PIPE
import pathlib
import atools_ml
from atools_ml.prep import dimensionality_reduction

import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

TARGETS = ['COF', 'intercept']
IDENTIFIERS = ['terminal_group_1', 'terminal_group_2', 'terminal_group_3',
               'backbone', 'frac-1', 'frac-2']

# get project root directory
proj = signac.get_project()
root_dir = proj.root_directory()

# define group that combines the different operations
neural_network = FlowProject.make_group(name='neural_network')

@FlowProject.label
def models_trained(job):
    # post condition for train_evaluate
    # checks if pickle files for models and scalers are in job workspace
    # check external hard drive for models
    
    for target in TARGETS:
        if not os.path.isfile('/mnt/d/neural-networks-with-signac/workspace/' + job.id + '/{}_trained.pickle'.format(target)):
            return False
        if not os.path.isfile(job.fn('{}_scaler.pickle'.format(target))):
            return False
    return True


@FlowProject.label
def features_in_ws(job):
    # post condition for train_evaluate
    # checks if features json file for both targets in job workspace
    
    return all([os.path.isfile(job.fn('{}_features.json'.format(target))) for target in TARGETS])


@FlowProject.label
def scores_in_doc(job):
    # post condition for train_evaluate
    # checks if job document contains all scores for both COF and
    #intercept
    
    for target in TARGETS:
        for score_name in ['r2', 'rmse', 'mae']:
            for data_set in ['test', 'train']:
                if not job.document.get('_'.join([target, score_name, data_set])):
                    return False
    return True


@neural_network
@FlowProject.operation
@FlowProject.post(models_trained)
@FlowProject.post(features_in_ws)
@FlowProject.post(scores_in_doc)
def train_evaluate(job):
    '''
    train MLP Regressor models for COF and intercept
    for the parameters given in the job statepoints
    
    evaluate using R^2, root mean square error, and
    mean absolute error
    '''
    for target in TARGETS:
        
        # read training data
        with open(root_dir + '/csv-files/{}_training_4.csv'.format(target)) as f:
            train = pd.read_csv(f, index_col=0)
        # read testing data
        with open(root_dir + '/csv-files/{}_testing.csv'.format(target)) as f:
            test = pd.read_csv(f, index_col=0)
        
        # Reduce the number of features by running data thru dimensionality reduction
        features_all = list(train.drop([target] + IDENTIFIERS, axis=1))
        train_red = dimensionality_reduction(train, features_all,
                                             filter_missing=True,filter_var=True,
                                             filter_corr=True,
                                             missing_threshold=0.4,
                                             var_threshold=0.02,
                                             corr_threshold=0.9)
        features = list(train_red.drop([target] + IDENTIFIERS, axis=1))
        
        # split train and test data into features (X) and target (y)
        X_train, y_train = train[features], train[target]
        X_test, y_test = test[features], test[target]
        
        # normalize input features                        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # train multi-layer perceptron neural network
        hidden_layers = [job.sp.num_perceptrons]*job.sp.num_layers
        MLP = MLPRegressor(hidden_layer_sizes=hidden_layers, alpha=job.sp.alpha,
                           random_state=43, tol=1e-6, max_iter=1000)
        MLP.fit(X_train_scaled, y_train)
        
        # score the model on train and test data using RMSE, MAE, R^2
        # store the scores in job document
        r2_test = MLP.score(X_test_scaled, y_test)
        r2_train = MLP.score(X_train_scaled, y_train)
        job.doc['{}_r2_test'.format(target)] = r2_test
        job.doc['{}_r2_train'.format(target)] = r2_train
        
        y_test_pred = MLP.predict(X_test_scaled)
        y_train_pred = MLP.predict(X_train_scaled)
        rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)
        rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
        job.doc['{}_rmse_test'.format(target)] = rmse_test
        job.doc['{}_rmse_train'.format(target)] = rmse_train
        
        mae_test = mean_absolute_error(y_test, y_test_pred)
        mae_train = mean_absolute_error(y_train, y_train_pred)
        job.doc['{}_mae_test'.format(target)] = mae_test
        job.doc['{}_mae_train'.format(target)] = mae_train
        
        # add features to json file in job workspace
        with open(job.fn('{}_features.json'.format(target)), 'w') as f:
            json.dump(features, f)
        
        # pickle out the model and scaler
        with open(job.fn('{}_trained.pickle'.format(target)), 'wb') as f:
            pickle.dump(MLP, f)
        with open(job.fn('{}_scaler.pickle'.format(target)), 'wb') as f:
            pickle.dump(scaler, f)
    
    # copy the job directory to external hard drive
    job_dir_path = pathlib.Path(root_dir + '/workspace/' + job.id)
    hard_drive_path = pathlib.Path('/mnt/d/neural-networks-with-signac/workspace/')
    process = Popen(['cp', '-r', job_dir_path, hard_drive_path], stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()
    
    # remove trained model pickle files from job directory
    # because they are backed up to external hard drive and take up a lot of space
    for target in TARGETS:
        path_to_pickle = pathlib.Path(str(job_dir_path) + '/{}_trained.pickle'.format(target))
        process = Popen(['rm', path_to_pickle], stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
            
        
if __name__ == '__main__':
    FlowProject().main()
        
        
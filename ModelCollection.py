# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:16:27 2019

@author: Paulo
"""


import os
from os.path import join
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC # classifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, balanced_accuracy_score, f1_score, accuracy_score, cohen_kappa_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RepeatedKFold, LeaveOneGroupOut
from sklearn.model_selection import GroupKFold, LeavePGroupsOut

import keras
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense, Dropout
from keras.models import Sequential, model_from_json
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, TensorBoard, History, ModelCheckpoint, CSVLogger
from keras.regularizers import l2, l1
from keras.constraints import MinMaxNorm


#
################################################################################
#
#param_grid = {'C':[5, 10, 12, 15, 25, 50], 'gamma':[0.01, 0.001, 0.0001, 0.00001, 0.000001], 'kernel':['rbf']}
#
#
#gs = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5, return_train_score=True)
#
#gs.fit(x_train, y_train)
#
#pred = gs.predict(x_test)
#print(confusion_matrix(y_test, pred))
#print(classification_report(y_test, pred))

ES_PATIENCE = 50
RL_PATIENCE = 15
RL_FACTOR = 0.3
RL_MINLR = 10e-6

#class Validation_Callback(Callback):
#    def __init__(self, monitor='val_loss', frequency, verbose=0, value=0.5):
#            super(Callback, self).__init__()
#            self.monitor = monitor
#            self.value = value
#            self.verbose = verbose
#
#    def on_epoch_end(self, epoch, logs={}):
#        self.model


def categorical_cubic_hinge(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)
    return K.maximum(0., (neg - pos + 1.)**3)

def categorical_squared_hinge(y_true, y_pred):
    pos = K.sum(y_true * y_pred, axis=-1)
    neg = K.max((1. - y_true) * y_pred, axis=-1)
    return K.maximum(0., (neg - pos + 1.)**2)

class DataSelection():

    def __init__(self, sub_indexes):
        self.sub_dataset_indexes = sub_indexes

    def select_subs(self, sub_indexes=None, subs='all', mode='include'):

        if sub_indexes is None:
            sub_indexes = self.sub_dataset_indexes

        if subs == 'all':
            return np.ones_like(sub_indexes).astype(bool)

        if mode == 'include':
            ctt = [True if x in subs else False for x in sub_indexes]
        else:
            ctt = [True if not x in subs else False for x in sub_indexes]

        return np.array(ctt).astype(bool)

    def select_movs(self, labels, movs='all', mode='include'):

        if movs == 'all':
            return np.ones_like(labels).astype(bool)

        if mode == 'include':
            ctt = [True if x in movs else False for x in labels]
        else:
            ctt = [True if not x in movs else False for x in labels]

        return np.array(ctt).astype(bool)

    def select(self, labels, subs='all', movs='all', sub_indexes=None, sub_mode='include', mov_mode='include'):
        return np.logical_and(self.select_subs(sub_indexes, subs, mode=sub_mode), self.select_movs(labels, movs, mode=mov_mode))

class LeaveOneSubjectOut():
    def __init__(self, subjects_indexes):
        self.subjects_indexes = subjects_indexes
        self.splitter = LeaveOneGroupOut()

    def split(self, X=None, y=None, groups=None):
        if groups == None:
            groups = self.subjects_indexes
        return self.splitter.split(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        if groups == None:
            groups = self.subjects_indexes
        return self.splitter.get_n_splits(X, y, groups)

class LeavePSubjectsOut():
    def __init__(self, subjects_indexes):
        self.subjects_indexes = subjects_indexes
        self.splitter = LeavePGroupsOut(np.unique(subjects_indexes))

    def split(self, X=None, y=None, groups=None):
        if groups == None:
            groups = self.subjects_indexes
        return self.splitter.split(X, y, groups)

    def get_n_splits(self, X=None, y=None, groups=None):
        if groups == None:
            groups = self.subjects_indexes
        return self.splitter.get_n_splits(X, y, groups)

class Splits():
    def __init__(self, sub_indexes, train_size=0.33, n_splits=10, mode='loso'):
        # bootstrap ou loso
        self.si = sub_indexes
        self.train_size = train_size
        self.n_splits = n_splits
        self.mode = mode
        self.create_splits()

    def create_splits(self, splits=None):

        if self.mode == 'bootstrap':
            unique = np.unique(self.si)

            rs = ShuffleSplit(n_splits=self.n_splits, test_size=1-self.train_size)
            splits = []
            for train, test in rs.split(unique):
                train = unique[train]
                test = unique[test]
                train_ = np.nonzero([x in train for x in self.si])
                test_ = np.nonzero([x in test for x in self.si])
                splits.append((train_, test_))
            self.splits = splits
            self.splitter = None
        elif self.mode == 'groupkfold':
            self.splitter = GroupKFold(n_splits=self.n_splits)
        elif self.mode == 'loso':
            self.splitter = LeaveOneGroupOut()
        elif self.mode == 'lpso':
            self.splitter = LeavePGroupsOut(n_groups=len(np.unique(self.si))//self.n_splits)

    def get_n_splits(self, X=None, y=None, groups=None):
        if self.splitter:
            return self.splitter.get_n_splits(X, y, groups)
        return self.n_splits

    def split(self, X=None, y=None, groups=None):
        if self.splitter:
            for i, j in self.splitter.split(X, y, groups):
                yield i, j
        else:
            for tt in self.splits:
                yield tt

class Model():

    _GLOBAL_NAME = 0

    def __init__(self, estimator, cv=5, param_grid=None, name=None, fit_params=dict(), scaler=None):
        if name is None:
            name = f"Name_{Model._GLOBAL_NAME}"
            Model._GLOBAL_NAME += 1
        self.__params = param_grid
        self.model_ = GridSearchCV(estimator, param_grid=param_grid, cv=cv, refit=False)
        if scaler != None:
            self.model = Pipeline([("scaler", scaler),("model",self.model_)])
        else:
            self.model = Pipeline([("model",self.model_)])
        self.name = name
        self.fitted = False
        self.fit_params = fit_params

    def fit(self, X=None, y=None):
        self.model.fit(X, y, **self.fit_params)
        self.fitted = True

    def isfitted(self):
        return self.fitted

    def predict(self, features):
        return self.model.predict(features)

    def evaluate_metrics(self, features, labels, metrics="default"):
        if not self.isfitted():
            print(f"Tried to take metrics from {self.name}: instance not fitted yet.")
            return None
        model = self.model
        def precision(a, b):
            return precision_score(a, b, average="weighted")
        def recall(a, b):
            return recall_score(a, b, average="weighted")
        def f1(a, b):
            return f1_score(a, b, average="weighted")

        if metrics == "default": metrics = [precision, recall, f1, accuracy_score, balanced_accuracy_score, cohen_kappa_score]

        prediction = model.predict(features)

        res = dict()
        values = (labels, prediction)

        for metric  in metrics:
            try:
                res[metric.__name__] = metric(*values)
            except:
                res[metric.__name__] = metric(labels.argmax(axis=1), prediction.argmax(axis=1))

#        self.report = classification_report(labels, prediction)
#        self.confusion_matrix = confusion_matrix(labels, prediction)
        return res

    def get_model(self):
        return self.model

    def generate_report(self, features, labels):
        pred = self.get_model().predict(features)
        print(classification_report(labels, pred))
        return classification_report(labels, pred)

    def __getattr__(self, attr):
        try:
            return getattr(self, attr)
        except:
            return getattr(self.model, attr)

    def save_model(self, path):
        if self.keras_model:
            model_json = self.model.to_json()
            with open(join(path, self.name + ".json"), "w") as json_file:
                json_file.write(model_json)
            self.model.save_weights(join(path, self.name + ".h5"))
        else:
            joblib.dump(self.model, join(path, "model.joblib"))

    def load_model(self, path):
        if self.keras_model:

            with open(join(path, "model.json"), "r") as json_file:
                loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(join(path, "model.h5"))
        else:
            self.model = joblib.load(join(path, "model.joblib"))

class Model_Set():

    """ This class is used for handling multi-model training, prediction, and
        metrics evaluation."""

    def __init__(self, create_default_set=True):
        self.models = []
        self.data = {}
        self.data_count = 0
        if create_default_set: self.create_set()

    def set_checkpoint_filepath(self, path):
        self.checkpoint_filepath = path

    def add_data(self, data):
        # data = (features, labels)
        if isinstance(data, dict):
            self.data.update(data)
        elif isinstance(data, np.ndarray):
            self.data.update({'DATA_'+str(self.data_count):data})
            self.data_count += 1

    def add(self, model):
        if isinstance(model, Model):
            self.models.append(model)
        elif isinstance(model, list) or isinstance(model, tuple):
            for model_ in model:
                if isinstance(model_, Model):
                    continue
                else:
                    break
            self.models.extend(model)
        else:
            print("Input must be an object of class `Model`, or an \
                  iterable of `Model`s.")

    def get_best_model(self):
        acc = [m.accuracy for m in self.models]
        return self.models[np.argmax(acc)].get_model()

    def fit(self, X=None, y=None, n_jobs=1):
        features, labels = X, y
        # [TODO] parallel fitting
        report = dict
        for i, model in enumerate(self.models):
            print('Fiting model #{0} of {1}'.format(i, len(self.models)))
            model.fit(features, labels, verbose=0)
            print('\tEvaluating the model...')
            result = model.evaluate_metrics(features, labels)
            report.update({model:result})
        return report

    def fit_from_data(self, n_jobs=1):
        # [TODO] parallel fitting
        report = dict()
        for i, model in enumerate(self.models):
            for data_id in self.data:
                data = self.data[data_id]
                print('Fiting model #{0} of {1} with {2}'.format(i, len(self.models), data_id))
                model.fit(*data, verbose=0)
                print('\tEvaluating the model...')
                result = model.evaluate_metrics(*data)
                report.update({(model, data_id):result})
        return report

    def predict(self, features):
        results = dict()
        for model in self.models:
            results[model] = model.predict(features)
        return results

    def evaluate_metrics(self, features, labels):
        results = dict()
        for model in self.models:
            results[model] = model.evaluate_metrics(features, labels)
        return results

    def get_models(self):
        return self.models

    def create_set(self, models=None):

        if not models:
            models = [Model_Collection.NEURAL_NETWORK_2LAYERS_COMPLETE,
                      Model_Collection.NEURAL_NETWORK_3LAYERS_COMPLETE,
                      Model_Collection.KNN,
                      Model_Collection.SVC_POLY,
                      Model_Collection.SVC_RBF,
                      Model_Collection.LOGISTIC_REGRESSION]
        for model in models:
            self.add(Model(**model))

    def generate_report(self):
        pass

def Classifier(name):

    """ This class is a wrapper for the KerasClassifier class.
        It generalizes the model creation so the application can use any
        Dense-Dropout-Dense based neural network topology."""

    def create_nn(units,
                  input_dim,
                  output_dim,
                  activations,
                  regularizers,
                  initializers,
                  dropouts,
                  lr,
                  loss_func,
                  bregularizers=None,
                  n_layers=None,
                  constraints=None):
        # units = N
        # activations = N
        # regularizers = N+1
        # initializers = N+1
        # dropouts = N
        if not n_layers:
            n_layers = len(units)
        if  not isinstance(activations, list):
            activations = [activations for i in range(n_layers)]
        if not isinstance(regularizers, list):
            regularizers = [regularizers for i in range(n_layers+1)]
        if not isinstance(bregularizers, list):
            bregularizers = [bregularizers for i in range(n_layers+1)]
        if not isinstance(constraints, list):
            regularizers = [constraints for i in range(n_layers+1)]
        if not isinstance(initializers, list):
            initializers = [initializers for i in range(n_layers+1)]
        if not isinstance(dropouts, list):
            dropouts = [dropouts for i in range(n_layers)]

        model = Sequential()
        for i in range(n_layers):
            if i == 0:
                model.add(Dense(units=units[i], input_dim=input_dim,
                                activation=activations[i],
                                kernel_initializer=initializers[i],
                                kernel_regularizer=regularizers[i],
                                bias_regularizer=bregularizers[i]))
                model.add(Dropout(dropouts[i]))
            else:
                model.add(Dense(units=units[i], activation=activations[i],
                                kernel_initializer=initializers[i],
                                kernel_regularizer=regularizers[i],
                                bias_regularizer=bregularizers[i]))

                model.add(Dropout(dropouts[i]))

        model.add(Dense(units=output_dim, activation='softmax',
                        kernel_initializer=initializers[-1],
                        kernel_regularizer=regularizers[-1],
                        bias_regularizer=bregularizers[i]))

        optimizer = keras.optimizers.Adam(lr=lr)
#        callbacks = [EarlyStopping(monitor='val_loss', patience=ES_PATIENCE, min_delta=10e-7, verbose=0),
#                     ReduceLROnPlateau(monitor='val_loss', patience=RL_PATIENCE, min_delta=10e-7, verbose=0, factor=RL_FACTOR, min_lr=RL_MINLR),
#                     TensorBoard(log_dir='./tensorboard/'+name+'.tbl', histogram_freq=1),
#                     CSVLogger('./csv_logs/'+name+'.csv'),
#                     ModelCheckpoint('./checkpoints/'+name+'.hdf5', monitor='accuracy', save_best_only=True)]

        model.compile(optimizer=optimizer,
                      loss=loss_func,
                      metrics=['accuracy'])
        return model

    classifier = KerasClassifier(build_fn=create_nn)
    return classifier

class ModelCollection():

    BEST_MODEL_1 = {'estimator':Classifier('BEST_MODEL_1'),
                                       'param_grid':
                                       {'units':[[200,150, 100], [100,100, 100], [200, 200, 200]],
                                        'input_dim':[1110],
                                        'output_dim':[49],
                                        'activations':['relu', 'selu'],
                                        'regularizers':[None, [l2(l=1e-5), l2(l=1e-5), l2(l=1e-5)]],
                                        'bregularizers':[None, [l2(l=1e-4), l2(l=1e-4), l2(l=1e-4)]],
                                        'initializers':['glorot_normal'],
                                        'constraints':[MinMaxNorm(0,0.5)],
                                        'dropouts':[[0.3, 0.2],[0.4, 0.3]],
                                        'lr':[1e-5, 1e-6],
                                        'loss_func':['categorical_crossentropy'],
                                        'batch_size':[256, 1024],
                                        'epochs':[5000]}}

    NEURAL_NETWORK_3LAYERS_COMPLETE = {'estimator':Classifier('3_LAYER_NN_A'),
                                       'param_grid':
                                       {'units':[[156, 156, 156], [156, 102, 49], [121, 96, 49], [148, 128, 49]],
                                        'input_dim':[156],
                                        'output_dim':[49],
                                        'activations':['relu', 'tanh'],
                                        'regularizers':[None, [l2(l=0.0001), l2(l=0.0001), l2(l=0.0001)]],
                                        'initializers':['glorot_normal', 'glorot_uniform'],
                                        'dropouts':[0.2, 0.3],
                                        'lr':[0.001, 0.0001],
                                        'loss_func':['sparse_categorical_crossentropy', categorical_cubic_hinge, categorical_squared_hinge],
                                        'batch_size':[32, 64, 128],
                                        'epochs':[1000, 1500, 2000]}}

    NEURAL_NETWORK_2LAYERS_COMPLETE = {'estimator':Classifier('2_LAYER_NN_A'),
                                       'param_grid':
                                       {'units':[[156, 156], [156, 49], [106, 49]],
                                        'input_dim':[156],
                                        'output_dim':[49],
                                        'activations':['relu', 'tanh'],
                                        'regularizers':[None, [l2(l=0.0001), l2(l=0.0001)]],
                                        'initializers':['glorot_normal', 'glorot_uniform'],
                                        'dropouts':[0.2, 0.3],
                                        'lr':[0.001, 0.0001],
                                        'loss_func':['sparse_categorical_crossentropy', categorical_cubic_hinge, categorical_squared_hinge],
                                        'batch_size':[32, 64, 128],
                                        'epochs':[1000, 1500, 2000]}}

    NEURAL_NETWORK_3LAYERS_COMPACT = {'estimator':Classifier('3_LAYER_NN_B'),
                                      'param_grid':
                                      {'units':[[156, 156, 156], [156, 102, 49], [148, 128, 49]],
                                       'input_dim':[156],
                                       'output_dim':[49],
                                       'activations':['relu'],
                                       'regularizers':[None],
                                       'initializers':['glorot_normal', 'glorot_uniform'],
                                       'dropouts':[0.2, 0.3],
                                       'lr':[0.0001],
                                       'loss_func':['sparse_categorical_crossentropy'],
                                       'batch_size':[64, 128],
                                       'epochs':[1000]}}

    NEURAL_NETWORK_2LAYERS_COMPACT = {'estimator':Classifier('2_LAYER_NN_B'),
                                      'param_grid':
                                      {'units':[[156, 156], [156, 49], [106, 49]],
                                       'input_dim':[156],
                                       'output_dim':[49],
                                       'activations':['relu'],
                                       'regularizers':[None],
                                       'initializers':['glorot_normal', 'glorot_uniform'],
                                       'dropouts':[0.2, 0.3],
                                       'lr':[0.0001],
                                       'loss_func':['sparse_categorical_crossentropy'],
                                       'batch_size':[64, 128],
                                       'epochs':[1000]}}
    NEURAL_NETWORK_MINIMALIST_1 = {'estimator':Classifier('1_MINIMALIST'),
                                   'param_grid':
                                   {'units':[[156, 156], [156, 49], [106, 49]],
                                    'input_dim':[156],
                                    'output_dim':[49],
                                    'activations':['relu'],
                                    'regularizers':[None],
                                    'initializers':['glorot_normal', 'glorot_uniform'],
                                    'dropouts':[0.2, 0.3],
                                    'lr':[0.0001],
                                    'loss_func':['sparse_categorical_crossentropy'],
                                    'batch_size':[64, 128],
                                    'epochs':[1000]}}

    SVC_RBF = {'estimator':SVC(),
               'param_grid':
               {'C':[2**i for i in range(1, 8)],
                'gamma':[2**-i for i in range(4, 16)],
                'kernel':['rbf']}}

    SVC_LINEAR = {'estimator':SVC(),
                  'param_grid':
                  {'C':[2**i for i in range(1, 8)],
                   'gamma':[2**-i for i in range(4, 16)],
                   'kernel':['linear']}}

    SVC_SIG = {'estimator':SVC(),
               'param_grid':
               {'C':[2**i for i in range(0, 7)],
                'coef0':[np.linspace(-5, 5, 11)],
                'gamma':[2**-i for i in range(5, 15)],
                'kernel':['sigmoid'],
                'class_weight':['balanced', None]}}

    SVC_POLY = {'estimator':SVC(),
                'param_grid':
                {'C':[2**i for i in range(0, 7)],
                 'coef0':[np.linspace(-5, 5, 11)],
                 'gamma':[2**-i for i in range(5, 15)],
                 'degree':[1, 2, 3, 4],
                 'kernel':['poly'],
                 'class_weight':['balanced', None]}}

    KNN = {'estimator':KNeighborsClassifier(),
           'param_grid':
           {'n_neighbors':[1, 3, 7, 15, 30, 60, 120, 250, 500],
            'weights':['uniform', 'distance'],
            'p':[1, 2],
            'metric':['minkowski']}}

    LOGISTIC_REGRESSION = {'estimator':LogisticRegression(),
                           'param_grid':
                           {'C':[i for i in range(1, 50, 2)],
                            'class_weight ':[None, 'balanced'],
                            'solver':['newton-cg', 'saga', 'lbfgs', 'sag'],
                            'multi_class':['ovr', 'multinomial'],
                            'l1_ratio':[0, 0.2, 0.5, 0.8, 1]}}


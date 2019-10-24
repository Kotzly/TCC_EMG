# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 17:57:26 2019

@author: Paulo
"""

import numpy as np
import os

from keras.utils import Progbar
from keras.utils import to_categorical

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.regularizers import l2, l1_l2
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TerminateOnNaN, EarlyStopping
from keras.constraints import MinMaxNorm
import keras.backend as K
from sklearn.utils import class_weight

from os.path import join
import warnings

warnings.simplefilter("ignore")


from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler
from cleaners import DataClipper, FeatureSelector
from ModelCollection import DataSelection
import joblib

from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report, cohen_kappa_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

from numpy.random import seed
from tensorflow import set_random_seed

def dist(a, b):
  res = np.sqrt(((a-b)**2).sum(axis=1))
  return res

def dist_balanced(distance):
    return 1/(distance + 1e-3)

def abs_KL_div(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), None)
    y_pred = K.clip(y_pred, K.epsilon(), None)
    return K.sum( K.abs( (y_true- y_pred) * (K.log(y_true / y_pred))), axis=-1)

TRAIN_IDS = np.array([39, 34, 19, 30, 21, 40,  6, 18, 12, 31,  4, 27, 20, 15,  7, 22,  8, 35, 10, 37, 14, 23, 25,  2, 17, 38, 26, 16])
VALID_IDS = np.array([ 9, 29,  5,  1, 13, 36])
TEST_IDS = np.array([32, 33,  3, 11, 24, 28])

vote_funcs = {"dist": dist_balanced,
              "uniform": lambda x: 1}

class KNN():

    def __init__(self, n_neighbors=370, mode="dist", verbose=False):
        self.k = n_neighbors
        self.vote_func = vote_funcs[mode]
        self.fitted = False
        self.verbose = verbose

    def fit(self, X, y):
        self.X = X
        self.y = y
        if np.ndim(y)<2:
            self.encoder = LabelEncoder()
            self.y = to_categorical(self.encoder.fit_transform(y))
        self.fitted = True
        return self

    def predict(self, X):
        bar = Progbar(len(X))
        pred_proba = []
        for i in range(len(X)):
          if self.verbose: bar.add(1)
          ordered_dists = dist(self.X, X[i])
          ordered_dists = sorted(zip(ordered_dists, range(len(ordered_dists))), key=lambda x:x[0])
          k_indexes = [x[1] for x in ordered_dists[:self.k]]
          votes = np.zeros(self.y.shape[1])
          for p, x in enumerate(k_indexes):
              votes[self.y[x].argmax()] += self.vote_func(ordered_dists[p][0])
          pred_proba.append(votes)
        return to_categorical(np.array(pred_proba).argmax(axis=1))

class LogisticRegression():

    def __init__(self, inputs, outputs, save_path=".", reg_l1=0, reg_l2=0, lr=1e-3):
        self.inputs = inputs
        self.outputs = outputs
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.lr = lr
        self.save_path = save_path
        self.seed = 666
        self.create_model()
        self.create_callbacks()
        self.fitted = False

    def create_callbacks(self, patience_rlop=10, patience_es=20):

        cpkt_path = os.path.join(self.save_path, "model.cpkt")
        log_path = os.path.join(self.save_path, "log.csv")

        self.callbacks = [ReduceLROnPlateau(monitor="val_loss", patience=patience_rlop, factor=0.1, min_delta=0.0001, verbose=0),
                          EarlyStopping(monitor="val_loss", patience=patience_es, min_delta=0.0001, restore_best_weights=True, verbose=0),
                          TerminateOnNaN()]
        if not self.save_path is None:
          self.callbacks += [ModelCheckpoint(cpkt_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False),
                             CSVLogger(log_path, separator=',', append=False)]

    def create_model(self):
        set_random_seed(666)
        seed(self.seed)
        model = Sequential()
        model.add(Dense(input_shape=(self.inputs,),
                        units=self.outputs,
                        activation="softmax",
                        kernel_initializer="glorot_normal",
                        kernel_regularizer=l1_l2(l1=self.reg_l1, l2=self.reg_l2),
                        bias_regularizer=l1_l2(l1=self.reg_l1, l2=self.reg_l2)))
        loss = "categorical_crossentropy" if self.outputs>1 else "binary_crossentropy"
        model.compile(loss=loss, metrics=["acc"], optimizer=Adam(lr=self.lr))
        self.model = model

    def fit(self, *args, **kwargs):
        self.fitted = False
        set_random_seed(self.seed)
        seed(self.seed)
        kwargs.update(callbacks=self.callbacks)
        self.model.fit(*args, **kwargs)
        self.fitted = True
        if not self.save_path is None:
          self.model.save(os.path.join(self.save_path, "model.K"))

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        self.model.save(path)


class Neural_Network():

    def __init__(self, input_dim, output_dim, lr=1e-5, activation="selu", n_layers=3,
                 max_norm=.5, dropouts=[.3, .2, 0], units=[200, 150,100],
                 rk =[1e-4]*3, rb=[1e-3]*3, save_path="."):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_kwargs = dict(lr=lr, activation="selu", n_layers=n_layers,
                                 max_norm=max_norm, dropouts=dropouts,
                                 units=units, rk=rk,
                                 rb=rb)
        self.callbacks = []
        self.save_path = save_path
        self.seed = 666
        self.create_model()
        self.create_callbacks()
        self.fitted = False

    def create_callbacks(self, patience_rlop=10, patience_es=30):

        cpkt_path = os.path.join(self.save_path, "model.cpkt")
        log_path = os.path.join(self.save_path, "log.csv")

        self.callbacks = [ReduceLROnPlateau(monitor="val_loss", patience=patience_rlop, factor=0.1, min_delta=0.0001, verbose=0),
                          EarlyStopping(monitor="val_loss", patience=patience_es, min_delta=0.0001, restore_best_weights=True, verbose=0),
                          TerminateOnNaN()]
        if not self.save_path is None:
          self.callbacks += [ModelCheckpoint(cpkt_path, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False),
                             CSVLogger(log_path, separator=',', append=False)]

    def create_model(self, clear_session=False):
        if clear_session: K.clear_session()
        set_random_seed(666)
        seed(self.seed)
        model = Sequential()
        activation = self.model_kwargs["activation"]
        max_norm = self.model_kwargs["max_norm"]
        n_layers = self.model_kwargs["n_layers"]
        d = self.model_kwargs["dropouts"]
        units = self.model_kwargs["units"]
        rk = self.model_kwargs["rk"]
        rb = self.model_kwargs["rb"]
        lr = self.model_kwargs["lr"]

        model.add(Dense(input_shape=(self.input_dim,),
                          units=units[0],
                          kernel_initializer="glorot_normal",
                          kernel_regularizer=l2(rk[0]),
                          kernel_constraint=MinMaxNorm(0, max_norm),
                          bias_regularizer=l2(rb[0]),
                          activation=activation))
        model.add(Dropout(d[0]))

        for i in range(n_layers-1):
          model.add(Dense(units=units[i+1],
                          kernel_initializer="glorot_normal",
                          kernel_regularizer=l2(rk[i+1]),
                          bias_regularizer=l2(rb[i+1]),
                          kernel_constraint=MinMaxNorm(0, max_norm),
                          activation=activation))
          model.add(Dropout(d[i+1]))

        model.add(Dense(units=self.output_dim,
                      kernel_initializer="glorot_normal",
                      kernel_regularizer=l2(rk[-1]),
                      bias_regularizer=l2(rb[-1]),
                      kernel_constraint=MinMaxNorm(0, max_norm),
                      activation="softmax"))
        optimizer = Adam(lr=lr, clipnorm=.2)
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"])
        self.model = model

    def fit(self, *args, **kwargs):
        self.fitted = False
        set_random_seed(self.seed)
        seed(self.seed)
        kwargs.update(callbacks=self.callbacks)
        self.model.fit(*args, **kwargs)
        self.fitted = True
        if not self.save_path is None:
          self.model.save(os.path.join(self.save_path, "model.K"))

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def save(self, path):
        self.model.save(path)

def make_metrics(label, prediction, ids=None, folder=".", mode="w", identifier=""):

    bas, f1, k, p, r, auc = [], [], [], [], [], []
    for sub in np.unique(ids):
        ctt = ids==sub
        data = (label[ctt].argmax(1), prediction[ctt].argmax(1))
        bas.append(balanced_accuracy_score(*data))
        f1.append(f1_score(*data, average="weighted"))
        k.append(cohen_kappa_score(*data))
        try:
          auc.append(roc_auc_score(label[ctt], prediction[ctt], average="weighted"))
        except:
          print(f"AUC error at subject {sub}")
          auc.append(0)
        p.append(precision_score(*data, average="weighted"))
        r.append(recall_score(*data, average="weighted"))

    mean = np.mean
    std = np.std

    with open(os.path.join(folder, identifier+"results_detailed.txt"), mode) as file:
      print("Balanced Accuracy, F1 Score, Cohen Kappa, AUC, Precision, Recall", file=file)
      for i in range(len(bas)):
        print(f"{bas[i]}, {f1[i]}, {k[i]}, {auc[i]}, {p[i]}, {r[i]}",file=file)

    with open(os.path.join(folder, identifier+"results.txt"), mode) as file:
        print("N. examples:", prediction.shape[0], file=file)
        print("Balanced Accuracy:", min(bas), max(bas), mean(bas), std(bas), file=file)
        print("F1 Score:", min(f1), max(f1), mean(f1), std(f1), file=file)
        print("Cohen Kappa:", min(k), max(k), mean(k), std(k), file=file)
        print("AUC:", min(auc), max(auc), mean(auc), std(auc), file=file)
        print("Precision:", min(p), max(p), mean(p), std(p), file=file)
        print("Recall:", min(r), max(r), mean(r), std(r), file=file)
    with open(os.path.join(folder, identifier+"report.txt"), mode) as file:
        print(classification_report(label.argmax(1), prediction.argmax(1)), file=file)

def create_folds(features, labels, ids, mixed_subjects=False,
                 save_path=".", feature_cols=None, movs=range(1, 50),
                 train_ids=TRAIN_IDS, valid_ids=VALID_IDS,
                 test_ids=TEST_IDS):

    x_train = []
    x_valid = []
    x_test = []
    y_train = []
    y_valid = []
    y_test = []

    ids_train, ids_valid, ids_test = [], [], []

    selector = DataSelection(ids)
    ctt = selector.select(labels, subs="all", movs=movs)

    print(sum(ctt), "out of", features.shape[0], "selected")

    print("Selecting movements")

    features_ = features[ctt]
    features_ = features_[:, feature_cols]
    ids_ = ids[ctt]
    labels_ = labels[ctt]

    print("Folding subjects")
    if not mixed_subjects:

      ctt_train = [x in train_ids for x in ids_]
      ctt_valid= [x in valid_ids for x in ids_]
      ctt_test = [x in test_ids for x in ids_]

      x_train = features_[ctt_train]
      x_valid = features_[ctt_valid]
      x_test = features_[ctt_test]

      y_train = labels_[ctt_train]
      y_valid = labels_[ctt_valid]
      y_test = labels_[ctt_test]

      ids_train = ids_[ctt_train]
      ids_valid = ids_[ctt_valid]
      ids_test = ids_[ctt_test]

    else:
      x_train, x_test, y_train, y_test, ids_train, ids_test = train_test_split( features_, labels_, ids_, test_size=0.3, random_state=42)
      x_test, x_valid, y_test, y_valid, ids_test, ids_valid = train_test_split( x_test, y_test, ids_test, test_size=0.5, random_state=42)

    print("TRAIN #", x_train.shape[0])
    print("VALID #", x_valid.shape[0])
    print("TEST #", x_test.shape[0])

    encoder = LabelEncoder()
    onehot =  OneHotEncoder()
    y_train = np.array(onehot.fit_transform(encoder.fit_transform(y_train).reshape(-1, 1)).todense())
    y_valid = np.array(onehot.transform(encoder.transform(y_valid).reshape(-1, 1)).todense())
    y_test = np.array(onehot.transform(encoder.transform(y_test).reshape(-1, 1)).todense())

    print("Scaling")

    scaler1 = RobustScaler()
    x_train = scaler1.fit_transform(x_train)
    x_valid = scaler1.transform(x_valid)
    x_test = scaler1.transform(x_test)

    scaler2 = PowerTransformer()
    x_train = scaler2.fit_transform(x_train)
    x_valid = scaler2.transform(x_valid)
    x_test = scaler2.transform(x_test)

#     if not isinstance(y_train, np.ndarray): y_train = y_train.todense()
#     if not isinstance(y_valid, np.ndarray): y_valid = y_train.todense()
#     if not isinstance(y_test, np.ndarray): y_test = y_test.todense()

    return (x_train, y_train, ids_train), \
            (x_valid, y_valid, ids_valid), \
            (x_test, y_test, ids_test), \
            (onehot, encoder, scaler1, scaler2)

def save_transformers(transformers, names=[], path="."):
    for t, name in zip(transformers, names):
        name = name + ".joblib" if not "." in name else name
        file = os.path.join(path, name)
        joblib.dump(t, file)

def get_movs(x, y, n_movs=49, n_features=10):
  cor = np.corrcoef(np.nan_to_num(np.concatenate([x, y], axis=1)).T)
  corvalid = np.nan_to_num(cor[-49:, :-49])

  cttfeat = zip(abs(corvalid).mean(axis=0), range(corvalid.shape[1]))
  cttfeat = sorted(cttfeat, reverse=True, key=lambda x:x[0])
  cttfeat = np.array([x[1] for x in cttfeat[:n_features]])

  cttmovs = zip(abs(corvalid).mean(axis=1), range(49))
  cttmovs = sorted(cttmovs, key=lambda x:x[0], reverse=True)
  cttmovs = np.array([x[1] for x in cttmovs[:n_movs]])

  return cttfeat, cttmovs

def save_feats_classes(feats, classes, path, feat_names=None):
    if feat_names != None:
        with open(os.path.join(path, "features.txt"), "w") as file:
            for feat in np.array(feat_names)[feats]:
                file.write(feat+"\n")
    with open(os.path.join(path, "features_number.txt"), "w") as file:
        for feat in feats:
            file.write(str(feat)+"\n")
    with open(os.path.join(path, "classes.txt"), "w") as file:
        for c in classes:
            file.write(str(c)+"\n")

def get_data(path, subjects=range(1, 41), preprocessing="raw", window_content="pure", window_stride="2048_512"):

    features, labels, ids = [], [], []

    print("Subjects: ")
    for subject in subjects:
      print(subject, end=" ")
      features.append(np.load(join(path, "Features_Files", preprocessing, window_content, window_stride, f"features_{subject}.npy")))
      labels.append(np.load(join(path, "Windows_Files", preprocessing, window_content, window_stride, f"labels_{subject}.npy")))
      ids.append(np.load(join(path, "Windows_Files", preprocessing, window_content, window_stride, f"ids_{subject}.npy")))

    features = np.concatenate(features)
    labels = np.concatenate(labels)
    ids = np.concatenate(ids)

    return features, labels, ids

def get_unique_names(path):

    with open(join(path, "Features_Files", "raw", "pure", "2048_512", f"features_names.txt")) as file:
        names = file.read().splitlines()

    cols = []
    marked = []
    names_ = []
    for i in range(len(names)):
      if not names[i] in marked and not names[i].startswith("KL"):
        cols.append(True)
        marked.append(names[i])
        names_.append(names[i])
      else:
        cols.append(False)
    names = names_
    return names, cols

def select_feats_and_movs(X, Y, ids,  feats, movs):
    ctt = [x in movs for x in Y.argmax(1)]
    X_ = X[ctt][:, feats]
    Y_ = Y[ctt]
    ids_ = ids[ctt]
    onehot= OneHotEncoder()
    encoder = LabelEncoder()
    Y_ = onehot.fit_transform(encoder.fit_transform(Y_.argmax(axis=1)).reshape(-1, 1))
    return X_, np.array(Y_.todense()), ids_, onehot, encoder

def select_from_folds(*folds, feats=None, movs=None):
    res = []
    for fold in folds:
        res.append(select_feats_and_movs(*fold, feats, movs))
    return tuple(res)

def make_dirs(*paths):
  for path in paths:
    try:
      os.mkdir(path)
    except:
      pass

def need_to(path):
  try:
    files = os.listdir(path)
  except:
    return True
  return not "test_results.txt" in files
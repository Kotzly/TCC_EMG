# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 11:42:09 2019

@author: Paulo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler


class data_cleaner():

    def __init__(self,mode='clip'):
        self.mode=mode
        self.quantiles = dict()

    def fit(self,df,features=None,quantile=0.99):
        dataframe=df.copy()

        if not features: features = dataframe.columns

        quantile+=(1-quantile)/2

        self.quantiles.update({feat:(np.quantile(dataframe[feat],1-quantile),np.quantile(dataframe[feat],quantile)) for feat in features})
        self.features=features
    def transform(self,df,features='all',mode='auto'):
        dataframe=df.copy()

        if not features: features = dataframe.columns
        if type(features)==str and features=='all': features=self.features
        if mode=='auto': mode=self.mode
        for feat in features:
            if mode=='clip':
                dataframe[feat]=dataframe[feat].clip(self.quantiles[feat][0],self.quantiles[feat][1])
            if mode=='remove':
                dataframe[feat]=dataframe[feat][(dataframe[feat]>=self.quantiles[feat][0]) & (dataframe[feat]<=self.quantiles[feat][1])]
            if mode=='mean':
                ctt = (dataframe[feat]<=self.quantiles[feat][0]) | (dataframe[feat]>=self.quantiles[feat][1])
                dataframe[feat][ctt]=dataframe[feat].mean()
            if mode=='zero':
                ctt = (dataframe[feat]<=self.quantiles[feat][0]) | (dataframe[feat]>=self.quantiles[feat][1])
                dataframe[feat][ctt]=0.0
        return dataframe

    def fit_transform(self,df,features=None,mode='auto',quantile=0.99):
        self.fit(df,features,quantile)
        return self.transform(df,features,mode)

class DataClipper():
    def __init__(self, quantiles=[0.05, 0.95]):
        self.quantiles = quantiles
        self.fitted = False

    def fit(self,data):
        self.columns_quantiles = np.quantile(data,self.quantiles, axis=0)
        self.fitted = True
    def transform(self, data):
        if not self.fitted:
            return None
        if isinstance(data, np.ndarray):
            new_data = np.clip(data, *self.columns_quantiles)
        elif isinstance(data, pd.DataFrame):
            new_data = data.clip(*self.columns_quantiles, axis=1)
        return new_data

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

class FeatureSelector():
    def __init__(self, features=[], sensors=[]):
        self.features = features
        self.sensors = sensors
        self.accepted = []

    def fit(self, features_names):
        accepted = []
        for i in range(len(features_names)):
            feature = features_names[i].split("_")

            if len(feature)==2:
                if feature[0] in self.features:
                    if int(feature[1][1:]) in self.sensors:
                        accepted.append(True)

            elif len(feature)==3:
                if feature[0] in self.features:
                    if feature[2][0]=="C":
                        if int(feature[1][1:]) in self.sensors:
                            accepted.append(True)
                    else:
                        if (int(feature[1]) in self.sensors) and (int(feature[2]) in self.sensors):
                            accepted.append(True)
            if len(accepted)==i:
                accepted.append(False)
        self.accepted = accepted
        self.features_names = np.array(features_names)[accepted]

    def transform(self, data):
        if len(self.accepted) != data.shape[1]:
            print("Wrong number of features")
            return None
        if isinstance(data, pd.DataFrame):
            return data.iloc[:, self.accepted]
        elif isinstance(data, np.ndarray):
            return data[:, self.accepted]
        else:
            return None


    def fit_transform(self, data, features_names=[]):
        if isinstance(data, pd.DataFrame):
            features_names = data.columns
        self.fit(features_names)
        return self.transform(data)

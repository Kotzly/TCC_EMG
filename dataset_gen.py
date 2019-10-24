# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:56:38 2019

@author: Paulo
"""

from dataset import Dataset
from preprocessing import dwt_filter
from sklearn.decomposition import FastICA, PCA, NMF, FactorAnalysis
from feature_extraction import SignalParser
import h5py
from os.path import join
import numpy as np
import os
import time

signal_parser = SignalParser(verbose=1)
preps = [FastICA(max_iter=500),NMF(),FactorAnalysis(),PCA(n_components=11)]
prep_names = ['ICA','NMF','FA','PCA']
if not os.path.exists('preprocessed_files'):
    os.mkdir('preprocessed_files')
for name in prep_names:
    path = join('preprocessed_files',name)
    if not os.path.exists(path):
        os.mkdir(path)

root= '.'
if __name__ is "__main__":
    data = Dataset()
    old_time = time.clock()
    for i in range(1,3):
        data.get_data(subjects=[i])
        emg = dwt_filter(data[i].emg,axis=1,threshold=True,
                         level=4,filter_level=3,thr_mode='hard')
        for prep_name,preprocessing in zip(prep_names,preps):
            print(f'Generating Dataset with preprocessing {prep_name}')
            print(f'\tFitting {prep_name}...')
            if prep_name is 'NMF':
                data[i].emg = preprocessing.fit_transform(emg**2)
            else:
                data[i].emg = preprocessing.fit_transform(emg)
            print(f'\tExtracting windows...')
            windows,labels,sub_indexes = data.get_dataset_file(window_size=512,overlap=256,subjects=[i])
            channel_feat,corr_feat = signal_parser.extract(windows)
            names = channel_feat[0] + corr_feat[0]
            chfs,cofs = channel_feat[1].shape[0],corr_feat[1].shape[0]
            print(f'\tConcatenating windows...')
            features = np.concatenate([channel_feat[1].reshape(chfs,-1),
                                       corr_feat[1].reshape(cofs,-1)],axis=0)
            print(f'\tWriting to files...')
            with h5py.File(join(root,'preprocessed',prep_name,'dataset.hdf5'),'a') as hdf5_file:
                try:
                    grp = hdf5_file.create_group(f'subject_{i}')
                except:
                    grp = hdf5_file[f'subject_{i}']
                grp.create_dataset(prep_name,data=features)
                grp.create_dataset('ground_truth',data=labels)
                grp.create_dataset('sub_indexes',data=sub_indexes)
                grp.create_dataset('feature_names',data=names)

        new_time = time.clock()
        print("Spent {} at subject {}".format(new_time-old_time,i))
        old_time = new_time

class DumbEstimator():
    def __init__(self):
        pass

    def predict(self, X=None):
        return X

    def fit(self, X=None, y=None):
        return self

    def decision_function(self, X=None):
        return X


def run_preprocessing_pipeline(processing_pipeline, feature_pipeline, load_fold, save_folder, dataset, subjects=None):
    features = list()
    if subjects == None:
        subjects = dataset.loaded_subjects_ids()
    for sub in subjects:
        if dataset.is_subject_loaded(sub):
            emg = dataset[sub].emg
        else:
            dataset.get_data(subjects=[sub])
            emg = dataset[sub]
            dataset.remove(sub)
        processed = processing_pipeline.fit_transform(emg)
        dataset[sub].emg = processed
        feature = feature_pipeline.fit_transform(processed)
        features.append(feature)

    features = np.concatenate(features, axis=1)
    return features
# pipeline
#    Aquisition - Dataset
#    Filtering - Bandpass
#    Denoising - DWTFilter
#    ICA - OrderedFastICA
#    Normalization - ?
#    Extraction - SignalParser
#    Scaling - Sklearn Preprocessing

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:56:15 2018

@author: PauloAugusto
"""
import os
import zipfile
import numpy as np
import keras
from preprocessing import envelope
from scipy.io import loadmat
from scipy.stats import mode

number_of_subjects = 1
my_folder = "C:/Users/Paulo/Desktop/TCC/EMG_Database/"

class Subject():
    """ Class used internally for subject data storage.
    """
    def __init__(self, sub_id=None, archive=None, subject=None, database=None,
                 exercise=None, repetition=None, rerepetition=None,
                 stimulus=None, restimulus=None):
        self.archive = archive
        self.subject = subject

    def __getitem__(self, item):
        return getattr(self, item)

    def append(self, attr, arr):
        if not hasattr(self, attr):
            setattr(self, attr, arr)
        else:
            setattr(self, attr, np.vstack([self[attr], arr]))

class Dataset():

    def __init__(self, folder=my_folder):
        self.subjects = []
        self.archives = []
        self.folder = folder
        self.labels = None
        self.ids = None

    def remove(self, sub):
        if isinstance(sub, Subject):
            self.subjects.remove(sub)
        else:
            self.subjects.remove(self[sub])

    def __getitem__(self, i):
        for sub in self.subjects:
            if sub.subject == i:
                return sub
        return None

    def is_subject_loaded(self, i):
        if self[i] != None:
            return True
        return False

    def loaded_subjects_ids(self):
        ids = list()
        for sub in self.subjects:
            ids.append(sub.subject)
        return ids

    def get_data(self, verbose=5, folder=None, dataset_number=2,
                 get_acc=False, get_glove=False, get_inclin=False,
                 subjects=None, get_envelope=False):
        """ Function for reading .zip file from the NinaPro Database.
        """

        if not subjects:
            subjects = ['DB{}_s{}'.format(dataset_number, i) for i in range(1,41)]
        else:
            subjects = ['DB{}_s{}'.format(dataset_number, i) for i in np.array(subjects)]

        if not folder: folder = self.folder

        for sub in subjects:
            print('Acessing ', folder + sub + '.zip')

            with zipfile.ZipFile(folder + sub + '.zip', "r") as zip_ref:
                zip_ref.extractall(folder)

            if verbose > 0:
                print('Loading ' + sub)

            archives = ['S{0}_E{1}_A1.mat'.format(sub[sub.find('s')+1:], ex) for ex in range(1, 4)]
            for archive in archives:
                if verbose > 1:
                    print('\t Loading ' + archive)
                data = loadmat(folder + sub + '/' + archive)

                attributes = ['emg', 'stimulus', 'repetition',
                              'restimulus', 'rerepetition']

                if get_glove: attributes += ['glove']
                if get_inclin: attributes += ['inclin']
                if get_acc: attributes += ['acc']

                if get_envelope:
                    data['emg'] = envelope(data=data['emg'])

                subject = self[data['subject'].squeeze()]
                if not subject:
                    subject = Subject(archive=sub,
                                      subject=data['subject'].squeeze(),
                                      database=dataset_number)
                    self.subjects.append(subject)

                for attr in attributes:
                    subject.append(attr, data[attr])

                self.archives.append(archive)
            for arq in os.listdir(folder+sub):
                os.remove(folder+sub+'/'+arq)
            os.rmdir(folder+sub)

    def clean_history(self):
        self.archives = list()
        self.subjects = list()

    def get_dataset(self, test_size=None, window_size=256,
                    subjects=None, processing_func=None,
                    stride=128, pure_samples=True, save_memory=True):

        """
            Args:
                test_size: not implemented
                window_size: size of the windows
                overlap: should be "strides", how much samples to skip after each window
                subjects: which subjects to parse
                processing_func: a function that receives an array of dimensions
                    (n_samples,window_size,channels) and returns (n_samples,features,channels)
                    for parsing all windows.
                save_memory: alters the indexing mode when selecting the windows. Currently it is faster and consuming less memory.
            Returns:
                An (n_samples,window_size,channels) or (n_samples,features,channels) array,
                depending if "feature_processing_func" was passed or not.
        """
        if processing_func is None:
            processing_func = lambda x: x

        if subjects is None:
            subjects = self.subjects
        else:
            subjects = [self[i] for i in subjects if self[i] is not None]

        data = []
        restimuluses = []
        subs = []

        if save_memory:
            for subject in subjects:
                i = 0
                while i < min(len(subject.emg), len(subject.restimulus))-window_size:
                    rest = subject.restimulus[i:i+window_size]
                    if pure_samples:
                        if np.all(rest == rest[0]):
                            restimuluses.append(rest[0])
                            data.append(processing_func(subject.emg[i:i+window_size]))
                            subs.append(subject.subject)
                    else:
                        data.append(processing_func(subject.emg[i:i+window_size]))
                        rest = mode(subject.restimulus[i:i+window_size], axis=0)[0].squeeze()
                        restimuluses.append(rest)
                        subs.append(subject.subject)
                    i += stride
            data = np.stack(data)
            restimuluses = np.array(restimuluses)
            subs = np.array(subs)
        else:
            for subject in subjects:
                emg = subject.emg
                restimulus = subject.restimulus
                indexes = np.mgrid[0:emg.shape[0]-window_size:stride,
                                   0:window_size:1]
                indexes = indexes[0]+indexes[1]
                if pure_samples:
                    pre_rest = restimulus[indexes]
                    pure_indexes = np.all(pre_rest.squeeze() == pre_rest[:, 0], axis=1)
                    emg = emg[indexes]
                    emg = emg[pure_indexes]
                    restimulus = restimulus[indexes][pure_indexes, 0]
                else:
                    emg = emg[indexes]
                    restimulus = mode(restimulus[indexes], axis=1)[0].squeeze()

                data.append(processing_func(emg))
                restimuluses.append(restimulus)
                subs += [subject.subject*1]*restimuluses[-1].shape[0]

                data = np.concatenate(data, axis=0)
                restimuluses = np.concatenate(restimuluses, axis=0).squeeze()
                subs = np.array(subs)

        self.labels = restimuluses
        self.ids = subs

        return data, restimuluses, subs

def load_data_and_save_windows(load_folder, save_folder, verbose=0, **get_dataset_kws):
    """ Load the NinaPro .zip files, create the windows and save
    them to the `save_folder` folder.
    Parameters:
        load_folder: Folder where the .zip files are located.
        save_folder: Folder where the .npy files will be saved.
        verbose: Verbose.
        get_dataset_kws: Keyword arguments for the get_dataset function.
    """
    for subject in get_dataset_kws.pop("subjects"):

        data = Dataset(load_folder)
        data.get_data(subjects=[subject], verbose=verbose)
        windows, labels, ids = data.get_dataset(**get_dataset_kws)

        np.save(os.path.join(save_folder, f"data_{subject}"), windows)
        np.save(os.path.join(save_folder, f"labels_{subject}"), labels)
        np.save(os.path.join(save_folder, f"ids_{subject}"), ids)

def load_saved_windows(load_folder, subjects=None):
    """ Load the subjects windows previously saved.
        The windows files must be named `data_{subject}.npy`.
        The labels files must be named `labels_{subject}.npy`.
        The ids files must be named `ids_{subject}.npy`.
        Parameters:
            load_folder: Where the .npy files are stored.
            subjects: subject to load.
        Returns:
            data: An array with shape (n_samples, window_size, n_channels)
            labels: An array with shape (n_samples,)
            ids: An array with shape (n_samples,)
    """

    if subjects == None:
        return (), (), ()

    data, labels, ids = [], [], []

    for subject in subjects:
        data.append(np.load(os.path.join(load_folder, f"data_{subject}.npy")))
        labels.append(np.load(os.path.join(load_folder, f"labels_{subject}.npy")))
        ids.append(np.load(os.path.join(load_folder, f"ids_{subject}.npy")))

    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0).flatten()
    ids = np.concatenate(ids, axis=0).flatten()

    return data, labels, ids

class DataGenerator(keras.utils.Sequence):
    """Generates data for Keras
    """
    def __init__(self, dataset,
                 batch_size=32,
                 time_steps=100,
                 mode='striding',
                 stride=100):
        'Initialization'
        self.dataset = dataset
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.mode = mode
        self.stride = stride
        self.on_epoch_end()
        self.__set_data_generator()

    def __len__(self):

        if self.lenght:
            return self.lenght

        print('No length!')
        return 0

    def __getitem__(self, index):
        'Generate one batch of data'
        X, y = self.__data_generator(index)

        return X, y

    def __set_data_generator(self):
        self.__data_generator = {'continuous':self.__data_generation_continuous,
                                 'striding':self.__data_generation_strides,
                                 'continuous_batch':self.__data_generation_continuous_batch}[self.mode]
        self.lenght = {'continuous':len(self.emg)-self.time_steps,
                       'striding':int((len(self.emg)-self.time_steps)/self.stride)+1,
                       'continuous_batch':int(self.batch_size*len(self.emg)/(self.batch_size+self.time_steps))}[self.mode]

    def set_mode(self, mode):
        if mode not in ('continuous', 'striding', 'continuous_batch'):
            print('Choose between ("continuous","striding","continuous_batch")')
            return
        self.mode = mode
        self.__set_data_generator()

    def on_epoch_end(self):
        self.sub = np.random.choice(self.dataset.subjects)

        a, b = self.sub.stimulus.shape[0], self.sub.emg.shape[0]
        c = np.min([a, b])
#        print(c,a,b)
        self.emg = self.sub.emg[:c, :][(self.sub.stimulus != 0).flatten()[:c], :]
        self.restimulus = self.sub.restimulus[:c, :][(self.sub.stimulus != 0).flatten()[:c], :]
#        self.emg = self.sub.emg[(self.sub.stimulus!=0).flatten(),:]
#        self.restimulus = self.sub.stimulus[(self.sub.stimulus!=0).flatten(),:]



    def __data_generation_continuous(self, index):

        features, labels = [], []
        i = self.time_steps + index
        while len(features) < self.batch_size:
            features.append(self.emg[i-self.time_steps:i, :])
            labels.append(mode(self.restimulus[i-self.time_steps:i, 0])[0].squeeze())
            i += 1
        out = np.expand_dims(np.array(features), axis=-1).transpose(2, 0, 1, 3)
        return list(out), np.array(labels)

    def __data_generation_continuous_batch(self, index):

        features, labels = [], []
        i = self.time_steps + index*(self.batch_size+self.time_steps)

        while len(features) < self.batch_size:
            features.append(self.emg[i-self.time_steps:i, :])
            labels.append(mode(self.restimulus[i-self.time_steps:i, 0])[0].squeeze())
            i += 1
        out = np.expand_dims(np.array(features), axis=-1).transpose(2, 0, 1, 3)
        return list(out), np.array(labels)

    def __data_generation_strides(self, index):

        features, labels = [], []
        i = index*self.stride+self.time_steps

        while len(features) < self.batch_size:
            features.append(self.emg[i-self.time_steps:i, :])
            labels.append(mode(self.restimulus[i-self.time_steps:i, 0])[0].squeeze())
            i += self.stride
        return list(np.expand_dims(np.array(features), axis=-1).transpose(2, 0, 1, 3)), np.array(labels)

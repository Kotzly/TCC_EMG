# -*- coding: utf-8 -*-
"""
Created on Thu May  2 20:05:34 2019

@author: Paulo
"""

import pywt

import numpy as np
from numpy.linalg import norm

import scipy.signal as sig
from sklearn.decomposition import FastICA

BAND = [5, 240]
FS = 2000
ENVELOPE_FC = 10
TRESHOLDING_DEFAULT_MODE = "soft"


def thresholding(arr, thr=1, mode=TRESHOLDING_DEFAULT_MODE):
    data = arr.copy()
    ddata = data.copy()
    if mode == 'soft':
        ddata = np.sign(ddata)*(np.abs(ddata) - thr)
    elif mode == 'hard':
        pass
    elif mode == 'zero':
        ddata[ddata != 0] = 0
    data[np.abs(data) <= thr] = 0
    data[np.abs(data) > thr] = ddata[np.abs(data) > thr]
    return data

def low_pass(data, fc=ENVELOPE_FC,filter_type="lfilt", fs=FS, axis=0):
    """ Low pass filtering.
    """
    b, a = sig.butter(4, fc/fs, btype='low')
    z = None
    if filter_type=="lfilt": z = sig.lfilter(b, a, data, axis=axis)
    if filter_type=="filtfilt": z = sig.filtfilt(b, a, data, axis=axis)
    return z

def band_pass(data, fc=BAND, fs=FS, filter_type="lfilt", axis=0):
    """ Band pass filtering.
        lfilt or filtfilt.
    """
    b, a = sig.butter(4, [fc[0]/fs, fc[1]/fs], btype='band')
    z = None
    if filter_type=="lfilt": z = sig.lfilter(b, a, data, axis=axis)
    if filter_type=="filtfilt": z = sig.filtfilt(b, a, data, axis=axis)
    return z

def envelope(data, axis=0, filter_type="lfilt"):
    return low_pass(np.abs(data), axis=axis, filter_type=filter_type)

def filterData(vector, band=BAND, fs=FS):
    high = band[1]#240
    low = band[0]#5
    b, a = sig.butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    zi = sig.lfilter_zi(b, a)
    z, _ = sig.lfilter(b, a, vector, zi=zi*vector[0])
    vector = z
    return z

def dwt_filter(signal, wavelet='db2', axis=0, level=4, thresholds=None,
               filter_start=0, filter_end=None, thr_mode='soft',
               n_std=1, pos_filtering={'fc':BAND, 'fs':FS, 'axis':0},
               return_thr=False):
# An Optimal Wavelet Function Based on Wavelet Denoising for Multifunction Myoelectric Control
# A Comparative Study of Wavelet Denoising for Multifunction Myoelectric Control
# If not threshold, reconstruct using "filter_level" decomposition levels
# If threshold, threshold the decomposition levels using
# thr_mode and n_std (number o standard deviations)

    """ Discrete Wavelet filtering.
        The default kws were chosen based on scientif papers (See comments).
    """
    if not filter_end: filter_end = level
    def thresh(data):
        return thresholding(data, thr, thr_mode)
    dec = pywt.wavedec(signal, wavelet, level=level, axis=axis)
    a = dec[0]
    dec = dec[1:]
    if thresholds == None: _thresholds = []
    for i, i_level in enumerate(range(filter_start, filter_end)):
        if thresholds == None:
            sigma = np.abs(dec[i_level]).mean(axis=axis)
            thr = n_std*sigma*np.sqrt(2*np.log(len(dec[i_level])))
            thr = np.expand_dims(thr, axis)
            _thresholds.append(thr)
        else:
            _thresholds = thresholds
            thr = thresholds[i]
        dec[i_level] = pywt.threshold(dec[i_level], thr, mode=thr_mode)
#            dec[i]=thresholding(dec[i],thr,thr_mode)
    for i in range(filter_end, level):
        dec[i] *= 0
    dec = [a]+dec
    rec = pywt.waverec(dec, wavelet=wavelet, axis=axis)
    if pos_filtering:
        rec = band_pass(rec, **pos_filtering)
    if return_thr:
        return rec, _thresholds
    return rec

class DwtFilter():
    def __init__(self, **params):
        self.params = params
        self.thr = None
    def fit(self, X=None, y=None):
        self.thr = dwt_filter(X, thresholds=self.thr, **self.params)[1]
        return self
    def transform(self, X):
        return dwt_filter(X, **self.params)
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def orde(m):
    size = m.shape[0]
    res = np.zeros((size, size))
    for i in range(size):
        res[np.abs(m[i, :]).argmax(), :] = m[i, :]
    return res

class multi_run_ICA():
#Multi run ICA and surface EMG based signal processing system for recognising hand gestures

    def __init__(self, n_iter=200, max_iter=500, n_components=12, algorithm='parallel'):
        self.ica = None
        self.sir_log = []
        self.n_iter = n_iter
        self.max_iter = max_iter
        self.n_components = n_components
        self.algorithm = algorithm
    def fit(self, data):
        max_sir = None
        self.ica = None
        for _ in range(self.n_iter):
            fca = FastICA(n_components=self.n_components,
                          max_iter=self.max_iter,
                          algorithm='deflation')
            fca.fit(data)
#            try:

            new_components = sorted(zip(fca.components_, np.argmax(np.abs(fca.components_), axis=1)), key= lambda x: x[1])
            new_components = np.stack(new_components)
            new_components = np.array([x[0] for x in new_components])
            fca.components_ = new_components

            m = new_components/(abs(norm(new_components, axis=1)).reshape(-1,1)) - np.identity(data.shape[1])
            sir = norm(m, axis=1)
#             except:
#                continue
            sir = -10*np.log10(sir).sum()
            # in the paper its >
            if max_sir is None or sir > max_sir:
                self.max_sir = sir
                max_sir = sir
                self.sir_log.append(sir)
                self.ica = fca
        return self.ica
    def transform(self, data):
        return self.ica.transform(data)

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

class OrderedFastICA():

    """ This class is a wrapper for sklearn's FastICA. The main difference is that when using the `transform` method this class tries to return the sources by the order of the components found.
    For instance: for each line in the unmixing matrix, which is the linear combination of the inputs that makes the sources, it finds the component that has the highest absolute value, and put them in order. If in the first line the higher component is from the 4º input, this line will be put in the 4º line.
    """

    def __init__(self, **ica_kws):
        if ica_kws == {}:
            ica_kws["algorithm"] = "deflation"
        self.ica_kws = ica_kws
        self.ica = FastICA(**ica_kws)

    def fit(self, x=None):
        ica = self.ica
        ica.fit(x)
        new_components = sorted(zip(ica.components_, np.argmax(np.abs(ica.components_), axis=1)), key= lambda x: x[1])
        new_components = np.stack(new_components)
        new_components = np.array([x[0] for x in new_components])
        ica.components_ = new_components
        return self.ica

    def transform(self, x=None):
        return self.ica.transform(x)

    def fit_transform(self, x=None):
        self.fit(x)
        return self.transform(x)

class RelativeNormalizer():

    """ Class for normalizing subjects sensors.
        Works with B-vectors.
    """

    def __init__(self, mode):
        """absmean, mean, energy, rms, abssum"""
        self.mode = mode

    def fit(self, x=None):
        pass

    def transform(self, x=None):
        if self.mode == "absmean":
            norm = abs(x).mean(axis=1)
        elif self.mode == "absum":
            norm = abs(x).sum(axis=1)
        elif self.mode == "rms":
            norm = np.sqrt(np.sum(x**2, axis=1)/float(x.shape[1]))
        elif self.mode == "energy":
            norm =  np.power(x, 2).sum(axis=1)
        elif self.mode == "mean":
            norm = x.mean(axis=1)
        else:
            raise ValueError("This normalizer don't have a mode!")
        return x/np.expand_dims(norm, axis=1)

    def fit_transform(self, x=None):
        self.fit(x)
        return self.transform(x)

#new_components = np.stack(sorted(zip(ica.components_,np.argmax(np.abs(ica.components_),axis=1)), key= lambda x: x[1]))
#new_components = np.array([x[0] for x in new_components])
#ica.components_ = new_components

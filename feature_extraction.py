# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 22:04:13 2018

@author: PauloAugusto
"""
import os
from preprocessing import BAND, FS, envelope

import numpy as np
import scipy.stats as st
import scipy.signal as sig
import scipy.special as spe
from scipy.stats import zscore
from scipy.ndimage.measurements import center_of_mass as _center_of_mass
import pandas as pd
import pywt


def energy(v, axis=1):
    """
        Calculates the energy of vector along an axis.
        Parameters:
            v: The signal to be calculated.
            axis: axis to calculate the energy.
        Returns:
            An array of 1 dimension less than "v" (the dimension specificied in axis).
    """

    sqrd = np.power(v, 2)
    return sqrd.sum(axis=axis)

def rel_energy(v, axis=1):
    """
        Calculates the relative energy of a vector relative to another vectors.
        Parameters:
            v: An numpy array. The signal to be calculated. Needs to have at
            least 3 dimensions.
            axis: axis to calculate the energy. Cannot be the last axis
            (v.ndim-1).
        Returns:
            An array with 1 less dimensions than v.
            Dimensions (axis) and (axis+1) are removed.
    """

    enr = energy(v, axis)
    n = enr.shape[1]
    rel = [enr[:, i]/enr[:, j] for i in range(n) for j in range(i, n) if i != j]
    return np.array(rel).T

def pearsonp(v, axis=1):
    """
        Calculates the Pearson correlation coefficient of a number channels
        relative to one another (n_samples, signal, channels).
        Parameters:
            v: An numpy array. The signals to be calculated. Needs to have at
            least 3 dimensions.
            axis: axis to calculate the correlation. Cannot be the last axis
            (v.ndim-1).
        Returns:
            An array with 1 less dimensions than v.
            Dimensions (axis) and (axis+1) are removed.
    """

    n = v.shape[2]
    rel = [[st.pearsonr(x.take(i, axis=axis), x.take(j, axis=axis))[0] for i in range(n) for j in range(i, n) if i != j] for x in v]
    return np.array(rel)

def wilcoxon(v, axis=1):
    """
        Calculates the Wilcoxon correlation coefficient of a number channels
        relative to one another.
        Parameters:
            v: An numpy array. The signals to be calculated. Needs to have at
            least 3 dimensions (n_samples, signal, channels).
            axis: axis to calculate the correlation. Cannot be the last axis
            (v.ndim-1).
        Returns:
            An array with 1 less dimensions than v.
            Dimensions (axis) and (axis+1) are removed.
    """
    n = v.shape[2]
    rel = [[st.ranksums(x.take(i, axis=axis), x.take(j, axis=axis))[0] for i in range(n) for j in range(i, n) if i != j] for x in v]
    return np.array(rel)

def spearmeanr(v, axis=1, sensor_axis=2):
    """
        Calculates the Spearman correlation coefficient of a number channels
        relative to one another.
        Parameters:
            v: An numpy array. The signals to be calculated. Needs to have at
            least 3 dimensions (n_samples, signal, channels).
            axis: axis to calculate the correlation. Cannot be the last axis
            (v.ndim-1).
        Returns:
            An array with 1 less dimensions than v.
            Dimensions (axis) and (axis+1) are removed.
    """
    m = np.empty((v.shape[sensor_axis], v.shape[sensor_axis]))
    m = np.triu_indices_from(m, 1)
#    return st.spearmanr(v[0], axis=0)
    rel = [st.spearmanr(x, axis=0)[0][m] for x in v]
    return np.array(rel)

def kl_div(v, axis=1):
    """
        Calculates the Kullback-Leibler divergence of a number channels
        relative to the others.
        Parameters:
            v: An numpy array. The signals to be calculated. Needs to have at
            least 3 dimensions (n_samples, signal, channels).
            axis: axis to calculate the divergence. Cannot be the last axis
            (v.ndim-1).
        Returns:
            An array with 1 less dimensions than v.
            Dimensions (axis) and (axis+1) are removed.
    """
    n = v.shape[2]
    rel = [[spe.kl_div(x.take(i, axis=axis-1), x.take(j, axis=axis-1))[0] for i in range(n) for j in range(i, n) if i != j] for x in v]
    return np.array(rel)

def ar_coef(v, nr=5, axis=1):
    """
        Calculates auto-regressive coefficients of a signal, given an axis.
        Parameters:
            v: An numpy array. The signals to be calculated. Needs to have at
            least 3 dimensions (n_samples, signal, channels).
            nr: The number of coefficients to use.
            axis: Axis to calculate the correlation. Cannot be the last axis
            (v.ndim-1).
        Returns:
            An array with the same dimensionality as v. It has nr coefficients
            plus the interceipt.
    """
    from sklearn.linear_model import LinearRegression
    from keras.utils import Progbar
    shape = np.prod([x for i, x in enumerate(v.shape) if i != axis])
    bar = Progbar(shape)
    lr = LinearRegression()
    def get_ar(arr, n=nr):
        s = arr.shape[0]
        s = s-s%n
        x = arr[:s][np.array([[i for i in range(j-n, j)] for j in range(n, s)])]
        y = arr[:s][np.array(range(n, s))]
        lr.fit(x, y)
        bar.add(1)
        return [lr.intercept_, *lr.coef_]
    temp = v
    if temp.ndim == 1:
        temp = temp.reshape(-1, 1)
    return np.apply_along_axis(get_ar, axis, temp)

def rms(v, axis=0):
    """
        Calculates RMS value of a signal, given an axis.
        Parameters:
            v: An numpy array. The signals to be calculated. Needs to have at
            least 3 dimensions (n_samples, signal, channels).
            axis: Axis to use.
        Returns:
            An array of the RMS values calculated, along axis "axis".
    """
    temp = np.sqrt(np.sum(v**2, axis=axis)/float(v.shape[axis]))
    return temp


def rel_rms(v, axis=1):
    rms_values = rms(v, axis)
    n = rms_values.shape[1]
    rel = [rms_values[:, i]/rms_values[:, j] for i in range(n) for j in range(i, n) if i != j]
    return np.array(rel).T

def center_of_mass_by_frequency(v, axis=0, sensor_axis=2):

    if not isinstance(v, np.ndarray):
        v = np.array(v)
    size = v.shape[axis]
    ticks = np.linspace(0, BAND[1], size)
    shape = [v.shape[i] if i==axis else 1 for i in range(len(v.shape))]
    return (v*ticks.reshape(*shape)).sum(axis=axis)/v.sum(axis=axis)

def center_of_mass_by_index(v, axis=0, sensor_axis=2):

    if not isinstance(v, np.ndarray):
        v = np.array(v)
    size = v.shape[axis]
    ticks = np.array(range(size))
    shape = [v.shape[i] if i==axis else 1 for i in range(len(v.shape))]
    return (v*ticks.reshape(*shape)).sum(axis=axis)/v.sum(axis=axis)


def spectral_centroid_time(v, axis=1, sensor_axis=2, fs=FS):
    """ Calculates the spectral centroid of a temporal signal. Uses the mean
        value of the predominant frequencies calculated using STFT.
        Parameters:
            v: An numpy array. The signals to be calculated. Needs to have at
            least 3 dimensions (n_samples, signal, channels).
            axis: Axis to use.
        Returns:
            An array of the centroids, along axis "axis".
    """
    fft_v = np.abs(np.fft.fft(v, axis=axis))
    centroids = center_of_mass_by_index(fft_v, axis=axis, sensor_axis=sensor_axis).astype(int)
    frequencies = np.linspace(0, fs, v.shape[axis])
    return frequencies[centroids]

def spectral_centroid_frequency(v, axis=1, sensor_axis=2):
    """ Calculates the spectral centroid of a temporal signal. Uses the mean
        value of the predominant frequencies calculated using STFT.
        Parameters:
            v: An numpy array. The signals to be calculated. Needs to have at
            least 3 dimensions (n_samples, signal, channels).
            axis: Axis to use.
        Returns:
            An array of the centroids, along axis "axis".
    """
    centroids = center_of_mass_by_frequency(v, axis=axis, sensor_axis=sensor_axis)
    return centroids

def willison_amplitude(v, axis=1, threshold=0.01):
    def  bin_treshold(arr, thr):
        return (arr >= thr).astype(np.float32)
    temp = v.swapaxes(0, axis)
    temp = bin_treshold(np.abs(temp[:-1]-temp[1:]), threshold).sum(axis=0)
    return temp.squeeze()

def fft(v, axis=1, fs=FS):
    fft_array = np.fft.fft(v, axis=axis)
    fft_array = np.abs(fft_array)
    fc_index = int(BAND[1]*fft_array.shape[axis]/FS)
    fft_array = np.swapaxes(fft_array, 0, axis)
    fft_array = fft_array[:fc_index]
    fft_array = np.swapaxes(fft_array, 0, axis)
    ticks = np.linspace(0, BAND[1], len(fft_array))
    return fft_array, ticks

def filterData(vector, band=BAND, fs=FS):
    high = band[1]
    low = band[0]
    b, a = sig.butter(4, [low/(fs/2), high/(fs/2)], btype="band")
    zi = sig.lfilter_zi(b, a)
    z, _ = sig.lfilter(b, a, vector, zi=zi*vector[0])
    vector = z
    return z

def zero_cross_count(v, axis=0):
    temp = np.swapaxes(v, 0, axis)
    return np.sum(np.abs(np.sign(temp[:-1])-np.sign(temp[1:])), axis=0)/2

def hist(v, bins=10, ranges=None, axis=1):
    if ranges==None:
        ranges=[v.min(axis=axis).min(), v.max(axis=axis).max()]
    return np.apply_along_axis(lambda a: np.histogram(a, bins=bins, range=ranges)[0], axis, v)

def percentiles(v, percents=[10*i for i in range(1, 11)], axis=1):
    result = np.percentile(v, percents, axis=axis)
    return np.swapaxes(result, 0, axis)

def corrcoef(v, axis=None):
    result = []
    if len(v.shape) == 2:
        return np.corrcoef(v, rowvar=False)
    for i in range(v.shape[0]):
        m = np.corrcoef(np.array(v[i, :, :]), rowvar=False)
        m = m[np.triu_indices_from(m, 1)].reshape(1, -1)
        result.append(m)
    return np.concatenate(result, axis=0)

def cwt_features(v, scales=range(1, 16), wavelet="mexh", axis=1):
    from keras.utils import Progbar
    td = v.shape[2] if v.ndim is 3 else 1
    bar = Progbar(v.shape[0]*td)
    def cwt(arr):
        res = pywt.cwt(arr, scales, wavelet)[0]
        std = res.std()
        mean = res.mean()
        energy = (res**2).sum()
        abssum = np.abs(res).sum()
        absmax = np.max(np.abs(res))
        hist = np.histogram(res, bins=10)[0]
        bar.add(1)
        return np.array([std, mean, energy, abssum, absmax, *hist])
    return np.apply_along_axis(cwt, axis, v)

def dwt_energy(v, level=8, wavelet="db1", axis=1):
    dec = pywt.wavedec(v, wavelet, level=level, axis=axis)
    energy = [np.expand_dims((s**2).sum(axis=axis), axis=1) for s in dec]
    energy = np.concatenate(energy, axis=1)
    return energy

def dwt_features(v, level=8, wavelet="db1", axis=1, debug=False, return_names=False):
    from scipy.stats import kurtosis, skew, gmean, iqr
    if debug: print("Calculating DWT")
    dec = pywt.wavedec(v, wavelet, level=level, axis=axis)
    res = []
    for i, lvl in enumerate(dec):
        if debug:
            print("Calculating features for level", (["A"] + ["D"+str(j+1) for j in range(level)])[i])
        res.extend([lvl.mean(axis),
                    lvl.std(axis),
                    kurtosis(v, axis),
                    skew(v, axis),
                    gmean(np.abs(v), axis),
                    *[iqr(v, axis=axis, rng=(i*20, (i+1)*20)) for i in range(5)]])
    res = np.array(res).transpose(1, 0, 2)#.reshape(v.shape[0],v.shape[2],-1).transpose(0,2,1)
    if return_names:
        names = [feat + "_lvl"+str(lvl) for feat in ["mean", "std", "kurt", "skew", "gmean", *["iqr"+str(ir) for ir in range(10)]]
                 for lvl in range(level+1)]
        return res, names
    return res

def absmean(v, axis=1):
    return np.abs(v).mean(axis=axis)

class SignalParser():

    """ Class for extracting EMG Signals.
        Parameters:
            extractors: a list with 2 elements: a list with the temporal features extractors and a list with the spectral features extractor.
            n_sensors: number of sensor/channels that the extractors will use.
        Returns:
            2 lists [features_names, features]. The first one has features that can be calculated using each channel individually, and the second has features that contains relative characteristics amongst channels.
    """

    class extractor():
        """Function must return (n_samples, n_channels) or (n_samples, n_features, n_channels)"""
        def __init__(self, function, name, verbose=0, use_envelope=False):
            self.name = name
            self.function = function
            self.verbose = verbose
            self.use_envelope = use_envelope

        def extract(self, signal, axis=1, debug=False):
            if self.use_envelope:
                signal = envelope(signal, axis=axis)
            n_sensors = signal.shape[2]
            if self.verbose > 0:
                print(f"\tExtracting {self.name}")
            feature = np.array(self.function(signal, axis=axis))
            if feature.ndim is 3:
                names = [self.name+f"_S{i}_C{j}" for j in range(feature.shape[1]) for i in range(n_sensors)]
            elif feature.ndim is 2:
                if debug:
                    print(f"Warning: {self.name} returned shape {feature.shape}")
                    print(f"Transformed to {feature.shape}")
                feature = np.expand_dims(feature, 1)
                n_channels = feature.shape[2]
                if n_channels is n_sensors:
                    names = [self.name + f"_S{i}" for i in range(n_sensors)]
                else:
                    names = [self.name + f"_{i}_{j}" for i in range(n_sensors)
                             for j in range(i, n_sensors) if i != j]

            else:
                print(self.name, feature.shape)
                raise ValueError("Wrong output shape at {}".format(self.name))
            return names, feature

        def __call__(self, *args, **kwargs):
            return self.extract(*args, **kwargs)

    time_extractors = [extractor(st.kurtosis, "KRTT"),
                       extractor(st.skew, "SKWT"),
                       extractor(np.std, "STDT"),
                       extractor(np.var, "VART"),
                       extractor(np.mean, "MENT"),
                       extractor(absmean, "ABMT"),
                       extractor(rms, "RMST"),
                       extractor(zero_cross_count, "ZCCT"),
                       extractor(willison_amplitude, "WAPT"),
                       extractor(energy, "ENGT"),
                       extractor(hist, "HSTT"),
                       extractor(percentiles, "PRCT"),
                       extractor(corrcoef, "CORT"),
                       extractor(rel_energy, "RENT"),
                       extractor(pearsonp, "PSPT"),
                       extractor(wilcoxon, "WCXT"),
                       extractor(spearmeanr, "SPRT"),
                       extractor(dwt_energy, "DWET"),
                       extractor(cwt_features, "CWFT")]

    frequency_extractors = [extractor(spectral_centroid_frequency, "SPCF"),
                            extractor(st.kurtosis, "KRTF"),
                            extractor(st.skew, "SKWT"),
                            extractor(np.std, "STDF"),
                            extractor(np.var, "VARF"),
                            extractor(np.mean, "MENF"),
                            extractor(hist, "HSTF"),
                            extractor(percentiles, "PRCF"),
                            extractor(corrcoef, "CORF"),
#                            extractor(kl_div, "KLDF")
                            ]

    def __init__(self, extractors=None, n_sensors=12, verbose=0):
        self.n_sensors = n_sensors
        self.verbose = verbose

        self.time_extractors = []
        self.frequency_extractors = []
        if extractors:
            for ext_name in extractors[0]:
                for ext in SignalParser.time_extractors:
                    if ext.name == ext_name:
                        self.time_extractors.append(ext)
            for ext_name in extractors[1]:
                for ext in SignalParser.frequency_extractors:
                    if ext.name == ext_name:
                        self.frequency_extractors.append(ext)
        else:
            self.time_extractors = SignalParser.time_extractors
            self.frequency_extractors = SignalParser.frequency_extractors
        for ext in self.time_extractors + self.frequency_extractors:
            ext.verbose = self.verbose

    def extract(self, signal, axis=1, debug=False, features_and_names=False):
        if self.verbose > 2:
            print("Extracting features...")
        features = []
        names = []
        for f in self.time_extractors:
            try:
                name, feature = f(signal, axis=axis, debug=debug)
            except Exception as e:
                print(f.name, signal.shape, str(e))
                raise e
            features.append(feature)
            names.append(name)

        f_signal = fft(signal)[0]

        for f in self.frequency_extractors:
            name, feature = f(f_signal, axis=axis, debug=debug)
            features.append(feature)
            names.append(name)

        channel_features = []
        correlated_features = []
        channel_features_names = []
        correlated_features_names = []

        n = self.n_sensors
        for name, feature in zip(names, features):
            if feature.shape[2] == n:
                channel_features.append(feature)
                channel_features_names.extend(name)

            elif feature.shape[2] == int(n*(n-1)/2):
                correlated_features.append(feature)
                correlated_features_names.extend(name)

            else:
                raise Exception(f"The number of sensors found is not equal to the number of sensors declared {self.n_sensors}")

#        channel_res = (channel_features_names, np.concatenate(channel_features, axis=1))
#        corr_res = (correlated_features_names, np.concatenate(correlated_features, axis=1))
#
#        if not features_and_names:
#            out1, out2 = channel_res, corr_res
#        else:
#            features = np.concatenate([channel_res[1].reshape(channel_res[1].shape[0], -1), corr_res[1].reshape(corr_res[1].shape[0], -1)], axis=-1)
#            names = channel_res[0] + corr_res[0]
#            out1, out2 = features, names
        out1 = None
        out2 = None

        if not features_and_names:
            if channel_features:
                out1 = (channel_features_names, np.concatenate(channel_features, axis=1))
            if correlated_features:
                out2 = (correlated_features_names, np.concatenate(correlated_features, axis=1))
        else:
            temp = []
            if channel_features:
                 chf = np.concatenate(channel_features, axis=1)
                 temp.append(chf.reshape(chf.shape[0], -1))
            if correlated_features:
                 crf = np.concatenate(correlated_features, axis=1)
                 temp.append(crf.reshape(crf.shape[0], -1))
            features = np.concatenate(temp, axis=1)
            names = channel_features_names + correlated_features_names
            out1, out2 = features, names

        return out1, out2

    def extract_features(self, signal, axis=1, save_folder=None, name=None):
        features, names = self.extract(signal, axis=axis, features_and_names=True)
        if save_folder:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            if name:
                np.save(os.path.join(save_folder, name), features)
            else:
                np.save(os.path.join(save_folder, "features"), features)
            with open(os.path.join(save_folder, "features_names.txt"), "w") as file:
                file.writelines("\n".join(names))
        return features, names

    def __call__(self, signal):
        return self.extract(signal, features_and_names=True)

SignalParser.time_extractors_names = [ext.name for ext in SignalParser.time_extractors]
SignalParser.frequency_extractors_names = [ext.name for ext in SignalParser.frequency_extractors]

class Kinesin():
    def __init__(self):
        import keras
        self.parser = SignalParser()
        self.bar = keras.utils.Progbar(1000)

    def walk(self, signal, stimulus, sub_id="", verbose=1):
        # signal = (n_sample, n_channels)
        parser = self.parser

        signal_len = len(signal)
        rest_len = len(stimulus)
        lenght = np.min([signal_len, rest_len])

        last_label = stimulus[0, 0]
        i0 = 0
        lines = []

        column_names = ["Title", "Stimulus"]
        names = parser.extract(np.random.randint(0, 10, (10, 12)))[0]
        column_names.extend(names)

        one_percent = (lenght//1000)

        if verbose > 0: print("Now walking at subject ID#", sub_id)
        self.bar.update(0)

        for i in range(lenght):
            if verbose > 0:
                if i%one_percent == 0:
                    self.bar.add(1)
            if not stimulus[i, 0] == last_label or i == lenght-1:
                lines.append([sub_id, last_label])
                lines[-1].extend(parser(signal[i0:i, :]))
                last_label = stimulus[i, 0]
                i0 = i

        return np.array(column_names), np.array(lines)
#sp = SignalParser(extractors=[['KRTT',
#                               'SKWT',
#                               'STDT',
#                               'VART',
#                               'ABMT',
#                               'RMST',
#                               'ZCCT',
#                               'WAPT',
#                               'ENGT',
#                               'PRCT',
#                               'CORT',
#                               'RENT',],
#                              ['SPCF',
#                               'KRTF',
#                               'STDF',
#                               'VARF',
#                               'MENF',
#                               'HSTF',
#                               'PRCF',
#                               'CORF',
#                               'KLDF']])

#from sklearn.model_selection import train_test_split
#from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score
#from sklearn.preprocessing import RobustScaler
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.linear_model import LogisticRegression

#rs = RobustScaler()
#mlp = MLPClassifier(hidden_layer_sizes=(100,100,2,50),early_stopping=True, validation_fraction=0.2)
#mlp = RandomForestClassifier(n_estimators=200,max_depth=10,max_features=10)
#mlp = KNeighborsClassifier(n_neighbors=21)
#mlp = LogisticRegression(penalty="l1",solver = ["newton-cg", "lbfgs", "liblinear",
#                                                "sag", "saga"][2])
#
#x_train, x_test,y_train, y_test = train_test_split(f1[0][b[1].ravel()!=0,-66:],b[1][b[1].ravel()!=0], test_size=0.4)
#rs.fit(x_train)
#x_train = rs.transform(x_train)
#x_test = rs.transform(x_test)
#mlp.fit(x_train, y_train)
#pred_train = mlp.predict(x_train)
#pred_test = mlp.predict(x_test)
#print("Acc:", balanced_accuracy_score(y_train, pred_train))
#print("Acc:", balanced_accuracy_score(y_test, pred_test))
#print("Kappa:", cohen_kappa_score(y_test, pred_test))

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 20:56:15 2018

@author: PauloAugusto
"""

import pandas as pd
from extract_parameters import features_extractors,features_names,filterData
from scipy.io import loadmat
import scipy.stats as st 
import matplotlib.pyplot as plt
import numpy as np
import os
import zipfile
from sys import getsizeof
import pickle

fs=2000
band=[6,240]
#verbose=3
number_of_subjects=1
my_folder='C:/Users/Paulo/Desktop/TCC/EMG_Database/'
def get_data(verbose,folder=my_folder,subjects = None,sensors=None,repetitions=None,stimuluses=None,return_samples=False,return_processed_data=False,return_df=False,return_fft_samples=False,dataset_number=2):
    """Function for reading .zip file from the NinaPro Database.
    return_df mode works the best for all cases
    return_samples works the est for intact subjects (who performed all repetitions and all exercises)
    return_processed_data also returns a data_frame with some functions applied to the signals (not recommended, instead process it later)"""
    
    features_columns=['Title','Sensor','Stimulus', 'Repetition']+features_names
    data_frame=pd.DataFrame(columns=features_columns)
    if not subjects:    subjects=['DB{}_s{}'.format(dataset_number,1+i) for i in range(40)]
    else:               subjects=['DB{}_s{}'.format(dataset_number,1+i) for i in np.array(subjects)-1]
    
    if not sensors: sensors=range(12)
    if not repetitions: repetitions=range(1,7)
    if not stimuluses: stimuluses=range(1,50)
    samples=[]
    df = []
    for sub in subjects:
        samples.append([])
        print(folder + sub + '.zip')
        with zipfile.ZipFile(folder + sub + '.zip',"r") as zip_ref:
            zip_ref.extractall(folder)
        if verbose>0:   print('Loading ' + sub)
        memcon=np.array(samples)
        if verbose>3:   print('\t Sample Memory Usage:' + str((memcon.size*8)/1024.0)+' KB')
        archives=['S{0}_E{1}_A1.mat'.format(sub[sub.find('s')+1:],ex) for ex in range(1,4)]
        for archive in archives:
            if verbose>1:   print('\t Loading ' + archive)
            if verbose>3:   print('\t Data Frame Memory Usage:' + str(data_frame.memory_usage()[0]/1024.0)+' KB')
            data=loadmat(folder + sub + '/' + archive )
            data['emg']=data['emg'].T
            data['rerepetition']=data['rerepetition'].T.flatten()
            data['restimulus']=data['restimulus'].T.flatten()
            
            for i in range(len(data['emg'])):
                data['emg'][i]=filterData(data['emg'][i],band,fs)
            
            min_sti=data['restimulus'][data['restimulus']>0].min()
            max_sti=data['restimulus'].max()
            for sensor in sensors:
                sensor_i=sensors.index(sensor)
                if archive ==archives[0]: samples[-1].append([])
                if verbose>2:   print('\t\tLoading sensor #' + str(sensor+1) )
                for stimulus in range(min_sti,max_sti+1):
                    if not stimulus in stimuluses: continue
                    stimulus_i=stimuluses.index(stimulus)
                    samples[-1][sensor_i].append([])
                    for repetition in repetitions:
                        repetition_i=repetitions.index(repetition)
                        movement_window=data['emg'][sensor][:min(len(data['emg'][sensor]),len(data['restimulus']),len(data['rerepetition']))][(np.equal(data['rerepetition'],repetition) &
                                                            np.equal(data['restimulus'],stimulus))
                                                            [:min(len(data['emg'][sensor]),len(data['restimulus']),len(data['rerepetition']))]]
                        if return_df:
                            df.append([int(sub[sub.find('s')+1:]),sensor+1,stimulus,repetition,movement_window.copy()])
                        
                        if return_processed_data:
                            slc = range(round(band[0]*len(movement_window)/fs),round(band[1]*len(movement_window)/fs)+1)#
                            movement_fft=np.abs(np.fft.fft(movement_window))[slc]#

                            features=[subjects.index(sub)+1,sensor+1,stimulus,repetition]
                            features+=[extractor(movement_window) for extractor in features_extractors[:7] ]
                            features+=[extractor(movement_fft) for extractor in features_extractors[7:]]
                            data_frame.loc[len(data_frame)]=features
                        if return_samples: 
                            data_to_append = [movement_window.copy()]
                            if return_fft_samples: data_to_append.append(np.abs(np.fft.fft(movement_window)))
                            samples[-1][sensor_i][stimulus_i].append(data_to_append)
        [os.remove(folder+sub+'/'+arq) for arq in os.listdir(folder+sub)]
        os.rmdir(folder+sub)
    signal_features=data_frame.keys().copy()

    si=pd.read_csv(folder+'Ninapro_DB{}_SubjectsInformation.csv'.format(dataset_number))
    data_frame=data_frame.merge(si,on='Title')
    
    df=pd.DataFrame(data=df,columns=['Title','Sensor','Stimulus','Repetition','Raw_Data'])
    df=df.merge(si,on='Title')

    keys=data_frame.keys()
    
    return_list = []
    if return_df: return_list.append(df)
    if return_samples: return_list.append(np.array(samples))
    if return_processed_data: return_list.append(data_frame)
    
    if type(return_list)==list and len(return_list)==1: return return_list[0]
    else:                                               return return_list
    
    
def print_covs(data_frame,stimulus=None,parameter=None):
    global features_names
    if not stimulus:
        m=data_frame[['Stimulus']+features_names]
    else:
        m=data_frame[features_names][data_frame['Stimulus']==stimulus]
    covs=m.cov().as_matrix()
    max_covs=[]
    fn=['Stimulus'] + features_names
    for i in range(15):
        line=covs[i].copy()
        line[0],line[i]=+np.inf,+np.inf
        max_covs.append((fn[i],
                         fn[np.argmin(line)]))
    for dupla in max_covs:
        if (parameter and max_covs.index(dupla)==parameter) or parameter==None:
            plt.scatter(m[dupla[0]],m[dupla[1]])
            plt.title(dupla[0]+' vs '+dupla[1])
            plt.xlabel(dupla[0])
            plt.ylabel(dupla[1])
            plt.axis([np.min(m[dupla[0]]),np.max(m[dupla[0]]),np.min(m[dupla[1]]),np.max(m[dupla[1]])])
            plt.show()
def save_data(data,path):
    try:
        with open(path,'wb') as savefile:
            pickle.dump(data,savefile) 
        return 1
    except:
        return 0
def load_data(path):
    with open(path,'rb') as infile:
        som = pickle.load(infile) 
    return som

def print_3d(data,n1,n2,n3,x,y,z,s1,s2,s3):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    ctt1=np.equal(data['Sensor'],n1) & np.equal(data['Title'],s1) & np.less(data['Stimulus'],41)
    ctt2=np.equal(data['Sensor'],n2) & np.equal(data['Title'],s2) & np.greater_equal(data['Stimulus'],41)
    ctt3=np.equal(data['Sensor'],n3) & np.equal(data['Title'],s3) & np.less(data['Stimulus'],41)
    ax.scatter3D(data[x][ctt1],data[y][ctt1],data[z][ctt1],c='r')
    ax.scatter3D(data[x][ctt2],data[y][ctt2],data[z][ctt2],c='b')
    ax.scatter3D(data[x][ctt3],data[y][ctt3],data[z][ctt3],c='g')
    ax.set_zlabel(z)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
    
#    'Energy','Zero Crosses','Time Skewness', Weight
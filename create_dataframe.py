# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 19:00:00 2019

@author: Paulo
"""

import numpy as np
import pandas as pd
import emg_loader as el
import matplotlib.pyplot as plt
import signal_funcs as sf
import scipy.stats as st

###############################################################################
#emg_data = el.get_data(2,subjects=list(np.random.randint(1,41,3)),stimuluses=[1,2,3],return_processed_data=False,return_samples=True)
#emg_data = np.array(emg_data)
#shape =  emg_data.shape[:-1]
#i = np.zeros_like(shape)
#[Subject,Sensor,Stimulus,Repetition]
###############################################################################
i=np.array([0,0,0,0])
time_functions=[np.mean,np.std,np.var,
                sf.energy,st.kurtosis,
                st.skew]
frequency_functions=[st.kurtosis,st.skew,
                     np.var,np.mean,
                     np.std,sf.rms]
band=[6,240]
fs=2000
sc_f = sf._spectral_centroid()

class helper():
    def __init__(self,data):
        self.data=data
    def __getitem__(self,b):
        df = self.data
        ctt = (df['Title']==b[0]) & (df['Sensor']==b[1]) & (df['Stimulus']==b[2]) & (df['Repetition']==b[3]) 
        r = df[ctt]['Raw_Data'].values
        if len(r)>0: return r[0]
        else:        return r

#    if type(data)==np.ndarray: sh = [range(a) for a in data.shape]
#    if type(data)==pd.DataFrame: 
#        sh = [data[feat].unique() for feat in data.columns[:4]]
#        data = helper(data)

def recursion(init=0,matrix_data=[],data=np.array([]),
              time_functions=time_functions,
              frequency_functions=frequency_functions,
              initial_sub=0,verbose=0):
    global i,band
    shape = data.shape[:-1]
    if init==len(shape):
        
        freq_window = np.abs(np.fft.fft(data[i[0],i[1],i[2],i[3],0]))
        slc1 = round(band[0]*len(freq_window)/fs)
        slc2 = round(band[1]*len(freq_window)/fs)
        fft_values = freq_window[slc1:slc2]        
        sc=sc_f(fft_values)*fs/len(freq_window)+band[0]
        
        time_data = [func(data[i[0],i[1],i[2],i[3],0]) for func in time_functions]
        frequency_data = [func(fft_values) for func in frequency_functions]
        data_to_append = time_data + frequency_data + [sc]
        
        initial_sub=np.array([initial_sub]+[0 for _ in range(len(i)-1)])
        
        matrix_data.append([*(i+1)+initial_sub,*data_to_append])
#        if (i==shape).all(): 
        return
    for i[init] in range(shape[init]):
        if (init==0) and (verbose>=0): print('Processing Subject: ',i[init])
        if (init==1) and (verbose>=1): print('\tProcessing Sensor: ',i[init])
        if (init==2) and (verbose>=2): print('\t\tProcessing Stimulus: ',i[init])
        if (init==3) and (verbose>=3): print('\t\t\tProcessing Repetition: ',i[init])
        recursion(init+1,matrix_data,data,initial_sub=initial_sub,verbose=verbose)
    return np.array(matrix_data)

def create_dataframe(data,time_functions,frequency_functions,initial_sub=0,verbose=0):
    print('Use create_dataframe_nonrecursive for better performance')
    columns  = ['Title','Sensor','Stimulus','Repetition']
    columns += ['Time_' + _.__name__.capitalize() for _ in time_functions]
    columns += ['Frequency_' + _.__name__.capitalize() for _ in frequency_functions]
    columns += ['Spectral Centroid']
    matrix = recursion(init=0,matrix_data=[],data=data,
                       time_functions=time_functions,
                       frequency_functions=frequency_functions,
                       initial_sub=initial_sub,
                       verbose=verbose)
    return pd.DataFrame(data=matrix,columns=columns)

def create_dataframe_nonrecursive(data,time_functions,frequency_functions,initial_sub=0,verbose=0):
    columns  = ['Title','Sensor','Stimulus','Repetition']
    columns += ['Time_' + _.__name__.capitalize() for _ in time_functions]
    columns += ['Frequency_' + _.__name__.capitalize() for _ in frequency_functions]
    columns += ['Spectral Centroid']
#    matrix = recursion(init=0,matrix_data=[],data=data,
#                       time_functions=time_functions,
#                       frequency_functions=frequency_functions,
#                       initial_sub=initial_sub,
#                       verbose=verbose)
#    class helper():
#        def __init__(self,data):
#            self.data=data
#        def __getitem__(self,b):
#            df = self.data
#            ctt = (df['Title']==b[0]) & (df['Sensor']==b[1]) & (df['Stimulus']==b[2]) & (df['Repetition']==b[3]) 
#            r = df[ctt]['Raw_Data'].values
#            if len(r)>0: return r[0]
#            else:        return r

    if type(data)==np.ndarray: sh = [range(a) for a in data.shape]
    if type(data)==pd.DataFrame: 
        sh = [data[feat].unique() for feat in data.columns[:4]]
        data = helper(data)
    
    
    matrix=[]
    
    for a in sh[0]:
        if (verbose>=0): print('Processing Subject: ',a+initial_sub)
        for b in sh[1]:
            if (verbose>=1): print('\tProcessing Sensor: ',b)
            for c in sh[2]:
                if (verbose>=2): print('\t\tProcessing Stimulus: ',c)
                for d in sh[3]:
                    if (verbose>=3): print('\t\t\tProcessing Repetition: ',d)
                    window = data[a,b,c,d,0]
                    if len(window)==0: continue
                    freq_window = np.abs(np.fft.fft(window))
                    slc1 = round(band[0]*len(freq_window)/fs)
                    slc2 = round(band[1]*len(freq_window)/fs)
                    fft_values = freq_window[slc1:slc2]        
                    sc=sc_f(fft_values)*fs/len(freq_window)+band[0]
                    
                    time_data = [func(data[a,b,c,d,0]) for func in time_functions]
                    frequency_data = [func(fft_values) for func in frequency_functions]
                    data_to_append = time_data + frequency_data + [sc]
                    
                    matrix.append([*(a+initial_sub,b+1,c,d),*data_to_append])              
    return pd.DataFrame(data=matrix,columns=columns)

def create_relative(df,mode='all',on='Energy'):
    global count
    count=0
    
    def get_sensor_rel_energy(df,stimulus,title,rep,sensor,mode='all',on='Energy'):
        global count
        if mode=='all':
            ctt_data = df[df['Stimulus']==stimulus]
            ctt_data = ctt_data[ctt_data['Title']==title]
            ctt_data = ctt_data[ctt_data['Repetition']==rep]
            total_en = ctt_data[on].sum()
            temp = pd.Series((ctt_data[on]/total_en).values)
        elif mode=='relation':
            ctt_data = df[df['Stimulus']==stimulus]
            ctt_data = ctt_data[ctt_data['Title']==title]
            ctt_data = ctt_data[ctt_data['Repetition']==rep]
            temp = np.array([ctt_data[ctt_data['Sensor']==i][on].values/ctt_data[ctt_data['Sensor']==j][on].values for i in range(1,13) for j in range(i,13) if i!=j]).reshape(66)
            temp = pd.Series(temp)

        if count%(len(df)//100+1)==0:
            print(str(count*100/len(df))+'%')
        count+=1
        return temp
    if mode=='relation': ctt = [True for i in range(len(df))]
    if mode=='all':      ctt = (df['Sensor']==1)
    energy_features = df[['Stimulus','Title','Repetition','Sensor']].apply(lambda x: get_sensor_rel_energy(df,*x,mode=mode,on=on),axis=1)[ctt]
    return energy_features

def create_correlation(data,on='Energy',verbose=0):
    sh = [range(i) for i in data.shape[:-1]]
    matrix=[]
    
    if type(data)==np.ndarray: sh = [range(a) for a in data.shape]
    if type(data)==pd.DataFrame: 
        sh = [data[feat].unique()-1 for feat in data.columns[:4]]
        data = helper(data)
    
    for sub in sh[0]:
        if verbose>0: print('Subject: ',sub)
        for sti in sh[2]:
            if verbose>1: print('\tStimulus: ',sti)
            for rep in sh[3]:
                if verbose>2: print('\t\tRepetition: ',rep)
                data_to_append=[sub+1,sti+1,rep+1]
                c=0
                for i in range(12):
                    for j in range(i,12):
                        if not i==j:
                            x = data[sub,i,sti,rep,0]
                            y = data[sub,j,sti,rep,0]
                            size=np.min([len(x),len(y)])
                            cov = (1.0/size)*np.sum((x-x.mean())*(y-y.mean()))
                            cor = 2*cov/(x.var()+y.var()+(x.mean()+y.mean())**2)
                            data_to_append.append(cor)  
                matrix.append(data_to_append)
    return pd.DataFrame(data = matrix,columns=['Title','Stimulus','Repetition']+['Var_{}_{}'.format(i,j) for i in range(1,13) for j in range(i,13) if not i==j])
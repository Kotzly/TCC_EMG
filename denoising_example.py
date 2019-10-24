# -*- coding: utf-8 -*-
"""
Created on Sat May 11 22:43:30 2019

@author: Paulo
"""

import numpy as np
#import dataset
import seaborn as sns
import matplotlib.pyplot as plt

#if not 'data' in globals() or not data:
#    data = dataset.Dataset()
#    data.get_data(subjects=[1,2,3])
#
#emg = [x.emg for x in data.subjects]
#rest = [x.restimulus for x in data.subjects]
#
#emg = [i[:min(len(i),len(j))] for i,j in zip(emg,rest)]
#rest = [j[:min(len(i),len(j))] for i,j in zip(emg,rest)]
#
#emg = np.concatenate(emg,axis=0)
#rest = np.concatenate(rest,axis=0).reshape(-1)
#
#fig, axes = plt.subplots(10,5,figsize=(10,20),sharex=True,sharey=True)
#for i in range(50):
##    print(i)
##    ax = axes[i//5,i%5]
#    cc = np.corrcoef(d[b[1]==i].T)
##    sns.heatmap(cc,ax=ax,vmin=-1,vmax=1)
#    sns.heatmap(cc,vmin=-1,vmax=1)
##    ax.set_title('Movement '+str(i))
#    plt.title('Movement '+str(i))
#    plt.show()
#    plt.tight_layout()

import seaborn as sns
from matplotlib.patches import Rectangle
sns.set()
plt.style.use("seaborn-dark")

start = 260000
end = 275000
sensor = 9

dataset = Dataset()
dataset.get_data(subjects=[1])

from preprocessing import dwt_filter, band_pass
filtered = band_pass(dataset[1].emg)
denoised = dwt_filter(filtered)

data = [dataset[1].emg[start:end,sensor],
        filtered[start:end,sensor],
        denoised[start:end,sensor]]
fig, axes = plt.subplots(3, 1, figsize=(12,8))
limup, limdown = dataset[1].emg[start:end,sensor].max(), dataset[1].emg[start:end,sensor].min()
axes[0].plot(data[0], linewidth=1)
axes[1].plot(data[1], linewidth=1)
axes[2].plot(data[2], linewidth=1)
#axes[3].plot(dataset[1].restimulus[start:end])

axes[0].set_title("Sinal puro")
axes[1].set_title("Sinal filtrado")
axes[2].set_title("Sinal filtrado com denoising")

for ax in axes: ax.set_ylim([limup, limdown])
for i, ax, d in zip(range(3), axes, data):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim([0, end-start])
    ax.grid()
    plt.sca(ax)
    r = Rectangle((1/15, .25), 1/16, .5, transform=ax.transAxes, fill=False, color="r", linewidth=2)
    a = plt.axes([0.7, 0.78-i*.266, .2, .1])
    plt.plot(d[1000:2000])
    a.set_xticklabels([])
    a.set_yticklabels([])
    a.set_ylabel("Zoom")
    a.grid()
    ax.add_patch(r)

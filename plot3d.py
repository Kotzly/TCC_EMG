# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 22:24:50 2019

@author: Paulo
"""
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

def plot3d(x, y, z, size=4, colors='blue', alpha=0.4):

    init_notebook_mode(connected=False)

    trace1 = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=size,
            color=colors,
            colorscale='Viridis',
            opacity=alpha
        )
    )

    data = [trace1]
    layout = go.Layout(
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0
        )
    )
    fig = go.Figure(data=data, layout=layout)
    plot(fig, filename='temp.html')
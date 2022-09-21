# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 13:20:13 2022

@author: Toni Takala
@author: Mika Valkama
"""

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import itertools
from scipy.spatial.distance import cdist
from tqdm import tqdm

from dataparser import prepare_data

CSV_FILE="./data/processed.csv"
PDF_FILE="./analysis.pdf"

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def test_elbow(X, ax, labels, determine_elbow=0.4, max_clusters=10):
    '''
    This elbow plotting is modified from original source:
        https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
    '''
    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}
    K = range(1, max_clusters)
     
    for k in K:
        # Building and fitting the model
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
     
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
        inertias.append(kmeanModel.inertia_)
     
        mapping1[k] = sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]
        mapping2[k] = kmeanModel.inertia_
    
    elbow_point = 0
    elbow_descent_rate = 1.00
    
    for idx in range(0, len(mapping2.items())):                
        elbow_point = idx
        if idx < len(mapping2.items())-1:
            elbow_descent_rate = abs((list(mapping2.values())[idx+1] - list(mapping2.values())[idx]) / (list(mapping2.values())[idx]))
                
            if elbow_descent_rate < determine_elbow: 
                elbow_point = (idx+1)
                break
        
    # create plot
    ax.set_title(f"Elbow method")
    ax.plot(list(range(1,len(mapping2)+1)), list(mapping2.values()))
            
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    return elbow_point


def kmeans(X, X_headers, clusters, ax):
    """
    A reference for a very good and simple PCA applied (dimensionality reduction) K-means clustering algorithm source:
    https://www.askpython.com/python/examples/plot-k-means-clusters-python
    """
    kmeans = KMeans(n_clusters=clusters, random_state=0)
    kmeans.fit(X, y=None, sample_weight=None)
    y_kmeans = kmeans.predict(X)
    centers = kmeans.cluster_centers_

    ax.scatter(X[:,0], X[:,1], s = 50, alpha = 0.35)
    
    ax.set_title(f"Data & clusters, elbow: {str(clusters)}")
    plt.xlabel(X_headers[0])
    plt.ylabel(X_headers[1])
    
    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    ax.scatter(centers[:,0], centers[:,1], c = np.unique(y_kmeans), s = 200, alpha = 1, cmap = 'rainbow')    


def confidence_ellipse(x, y, ax, n_std=1.0, facecolor='none', **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=facecolor, **kwargs)

    # Calculate the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculate the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    
    return ax.add_patch(ellipse)


def test_confidence_graph(X, X_headers, ax):
    ax.scatter(X[:,0], X[:,1], s = 1)

    ax.axvline(c='gray', lw=1)
    ax.axhline(c='gray', lw=1)           
    
    confidence_ellipse(X[:,0], X[:,1], ax, n_std = 1, label=r'$1\sigma$', edgecolor='firebrick')
    confidence_ellipse(X[:,0], X[:,1], ax, n_std = 2, label=r'$2\sigma$', edgecolor='fuchsia', linestyle='--')
    confidence_ellipse(X[:,0], X[:,1], ax, n_std = 3, label=r'$3\sigma$', edgecolor='blue', linestyle=':')

    ax.set_title('Different standard deviations')
    ax.legend()
    
    ax.set_xlim([-0.1, 1.5])
    ax.set_ylim([-0.1, 1.5])
    
    plt.xlabel(X_headers[0])
    plt.ylabel(X_headers[1])
    

def generate_analysis(df, df_headers):
    '''
    Test Kmeans elbow and apply elbow point to the data clustering
    (merge datapoints column-wise before applying to kmeans...)
    '''
    
    with PdfPages(PDF_FILE) as pdf:
        allIndices = list(range(0, len(df_headers)))

        # need this to figure out how many combinations we have
        # to get functioning progress bar
        combs = itertools.combinations(allIndices, 2)
        count = 0
        for p in combs:
            count += 1

        combs = itertools.combinations(allIndices, 2)

        t = tqdm(total=count)

        for p in combs:
            t.update()
            a = df.iloc[:,p[0]].to_numpy()
            b = df.iloc[:,p[1]].to_numpy()

            if len(a) == 0 or len(b) == 0:
                continue

            Xlabels = [df_headers[p[0]], df_headers[p[1]]]
            Xpair = [a, b]
            value_pair_array = np.concatenate((Xpair[0].reshape(-1,1), Xpair[1].reshape(-1,1)), axis=1)
            
            fig = plt.figure(figsize=(11.69, 8.27), constrained_layout=True) # A4 in inches
            gs = fig.add_gridspec(2, 2)
            
            # elbow method
            ax = fig.add_subplot(gs[0, 0])
            kmeans_elbow_point = test_elbow(np.array(value_pair_array), determine_elbow=0.43, ax=ax, labels=Xlabels)

            # Confidence graph
            ax = fig.add_subplot(gs[:, 1])
            test_confidence_graph(value_pair_array, Xlabels, ax)

            # K-means
            ax = fig.add_subplot(gs[1, 0])
            kmeans(value_pair_array, Xlabels, kmeans_elbow_point, ax)
            
            fig.tight_layout()

            pdf.savefig()
            plt.close()
        
        t.close()


dfVals, dfHeaders = prepare_data(CSV_FILE, True)
generate_analysis(dfVals, dfHeaders)
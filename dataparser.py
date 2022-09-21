# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:21:58 2022

@author: Toni Takala
"""

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import pandas as pd
import numpy as np
from datetime import datetime

CSV_FILE="./data/processed.csv"

ALL_FIELDS = ['valA', 'valB', 'valC', 'valD', 'dateA', 'dtA', 'dtB', 'dtC', 'dtD', 'dtE', 'dtF', 'dateB', 'dtG', 'dtH', 'dtI', 'dtJ', 'valE', 'valF', 'valG', 'valH', 'valI', 'valJ', 'valH', 'valI', 'valJ', 'valK', 'valL', 'valM', 'valN', 'valO', 'calc_DtBmDtA', 'calc_DtCmDtB', 'calc_DtDmDtC', 'calc_DtFmDtD', 'calc_DtHmDtG', 'calc_DtImDtG', 'calc_DtJmDtG', 'calc_DtImDtH', 'calc_DtJmDtI']
VALUE_FIELDS = ['valA', 'valB', 'valC', 'valD', 'valE', 'valF', 'valG', 'valH', 'valI', 'valJ', 'valH', 'valI', 'valJ', 'valK', 'valL', 'valM', 'valN', 'valO', 'calc_DtBmDtA', 'calc_DtCmDtB', 'calc_DtDmDtC', 'calc_DtFmDtD', 'calc_DtHmDtG', 'calc_DtImDtG', 'calc_DtJmDtG', 'calc_DtImDtH', 'calc_DtJmDtI']    

 # Helper method to find difference between arrays
def find_missing(a, b):
    for x in b:
        a.remove(x)
    return a
    
DATE_FIELDS = find_missing(ALL_FIELDS, VALUE_FIELDS)

DEBUG = False

def fetch_value_data_from_df(df):
    dfVals = df[VALUE_FIELDS].astype(float)
    dfHeaders = list(dfVals)
    
    return dfVals, dfHeaders

    
def normalize_data(df):
    '''
    Perform data scaling by normalization. By default we want positive range.
    Min-Max normalization algorithm: 
        Xn = (X - Xminimum) / ( Xmaximum - Xminimum)  
    '''
    return (df - df.min()) / (df.max() - df.min())


def prepare_data(csv_file=CSV_FILE, normalize=True):
    # Read and parse a dataframe from CSV-file
    dci = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    df = pd.read_csv(csv_file, delimiter=";", parse_dates=dci)
    
    # Fetch values and headers from dataframe
    dfVals, dfHeaders = fetch_value_data_from_df(df)

    # Perform scaling with normalization (or standardization...)
    if normalize:
        dfValsScaled = normalize_data(dfVals)

        return dfValsScaled, dfHeaders
    else:
        return dfVals, dfHeaders


if __name__ == "__main__": 
    dfVals, dfHeaders = prepare_data(normalize=False)

    if DEBUG: 
        print(dfHeaders)
        print(dfVals.dtypes)
        print(dfVals.head())

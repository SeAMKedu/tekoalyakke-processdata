# -*- coding: utf-8 -*-
"""

@author: Mika Valkama
"""

import pandas as pd
from ctgan import CTGANSynthesizer
from dataparser import prepare_data

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

pd.options.mode.chained_assignment = None  # default='warn'

CSV_FILE="./data/processed.csv"
OUT_FILE="./data/gan.csv"

dfVals, dfHeaders = prepare_data(CSV_FILE, normalize=False)

disdim = (32, 16, 8)
gendim = (16, 32, 32, 8)
epochs = 100
batch_size = 32
verbose = True
ctgan = CTGANSynthesizer(epochs=200, verbose=True)
ctgan.fit(dfVals.values)

df = pd.DataFrame(columns=dfHeaders)

print(f"Generating new data")

newdata = ctgan.sample(10000)

print(f"Storing in df")
# this is not the most efficient way of doing this...
for s in newdata:
    df.loc[len(df)] = s

df.reset_index(drop=True)

df.to_csv(OUT_FILE, sep=";")

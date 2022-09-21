"""
Created on Wed May 18 12:24:59 2022

@author: Toni Takala
@author: Mika Valkama
"""

# matplotlib 3.6.0 has a new annoying warning, so hiding it for now,
# if you use newer, comment two lines below and fix the issue
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import time
from datetime import timedelta

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from dataparser import prepare_data, normalize_data

print(tf.config.list_physical_devices())

# Suppresses numpy's scientific notation (eg. 1.05245e+02)
np.set_printoptions(suppress=True)

def train_model(training, labels, epochs=900):
    print("Training model")

    start = time.time()

    model = setup_model(training, labels)
    
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    
    training = np.asarray(normalize_data(training))

    history = model.fit(
        training,
        labels,
        validation_split    = 0.1,
        epochs              = epochs,
        batch_size          = int(len(training) * 0.25),
        verbose             = 0,
        callbacks           = [reduce_lr, earlystop],
        shuffle             = True
    )
    
    print(f"Training took (H:MM:SS) {str(timedelta(seconds=(time.time() - start)))}\n\n")

    return model, history

def setup_model(X, labels):
    # Every dataset will most probably need to create it's own NN architecture.
    # This fits to the specified data.
    num_input = X.shape[1]
    num_classes = len(labels[0])

    model = tf.keras.Sequential([
        tf.keras.layers.Input((num_input,), name='feature'),
        tf.keras.layers.Dense(96, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(96, activation='relu'),
        #tf.keras.layers.GaussianNoise(.07),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(58, activation='relu'),
        tf.keras.layers.Dense(36, activation='relu'),
        tf.keras.layers.Dense(9, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    ftrl_optimizer = tf.keras.optimizers.Ftrl(learning_rate = 0.001, learning_rate_power = -0.5, name = "Ftrl")
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, epsilon = 1e-05, amsgrad = False, name = 'Adam')

    model.compile(
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False),
        optimizer = adam_optimizer,
        metrics = ["accuracy", "categorical_accuracy"]
    )

    return model

def test_model(model, testing, dfVals, column, labels, human_readable):
    print(f"Evaluate on test data, rows {len(testing)}")

    results = model.evaluate(testing, labels, batch_size=8)
    print("test loss, test acc:", results)

    print("\nTesting predictions")

    testingnp = np.asarray(normalize_data(testing))[:10]
    predictions = model.predict(testingnp)

    print("\nValue   Label  Prediction Correct")
    for i in range(len(predictions)):
        pred = predictions[i]

        poh = [0] * len(labels[0])
        lidx = np.argmax(labels[i])
        pidx = np.argmax(pred)
        poh[pidx] = 1

        #print(f"{testing.iloc[i]['valA']}\t{labels[i]} - {poh}\t{np.array_equal(labels[i], poh)}\t{human_readable[idx]}")
        x = testing.index[i]
        result = dfVals.loc[x][column]
        print(f"{result:>5.1f}\t{human_readable[lidx]:6s} {human_readable[pidx]:10s} {(np.array_equal(labels[i], poh))}")

    print()

def plot_training_results(history, name):
    fig = plt.figure(figsize=[15.0, 5.0])
    gs = fig.add_gridspec(1, 3)
    
    ax = fig.add_subplot(gs[0])
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    ax = fig.add_subplot(gs[1])
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')

    ax = fig.add_subplot(gs[2])
    ax.plot(history.history['categorical_accuracy'])
    ax.plot(history.history['val_categorical_accuracy'])
    plt.title('Categorical accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='upper left')

    plt.savefig(f"./img/{name}_training_result.png")
    plt.close()


def split_to_classes(df, boundaries, column):
    def classify(value):
        for b in reversed(range(len(boundaries))):
            if value >= boundaries[b]:
                return b
        
    df["cls"] = df.apply(lambda row: classify(row[column]), axis=1)

    return df
    

def preprocess(dfVals, boundaries, column, excluded_columns):
    # classify each row based on selected column
    dfVals = split_to_classes(dfVals, boundaries, column)

    # one hot encoding for classification
    dfVals = pd.get_dummies(dfVals, columns=["cls"])

    temp_columns = []
    if 'cls_0' in dfVals.columns:
        for i in range(len(boundaries)):
            temp_columns.append(f"cls_{i}")
    else:
        for i in range(len(boundaries)):
            temp_columns.append(f"cls_{i}.0")   # not sure why the .0 appears on column name some times

    # print out number of items in bins
    print(f"{dfVals[temp_columns].sum()}")

    # combine created columns to single array and drop the intermediate columns
    dfVals["cls"] = dfVals[temp_columns].values.tolist()
    dfVals.drop(columns=temp_columns, inplace=True)

    # exclude columns selected columns from training and testing material
    cleaned = dfVals.drop(columns=excluded_columns)
    
    # split for training and testing sets
    train, test = train_test_split(cleaned, test_size=0.05)

    cls_train = np.stack(train['cls'].to_numpy())
    cls_test = np.stack(test['cls'].to_numpy())

    # remove classification column from training and testing
    train.drop(columns=["cls"], inplace=True)
    test.drop(columns=["cls"], inplace=True)

    return dfVals, train, test, cls_train, cls_test


def define_boundaries(dfVals, column, quantiles):
    boundaries = []

    for q in quantiles:
        boundaries.append(dfVals[column].quantile(q))

    print(f"Calculated boundaries {boundaries}")

    return boundaries


def process(dfVals, name):
    # Split into bins for labeling accordingly
    human_readable = ["Bad", "Ok", "Good", "Oh my"]
    #bin_boundaries = define_boundaries(dfVals, "valA", [0.1, 0.3, 0.7, 0.9])
    bin_boundaries = [600,925,975,1100] # manually picked due to so little data

    # which column is used for classification
    classification_column = "valD"

    # don't use these in training. 
    # they contain values that are too directly related to classification measurement
    excluded_columns = ["valA", "valB", "valC", "valD", "valG"]

    dfVals, train, test, cls_train, cls_test = preprocess(dfVals, bin_boundaries, classification_column, excluded_columns)

    model, history = train_model(train, cls_train, epochs=400)

    tf.keras.utils.plot_model(model, to_file="./img/model.png", show_shapes=True)

    plot_training_results(history, name)

    test_model(model, test, dfVals, 'valD', cls_test, human_readable)


CSV_FILE="./data/processed.csv"
GAN_FILE="./data/gan.csv"

dfVals, dfHeaders = prepare_data(CSV_FILE, normalize=False)
process(dfVals, "just_real")

dfVals, dfHeaders = prepare_data(GAN_FILE, normalize=False)
process(dfVals, "gan")

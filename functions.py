import requests
import os
import sys
from datetime import datetime
from datetime import timedelta
import time
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import keras
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

import ta
import pytz
import tensorboard
from statistics import mean
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


periods = {
    '60': '1m',  # 1 Minute
    '180': '3m', # 3 Minutes
    '300': '5m',
    '900': '15m',
    '1800': '30m',
    '3600': '1h', # 1 Hour
    '7200': '2h',
    '14400': '4h',
    '21600': '6h',
    '43200': '12h',
    '86400': '1d', # 1 Day
    '259200': '3d',
    '604800': '1w', # 1 Week
}

def disable_console():
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    
def enable_console():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__

# Normalizing/Scaling the DF
def normalize_df(df):
    
    # Scale fitting the close prices separately for inverse_transformations purposes later
    # close_scaler = RobustScaler()

    # close_scaler.fit(df[['Close']])
    
    scaler = preprocessing.StandardScaler()
    
    scaler.fit(df[['Close']])

    # scaler = RobustScaler()
    
    # df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
    
    df = pd.DataFrame(scaler.transform(df), columns=df.columns, index=df.index)
    
    return df, scaler

#give normalized df
def get_next_predicted_price(df, n_per_in, n_features, model, close_scaler):

    yhat = model.predict(np.array(df.tail(n_per_in)).reshape(1, n_per_in, n_features))
    
    yhat = close_scaler.inverse_transform(yhat)[0]

    # Creating a DF of the predicted prices
    
    # preds = pd.DataFrame(yhat, 
                         # index=pd.date_range(start=df.index[-1]+timedelta(hours=1), 
                                             # periods=len(yhat), 
                                             # freq="H"), 
                         # columns=(['Close']))
                         
    yhat2 = model.predict(np.array(df.head(len(df)-1).tail(n_per_in)).reshape(1, n_per_in, n_features))
    
    yhat2 = close_scaler.inverse_transform(yhat2)[0]
                         
    return yhat[0], yhat2[0]
    
def plot_predictions_actual_prev(predictions, actual, prev):

    predictions.to_csv('predictions.csv')
    actual.to_csv('actual.csv')
    
    # Plotting
    plt.figure(figsize=(16,6))

    # Plotting those predictions
    plt.plot(predictions, label='Predicted')

    # Plotting the actual values
    plt.plot(actual, label='Actual')
    
    # Plotting the actual values
    plt.plot(prev, label='PrevPredictions')

    plt.title(f"Predicted vs Actual Closing Prices")
    plt.ylabel("Price")
    plt.legend()
    plt.show()   

def plot_predictions_actual(predictions, actual):

    predictions.to_csv('predictions.csv')
    actual.to_csv('actual.csv')
    
    # Plotting
    plt.figure(figsize=(16,6))

    # Plotting those predictions
    plt.plot(predictions, label='Predicted')

    # Plotting the actual values
    plt.plot(actual, label='Actual')

    plt.title(f"Predicted vs Actual Closing Prices")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def transform_back(df, close_scaler):
    actual = pd.DataFrame(close_scaler.inverse_transform(df[["Close"]]), 
                          index=df.index, 
                          columns=(['Close']))
                          
    return actual

def train_model(df, epochs, batch_size, model, n_per_in, n_per_out, loadWeights=False, earlyStop=True, saveWeights=True, pathToWeights="models/cp.ckpt", displayResults=False, close_scaler=None):

    
        
    # Splitting the data into appropriate sequences
    X, y = split_sequence(df.to_numpy(), n_per_in, df, close_scaler)
    
    callbacks = []
    
    if(earlyStop):
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=9)
        callbacks.append(es_callback)
        
    if(saveWeights):
        checkpoint_dir = os.path.dirname(pathToWeights)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=pathToWeights,
                                                         save_weights_only=True,
                                                         verbose=1)
        callbacks.append(cp_callback)
    
    
    
    
    if(loadWeights):
        print("Shape of X: " + str(X.shape))
        print("Shape of y: " + str(y.shape))
        print("one example training sample: ")
        
        # if(close_scaler != None):
            # for i in range(0, 5*n_per_out, n_per_out):
                # test = pd.DataFrame(X[int(i/n_per_out)], 
                                  # index=df.index[i:n_per_in+i], 
                                  # columns=df.columns)
                                  
                # goodtest = pd.DataFrame(close_scaler.inverse_transform(test[["Close"]]), 
                              # index=test.index, 
                              # columns=(['Close']))
                                  
                # goodtest.to_csv('test X ' + str(int(i/n_per_out)) + '.csv')
                
                # print(str(df.index[n_per_in + i]) + " next price: " + str(close_scaler.inverse_transform(np.array(y[int(i/n_per_out)]).reshape(-1, 1))))
               
        model.load_weights(pathToWeights)
        
        # if(close_scaler != None):
        
            # for i in range(0, 5*n_per_out, n_per_out):
            
                # print("Shape of Xout: " + str(X[int(i/n_per_out)].reshape(1, n_per_in, 5).shape))
            
                # yTest = model.predict(X[int(i/n_per_out)].reshape(1, n_per_in, 5))
                
                # print("Predicted: " + str(df.index[n_per_in + i]) + " next price: " + str(close_scaler.inverse_transform(np.array(yTest).reshape(-1, 1))))
    else:
       
                          
        if(len(callbacks) > 0):
            res = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.10, verbose=1, callbacks=callbacks)
        else:
            res = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.10, verbose=1)
            
        # model.load_weights("models/cp.ckpt")    
        
        
        
    
        if(displayResults):
            visualize_training_results(res)

def train_model_binary(df, epochs, batch_size, model, n_per_in, n_per_out, loadWeights=False, earlyStop=True, saveWeights=True, pathToWeights="models/cp.ckpt", displayResults=False, close_scaler=None):

    
        
    # Splitting the data into appropriate sequences
    X, y = split_sequence_binary(df.to_numpy(), n_per_in, df, close_scaler)
    
    callbacks = []
    
    if(earlyStop):
        es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=epochs, restore_best_weights=True)
        callbacks.append(es_callback)
        
    if(saveWeights):
        checkpoint_dir = os.path.dirname(pathToWeights)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=pathToWeights,
                                                         save_weights_only=True,
                                                         verbose=1)
        callbacks.append(cp_callback)
        
    # cp_board = tf.keras.callbacks.TensorBoard(
        # log_dir="logs",
        # histogram_freq=1,  # How often to log histogram visualizations
        # embeddings_freq=1,  # How often to log embedding visualizations
        # update_freq="batch",
    # )
    
    # callbacks.append(cp_board)
    
    print("Shape of X: " + str(X.shape))
    print("Shape of y: " + str(y.shape))
    if(loadWeights):
        print("Shape of X: " + str(X.shape))
        print("Shape of y: " + str(y.shape))
        print("one example training sample: ")
        
        # if(close_scaler != None):
            # for i in range(0, 5*n_per_out, n_per_out):
                # test = pd.DataFrame(X[int(i/n_per_out)], 
                                  # index=df.index[i:n_per_in+i], 
                                  # columns=df.columns)
                                  
                # goodtest = pd.DataFrame(close_scaler.inverse_transform(test[["Close"]]), 
                              # index=test.index, 
                              # columns=(['Close']))
                                  
                # goodtest.to_csv('test X ' + str(int(i/n_per_out)) + '.csv')
                
                # print(str(df.index[n_per_in + i]) + " next price: " + str(close_scaler.inverse_transform(np.array(y[int(i/n_per_out)]).reshape(-1, 1))))
               
        model.load_weights(pathToWeights)
        
        # if(close_scaler != None):
        
            # for i in range(0, 5*n_per_out, n_per_out):
            
                # print("Shape of Xout: " + str(X[int(i/n_per_out)].reshape(1, n_per_in, 5).shape))
            
                # yTest = model.predict(X[int(i/n_per_out)].reshape(1, n_per_in, 5))
                
                # print("Predicted: " + str(df.index[n_per_in + i]) + " next price: " + str(close_scaler.inverse_transform(np.array(yTest).reshape(-1, 1))))
    else:
       
                          
        if(len(callbacks) > 0):
            res = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.10, verbose=1, callbacks=callbacks)
        else:
            res = model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.10, verbose=1)
            
        # model.load_weights("models/cp.ckpt")    
        
        
        
    
        if(displayResults):
            visualize_training_results(res)

## How many periods looking back to learn
# n_per_in  = 48
## How many periods to predict
# n_per_out = 1
## Features 
# n_features = df.shape[1]
def create_model(n_per_in, n_per_out, n_features):

    ## Creating the NN

    #dropout rate

    d = 0.3

    # Instatiating the model

    model = Sequential()

    model.add(LSTM(256, input_shape=(n_per_in, n_features), return_sequences=True))
    model.add(Dropout(d))

    model.add(LSTM(256, input_shape=(n_per_in, n_features), return_sequences=False))
    model.add(Dropout(d))

    model.add(Dense(32,kernel_initializer="uniform",activation='relu'))
    # model.add(Dense(n_per_out,kernel_initializer="uniform",activation='linear'))
    model.add(Dense(n_per_out))
    # Model summary
    # model.summary()

    # Compiling the data with selected specifications
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    return model
    
def create_model_binary(n_per_in, n_per_out, n_features):

    ## Creating the NN

    #dropout rate

    d = 0.3

    # Instatiating the model

    model = Sequential()

    model.add(LSTM(256, input_shape=(n_per_in, n_features), return_sequences=True))
    model.add(Dropout(d))
    
    # model.add(LSTM(64, input_shape=(n_per_in, n_features), return_sequences=False))
    # model.add(LSTM(256, input_shape=(n_per_in, n_features), return_sequences=False))
    # model.add(Dropout(d))

    # model.add(Dense(32,kernel_initializer="uniform",activation='relu'))
    # model.add(Dense(n_per_out,kernel_initializer="uniform",activation='linear'))
    model.add(Dense(1, activation='sigmoid'))
    # Model summary
    # model.summary()

    # Compiling the data with selected specifications
    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    
    return model

#market = bitfinex
#symbol = btcusd
#period = 3600 (1h)
#historical = "bitcoin1h.csv"
def get_data(period, market, symbol, percentTestData=None, saveData=True, historical=None, size=None):

    resp = requests.get('https://api.cryptowat.ch/markets/' + market + '/' + symbol + '/ohlc', params={
        'periods': period
    })

    data = resp.json()

    newdf = pd.DataFrame(data['result'][period], columns=[
        'Date', 'Open', 'High', 'Low', 'Close', 'NA', 'Volume USD'
    ])

    newdf['Date'] = pd.to_datetime(newdf.Date, unit='s')

    newdf.drop(columns=['NA'], inplace=True)

    newdf.set_index('Date', inplace=True)
        
    df = None
        
    if(historical is not None):
        df = pd.read_csv(historical)
        
        df.drop(columns=['Symbol', 'Volume BTC'], inplace=True)

        df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d %I-%p")

        df.set_index('Date', inplace=True)

        df = df[::-1]

        dates = df.index.tolist()
        newdates = newdf.index.tolist()

        # Adding last len(newdates)-newdates.index(dates[-1]) - 1) items

        df = df.append(newdf.tail(len(newdates)-newdates.index(dates[-1]) - 1))

        #df.update(newdf)
        
       


        #df = df.tail(1000)

        df = df[['Close', 'Open', 'High', 'Low', 'Volume USD']]

        #df.to_csv('Combined Bitcoin ' + periods[period] + ' data.csv')
    else:
        df = newdf
    
    print("Shape df: " + str(df.shape))    
        
        
    df = df[['Close', 'Open', 'High', 'Low', 'Volume USD']]
    
    if(size is not None):
        df = df.tail(size)
    
    # Dropping any NaNs
    df.dropna(inplace=True)
    
    print("Shape df after dropping na: " + str(df.shape))
    
    # print("Len fulldf after na: " + str(len(df)))
    
    
    
    testDf = []
    
    fullDf = pd.DataFrame(df)
    
    if(percentTestData is not None):
        testDfPercent = percentTestData;

        testDfSize = int(testDfPercent * len(df))

        testDf = df.tail(testDfSize)

        df = df.head(len(df)-testDfSize)
    
    if(saveData):
        newdf.to_csv(market + ' ' + symbol + ' ' + periods[period] + ' data.csv')
        df.to_csv('Combined ' + market + ' ' + symbol + ' ' + periods[period] + ' data.csv')
     
    # print("Len fulldf end: " + str(len(fullDf)))
    
    return fullDf, testDf, df, df.shape[1]

def calc_gain(actual, pred, startAmount, tradeCost, n_per_in, n_per_out, noLoss, cashOut, conserveBuy, opposite=False,showInfo=True):
    """
    Calculates the potential gain if the model is used to trade currency
    
    """
    
    actual = actual.tail(len(pred))
    
    actualClose = actual["Close"].tolist()
    predClose = pred["Close"].tolist()
    
    # print("Actual Close: " + str(actualClose))
    # print("Pred Close: " + str(predClose))
    
    # print("Len actual: " + str(len(actualClose)) + " Len Pred: " + str(len(predClose)))
    
    #actualClose = actualClose[len(actualClose)-len(predClose):]
    
    if(showInfo == False):
        disable_console()
        
    cashOutAmount = 0
    
    lastAction = None
    sellPrice = None
    startPrice = actualClose[n_per_in-1]
    buyPrice = actualClose[n_per_in-1]
    currentAmount = startAmount
    
    for i in range(n_per_in-1, len(actualClose)-n_per_out, n_per_out):
        predList = []
        #print("i: " + str(i) + " predI: " + str(predClose[i]))
        for x in range(i + 1, n_per_out + i + 1):
            predList.append(predClose[x])
            
        # print("Mean for " + str(i+1) + ": " + str(mean(predList)) + " currBTC for " + str(i) + ": " + str(actualClose[i]))
        print("Current time: " + str(actual.index[i]))
        print("Current Predicted price: " + str(predClose[i]))
        print("Next Predicted price: " + str(mean(predList)))
        print("Current price: " + str(actualClose[i]))
        print("Next price: " + str(actualClose[i+1]))
        
        
        if(lastAction == "Buy" and cashOut and buyPrice < actualClose[i]):
            newAmount = (((actualClose[i] - buyPrice) / buyPrice) * currentAmount) + currentAmount - (currentAmount * tradeCost)
            print("CashOut Policy On!")
            print("Selling Debug:")
            print("Buy Price: " + str(buyPrice))
            print("Selling Price: "+ str(actualClose[i]))
            print("Cashing out for " + str(newAmount-startAmount))
            print()
            buyPrice = actualClose[i]
            cashOutAmount += newAmount-startAmount
            currentAmount = startAmount
            continue
        if(mean(predList) <= predClose[i]):
            if(opposite):
                if(lastAction == "Buy"):
                    print("Holding " + str(currentAmount) + " at BTC/USD: " + str(actualClose[i]))
                if(lastAction == "Sell" or lastAction == None):
                    if(conserveBuy and actualClose[i] > startPrice):
                        print("Conservebuy on!")
                        print("Start price: " + str(startPrice))
                        print("Current price: " + str(actualClose[i]))
                    else:
                        print("Buying BTC with " + str(currentAmount) + " at BTC/USD: " + str(actualClose[i]))
                        buyPrice = actualClose[i]
                        lastAction = "Buy"
            else:
                if(lastAction == "Sell"):
                    print("Holding " + str(currentAmount) + " at BTC/USD: " + str(actualClose[i]))
                if(lastAction == "Buy" or lastAction == None):
                    print("Selling Debug:")
                    print("Buy Price: " + str(buyPrice))
                    print("Selling Price: "+ str(actualClose[i]))
                    print("Percentage difference: " + str((actualClose[i] - buyPrice) / buyPrice))
                    if(abs((actualClose[i] - buyPrice) / buyPrice) < 2 * tradeCost):
                        print("Percentage below 2 * tradeCost threshold, not selling")
                    else:
                        if(noLoss == True and actualClose[i] < buyPrice):
                            print("No loss is on, not selling")
                        else:
                            newAmount = (((actualClose[i] - buyPrice) / buyPrice) * currentAmount) + currentAmount - (currentAmount * tradeCost)
                            print("Selling " + str(currentAmount) + " at BTC/USD: " + str(actualClose[i]) + " for a gain of " + str(newAmount - currentAmount))
                            currentAmount = newAmount
                            #sellPrice = actualClose[i]
                            lastAction = "Sell"
                    
               
        if(mean(predList) > predClose[i]):
            if(opposite):
                if(lastAction == "Sell"):
                    print("Holding " + str(currentAmount) + " at BTC/USD: " + str(actualClose[i]))
                if(lastAction == "Buy" or lastAction == None):
                    print("Selling Debug:")
                    print("Buy Price: " + str(buyPrice))
                    print("Selling Price: "+ str(actualClose[i]))
                    print("Percentage difference: " + str((actualClose[i] - buyPrice) / buyPrice))
                    if(abs((actualClose[i] - buyPrice) / buyPrice) < 2 * tradeCost):
                        print("Percentage below 2 * tradeCost threshold, not selling")
                    else:
                        if(noLoss == True and actualClose[i] < buyPrice):
                            print("No loss is on, not selling")
                        else:
                            newAmount = (((actualClose[i] - buyPrice) / buyPrice) * currentAmount) + currentAmount - (currentAmount * tradeCost)
                            print("Selling " + str(currentAmount) + " at BTC/USD: " + str(actualClose[i]) + " for a gain of " + str(newAmount - currentAmount))
                            currentAmount = newAmount
                            #sellPrice = actualClose[i]
                            lastAction = "Sell"
            else:
                if(lastAction == "Buy"):
                    print("Holding " + str(currentAmount) + " at BTC/USD: " + str(actualClose[i]))
                if(lastAction == "Sell" or lastAction == None):
                    if(conserveBuy and actualClose[i] > startPrice):
                        print("Conservebuy on!")
                        print("Start price: " + str(startPrice))
                        print("Current price: " + str(actualClose[i]))
                    else:
                        print("Buying BTC with " + str(currentAmount) + " at BTC/USD: " + str(actualClose[i]))
                        buyPrice = actualClose[i]
                        lastAction = "Buy"
        print()
        
    lastIndex = len(actualClose)-n_per_out-1
      
    enable_console()
    
    currentAmount += cashOutAmount
    
    print("\n\nFinal Trade\n\n")
 
    if(lastAction == "Sell"):
        print("Holding " + str(currentAmount) + " at BTC/USD: " + str(actualClose[lastIndex]))
    if(lastAction == "Buy" or lastAction == None):
        newAmount = (((actualClose[lastIndex] - buyPrice) / buyPrice) * currentAmount) + currentAmount - (currentAmount * tradeCost)
        print("Selling " + str(currentAmount) + " at BTC/USD: " + str(actualClose[lastIndex]) + " for a gain of " + str(newAmount - currentAmount))
        currentAmount = newAmount
        #sellPrice = actualClose[i]
        lastAction = "Sell"  
    print("\n\nFinal Statistics: \n\n")
    print("Start time: " + str(actual.index[n_per_in-1]))
    print("End time: " + str(actual.index[-1]))
    print("Number of hours elasped: " + str(len(actual)-n_per_in))
    
    
    print()
    print("BTC start: " + str(actualClose[n_per_in-1]))
    print("BTC end: " + str(actualClose[-1]))
    print("Percentage change of BTC: " + str((actualClose[-1] - actualClose[n_per_in-1]) / actualClose[n_per_in-1]))
    print("Percentage change of Amount: " + str((currentAmount - startAmount) / startAmount))
    
    print()
    print("Start Amount: " + str(startAmount))
    print("Current Amount: " + str(currentAmount))
    print("Gain/Loss: " + str(currentAmount - startAmount))
    
    return currentAmount        
        
def split_sequence_binary(seq, n_steps_in, df, close_scaler, n_steps_out=1):
    """
    Splits the multivariate time sequence
    """
    
    # Creating a list for both variables
    X, y = [], []
    
    
    for i in range(len(seq)):
        
        # Finding the end of the current sequence
        end = i + n_steps_in
        out_end = end + n_steps_out
        
        # Breaking out of the loop if we have exceeded the dataset's length
        if out_end > len(seq):
            break
        
        # Splitting the sequences into: x = past prices and indicators, y = prices ahead
        seq_x = seq[i:end, :]
        seq_test = seq[end-1:out_end, 0]
        seq_y = [0]
        if(seq_test[1]-seq_test[0] > 0):
            seq_y = [1]
            
        elif(seq_test[1]-seq_test[0] <= 0):
            seq_y = [0]
            
        if(len(y) < 5):
        
        
            xDf = pd.DataFrame(np.array(close_scaler.inverse_transform(seq_x)).reshape(n_steps_in, -1),
                               index=pd.date_range(start=df.index[i], 
                                                   periods=len(seq_x), 
                                                   freq="H"),
                               columns=df.columns)
                               
            yDf = pd.DataFrame(np.array(seq_y).reshape(-1, 1),
                               index=pd.date_range(start=df.index[end], 
                                                   periods=len(seq_y), 
                                                   freq="H"),
                               columns=['Close'])
            xDf.to_csv('xdf ' + str(i) + '.csv')
            yDf.to_csv('ydf ' + str(i) + '.csv')
        
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)        

def split_sequence(seq, n_steps_in, df, close_scaler, n_steps_out=1):
    """
    Splits the multivariate time sequence
    """
    
    # Creating a list for both variables
    X, y = [], []
    
    
    for i in range(len(seq)):
        
        # Finding the end of the current sequence
        end = i + n_steps_in
        out_end = end + n_steps_out
        
        # Breaking out of the loop if we have exceeded the dataset's length
        if out_end > len(seq):
            break
        
        # Splitting the sequences into: x = past prices and indicators, y = prices ahead
        seq_x, seq_y = seq[i:end, :], seq[end:out_end, 0]
        
        if(len(y) < 5):
        
        
            xDf = pd.DataFrame(np.array(close_scaler.inverse_transform(seq_x)).reshape(n_steps_in, -1),
                               index=pd.date_range(start=df.index[i], 
                                                   periods=len(seq_x), 
                                                   freq="H"),
                               columns=df.columns)
                               
            yDf = pd.DataFrame(close_scaler.inverse_transform(np.array(seq_y).reshape(-1, 1)),
                               index=pd.date_range(start=df.index[i], 
                                                   periods=len(seq_y), 
                                                   freq="H"),
                               columns=['Close'])
            xDf.to_csv('xdf ' + str(i) + '.csv')
            yDf.to_csv('ydf ' + str(i) + '.csv')
        
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)
    
def visualize_training_results(results):
    """
    Plots the loss and accuracy for the training and testing data
    """
    history = results.history
    plt.figure(figsize=(16,5))
    plt.plot(history['val_loss'])
    plt.plot(history['loss'])
    plt.legend(['val_loss', 'loss'])
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    
    plt.figure(figsize=(16,5))
    plt.plot(history['val_accuracy'])
    plt.plot(history['accuracy'])
    plt.legend(['val_accuracy', 'accuracy'])
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
          
def get_predictions(df, n_per_in, n_per_out, n_features, close_scaler, model, timeRemaining, numberPredictions=None):
    """
    Runs a 'For' loop to iterate through the length of the DF and create predicted values for every stated interval
    Returns a DF containing the predicted values for the model with the corresponding index values based on a business day frequency
    """
    
    #lengthPredictions = int(.8 * len(df)) #amount to chop 
    
    if(numberPredictions is not None):
        if(numberPredictions > len(df)):
            numberPredictions = len(df)
            
        shortDf = df.tail(numberPredictions)
    else:
        shortDf = df
        
        
    shortDf.to_csv("Short df")
        
    # Creating an empty DF to store the predictions
    predictions = pd.DataFrame(index=shortDf.index, columns=['Close'])
    prevPredictions = pd.DataFrame(index=shortDf.index, columns=['Close'])
    
    times = []
    
    avg = 1
    
    numberOfAvg = 0

    for i in range(0, len(shortDf)-n_per_in, n_per_out):
    
        start = time.time()
        # Creating rolling intervals to predict off of
        
        #print("I: " + str(i) + " Range: [" + str(i) + ", " + str(i+n_per_in) + "]")
        
        x = shortDf[i:i+n_per_in]
        
        # if(i > len(shortDf)-10-n_per_in):
            # print("xlen: " + str(len(x)))
            # print("start predict time: " + str(x.index[0]))
            # print("end predict time: " + str(x.tail(1).index[0]))
            # xFix = pd.DataFrame(close_scaler.inverse_transform(x[["Close"]]), 
                          # index=x.index, 
                          # columns=(['Close']))
            # xFix.to_csv('dataForPredict ' + str(i) + '.csv')

        # Predicting using rolling intervals
        yhat = model.predict(np.array(x).reshape(1, n_per_in, n_features))

        # Transforming values back to their normal prices
        # yhatNext = close_scaler.inverse_transform(yhat)[0][1]
        
        yhatNext = close_scaler.inverse_transform(yhat)[0][0]
        
        #print("YHat: " + str(yhat))

        # DF to store the values and append later, frequency uses business days
        pred_df = pd.DataFrame(yhatNext, 
                               index=pd.date_range(start=shortDf.index[i+n_per_in], 
                                                   periods=1, 
                                                   freq="H"),
                               columns=['Close'])
                               
        # if(i > len(shortDf)-10-n_per_in):
            # print("predicted time: " + str(pred_df.index[0]))
            # pred_df.to_csv('predictedNext ' + str(i) + '.csv')

        # Updating the predictions DF
        predictions.update(pred_df)
        
        # pred_dfPrev = pd.DataFrame(yhatPrev, 
                               # index=pd.date_range(start=shortDf.index[i+n_per_in-1], 
                                                   # periods=1, 
                                                   # freq="H"),
                               # columns=['Close'])
                               
        # if(i > len(shortDf)-10-n_per_in):
            # print("predicted time: " + str(pred_df.index[0]))
            # pred_df.to_csv('predictedNext ' + str(i) + '.csv')
            # pred_dfPrev.to_csv('predictedPrev ' + str(i) + '.csv')

        # Updating the predictions DF
        # prevPredictions.update(pred_dfPrev)
        
        diff = time.time() - start
        
        numberOfAvg += 1
        
        avg = (avg * ((numberOfAvg-1)/numberOfAvg)) + ((1/numberOfAvg) * diff)
        if(timeRemaining):
            print("Time remaining = " + str(int(avg * (len(shortDf)-n_per_in-i))))
            
    
    return predictions, shortDf, prevPredictions
    
    
def get_predictions_binary(df, n_per_in, n_per_out, n_features, close_scaler, model, timeRemaining, numberPredictions=None):
    """
    Runs a 'For' loop to iterate through the length of the DF and create predicted values for every stated interval
    Returns a DF containing the predicted values for the model with the corresponding index values based on a business day frequency
    """
    
    #lengthPredictions = int(.8 * len(df)) #amount to chop 
    
    if(numberPredictions is not None):
        if(numberPredictions > len(df)):
            numberPredictions = len(df)
            
        shortDf = df.tail(numberPredictions)
    else:
        shortDf = df
        
        
    shortDf.to_csv("Short df")
        
    # Creating an empty DF to store the predictions
    predictions = pd.DataFrame(index=shortDf.index, columns=['Close'])
    prevPredictions = pd.DataFrame(index=shortDf.index, columns=['Close'])
    
    times = []
    
    avg = 1
    
    numberOfAvg = 0

    for i in range(0, len(shortDf)-n_per_in, n_per_out):
    
        start = time.time()
        # Creating rolling intervals to predict off of
        
        #print("I: " + str(i) + " Range: [" + str(i) + ", " + str(i+n_per_in) + "]")
        
        x = shortDf[i:i+n_per_in]
        
        # if(i > len(shortDf)-10-n_per_in):
            # print("xlen: " + str(len(x)))
            # print("start predict time: " + str(x.index[0]))
            # print("end predict time: " + str(x.tail(1).index[0]))
            # xFix = pd.DataFrame(close_scaler.inverse_transform(x[["Close"]]), 
                          # index=x.index, 
                          # columns=(['Close']))
            # xFix.to_csv('dataForPredict ' + str(i) + '.csv')

        # Predicting using rolling intervals
        yhat = model.predict(np.array(x).reshape(1, n_per_in, n_features))

        # Transforming values back to their normal prices
        # yhatNext = close_scaler.inverse_transform(yhat)[0][1]
        if(yhat >= .5):
            yhat = 1
        if(yhat < .5):
            yhat = -1
        yhatNext = np.array([yhat])
        
        #print("YHat: " + str(yhat))

        # DF to store the values and append later, frequency uses business days
        pred_df = pd.DataFrame(yhatNext.reshape(-1,1), 
                               index=pd.date_range(start=shortDf.index[i+n_per_in], 
                                                   periods=1, 
                                                   freq="H"),
                               columns=['Close'])
                               
        # if(i > len(shortDf)-10-n_per_in):
            # print("predicted time: " + str(pred_df.index[0]))
            # pred_df.to_csv('predictedNext ' + str(i) + '.csv')

        # Updating the predictions DF
        predictions.update(pred_df)
        
        # pred_dfPrev = pd.DataFrame(yhatPrev, 
                               # index=pd.date_range(start=shortDf.index[i+n_per_in-1], 
                                                   # periods=1, 
                                                   # freq="H"),
                               # columns=['Close'])
                               
        # if(i > len(shortDf)-10-n_per_in):
            # print("predicted time: " + str(pred_df.index[0]))
            # pred_df.to_csv('predictedNext ' + str(i) + '.csv')
            # pred_dfPrev.to_csv('predictedPrev ' + str(i) + '.csv')

        # Updating the predictions DF
        # prevPredictions.update(pred_dfPrev)
        
        diff = time.time() - start
        
        numberOfAvg += 1
        
        avg = (avg * ((numberOfAvg-1)/numberOfAvg)) + ((1/numberOfAvg) * diff)
        if(timeRemaining):
            print("Time remaining = " + str(int(avg * (len(shortDf)-n_per_in-i))))
            
    
    return predictions, shortDf, prevPredictions


def val_rmse(df1, df2):
    """
    Calculates the root mean square error between the two Dataframes
    """
    df = df1.copy()
    
    # Adding a new column with the closing prices from the second DF
    df['close2'] = df2.Close
    
    # Dropping the NaN values
    df.dropna(inplace=True)
    
    # Adding another column containing the difference between the two DFs' closing prices
    df['diff'] = df.Close - df.close2
    
    # Squaring the difference and getting the mean
    rms = (df[['diff']]**2).mean()
    
    # Returning the sqaure root of the root mean square
    return float(np.sqrt(rms))

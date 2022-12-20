import requests
import pandas as pd
import os
import shutil
import sys
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import Sequential
from tensorflow.keras import regularizers
from keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import time
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import mean_squared_error
import random
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt
from pytz import timezone
import logging
from statistics import mean
import trade
import threading
import pprint
import math as math
from printer import *

market = "binance"
symbol = "btcusdt"
period = "3600"
apiKey = "<insert_apiKey_for_cryptowatch>"

# logging.basicConfig(filename='redsun2.log', filemode='w', level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
    
# logger = logging.get_logger("logger")

badPairs = ['aebtc',
 'ambbtc' 
 'antbtc',  
 'appcbtc',  
 'ardrbtc',  
 'arkbtc',  
 'arpabtc',  
 'bcdbtc',  
 'bcptbtc',  
 'blzbtc',  
 'btsbtc',  
 'cdtbtc',  
 'chzbtc',  
 'cmtbtc',  
 'cndbtc', 
 'ctxcbtc',  
 'cvcbtc',  
 'dashbtc',  
 'databtc',  
 'dltbtc',  
 'drepbtc',  
 'evxbtc', 
 'fiobtc',   
 'funbtc',  
 'grsbtc',  
 'hcbtc',  
 'hivebtc',  
 'icxbtc',  
 'iostbtc',  
 'iotxbtc',  
 'kmdbtc',  
 'lskbtc',  
 'mblbtc', 
 'mdtbtc',  
 'mithbtc',  
 'nanobtc',  
 'neblbtc',  
 'nknbtc',  
 'nmrbtc',  
 'nulsbtc',  
 'nxsbtc',  
 'ongbtc',  
 'ostbtc',  
 'oxtbtc', 
 'paxgbtc',  
 'pivxbtc',  
 'pntbtc',  
 'poebtc',  
 'pptbtc',  
 'qlcbtc',  
 'qspbtc',  
 'rcnbtc',  
 'renbtc',  
 'rlcbtc',  
 'rsrbtc',  
 'scbtc', 
 'snglsbtc', 
 'stmxbtc',  
 'stratbtc',  
 'sysbtc', 
 'tctbtc',  
 'tnbbtc',  
 'troybtc',  
 'umabtc',  
 'vibbtc',  
 'vibebtc',  
 'wnxmbtc',  
 'wprbtc',  
 'xvgbtc',  
 'zenbtc']
 
badPairs2 = ['unibtc',
 'hntbtc',
 'dntbtc',  
 'balbtc',  
 'ognbtc',  
 'tomobtc',  
 'linkbtc',  
 'yfibtc',  
 'gasbtc',  
 'powrbtc',  
 'storjbtc',  
 'pivxbtc',  
 'repbtc',  
 'troybtc',  
 'dogebtc', 
 'funbtc',  
 'hotbtc',  
 'iostbtc',  
 'vitebtc',  
 'poebtc',  
 'drepbtc',  
 'ostbtc', 
 'mthbtc',   
 'chrbtc',  
 'mdtbtc',  
 'solbtc',  
 'sntbtc',  
 'ankrbtc',  
 'utkbtc',  
 'avaxbtc',  
 'paxgbtc',  
 'mkrbtc', 
 'xtzbtc',
 'scbtc',
 'mblbtc',
 'tnbbtc',
 'cosbtc',
 'wprbtc',
 ]
 
badUSBinancsPairs = [
    'DAI/USD',
    'USDC/USD',
    'BUSD/USD',
    'USDT/USD',
    'PAXG/USD',
    'REP/USD',
    'DASH/USD',
    'ONT/USD',
    'ZEC/USD',
    'WAVES/USD',
    'KNC/USD',
    'ZRX/USD',
    'IOTA/USD',
    'BNB/USD',
    'XRP/USD',
    'VTHO/USD',
    'UST/USD'
    # 'VET/USD',
    # 'SUSHI/USD',
    # 'RVN/USD'
 ]

# class StreamToLogger(object):
    # """
    # Fake file-like stream object that redirects writes to a logger instance.
    # """
    # def __init__(self, logger, level):
       # self.logger = logger
       # self.level = level
       # self.linebuf = ''

    # def write(self, buf):
       # for line in buf.rstrip().splitlines():
          # self.logger.log(self.level, line.rstrip())
          # sys.stdout = sys.__stdout__
          # if(line.rstrip() != '\n'):
            # printing(line.rstrip())
          # sys.stdout = StreamToLogger(logger,logging.INFO)

    # def flush(self):
        # pass

# sys.stdout = StreamToLogger(logger,logging.INFO)
# sys.stderr = StreamToLogger(logger,logging.ERROR)

 
console_locked = False
 
def lock_console():
    global console_locked
    console_locked = True
    
def unlock_console():
    global console_locked
    console_locked = False

def disable_console(code="null"):

    if(console_locked == False):
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")
        disable_logger()
    
def enable_console(code="null"):
    # global logger
    if(console_locked == False):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        # sys.stdout = StreamToLogger(logger,logging.INFO)
        # sys.stderr = StreamToLogger(logger,logging.ERROR)
        enable_logger()

def convert_pair_to_filename(pair):
    return pair.replace("/", "-")

def convert_filename_to_pair(filename):
    return filename.replace("-", "/")

def get_data(pair, exchange, timeframe):

    global badPairs2
    
    if(pair in badUSBinancsPairs):
        printing("get_data: Skipping pair in badList2: {}".format(pair))
        return "get_data: Skipping pair in badList2: {}".format(pair), False
    
    try:
        newdf = trade.get_pair_data(pair, exchange, timeframe)
    except:
        printing("get_data: Skipping pair bad payload: {}".format(pair))
        return "get_data: Skipping pair bad payload: {}".format(pair), False
     
    currTime = datetime.utcnow()
       
    if(len(newdf) < 24):
        printing("get_data: Skipping pair needs 24 hours of data: {}".format(pair))
        return "get_data: Skipping pair needs 24 hours of data: {}".format(pair), False
    
    if(datetime.strptime(str(newdf.index[-1]), "%Y-%m-%d %H:%M:%S") - currTime > timedelta(hours=4) or datetime.strptime(str(newdf.index[-1]), "%Y-%m-%d %H:%M:%S") - currTime < timedelta(hours=-4)):
        printing("get_data: Skipping pair no recent data: {}".format(pair))
        return "get_data: Skipping pair no recent data: {}".format(pair), False
    
    # printing("Volume[0:5]: {}".format(newdf.Volume[0:5]))
    
    for i in range(len(newdf)):
        newdf.at[newdf.index[i], 'Volume'] = newdf.Volume[i] * newdf.Close[i]
        
    # printing("Volume[0:5]: {}".format(newdf.Volume[0:5]))
    
    
    
    # printing("Length of data for {} is {}".format(symbol, len(newdf)))
    
    
    
    # foundOldData = False
    # try:
        # olddf = pd.read_pickle('savedData/' + convert_pair_to_filename(pair) + '.pkl')
        # olddf = olddf.rename(columns = {'Volume USD' : 'Volume'})
        # foundOldData = True
    # except:
        # printing("get_data: No Old Data for Pair: {}".format(pair))
        
    # if(foundOldData):
        # dates = olddf.index.tolist()
        # newdates = newdf.index.tolist()
        # if(newdates.index(dates[-1]) > -1):
            # df = olddf.append(newdf.tail(len(newdates)-newdates.index(dates[-1]) - 1))
    # else:
        # 
        
    # df.to_pickle('data/' + convert_pair_to_filename(pair) + '.pkl')
    # df.to_csv('dataCSV/' + convert_pair_to_filename(pair) + '.csv')
    
    df = newdf
    
    return df, True
    
def split_sequence_binary(seq, n_steps_in, n_steps_out, getEvenSamples, future_price_intervals, isTestSequence):
    """
    Splits the multivariate time sequence
    """
    
    #printing("future_price_intervals: {}".format(future_price_intervals))
    
    #printing(seq.shape)
    
    # Creating a list for both variables
    X, y = [], []
    
    xScalers = []
    
    numberPositive = 0
    numberNegative = 0
    
    positiveSeqsX = []
    positiveSeqsy = []
    
    negativeSeqsX = []
    negativeSeqsy = []
    if(not getEvenSamples):
        print("length of seq: {}".format(len(seq)))
    
    for i in range(len(seq)):
        
        # Finding the end of the current sequence
        end = i + n_steps_in
        out_end = end + future_price_intervals
        
        # Breaking out of the loop if we have exceeded the dataset's length
        if out_end > len(seq):
            break
        
        # Splitting the sequences into: x = past prices and indicators, y = prices ahead
        seq_x = seq[i:end, :]
        
        scalerX = preprocessing.MinMaxScaler(feature_range=(0, 1))
        
        seq_x = scalerX.fit_transform(seq_x)
        
        xScalers.append(scalerX)
        
        seq_x_end_price = seq[end-1:end, 0][0]
        
        seq_future_end_price = seq[out_end-1:out_end, 0][0]
        
        seq_future_end_prices = seq[end:out_end, 0]
        
        seq_x_prices = seq[i:end, 0]
        
        #printing("seq_x_end_price: {}".format(seq_x_end_price))
        #printing("seq_future_end_price: {}".format(seq_future_end_price))
        
        seq_y = [0]
        try:
            if(seq_future_end_price-seq_x_end_price > 0):
                seq_y = [1]
            elif(seq_future_end_price-seq_x_end_price < 0):
                seq_y = [0]
            elif(isTestSequence == False):
                continue
            else:
                seq_y = [0]
        except Exception as ex:
            printing("seq_future_end_price: {} seq_x_end_price: {} ex: {}".format(seq_future_end_price, seq_x_end_price, ex))
            
        # seq_x[0][0] = seq_y[0]
            
        # if(len(y) < 15):
            # np.set_printoptions(precision=3, suppress=True)
            # printing()
            # printing("Shape seq_x: " + str(seq_x.shape))
            
            # printing("Slope?: " + str((seq_test[1]-seq_test[0])/seq_test[0]))
            
            # printing("Iteration: " + str(i) + " seq_x: " + str(seq_x[-2:]))
            
            # printing("Iteration: " + str(i) + " seq_test: " + str(seq_test))
            
            # printing("Iteration: " + str(i) + " seq_y: " + str(seq_y))
            # printing()
            
            
        
            # xDf = pd.DataFrame(seq_x.reshape(n_steps_in, -1),
                               # index=pd.date_range(start=df.index[i], 
                                                   # periods=len(seq_x), 
                                                   # freq="H"),
                               # columns=df.columns)
                               
            # yDf = pd.DataFrame(np.array(seq_y).reshape(-1, 1),
                               # index=pd.date_range(start=df.index[end], 
                                                   # periods=len(seq_y), 
                                                   # freq="H"),
                               # columns=['Close'])
            # xDf.to_csv('xdf ' + str(i) + '.csv')
            # yDf.to_csv('ydf ' + str(i) + '.csv')
        if(getEvenSamples): # and numberNegative <= numberPositive):
            if(seq_y[0] == 0): 
                negativeSeqsX.append(seq_x)
                negativeSeqsy.append(seq_y)
                # X.append(seq_x)
                # y.append(seq_y)
                numberNegative+=1
            if(seq_y[0] == 1):
                positiveSeqsX.append(seq_x)
                positiveSeqsy.append(seq_y)
                # X.append(seq_x)
                # y.append(seq_y)
                numberPositive+=1
        else:
            X.append(seq_x)
            y.append(seq_y)
    if(getEvenSamples):
        printing("numberNegative: " + str(numberNegative))
        printing("numberPositive: " + str(numberPositive))
        
    if(getEvenSamples):
        length = min(len(negativeSeqsX), len(positiveSeqsX))
        
        X = negativeSeqsX[-length:] + positiveSeqsX[-length:]
        y = negativeSeqsy[-length:] + positiveSeqsy[-length:]
        
    return np.array(X), np.array(y) 

def split_sequence_binary_basic(seq, n_steps_in, n_steps_out, getEvenSamples, future_price_intervals, isTestSequence):
    """
    Splits the multivariate time sequence
    """
    
    #printing("future_price_intervals: {}".format(future_price_intervals))
    
    #printing(seq.shape)
    
    # Creating a list for both variables
    X, y = [], []
    
    xScalers = []
    
    numberPositive = 0
    numberNegative = 0
    
    positiveSeqsX = []
    positiveSeqsy = []
    
    negativeSeqsX = []
    negativeSeqsy = []
    
    for i in range(len(seq)):
        
        # Finding the end of the current sequence
        end = i + n_steps_in
        out_end = end + future_price_intervals
        
        # Breaking out of the loop if we have exceeded the dataset's length
        if out_end > len(seq):
            break
        
        # Splitting the sequences into: x = past prices and indicators, y = prices ahead
        seq_x = seq[i:end, :]
        
        scalerX = preprocessing.MinMaxScaler(feature_range=(0, 1))
        
        seq_x = scalerX.fit_transform(seq_x)
        
        xScalers.append(scalerX)
        
        seq_x_end_price = seq[end-1:end, 0][0]
        
        seq_future_end_price = seq[out_end-1:out_end, 0][0]
        
        seq_future_end_prices = seq[end:out_end, 0]
        
        seq_x_prices = seq[i:end, 0]
        
        #printing("seq_x_end_price: {}".format(seq_x_end_price))
        #printing("seq_future_end_price: {}".format(seq_future_end_price))
        
        seq_y = [0]
        try:
            if(seq_future_end_price-seq_x_end_price > 0):
                seq_y = [1]
            elif(seq_future_end_price-seq_x_end_price < 0):
                seq_y = [0]
            elif(isTestSequence == False):
                continue
            else:
                seq_y = [0]
        except Exception as ex:
            printing("seq_future_end_price: {} seq_x_end_price: {} ex: {}".format(seq_future_end_price, seq_x_end_price, ex))
            
        # seq_x[0][0] = seq_y[0]
            
        # if(len(y) < 15):
            # np.set_printoptions(precision=3, suppress=True)
            # printing()
            # printing("Shape seq_x: " + str(seq_x.shape))
            
            # printing("Slope?: " + str((seq_test[1]-seq_test[0])/seq_test[0]))
            
            # printing("Iteration: " + str(i) + " seq_x: " + str(seq_x[-2:]))
            
            # printing("Iteration: " + str(i) + " seq_test: " + str(seq_test))
            
            # printing("Iteration: " + str(i) + " seq_y: " + str(seq_y))
            # printing()
            
            
        
            # xDf = pd.DataFrame(seq_x.reshape(n_steps_in, -1),
                               # index=pd.date_range(start=df.index[i], 
                                                   # periods=len(seq_x), 
                                                   # freq="H"),
                               # columns=df.columns)
                               
            # yDf = pd.DataFrame(np.array(seq_y).reshape(-1, 1),
                               # index=pd.date_range(start=df.index[end], 
                                                   # periods=len(seq_y), 
                                                   # freq="H"),
                               # columns=['Close'])
            # xDf.to_csv('xdf ' + str(i) + '.csv')
            # yDf.to_csv('ydf ' + str(i) + '.csv')
        if(getEvenSamples): # and numberNegative <= numberPositive):
            if(seq_y[0] == 0): 
                negativeSeqsX.append(seq_x)
                negativeSeqsy.append(seq_y)
                # X.append(seq_x)
                # y.append(seq_y)
                numberNegative+=1
            if(seq_y[0] == 1):
                positiveSeqsX.append(seq_x)
                positiveSeqsy.append(seq_y)
                # X.append(seq_x)
                # y.append(seq_y)
                numberPositive+=1
        else:
            X.append(seq_x)
            y.append(seq_y)
    if(getEvenSamples):
        printing("numberNegative: " + str(numberNegative))
        printing("numberPositive: " + str(numberPositive))
        
    if(getEvenSamples):
        length = min(len(negativeSeqsX), len(positiveSeqsX))
        
        X = negativeSeqsX[-length:] + positiveSeqsX[-length:]
        y = negativeSeqsy[-length:] + positiveSeqsy[-length:]
        
    return np.array(X), np.array(y)      
                    
def calc_gain_binary(actualPrice, pred, times, percentages, start_amount, trade_cost, n_per_in, noLoss, cashout, opposite, showInfo, symbol, saveData):

    last_action = None
    sell_price = None
    current_amount = start_amount
    startPrice = actualPrice[0]
    buy_price = actualPrice[0]
    percentageThreshold = .000
    
    maxAmountSeen = 0
    
    consolePrevEnabled = False
    if(sys.stdout == sys.__stdout__):
        consolePrevEnabled = True
    
    if(showInfo == False):
        disable_console()
       
    numberCorrect = 0
    totalNumber = 0
    
    if(saveData):
        dfCreated = False
        data_df = None
    
    disable_console()

    for i in range(0, len(actualPrice)):
    
        printing("Time: " + str(times[i]))
        actualChange = None
        if(i < len(actualPrice)-1 and actualPrice[i+1] == actualPrice[i]):
            continue
        if(i < len(actualPrice)-1 and actualPrice[i+1] > actualPrice[i]):
            actualChange = 1
        if(i < len(actualPrice)-1 and actualPrice[i+1] < actualPrice[i]):
            actualChange = 0
        
        printing("Predict: " + str(pred[i]) + " Actual: " + str(actualChange))
        
        if(i < len(actualPrice)-1 and actualChange == pred[i]):
            numberCorrect+=1
            
        if(i < len(actualPrice)-1):
            totalNumber+=1
            
        old_amount = current_amount
        
        # if(abs(percentages[i] - .50) >= percentageThreshold):
        if(pred[i] == 1):
            if(last_action == "Sell" or last_action == None):
                printing("Buying BTC with " + str(current_amount) + " at BTC/USD: " + str(actualPrice[i]))
                current_amount -= current_amount * trade_cost
                buy_price = actualPrice[i]
                last_action = "Buy"
            elif(last_action == "Buy"):
                printing("Holding " + str(current_amount) + " at BTC/USD: " + str(actualPrice[i]))
        elif(pred[i] == 0):
            if(last_action == "Buy" or last_action == None):
                printing("Selling Debug:")
                printing("Buy Price: " + str(buy_price))
                printing("Selling Price: "+ str(actualPrice[i]))
                printing("Percentage difference: " + str((actualPrice[i] - buy_price) / buy_price))
                # if(abs((actualPrice[i] - buy_price) / buy_price) < 2 * trade_cost):
                    # printing("Percentage below 2 * trade_cost threshold, not selling")
                # else:
                if(noLoss == True and actualPrice[i] < buy_price):
                    printing("No loss is on, not selling")
                else:
                    new_amount = (((actualPrice[i] - buy_price) / buy_price) * current_amount) + current_amount - (current_amount * trade_cost)
                    printing("Selling " + str(current_amount) + " at BTC/USD: " + str(actualPrice[i]) + " for a gain of " + str(new_amount - current_amount))
                    current_amount = new_amount
                    # sell_price = actualClose[i]
                    last_action = "Sell"
            elif(last_action == "Sell"):
                printing("Holding " + str(current_amount) + " at BTC/USD: " + str(actualPrice[i]))
        # else:
            # printing(str(percentages[i]) + " is below threshold of " + str(percentageThreshold))
        printing()
        if(current_amount > maxAmountSeen):
            maxAmountSeen = current_amount
        if(saveData):   
            if(dfCreated == False):
                data_df = pd.DataFrame({'Date': [times[i]], 'symbol': [symbol], 'last_action': [last_action], 'old_amount': [old_amount], 'new_amount': [current_amount], 'current_price': [actualPrice[i]], 'buy_price': [buy_price], 'change_percent': [((current_amount - old_amount) / old_amount)]})
                data_df.set_index('Date', inplace=True)
                dfCreated = True
            else:
                new_row = pd.Series(data={'symbol': symbol, 'last_action': last_action, 'old_amount': old_amount, 'new_amount': current_amount, 'current_price': actualPrice[i], 'buy_price': buy_price, 'change_percent': ((current_amount - old_amount) / old_amount)}, name=times[i])
                data_df = data_df.append(new_row)
        
    lastIndex = len(actualPrice)-1
    
    enable_console()

    printing("\n\nFinal Trade\n\n")
 
    if(last_action == "Sell"):
        printing("Holding " + str(current_amount) + " at BTC/USD: " + str(actualPrice[lastIndex]))
    if(last_action == "Buy" or last_action == None):
        new_amount = (((actualPrice[lastIndex] - buy_price) / buy_price) * current_amount) + current_amount - (current_amount * trade_cost)
        printing("Selling " + str(current_amount) + " at BTC/USD: " + str(actualPrice[lastIndex]) + " for a gain of " + str(new_amount - current_amount))
        current_amount = new_amount
        #sell_price = actualClose[i]
        last_action = "Sell"  
        
    
    printing("\n\nFinal Statistics: \n\n")
    printing("Start time: " + str(times[0]))
    printing("End time: " + str(times[-1]))
    printing("Number of hours elasped: " + str(len(actualPrice)))
    
    
    printing()
    printing(symbol + " start: " + str(actualPrice[0]))
    printing(symbol + " end: " + str(actualPrice[-1]))
    printing("Percentage change of " + symbol + ": " + str((actualPrice[-1] - actualPrice[0]) / actualPrice[0]))
    printing("Percentage change of Amount: " + str((current_amount - start_amount) / start_amount))
    
    printing()
    printing("Number correct: " + str(numberCorrect))
    printing("Total number: " + str(totalNumber))
    printing("Percent correct: " + str(numberCorrect/totalNumber))
    
    printing()
    printing("Start Amount: " + str(start_amount))
    printing("Current Amount: " + str(current_amount))
    printing("Gain/Loss: " + str(current_amount - start_amount))
    
    
    
    printing("--------------------------------------Max Amount Seen: {}".format(maxAmountSeen))
    
    
    if(showInfo == False and consolePrevEnabled):
        enable_console()
        
    if(saveData):
        filename = convert_pair_to_filename(symbol) + "_calc_gain_binary.csv"
        try:
            os.remove('simulatedTradesData/' + filename)
        except Exception as ex:
            printing("calc_gain_binary: no simulated data to remove")
        try:
            data_df.to_csv('simulatedTradesData/' + filename)
        except Exception as ex:
            printing("calc_gain_binary: couldn't save simulated data: {} ex: {}".format(symbol, ex))
            
    return current_amount

def get_predictions_binary(df, n_per_in, n_per_out, n_features, model, getEvenSamples, future_price_intervals, timeRemaining, numberPredictions=None, showInfo=False, perfect=False):
    """
    Runs a 'For' loop to iterate through the length of the DF and create predicted values for every stated interval
    Returns a DF containing the predicted values for the model with the corresponding index values based on a business day frequency
    """
    
    if(numberPredictions is not None):
        if(numberPredictions > len(df)):
            numberPredictions = len(df)
            
        shortDf = df.tail(numberPredictions+n_per_in)
    else:
        shortDf = df
    
    # printing("shortDf.Close[0:n_per_in+5]: {}".format(shortDf.Close[0:n_per_in+5]))
    # printing("shortDf.Close[-5:end]: {}".format(shortDf.Close[-5:len(shortDf.Close)]))
    
    times = []
    
    avg = 1
    
    numberOfAvg = 0
    
    predicted_values = []
    correct_values = []
    correctPrices = []
    predicted_percentages = []
    
    
    printing(shortDf[0:5])
    
    
    
    X, y = split_sequence_binary(shortDf.to_numpy(), n_per_in, n_per_out, getEvenSamples, future_price_intervals, True)
    
    consolePrevEnabled = False
    if(sys.stdout == sys.__stdout__):
        consolePrevEnabled = True
    
    if(showInfo == False):
        disable_console() 
        
    print("length of X: {}".format(len(X)))
    
    yhats = model.predict(X.reshape(len(X), n_per_in, n_features))
    
    printing("shape yhats: " + str(yhats.shape))
    printing("shape x: {}".format(X.shape))
    printing("len shortDf: {}".format(len(shortDf)))
    
    i = 0
    
    for yhat in yhats:
        
        start = time.time()
    
        correctPrediction = y[i][0]
        
        prob = yhat
        
        predicted_percentages.append(prob)
        
        if(yhat > .5):
            yhat = 1
        if(yhat <= .5):
            yhat = 0
            
        correct_values.append(correctPrediction)
        
        predicted_values.append(yhat)
        
        correctPrices.append(shortDf.Close[i+n_per_in-1])
        
        times.append(shortDf.index[i+n_per_in-1])
        
        diff = time.time() - start
        
        numberOfAvg += 1
        
        avg = (avg * ((numberOfAvg-1)/numberOfAvg)) + ((1/numberOfAvg) * diff)
        if(timeRemaining):
            printing("Time remaining = " + str(int(avg * (len(shortDf)-n_per_in-i))))
            
        i += 1
        
    # printing("correctPrices[0:5]: {}".format(correctPrices[0:5]))
    # printing("correctPrices[-5:0]: {}".format(correctPrices[-5:len(correctPrices)]))
    
    try:
        results = confusion_matrix(correct_values, predicted_values) 
      
        printing('get_predictions_binary: Confusion Matrix :')
        printing(results) 
        printing('get_predictions_binary: Accuracy Score : {}'.format(accuracy_score(correct_values, predicted_values)))
        printing('get_predictions_binary: Report : ')
        printing(classification_report(correct_values, predicted_values))
        
        mcc = matthews_corrcoef(correct_values, predicted_values)
        printing('get_predictions_binary: MCC:' + str(mcc))
    except Exception as ex:
        printing(ex)
    
    
    
    if(perfect):
        predicted_values = correct_values
        
    if(showInfo == False and consolePrevEnabled):
        enable_console()
        
    
    return predicted_values, correctPrices, times, mcc, predicted_percentages, correct_values, accuracy_score(correct_values, predicted_values)
    
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
    
def create_model(n_per_in, n_features):

    model = Sequential()

    model.add(tf.keras.layers.GRU(n_per_in, kernel_regularizer=regularizers.l2(0.0001), input_shape=(n_per_in, n_features)))
    
    model.add(tf.keras.layers.Dropout(0.5))
    
    # model.add(Dense(10, activation='relu'))
    
    model.add(Dense(10, activation='relu'))
    
    model.add(tf.keras.layers.Dropout(0.5))
            
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    return model
    
def create_model_erik_test(n_per_in, n_features):

    model = Sequential()

    model.add(tf.keras.layers.GRU(n_per_in, kernel_regularizer=regularizers.l2(0.0001), input_shape=(n_per_in, n_features)))
    
    model.add(tf.keras.layers.Dropout(0.5))
    
    # model.add(Dense(10, activation='relu'))
    
    model.add(Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='relu'))
    
    model.add(tf.keras.layers.Dropout(0.5))
    
    model.add(Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='relu'))
    
    model.add(tf.keras.layers.Dropout(0.5))
    
    model.add(Dense(512, kernel_regularizer=regularizers.l2(0.0001), activation='relu'))
    
    model.add(tf.keras.layers.Dropout(0.5))
            
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    return model
    
    
def create_model_basic(n_per_in):
    
    model = Sequential()
    
    model.add(Dense(n_per_in, activation='relu'))
            
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    return model
    
def get_percent_increasing(prices):

    if(len(prices) == 0):
        return 0

    numberInc = 0
    for i in range(0, len(prices)-1):
        if(prices[i] < prices[i+1]):
            numberInc+=1
            
    percent = float(numberInc)/len(prices)
        
    #printing(percent)
            
    return percent

def get_percent_decreasing(prices):

    if(len(prices) == 0):
        return 0

    numberDec = 0
    for i in range(0, len(prices)-1):
        if(prices[i] > prices[i+1]):
            numberDec+=1
            
    percent = float(numberDec)/len(prices)
    
    #printing(percent)
    
    return percent
    
# def get_BB_df(type, df, thresh, numberPrices):

    # lastPrices = []
    
    # newDf = pd.DataFrame()
    
    # for i in range(0, len(df)):
        
        # if(type == "Bear"):
            # if(get_percent_decreasing(lastPrices) > get_percent_increasing(lastPrices) + thresh):
                # newDf = newDf.append(df.iloc[[i]])
        # elif(type == "Bull"):
            # if(get_percent_decreasing(lastPrices) + thresh < get_percent_increasing(lastPrices)):
                # newDf = newDf.append(df.iloc[[i]])
        # else:
            # if(not (get_percent_decreasing(lastPrices) + thresh < get_percent_increasing(lastPrices)) and not (get_percent_decreasing(lastPrices) + thresh < get_percent_increasing(lastPrices))):
                # newDf = newDf.append(df.iloc[[i]])
        # if(len(lastPrices) >= numberPrices):
            # lastPrices.pop(0)
            # lastPrices.append(df.iloc[[i]]['Close'][0])
            # printing(df.iloc[[i]])
        # else:
            # printing(df.iloc[[i]])
            # printing(df.iloc[[i]]['Close'][0])
            # lastPrices.append(df.iloc[[i]]['Close'][0])
    
    # printing("Len newDf {}".format(len(newDf)))
    
    # return newDf
    
def get_BB_df(df, thresh, numberPrices):

    lastPrices = []
    
    newDfBear = pd.DataFrame()
    newDfBull = pd.DataFrame()
    newDfNeutral = pd.DataFrame()
    
    numberInc = 0
    numberDec = 0
    total = 0
    lastPrice = -1
    
    for i in range(0, len(df)):
    
    
        if(total > 0):
        
            percentInc = numberInc / float(total)
            percentDec = numberDec / float(total)
        
            if(percentDec > percentInc + thresh):
                newDfBear = newDfBear.append(df.iloc[[i]])
            if(percentDec + thresh < percentInc):
                newDfBull = newDfBull.append(df.iloc[[i]])
            else:
                newDfNeutral = newDfNeutral.append(df.iloc[[i]])
                    
        if(total >= numberPrices):
        
            closePrice = df.iloc[[i]]['Close'][0]
        
            if(lastPrices[0] < lastPrices[1]):
                numberInc -= 1
            if(lastPrices[0] > lastPrices[1]):
                numberDec -= 1
                
                
            if(closePrice > lastPrice):
                numberInc += 1 
            elif(closePrice < lastPrice):
                numberDec += 1 
                
            lastPrice = closePrice
        
            lastPrices.pop(0)
            lastPrices.append(closePrice)
            #printing(df.iloc[[i]])
        else:
            #printing(df.iloc[[i]])
            #printing(df.iloc[[i]]['Close'][0])
            
            closePrice = df.iloc[[i]]['Close'][0]
            
            
            total += 1
            if(closePrice > lastPrice):
                numberInc += 1 
            elif(closePrice < lastPrice):
                numberDec += 1 
                
            lastPrices.append(closePrice)
            lastPrice = closePrice
    
    printing("Len newDfBear {}".format(len(newDfBear)))
    
    return newDfBear, newDfBull, newDfNeutral
    
    
def train_model(market_data_frames, pairs, number_of_hours_to_simulate, getEvenSamples, n_per_in, future_price_intervals, percent_of_hours_to_simulate):

    epochs = 100
    batch_size = 128
    n_per_out = 1
    size = None

    bigSet = False
    
    testmarket_data_frames = []

    for df in market_data_frames:
    
        # printing("YOOOYOYOY")
        # if(numberOfPairs > 50):
            # break
        # printing()
        # printing(file)
        # df = pd.read_pickle("newData/" + file)
        # df = df[['Close', 'Open', 'High', 'Low', 'Volume USD']]
        
        # printing("Length of data for {} is {}".format(file[0:-4], len(df)))
        
        # df.to_csv(file[0:-4] + ".csv")
        
        # df = df[['Close']]
        # df = df.rename(columns={'Volume USD': 'Volume'})
        
        # if(len(df) < n_per_in):
            # continue
            
        # printing("Len DF: {}".format(len(df)))
        # printing(df.tail())
        
        if(percent_of_hours_to_simulate > 0):
        
            number_of_hours_to_simulate = int(percent_of_hours_to_simulate * len(df))
        
        if(number_of_hours_to_simulate > 0):
            testDf = df.tail(number_of_hours_to_simulate)
            trainDf = df.head(len(df)-number_of_hours_to_simulate)
            df = trainDf
        
        # currTime = datetime.now()
        
        # printing(currTime)
        # printing(testDf.index[-1])
        
        # printing(datetime.strptime(str(testDf.index[-1]), "%Y-%m-%d %H:%M:%S") - currTime)
        
        # if(datetime.strptime(str(testDf.index[-1]), "%Y-%m-%d %H:%M:%S") - currTime > timedelta(hours=48) or datetime.strptime(str(testDf.index[-1]), "%Y-%m-%d %H:%M:%S") - currTime < timedelta(hours=-48)):
            # printing("Skipping")
            # continue
            
        # testDf.to_csv(file[0:-4] + ".csv")
        # printing(testDf.tail(5))
        
        if(size is not None):
            df = df.tail(size)
        
        X, y = split_sequence_binary(df.to_numpy(), n_per_in, n_per_out, getEvenSamples, future_price_intervals, False) 
        
        
        if(len(X) > 0):
            # printing("Sample X: {}, Sample y: {}".format(X[0], y[0]))
            if(not bigSet):
                BigX = X    
                Bigy = y
                bigSet = True
            else:
                BigX = np.concatenate((BigX, X), axis=0)
                Bigy = np.concatenate((Bigy, y), axis=0)
                
    if(not bigSet):
        printing("LenX {}".format(len(X)))
        printing("Leny {}".format(len(y)))
        
        return None, None, None, None
        
    if(number_of_hours_to_simulate > 0):
    
        X_train, X_test, y_train, y_test = train_test_split(BigX, Bigy, test_size=0.33, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(BigX, Bigy, test_size=.1, random_state=42)

    n_features = market_data_frames[0].shape[1]

    model = create_model_erik_test(n_per_in, n_features)

    callbacks = []
    
    printing("train_model: Size of Train Data: {} \nSize of Test Data: {}".format(len(X_train), len(X_test))) 

    addNoise = False

    if(addNoise):
        mu, sigma = 0, 0.1
        noise = np.random.normal(mu, sigma, np.shape(X_train))
        # printing("noise: {}".format(noise))
        X_train = X_train + noise

    res = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1, callbacks=callbacks)
    
    #visualize_training_results(res)
    
    return model, n_per_in, n_per_out, n_features

def Sort_Tuple(tup, keyNum): 
  
    # reverse = None (Sorts in Ascending order) 
    tup.sort(key = lambda x: x[keyNum]) 
    return tup
    
def get_number_of_longest_even_sequences(df, n_per_in, number_of_sequences, diff_thresh):
    
    seq_pairs = []
    startIndex = 0
    endIndex = 0
    longestSeqSize = 0
    
    for i in range(len(df)-2, 0, -1):
        for j in range(len(df)-2, i-1, -1):
            #printing("si: {}, ei: {} i: {}".format(j-i, j, i))
            if(j-i >= n_per_in-1 and abs((df.Close[j]-df.Close[j-i])/df.Close[j-i]) < diff_thresh):
                seq_pairs.append((j-i, j))
                if(len(seq_pairs) >= number_of_sequences):
                    break
        if(len(seq_pairs) >= number_of_sequences):
            break

    return seq_pairs
    
def get_number_of_longest_diff_sequences(df, n_per_in, number_of_sequences, diff, diff_thresh):
    
    seq_pairs = []
    startIndex = 0
    endIndex = 0
    longestSeqSize = 0
    
    if(diff == 0):
        return get_number_of_longest_even_sequences(df, n_per_in, number_of_sequences, diff_thresh)
    else:
        for i in range(len(df)-2, 0, -1):
            for j in range(len(df)-2, i-1, -1):
                #printing("si: {}, ei: {} i: {}".format(j-i, j, i))
                if(j-i >= n_per_in-1 and ((df.Close[j]-df.Close[j-i])/df.Close[j-i]) >= diff-diff_thresh and ((df.Close[j]-df.Close[j-i])/df.Close[j-i]) <= diff+diff_thresh):
                    seq_pairs.append((j-i, j))
                    if(len(seq_pairs) >= number_of_sequences):
                        break
            if(len(seq_pairs) >= number_of_sequences):
                break

    return seq_pairs
    
   
    
def run_tests_custom(model, n_per_in, n_per_out, n_features, number_of_hours_to_simulate, start_amount, market_data_frames, pairs, getEvenSamples, future_price_intervals):
    
    accs = []
        
    gains = []
    
    gainsHour = []
    
    number_of_sequences_to_generate = 1
    
    sequence_difference = .0 # As a percentage
    
    diff_thresh = .025 # As a percentage
    
    for i in range(len(market_data_frames)):
    
        printing("\nYOYOYO\n")
    
        df = market_data_frames[i]
        
        df = df.tail(10000)
        
        numSimTrials = int(len(df)/number_of_hours_to_simulate)
        
        dfs = []
        
        for x in range(numSimTrials):
            dfs.append(df.head(len(df)-(x*number_of_hours_to_simulate)).tail(number_of_hours_to_simulate))
            
        
        printing("\nCUSTOM TESTS \n Pair: {}".format(pairs[i]))    
        
        pairAccs = []
        
        pairGains = []
        
        pairGainsHour = []
        
        avgMinAcc = []
        avgQ1Acc = []
        avgMedAcc = []
        avgQ3Acc = []
        avgMaxAcc = []
        
        printing("\nGENERATING {} SEQUENCES OF DIFF: {}\n".format(len(dfs)*number_of_sequences_to_generate, sequence_difference))  
            
        for df1 in dfs:   
        
            seq_pairs = get_number_of_longest_diff_sequences(df1, n_per_in, number_of_sequences_to_generate, sequence_difference, diff_thresh) #[(n_per_in-1, len(df1)-2)]
            
            new_market_data_frames = []
            new_pairs = []
            
            for startIndex, endIndex in seq_pairs:
            
                if(startIndex == -1):
                    continue
                
                if(endIndex-startIndex <= n_per_in):
                    continue
                
                df1 = df1.iloc[startIndex-(n_per_in-1):endIndex+2,:]
                
                printing("Begin: {} End: {} Diff: {}".format(df1.Close[n_per_in-1], df1.Close[-2], (df1.Close[-2]- df1.Close[n_per_in-1])/df1.Close[n_per_in-1]))
                
                new_market_data_frames.append(df1)
                new_pairs.append(pairs[i])
                 
                # printing("longest sequence size: {}".format(endIndex-startIndex))
                
                acc, gainpercent, gainpercentperhour = run_tests(model, n_per_in, n_per_out, n_features, 100000, start_amount, new_market_data_frames, pairs, getEvenSamples, future_price_intervals, 1.00)
                
                printing("Seq Length: {} Average Gain Percent/Hour: {} Percent Gained: {} Acc: {}".format(len(df1), gainpercentperhour, gainpercent, acc))
                
                pairAccs.append(acc)
                
                pairGains.append(gainpercent)
                
                pairGainsHour.append(gainpercentperhour)
        
        
        if(len(dfs) > 0):
            printing("Pair Total: {} Average Gain Percent/Hour: {} Average Percent Gained: {} Average Acc: {}".format(pairs[i], mean(pairGainsHour), mean(pairGains), mean(pairAccs)))
            
            data = np.asarray(pairAccs)
            
            quartiles = np.percentile(data, [25, 50, 75])
            # calculate min/max
            data_min, data_max = data.min(), data.max()
            # print 5-number summary
            printing("ACC PERCENTILES: ")
            printing('Min: %.3f' % data_min)
            printing('Q1: %.3f' % quartiles[0])
            printing('Median: %.3f' % quartiles[1])
            printing('Q3: %.3f' % quartiles[2])
            printing('Max: %.3f' % data_max) 
            
            avgMinAcc.append(data_min)
            avgQ1Acc.append(quartiles[0])
            avgMedAcc.append(quartiles[1])
            avgQ3Acc.append(quartiles[2])
            avgMaxAcc.append(data_max)
            
            accs.append(mean(pairAccs))
            gains.append(mean(pairGains))
            gainsHour.append(mean(pairGainsHour))
        
    
       
        
    printing("Overall Total: Average Gain Percent/Hour: {} Average Percent Gained: {} Average Acc: {}".format(mean(gainsHour), mean(gains), mean(accs)))
    
    printing("Overall Total: Average MIN ACC: {} Q1 ACC: {} Median ACC: {} Q3 ACC: {} MAX ACC: {}".format(mean(avgMinAcc), mean(avgQ1Acc), mean(avgMedAcc), mean(avgQ3Acc), mean(avgMaxAcc)))
    
    return mean(gains), mean(accs)
    
def run_tests(model, n_per_in, n_per_out, n_features, number_of_hours_to_simulate, start_amount, market_data_frames, pairs, getEvenSamples, future_price_intervals, percent_of_hours_to_simulate):

    totalMoneySpent = 0
    totalMoneyGained = 0
    totalMoneyGainedHold = 0
    totalPercents = []
    MCCS = []
    endAmounts = []
    ACCS = []
    GAINPERCENTS = []
    COINchange_percentS = []
    GAINPERCENTSPERHOUR = []

    for i in range(len(market_data_frames)):
    #for i in range(0, 1):
    
        if(number_of_hours_to_simulate == 0 and percent_of_hours_to_simulate > 0):
            number_of_hours_to_simulate = int(percent_of_hours_to_simulate * len(market_data_frames[i]))

        df = market_data_frames[i].tail(number_of_hours_to_simulate + n_per_in)
        pair = pairs[i]
        
        if(len(df.Close) < n_per_in+1):
            printing("run_tests: length({}) of df({}) too small for n_per_in({}) skipping".format(len(df.Close), pair, n_per_in))
            continue

        calcPredictions, actualPrices, calcTimes, mcc, percentages, correctValues, acc = get_predictions_binary(df, n_per_in, n_per_out, n_features, model, False, future_price_intervals, False, showInfo=True)
        
        listOfPercentages = []
        
        for i in range(len(calcPredictions)):
            listOfPercentages.append((correctValues[i], percentages[i]))
            
        listOfPercentages = Sort_Tuple(listOfPercentages, 1)
       
        # for real, percent in listOfPercentages:
            # printing("{} : {}".format(real, percent))
            
        # if(pair == "hotbtc"):
            # endAmount = calc_gain_binary(calcShortDf, calcPredictions, calcTimes, percentages, start_amount, .00075, n_per_in, False, False, False, True, pair)
        # else:
        try:
            endAmount = calc_gain_binary(actualPrices, calcPredictions, calcTimes, percentages, start_amount, 0, n_per_in, False, False, False, False, pair, True)
        except Exception as ex:
            printing("run_tests: couldn't calc_gain_binary ex: {}".format(ex))
            endAmount = 0
        
        # if(endAmount > start_amount * 20):
            # continue
            
        # if(endAmount > 0):
        
            # printing("run_tests: accuracy " + pair + " acc: " + str(acc))
            # printing("run_tests: CalcGains from " + str(calcTimes[0]) + " to " + str(calcTimes[-1]) + ": " + str(endAmount))
            # printing("run_tests: " + pair + " value beginning: " + str(calcShortDf[0]))
            # printing("run_tests: " + pair + " value end: " + str(calcShortDf[-1]))
            # printing("run_tests: Percent gained " + pair + ": " + str(round(((calcShortDf[-1]-calcShortDf[0])/calcShortDf[0]), 2)))
            # printing("run_tests: Percent gained CLC: " + str(round(((endAmount-start_amount)/start_amount), 2)))
            # printing("") 
        
        # printing("run_tests: accuracy " + pair + " acc: " + str(acc))
        # printing("") 
        
        # endAmountHeld = start_amount + start_amount*((calcShortDf[-1]-calcShortDf[0])/calcShortDf[0])
        
        # percentCoinMoved = ((calcShortDf[-1]-calcShortDf[0])/calcShortDf[0])
        # totalPercents.append(percentCoinMoved)
        
        # totalMoneySpent += start_amount
        # totalMoneyGained += endAmount
        # totalMoneyGainedHold += endAmountHeld
        MCCS.append(mcc)
        # endAmounts.append(endAmount)
        ACCS.append(acc)
        GAINPERCENTS.append(round(((endAmount-start_amount)/start_amount), 2))
        COINchange_percentS.append(round(((actualPrices[-1]-actualPrices[0])/actualPrices[0]), 2))
        GAINPERCENTSPERHOUR.append(GAINPERCENTS[len(GAINPERCENTS)-1]/len(actualPrices))
      
    # average_percent_total = mean(totalPercents)
      
    printing("")  
    # printing("run_tests: totalMoneySpent: " + str(totalMoneySpent))
    # printing("run_tests: totalMoneyGained: " + str(totalMoneyGained))
    # printing("run_tests: totalMoneyGainedHold: " + str(totalMoneyGainedHold))
    # printing("")
    # printing("run_tests: total profit: " + str(totalMoneyGained-totalMoneySpent))
    # printing("")
    # printing("run_tests: totalMarketchange_percent: {}".format(round(average_percent_total, 4)))
    # printing("run_tests: marketProfit: {}".format(average_percent_total*totalMoneySpent))
    # printing("")
    printing("run_tests: Average MCC: {}".format(mean(MCCS)))
    printing("run_tests: Average ACC: {}".format(mean(ACCS)))
    printing("run_tests: Average Percent Gained: {}".format(mean(GAINPERCENTS)))
    printing("run_tests: Average Coin Price Change: {}".format(mean(COINchange_percentS)))
    
    data = np.asarray(MCCS)
    
    printing("\nrun_tests: Percentile Data")
    
    quartiles = np.percentile(data, [25, 50, 75])
    # calculate min/max
    data_min, data_max = data.min(), data.max()
    # print 5-number summary
    printing("run_tests: MCC Percentiles")
    printing('Min: %.3f' % data_min)
    printing('Q1: %.3f' % quartiles[0])
    printing('Median: %.3f' % quartiles[1])
    printing('Q3: %.3f' % quartiles[2])
    printing('Max: %.3f' % data_max)
    
    # data = np.asarray(endAmounts)
    
    # quartiles = np.percentile(data, [25, 50, 75])
    # calculate min/max
    # data_min, data_max = data.min(), data.max()
    # print 5-number summary
    # printing("\nrun_tests: End Amounts Percentiles")
    # printing('Min: %.3f' % data_min)
    # printing('Q1: %.3f' % quartiles[0])
    # printing('Median: %.3f' % quartiles[1])
    # printing('Q3: %.3f' % quartiles[2])
    # printing('Max: %.3f' % data_max)
    
    
    return mean(ACCS), mean(GAINPERCENTS)-mean(COINchange_percentS), mean(GAINPERCENTSPERHOUR)
    

def run_simple_tests(model, n_per_in, n_per_out, n_features, number_of_hours_to_simulate, start_amount, market_data_frames, pairs, getEvenSamples, future_price_intervals, percent_of_hours_to_simulate):

    endAmounts = []
    accs = []
    testAccs = []
    testEndAmounts = []
    totalAccs = []
    totalEndAmounts = []
    

    for i in range(len(market_data_frames)):
    
        df = market_data_frames[i].tail(len(market_data_frames[i]))
        pair = pairs[i]
    
        disable_console()
        lock_console()
        
        calcPredictions, actualPrices, calcTimes, mcc, percentages, correctValues, acc = get_predictions_binary(df, n_per_in, n_per_out, n_features, model, False, future_price_intervals, False, showInfo=True)
        endAmount = calc_gain_binary(actualPrices, calcPredictions, calcTimes, percentages, start_amount, 0, n_per_in, False, False, False, False, pair, False)
        
        totalAccs.append(acc)
        totalEndAmounts.append(endAmount)
        
        df = market_data_frames[i].tail(int(percent_of_hours_to_simulate * len(market_data_frames[i])) + n_per_in)
        pair = pairs[i]
        
        calcPredictions, actualPrices, calcTimes, mcc, percentages, correctValues, acc = get_predictions_binary(df, n_per_in, n_per_out, n_features, model, False, future_price_intervals, False, showInfo=True)
        endAmount = calc_gain_binary(actualPrices, calcPredictions, calcTimes, percentages, start_amount, 0, n_per_in, False, False, False, False, pair, False)
        
        testAccs.append(acc)
        testEndAmounts.append(endAmount)
        
        unlock_console()
        enable_console()
        
        if(number_of_hours_to_simulate == 0 and percent_of_hours_to_simulate > 0):
            number_of_hours_to_simulate = int(percent_of_hours_to_simulate * len(market_data_frames[i]))

        df = market_data_frames[i].tail(number_of_hours_to_simulate + n_per_in)
        pair = pairs[i]
        
        if(len(df.Close) < n_per_in+1):
            printing("run_tests: length({}) of df({}) too small for n_per_in({}) skipping".format(len(df.Close), pair, n_per_in))
            continue
            
        disable_console()
        lock_console()
        
        calcPredictions, actualPrices, calcTimes, mcc, percentages, correctValues, acc = get_predictions_binary(df, n_per_in, n_per_out, n_features, model, False, future_price_intervals, False, showInfo=True)
        
        endAmount = calc_gain_binary(actualPrices, calcPredictions, calcTimes, percentages, start_amount, 0, n_per_in, False, False, False, False, pair, False)
        
        unlock_console()
        enable_console()
        
        endAmounts.append(endAmount)
        accs.append(acc)
        
        #print("run_simple_tests: pair: {}".format(pair))
        #print("run_simple_tests: acc: {}".format(acc))
        #print("run_simple_tests: endAmount: {}".format(endAmount))
        
    return mean(accs), mean(endAmounts), mean(testAccs), mean(totalAccs), mean(testEndAmounts), mean(totalEndAmounts)
    
def get_coin_by_symbol(coin_trade_data_list, symbol):
    for coin in coin_trade_data_list:
        if(coin.symbol == symbol):
            return coin
            
def update_data_and_trade(exchange, timeframe, apiKey, coin_trade_data_list, model, n_per_in, n_per_out, trade_cost, trade_symbol):

    trade_symbol_price = get_current_coin_value('USDT/' + trade_symbol)
    
    overflow_coin = get_coin_by_symbol(coin_trade_data_list, "OVERFLOW")

    for coin in coin_trade_data_list:
    
        if(coin.symbol == "OVERFLOW"):
            continue
    
        pair = coin.symbol
    
        df, flag = get_data(pair, exchange, timeframe)
        
        try:
        
            if(flag == False):
                coin.prune = True
                printing("update_data: removing coin {} from coinList, issue with get_data ex: {}".format(pair, df))
                coin.prune_reason = "update_data: removing coin {} from coinList, issue with get_data ex: {}".format(pair, df)
                continue
                
            df = df[['Close', 'Open', 'High', 'Low', 'Volume']]
            
            df['Close'] = df['Close'].astype(float)
            df['Open'] = df['Open'].astype(float)
            df['High'] = df['High'].astype(float)
            df['Low'] = df['Low'].astype(float)
            df['Volume'] = df['Volume'].astype(float)
            
            coin.data_df = df
            
            current_hour = datetime.utcnow().hour
            
            number_of_tries = 0
            
            while(datetime.strptime(str(coin.data_df.index[-1]), "%Y-%m-%d %H:%M:%S").hour != current_hour):
                printing("update_data: {} not up to date yet trying again in 15 seconds".format(pair))
                time.sleep(15)
                current_hour = datetime.utcnow().hour
                df, flag = get_data(pair, exchange, timeframe)
                if(flag == False):
                    coin.prune = True
                    printing("update_data: removing coin {} from coinList, issue with get_data ex: {}".format(pair, df))
                    coin.prune_reason = "update_data: removing coin {} from coinList, issue with get_data ex: {}".format(pair, df)
                    break
                    
                df = df[['Close', 'Open', 'High', 'Low', 'Volume']]
                
                df['Close'] = df['Close'].astype(float)
                df['Open'] = df['Open'].astype(float)
                df['High'] = df['High'].astype(float)
                df['Low'] = df['Low'].astype(float)
                df['Volume'] = df['Volume'].astype(float)
                
                coin.data_df = df
                
                number_of_tries += 1
                
                if(number_of_tries > 10):
                    coin.prune = True
                    printing("update_data: removing coin {} from coinList, data is not up to date, number_of_tries exceeded 10".format(pair))
                    coin.prune_reason = "update_data: removing coin {} from coinList, data is not up to date, number_of_tries exceeded 10".format(pair)
                    break
            
            if(coin.prune == True):            
                continue
                
            coin.data_df = coin.data_df[:-1]
            
            coin.data_df.at[coin.data_df.index[-1], 'Close'] = get_current_coin_value(coin.symbol)
            
            coinTradeThread = threading.Thread(target=execute_single_trade, args=(coin, trade_symbol_price, model, n_per_in, n_per_out, trade_cost, exchange, overflow_coin), daemon=True)
            
            coinTradeThread.start()
            
        except Exception as ex:
            printing("couldnt find coin with symbol {} error {}".format(pair, ex))
    
def update_data(exchange, timeframe, apiKey, coin_trade_data_list):

    for coin in coin_trade_data_list:
    
        pair = coin.symbol
        
        df, flag = get_data(pair, exchange, timeframe)
        
        try:
            if(flag == False):
                coin_trade_data_list.remove(coin)
                printing("update_data: removing coin {} from coinList, issue with data".format(pair))
                continue
                
            df = df[['Close', 'Open', 'High', 'Low', 'Volume']]
            
            coin.data_df = df
            
            printing("\nupdate_data: comparing coin price")
            compareCoinPrices(coin.symbol, coin.data_df.Close[-1])
            
        except:
            printing("couldnt find coin with symbol {}".format(pair))
            
def update_data_intial(exchange, timeframe, apiKey, pairs=None):

    # resp = requests.get('https://api.cryptowat.ch?apikey=' + apiKey)

    # data = resp.json()

    # printing(data)

    # resp = requests.get('https://api.cryptowat.ch/markets?apikey=' + apiKey)

    # data = resp.json()

    numberOfSymbols = 0
    
    if(pairs == None):
        pairs = trade.get_pairs('USD')

    # for result in data['result']:
        # if(result['exchange'] == market):
            # if(result['pair'][-3:] == 'usd'):
                # numberOfSymbols += 1
                # printing(result)
                # printing(result['pair'])
                # pairs.append(result['pair'])

    # printing("Number of Symbols: {}".format(numberOfSymbols))

    # files = os.listdir(os.curdir + "/data")

    # for file in files:
        # os.remove('data/' + file)
        
    # files = os.listdir(os.curdir + "/dataCSV")

    # for file in files:
        # os.remove('dataCSV/' + file)

    for pair in pairs:
        
        try:
            df, flag = get_data(pair, exchange, timeframe)
        except Exception as ex:
            printing("update_data_intial: problem {}".format(ex))
            
    files = os.listdir(os.curdir + "/savedData")

    for file in files:
        os.remove('savedData/' + file)
        
    files = os.listdir(os.curdir + "/savedDataCSV")

    for file in files:
        os.remove('savedDataCSV/' + file)        
   
    files = os.listdir(os.curdir + "/data")

    for file in files:
        shutil.copy('data/' + file, 'savedData/' + file)
        
    files = os.listdir(os.curdir + "/dataCSV")

    for file in files:
        shutil.copy('dataCSV/' + file, 'savedDataCSV/' + file)
            
def update_coin_price(coin_trade_data_list):
    for coin in coin_trade_data_list:
        pair = coin.symbol
        current_hour = datetime.utcnow().hour+1
        if(datetime.strptime(str(coin.data_df.index[-1]), "%Y-%m-%d %H:%M:%S").hour != current_hour):
            printing("Missing current data for coin {} current_hour: {} coinHour {}".format(pair, current_hour, datetime.strptime(str(coin.data_df.index[-1]), "%Y-%m-%d %H:%M:%S").hour))
            resp = requests.get('https://api.cryptowat.ch/markets/' + market + '/' + pair + '/price?apikey=' + apiKey)
            data = resp.json()
            price = data["result"]["price"]
            newTime = datetime.strptime(str(coin.data_df.index[-1]), "%Y-%m-%d %H:%M:%S") + timedelta(hours=1)
            new_row = pd.Series(data={'Close': price}, name=newTime)
            coin.data_df = coin.data_df.append(new_row)
            printing(coin.data_df.tail())
            
        
def get_pair_info(n_per_in):

    useExtraData = False

    market_data_frames = []
    pairs = []
    
    files = os.listdir(os.curdir + "/data")

    for file in files:
       
        df = pd.read_pickle("data/" + file)
        
        df = df[['Close', 'Open', 'High', 'Low', 'Volume']]#, 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av']]
        
        df['Close'] = df['Close'].astype(float)
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Volume'] = df['Volume'].astype(float)
        # df['quote_av'] = df['quote_av'].astype(float)
        # df['trades'] = df['trades'].astype(float)
        # df['tb_base_av'] = df['tb_base_av'].astype(float)
        # df['tb_quote_av'] = df['tb_quote_av'].astype(float)
        
        # for i in range(len(df)):
            # df.at[df.index[i], 'Volume'] = df.Volume[i] * df.Close[i]
        
        #df = df.tail(500)
        
        
        if(len(df) < n_per_in):
            continue
            
        # printing()
        # printing(file)
            
        # printing(df.tail())
        
        # if(file[0:-4] in badPairs):
            # continue
         
        if(file[0:-4] == 'bnbusd'):
            printing("get_pair_info: Skipping BNB so that current amount can be used for fees")
            continue
        
        market_data_frames.append(df)
        pairs.append(convert_filename_to_pair(file[0:-4]))
        
    if(useExtraData):
    
        files = os.listdir(os.curdir + "/extraData")

        for file in files:
           
            df = pd.read_csv("extraData/" + file)
            df['Date'] = pd.to_datetime(df.Date, format="%Y-%m-%d %I-%p")
            df.set_index('Date', inplace=True)
            df = df[['Close', 'Open', 'High', 'Low', 'Volume USD']]
            
            # df = df[['Close']]
            
            df = df[::-1]
            
            if(len(df) < n_per_in):
                continue
                
            printing()
            printing(file)
            
            printing(df.head())
                
            printing(df.tail())
            
            # if(file[0:-4] in badPairs):
                # continue
                
            olddf = df
            
            index = pairs.index(file[0:-4])
            
            newdf = market_data_frames[index]
                
            dates = olddf.index.tolist()
            newdates = newdf.index.tolist()
            
            if(dates.index(newdates[0]) > -1):
                olddf = olddf.head(dates.index(newdates[0]) + 1)
                
            dates = olddf.index.tolist()
            
            if(newdates.index(dates[-1]) > -1):
                market_data_frames[index] = olddf.append(newdf.tail(len(newdates)-newdates.index(dates[-1]) - 1))
                
            market_data_frames[index].to_csv('testinghighlevel' + file[0:-4] + '.csv')
        
    return market_data_frames, pairs
        
class CoinTradeInfo:
    def __init__(self, symbol, start_amount, data_df, trade_symbol):
        self._symbol=symbol
        self._trade_symbol=trade_symbol
        self._start_amount=start_amount
        self._current_amount=start_amount
        self._free_amount=start_amount
        self._last_action=None
        self._buy_price=0
        self._current_price=0
        self._sell_price=0
        self._data_df=data_df
        self._df=None
        self._number_of_coins=0
        self._sell_number_of_coins=0
        self._sell_worth=0
        self._number_of_successful_orders=0
        self._has_df = False
        self._prune = False
        self._prune_reason = "default prune reason"

    #symbol
    @property
    def symbol(self):
        return self._symbol

    @symbol.setter
    def symbol(self, value):
        self._symbol = value
        
    #trade_symbol
    @property
    def trade_symbol(self):
        return self._trade_symbol

    @trade_symbol.setter
    def trade_symbol(self, value):
        self._trade_symbol = value

    #start_amount
    @property
    def start_amount(self):
        return self._start_amount

    @start_amount.setter
    def start_amount(self, value):
        self._start_amount = value
        
    #current_amount
    @property
    def current_amount(self):
        return self._current_amount

    @current_amount.setter
    def current_amount(self, value):
        self._current_amount = value
        
    #free_amount
    @property
    def free_amount(self):
        return self._free_amount

    @free_amount.setter
    def free_amount(self, value):
        self._free_amount = value
    
    #last_action
    @property
    def last_action(self):
        return self._last_action

    @last_action.setter
    def last_action(self, value):
        self._last_action = value
        
    #current_price
    @property
    def current_price(self):
        return self._current_price

    @current_price.setter
    def current_price(self, value):
        self._current_price = value
        
    #buy_price
    @property
    def buy_price(self):
        return self._buy_price

    @buy_price.setter
    def buy_price(self, value):
        self._buy_price = value
        
    #sell_price
    @property
    def sell_price(self):
        return self._sell_price

    @sell_price.setter
    def sell_price(self, value):
        self._sell_price = value
        
    #df
    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, value):
        self._df = value
        
    #data_df
    @property
    def data_df(self):
        return self._data_df

    @data_df.setter
    def data_df(self, value):
        self._data_df = value
        
    #number_of_coins
    @property
    def number_of_coins(self):
        return self._number_of_coins

    @number_of_coins.setter
    def number_of_coins(self, value):
        self._number_of_coins = value
        
    #sell_number_of_coins
    @property
    def sell_number_of_coins(self):
        return self._sell_number_of_coins

    @sell_number_of_coins.setter
    def sell_number_of_coins(self, value):
        self._sell_number_of_coins = value
        
    #sell_worth
    @property
    def sell_worth(self):
        return self._sell_worth

    @sell_worth.setter
    def sell_worth(self, value):
        self._sell_worth = value
        
    #number_of_successful_orders
    @property
    def number_of_successful_orders(self):
        return self._number_of_successful_orders

    @number_of_successful_orders.setter
    def number_of_successful_orders(self, value):
        self._number_of_successful_orders = value
        
    #has_df
    @property
    def has_df(self):
        return self._has_df

    @has_df.setter
    def has_df(self, value):
        self._has_df = value
        
    #prune
    @property
    def prune(self):
        return self._prune

    @prune.setter
    def prune(self, value):
        self._prune = value
        
    #prune_reason
    @property
    def prune_reason(self):
        return self._prune_reason

    @prune_reason.setter
    def prune_reason(self, value):
        self._prune_reason = value
        
def save_coin_trade_history(coin_trade_data_list):

    files = os.listdir(os.curdir + "/tradeHistoryData")

    for file in files:
        os.remove('tradeHistoryData/' + file)
        
    files = os.listdir(os.curdir + "/tradeHistoryDataCSV")

    for file in files:
        os.remove('tradeHistoryDataCSV/' + file)
        
    for coin in coin_trade_data_list:
    
        if(coin.symbol == "OVERFLOW"):
            continue
    
        try:
            if(coin.has_df):
                df = coin.df
                df.to_pickle('tradeHistoryData/' + convert_pair_to_filename(coin.symbol) + '.pkl')
                df.to_csv('tradeHistoryDataCSV/' + convert_pair_to_filename(coin.symbol) + '.csv')
        except Exception as ex:
            printing("save_coin_trade_history: Couldn't save coin {} error {}".format(coin.symbol, ex))
        
def load_coin_trade_history():

    coin_trade_data_list = []

    files = os.listdir(os.curdir + "/tradeHistoryData")

    for file in files:
    
        df = pd.read_pickle("tradeHistoryData/" + file)
        
        coin_trade_data_list.append(df)
        
    return coin_trade_data_list
    
def print_and_log(statement):
    # global logger
    printing(statement)
    # logger.info(statement)

def calc_sell(actualPrice, buy_price, current_amount, trade_cost):

    new_amount = (((actualPrice - buy_price) / buy_price) * current_amount) + current_amount - (current_amount * trade_cost)
                
    # printing("Selling {} with {} at price {} for a gain of {}".format(symbol, current_amount, actualPrice, (new_amount-current_amount)))
    
    return new_amount
    
def get_current_coin_value(pair):
    
    # resp = requests.get('https://api.cryptowat.ch/markets/' + market + '/' + pair + '/price?apikey=' + apiKey)
    # data = resp.json()
    # price = data["result"]["price"]
    
    if(pair[-4:] == "usdt"):
        resp = requests.get('https://api.cryptowat.ch/markets/' + market + '/' + pair + '/price?apikey=' + apiKey)
        data = resp.json()
        price = data["result"]["price"]
        return float(price)
    
    return trade.get_coin_price(pair)
    
    # printing("get_current_coin_value: Coin: {} price: {}".format(pair, price))
    
def compare_predictions(predictA, predictB):
    for key in predictA.keys():
        printing("\n---------------------------------------------\n coin: {}".format(key))
        printing("len predictA: {} len predictB: {}".format(len(predictA[key]), len(predictB[key])))
        printing("predictA: {} predictB: {}".format(predictA[key], predictB[key]))
        
        correctA = 0
        totalA = 0
        predictions = []
        prices = []
        for tuple in predictA[key]:
            time, price, prediction = tuple
            predictions.append(prediction)
            prices.append(price)
        
        for i in range(len(predictions)-1):
            if(predictions[i] == 0 and prices[i] >= prices[i + 1]):
                correctA += 1
            elif(predictions[i] == 1 and prices[i] < prices[i + 1]):
                correctA += 1
            totalA += 1
            
        accA = correctA/totalA
        
        correctB = 0
        totalB = 0
        predictions = []
        prices = []
        for tuple in predictB[key]:
            time, price, prediction = tuple
            predictions.append(prediction)
            prices.append(price)
        
        for i in range(len(predictions)-1):
            if(predictions[i] == 0 and prices[i] >= prices[i + 1]):
                correctB += 1
            elif(predictions[i] == 1 and prices[i] < prices[i + 1]):
                correctB += 1
            totalB += 1
            
        accB = correctB/totalB
        
        printing("accA: {} accB: {}".format(accA, accB))
        
def compareCoinPrices(pair, dataPrice):

    actualPrice = get_current_coin_value(pair)
    
    if(dataPrice != actualPrice):
        printing("compareCoinPrices: {} differs! dataPrice: {} actualPrice: {}".format(pair, dataPrice, actualPrice))
                    
def add_trade_to_coin(coin, old_amount, price):
    
    if(coin.has_df == False):
        df = pd.DataFrame({'Date': [coin.data_df.index[-1]], 'symbol': [coin.symbol], 'number_of_coins': [coin.number_of_coins], 'last_action': [coin.last_action], 'old_amount': [old_amount], 'new_amount': [coin.current_amount], 'current_price': [coin.current_price], 'buy_price': [coin.buy_price], 'change_percent': [((coin.current_amount - old_amount) / old_amount)], 'amount_based_on_coins': [coin.number_of_coins * coin.current_price], 'number_of_successful_orders' : [coin.number_of_successful_orders], 'trade_symbol_value': [price]})
        df.set_index('Date', inplace=True)
        coin.df = df
        coin.has_df = True
    else:
        new_row = pd.Series(data={'symbol': coin.symbol, 'number_of_coins': coin.number_of_coins, 'last_action': coin.last_action, 'old_amount': old_amount, 'new_amount': coin.current_amount, 'current_price': coin.current_price, 'buy_price': coin.buy_price, 'change_percent': ((coin.current_amount - old_amount) / old_amount), 'amount_based_on_coins': coin.number_of_coins * coin.current_price, 'number_of_successful_orders' : coin.number_of_successful_orders, 'trade_symbol_value': price}, name=coin.data_df.index[-1])
        coin.df = coin.df.append(new_row)
    
def execute_single_trade(coin, trade_symbol_price, model, n_per_in, n_per_out, trade_cost, exchange, overflow_coin):
    
    seq = coin.data_df[-n_per_in:].to_numpy()
        
    # printing("seq {}".format(seq))
    
    seq_x = seq[0:n_per_in, :]
    
    # printing("seq_x {}".format(seq_x))
    
    scalerX = preprocessing.MinMaxScaler(feature_range=(0, 1))
    
    seq_x = scalerX.fit_transform(seq_x)
    
    X = np.array([seq_x])
    
    # printing(coin.symbol)
    # printing(X)
    
    predy = model.predict(X.reshape(len(X), n_per_in, 5))
    
    # printing(predy)
    
    y = predy[0][0]
    
    if(y >= .5):
        y = 1
    else:
        y = 0
        
    #printing("coin: {}".format(coin.symbol))
      
    # predictTuple = (coin.data_df.index[-1], coin.data_df.Close[-1], y)
      
    # predictedValues[coin.symbol].append(predictTuple)
    
    old_amount = coin.current_amount 
    
    coin.current_price = coin.data_df.Close[-1]
    
    trade.cancel_orders(coin.symbol, exchange)
    
    coin.current_amount = (coin.data_df.Close[-1] * coin.number_of_coins) + coin.free_amount
    
    if(y == 1):
        if(coin.last_action == "Buy"):
            coin.current_price = coin.data_df.Close[-1]
            coin.current_amount = (coin.data_df.Close[-1] * coin.number_of_coins) + coin.free_amount
            printing("\nexecute_single_trade: Holding {} of coin {} at price {} amount {}".format(coin.number_of_coins, coin.symbol, coin.data_df.Close[-1], coin.current_amount))
            coin.last_action = "Buy"
            coin.number_of_successful_orders += 1
        elif(coin.last_action == "Sell" or coin.last_action == None):
        
            if(coin.number_of_coins != 0):
                printing("execute_single_trade: Skipping buy coin: {} with coin.number_of_coins: {} did not complete sell".format(coin.symbol, coin.number_of_coins))
                coin.current_price = coin.data_df.Close[-1]
                coin.last_action = "Buy"
            else:
            
                try:
                    coin.current_price = coin.data_df.Close[-1]
                    
                    if(coin.last_action == None):
                        coin.sell_worth = coin.start_amount
                    
                    number_of_coins_to_buy = coin.free_amount/coin.data_df.Close[-1]
                    
                    if(number_of_coins_to_buy > 1):
                        number_of_coins_to_buy = math.floor(number_of_coins_to_buy)
                        if(number_of_coins_to_buy * coin.current_price <= 10): #Logic added to keep coins that would otherwise fall below threshold on binance, 10 USD
                            number_of_coins_to_buy = round(number_of_coins_to_buy)
                        
                    order = trade.buy_coin(coin.symbol, number_of_coins_to_buy, exchange, float(coin.data_df.Close[-1]))
                    
                    time.sleep(60 * 5)
                    
                    number_of_coins_to_buy = float(order['amount'])
                    
                    # if(number_of_coins_to_buy != coin.sell_number_of_coins and coin.sell_number_of_coins != 0 and abs(number_of_coins_to_buy-coin.sell_number_of_coins) < 1): # Removing this logic as it would end up spending more than intended?
                        # printing("execute_single_trade: number_of_coins_to_buy: {} != coin.sell_number_of_coins: {}, buying higher number".format(number_of_coins_to_buy, coin.sell_number_of_coins))
                        # number_of_coins_to_buy = max(number_of_coins_to_buy, coin.sell_number_of_coins)
                    
                    orderId = order['info']['orderId']
                    
                    currentMinute = datetime.utcnow().minute
                    
                    while(currentMinute < 52):
                        
                        newOrder = trade.query_order(orderId, coin.symbol, exchange)
                    
                        if(newOrder['info']['status'] == "FILLED" and float(newOrder['filled']) == float(newOrder['amount'])):
                            time.sleep(60 * 5)
                            coin.number_of_coins = trade.get_number_of_coins(coin.symbol)
                            coin.free_amount = coin.free_amount - (coin.number_of_coins * float(coin.data_df.Close[-1]))
                            coin.buy_price = coin.data_df.Close[-1]
                            coin.last_action = "Buy"
                            coin.number_of_successful_orders += 1
                            printing("\nexecute_single_trade: limitBuyOrder: Bought {} of coin {} at price {}".format(number_of_coins_to_buy, coin.symbol, coin.data_df.Close[-1]))
                            add_trade_to_coin(coin, old_amount, trade_symbol_price)
                            # overflowAmount = coin.sell_worth - coin.current_amount
                            # overflow_coin.current_amount += overflowAmount
                            # if(overflowAmount != 0):
                                # printing("execute_single_trade: Added {} to overFlow {}".format(overflowAmount, overflow_coin.current_amount))
                            return
                        else:
                            bid, ask = trade.get_bid_ask(coin.symbol, exchange)
                            if(ask < coin.sell_price): # if(ask < (1 - (2 * trade_cost)) * coin.sell_price):
                                trade.cancel_orders(coin.symbol, exchange)
                                if(number_of_coins_to_buy > 1):
                                    number_of_coins_to_buy = math.floor(number_of_coins_to_buy)
                                    if(number_of_coins_to_buy * coin.current_price <= 10): #Logic added to keep coins that would otherwise fall below threshold on binance, 10 USD
                                        number_of_coins_to_buy = round(number_of_coins_to_buy)
                                # if(number_of_coins_to_buy != coin.sell_number_of_coins and coin.sell_number_of_coins != 0 and abs(number_of_coins_to_buy-coin.sell_number_of_coins) < 1): # Removing this logic as it would end up spending more than intended?
                                    # printing("execute_single_trade: number_of_coins_to_buy: {} != coin.sell_number_of_coins: {}, buying higher number".format(number_of_coins_to_buy, coin.sell_number_of_coins))
                                    # number_of_coins_to_buy = max(number_of_coins_to_buy, coin.sell_number_of_coins)
                                tradeWorth = trade.buy_coin(coin.symbol, number_of_coins_to_buy, exchange)
                                time.sleep(60 * 5)
                                coin.number_of_coins = trade.get_number_of_coins(coin.symbol)
                                coin.free_amount = coin.free_amount - (coin.number_of_coins * ask)
                                coin.buy_price = ask
                                coin.last_action = "Buy"
                                coin.number_of_successful_orders += 1
                                printing("\nexecute_single_trade: marketBuyOrder: Bought {} of coin {} at price {}".format(number_of_coins_to_buy, coin.symbol, ask))
                                add_trade_to_coin(coin, old_amount, trade_symbol_price)
                                # overflowAmount = coin.sell_worth - coin.current_amount
                                # overflow_coin.current_amount += overflowAmount
                                # if(overflowAmount != 0):
                                    # printing("execute_single_trade: Added {} to overFlow {}".format(overflowAmount, overflow_coin.current_amount))
                                return
                                
                        time.sleep(60 * 5)        
                        
                        currentMinute = datetime.utcnow().minute
                        
                        
                    
                            
                    while(currentMinute < 57):
                    
                        time.sleep(5)
                    
                        currentMinute = datetime.utcnow().minute
                        
                    newOrder = trade.query_order(orderId, coin.symbol, exchange)
                    
                    if(newOrder['info']['status'] == "FILLED" and float(newOrder['filled']) == float(newOrder['amount'])):
                        time.sleep(60 * 2)
                        coin.number_of_coins = trade.get_number_of_coins(coin.symbol)
                        coin.free_amount = coin.free_amount - (coin.number_of_coins * float(coin.data_df.Close[-1]))
                        coin.buy_price = coin.data_df.Close[-1]
                        coin.last_action = "Buy"
                        coin.number_of_successful_orders += 1
                        printing("\nexecute_single_trade: limitBuyOrder: Bought {} of coin {} at price {}".format(number_of_coins_to_buy, coin.symbol, coin.data_df.Close[-1]))
                        add_trade_to_coin(coin, old_amount, trade_symbol_price)
                        # overflowAmount = coin.sell_worth - coin.current_amount
                        # overflow_coin.current_amount += overflowAmount
                        # if(overflowAmount != 0):
                            # printing("execute_single_trade: Added {} to overFlow {}".format(overflowAmount, overflow_coin.current_amount))
                        return
                    else:
                        bid, ask = trade.get_bid_ask(coin.symbol, exchange)
                        if(ask < coin.sell_price): # if(ask < (1 - (2 * trade_cost)) * coin.sell_price):
                            trade.cancel_orders(coin.symbol, exchange)
                            number_of_coins_to_buy = coin.free_amount/ask
                            if(number_of_coins_to_buy > 1):
                                    number_of_coins_to_buy = math.floor(number_of_coins_to_buy)
                                    if(number_of_coins_to_buy * coin.current_price <= 10): #Logic added to keep coins that would otherwise fall below threshold on binance, 10 USD
                                        number_of_coins_to_buy = round(number_of_coins_to_buy)
                            # if(number_of_coins_to_buy != coin.sell_number_of_coins and coin.sell_number_of_coins != 0 and abs(number_of_coins_to_buy-coin.sell_number_of_coins) < 1): # Removing this logic as it would end up spending more than intended?
                                # printing("execute_single_trade: number_of_coins_to_buy: {} != coin.sell_number_of_coins: {}, buying higher number".format(number_of_coins_to_buy, coin.sell_number_of_coins))
                                # number_of_coins_to_buy = max(number_of_coins_to_buy, coin.sell_number_of_coins)
                            tradeWorth = trade.buy_coin(coin.symbol, number_of_coins_to_buy, exchange)
                            time.sleep(60 * 2)
                            coin.number_of_coins = trade.get_number_of_coins(coin.symbol)
                            coin.free_amount = coin.free_amount - (coin.number_of_coins * ask)
                            coin.buy_price = ask
                            coin.last_action = "Buy"
                            coin.number_of_successful_orders += 1
                            printing("\nexecute_single_trade: marketBuyOrder: Bought {} of coin {} at price {}".format(number_of_coins_to_buy, coin.symbol, ask))
                            add_trade_to_coin(coin, old_amount, trade_symbol_price)
                            # overflowAmount = coin.sell_worth - coin.current_amount
                            # overflow_coin.current_amount += overflowAmount
                            # if(overflowAmount != 0):
                                # printing("execute_single_trade: Added {} to overFlow {}".format(overflowAmount, overflow_coin.current_amount))
                            return
                        else:
                            printing("\nexecute_single_trade: limitBuyOrder: Failed to buy {} of coin {} at price {}".format(number_of_coins_to_buy, coin.symbol, coin.data_df.Close[-1]))
                            
                            filled = float(newOrder['filled'])
                            
                            if(filled > 0):
                                printing("execute_single_trade: limitBuyOrder: Failed to buy {} of coin {}, but {} was filled, pruning coin".format(coin.number_of_coins, coin.symbol, filled))
                                trade.cancel_orders(coin.symbol, exchange)
                                coin.prune = True
                                coin.prune_reason = "Failed to buy {} of coin {}, but {} was filled, pruning coin".format(coin.number_of_coins, coin.symbol, filled)
                                # coin.current_amount = filled * coin.data_df.Close[-1]
                                # coin.buy_price = coin.data_df.Close[-1]
                                # coin.last_action = "Buy"
                                # add_trade_to_coin(coin, old_amount, btc_price)
                                # overflowAmount = coin.sell_worth - coin.current_amount
                                # overflow_coin.current_amount += overflowAmount
                                # if(overflowAmount != 0):
                                    # printing("execute_single_trade: Added {} to overFlow {}".format(overflowAmount, overflow_coin.current_amount))
                                # trade.cancel_orders(coin.symbol, exchange)
                                return
                                
                            trade.cancel_orders(coin.symbol, exchange)
                except Exception as ex:
                    printing("execute_single_trade: Could not buy coin: {} with number_of_coins_to_buy: {} error: {}".format(coin.symbol, number_of_coins_to_buy, ex))
                    coin.prune = True
                    coin.prune_reason = "Could not buy coin: {} with number_of_coins_to_buy: {} error: {}".format(coin.symbol, number_of_coins_to_buy, ex)
                    
    elif(y == 0):
        if(coin.last_action == None):
            coin.number_of_successful_orders += 1
        if(coin.last_action == "Sell"):
            printing("\nexecute_single_trade: Holding {} of coin {} at price {} amount {}".format(coin.number_of_coins, coin.symbol, coin.data_df.Close[-1], coin.current_amount))
            coin.current_price = coin.data_df.Close[-1]
            coin.last_action = "Sell"
            coin.number_of_successful_orders += 1
        elif(coin.last_action == "Buy"):
        
            coin.number_of_coins = trade.get_number_of_coins(coin.symbol)
        
            if(coin.number_of_coins == 0):
                printing("execute_single_trade: Skipping sell coin: {} with coin.number_of_coins: {} did not complete buy".format(coin.symbol, coin.number_of_coins))
                coin.current_price = coin.data_df.Close[-1]
                coin.last_action = "Sell"
            else:
                # compareCoinPrices(coin.symbol, coin.data_df.Close[-1])
                
                
                
                coin.sell_price = coin.data_df.Close[-1]
                try:
                    coin.current_price = coin.data_df.Close[-1]
                
                    order = trade.sell_coin(coin.symbol, coin.number_of_coins, exchange, float(coin.data_df.Close[-1]))
                    
                    orderId = order['info']['orderId']
                    
                    time.sleep(60 * 5)
                    
                    currentMinute = datetime.utcnow().minute
                    
                    while(currentMinute < 52):
                        
                        newOrder = trade.query_order(orderId, coin.symbol, exchange)
                    
                        if(newOrder['info']['status'] == "FILLED" and float(newOrder['filled']) == float(newOrder['amount'])):
                            coin.free_amount = coin.free_amount + float(newOrder['info']['cummulativeQuoteQty'])
                            coin.current_amount = coin.free_amount
                            coin.sell_worth = coin.current_amount
                            coin.number_of_successful_orders += 1
                            printing("\nexecute_single_trade: limitSellOrder: Sold {} of coin {} at price {}".format(coin.number_of_coins, coin.symbol, coin.data_df.Close[-1]))
                            coin.last_action = "Sell"
                            coin.sell_price = coin.data_df.Close[-1]
                            coin.sell_number_of_coins = coin.number_of_coins
                            coin.number_of_coins = 0
                            add_trade_to_coin(coin, old_amount, trade_symbol_price)
                            return
                        else:
                            bid, ask = trade.get_bid_ask(coin.symbol, exchange)
                            if(bid > coin.buy_price and float(newOrder['filled']) == 0): #if(bid > (1 + (2 * trade_cost)) * coin.buy_price):
                                trade.cancel_orders(coin.symbol, exchange)
                                coin.free_amount = coin.free_amount + trade.sell_coin(coin.symbol, coin.number_of_coins, exchange)
                                coin.current_amount = coin.free_amount
                                coin.sell_worth = coin.current_amount
                                coin.number_of_successful_orders += 1
                                printing("\nexecute_single_trade: marketSellOrder: Sold {} of coin {} at price {}".format(coin.number_of_coins, coin.symbol, bid))
                                coin.last_action = "Sell"
                                coin.sell_price = bid
                                coin.sell_number_of_coins = coin.number_of_coins
                                coin.number_of_coins = 0
                                add_trade_to_coin(coin, old_amount, trade_symbol_price)
                                
                                return
                                
                        time.sleep(60 * 5)
                        
                        currentMinute = datetime.utcnow().minute
                    
                    while(currentMinute < 57):
                    
                        time.sleep(5)
                        currentMinute = datetime.utcnow().minute
                        
                    newOrder = trade.query_order(orderId, coin.symbol, exchange)
                    
                    if(newOrder['info']['status'] == "FILLED" and float(newOrder['filled']) == float(newOrder['amount'])):
                        coin.free_amount = coin.free_amount + float(newOrder['info']['cummulativeQuoteQty'])
                        coin.current_amount = coin.free_amount
                        coin.sell_worth = coin.current_amount
                        coin.number_of_successful_orders += 1
                        printing("\nexecute_single_trade: limitSellOrder: Sold {} of coin {} at price {}".format(coin.number_of_coins, coin.symbol, coin.data_df.Close[-1]))
                        coin.last_action = "Sell"
                        coin.sell_price = coin.data_df.Close[-1]
                        coin.sell_number_of_coins = coin.number_of_coins
                        coin.number_of_coins = 0
                        add_trade_to_coin(coin, old_amount, trade_symbol_price)
                        
                        return
                    else:
                        bid, ask = trade.get_bid_ask(coin.symbol, exchange)
                        if(bid > coin.buy_price): #if(bid > (1 + (2 * trade_cost)) * coin.buy_price):
                            trade.cancel_orders(coin.symbol, exchange)
                            coin.free_amount = coin.free_amount + trade.sell_coin(coin.symbol, coin.number_of_coins, exchange)
                            coin.current_amount = coin.free_amount
                            coin.sell_worth = coin.current_amount
                            coin.number_of_successful_orders += 1
                            printing("\nexecute_single_trade: marketSellOrder: Sold {} of coin {} at price {}".format(coin.number_of_coins, coin.symbol, bid))
                            coin.last_action = "Sell"
                            coin.sell_price = bid
                            coin.sell_number_of_coins = coin.number_of_coins
                            coin.number_of_coins = 0
                            add_trade_to_coin(coin, old_amount, trade_symbol_price)
                            
                            return
                        else:
                            printing("\nexecute_single_trade: limitSellOrder: Failed to sell {} of coin {} at price {}".format(coin.number_of_coins, coin.symbol, coin.data_df.Close[-1]))
                            
                            filled = float(newOrder['filled'])
                            
                            if(filled > 0):
                                printing("execute_single_trade: limitSellOrder: Failed to sell {} of coin {}, but {} was filled, pruning coin".format(coin.number_of_coins, coin.symbol, filled))
                                coin.prune = True
                                coin.prune_reason = "Failed to sell {} of coin {}, but {} was filled, pruning coin".format(coin.number_of_coins, coin.symbol, filled)
                                # coin.current_amount = filled * coin.data_df.Close[-1]
                                # coin.sell_worth = coin.current_amount
                                # coin.last_action = "Sell"
                                # coin.sell_price = coin.data_df.Close[-1]
                                # coin.sell_number_of_coins = coin.number_of_coins
                                # coin.number_of_coins = 0
                                # add_trade_to_coin(coin, old_amount, btc_price)
                                
                                trade.cancel_orders(coin.symbol, exchange)
                                return                                                                                                                                                                                            
                                
                            trade.cancel_orders(coin.symbol, exchange)
                        
                except Exception as ex:
                    printing("execute_single_trade: Could not sell coin: {} with coin.current_amount: {} coin.number_of_coins: {} error: {}".format(coin.symbol, coin.current_amount, coin.number_of_coins, ex))
                    coin.prune = True
                    coin.prune_reason = "Could not sell coin: {} with coin.current_amount: {} coin.number_of_coins: {} error: {}".format(coin.symbol, coin.current_amount, coin.number_of_coins, ex)
            
    add_trade_to_coin(coin, old_amount, trade_symbol_price)
    
    return
    
def sell_all_coins():

    exchange = trade.get_exchange()
    
    pairs = load_coin_list()
    
    printing(pairs)
    
    walletInfo = trade.get_wallet(exchange)
    
    balances = {}
    
    for coin in walletInfo:
        balances[coin['asset'].upper() + '/USD'] = float(coin['free'])
    
    printing(balances)
    
    
    for pair in pairs:
    
        try:
                
            trade.cancel_orders(pair, exchange)
            
        except Exception as ex:
            printing("sell_all_coins: Couldn't cancel order of coin {} error {}".format(pair, ex))
        
        try:
            if(pair in balances and balances[pair] > 0): 
                trade.sell_coin(pair, balances[pair], exchange)
            
        except Exception as ex:
            printing("sell_all_coins: Couldn't complete selling of coin {} error {}".format(pair, ex))
            
def display_current_coin_value():
    
    coin_trade_data_list = load_coin_trade_history()
    
    btc_price = get_current_coin_value('btcusdt')
    
    totalAmount = 0
    
    for df in coin_trade_data_list:
        
        if(df.number_of_coins[-1] != 0):
            totalAmount += df.number_of_coins[-1] * trade.get_coin_price(df.symbol[-1]) * .99925 * btc_price
            # printing("adding amount number_of_coins {}, {} for {}".format(df.number_of_coins[-1], df.number_of_coins[-1] * trade.get_coin_price(df.symbol[-1]) * .99925 * btc_price, df.symbol[-1]))
        else:
            index = 1
            while(df.number_of_coins[-(index)] == 0):
                index += 1
            totalAmount += df.number_of_coins[-(index)] * df.current_price[-(index)] * .99925 * btc_price
            # printing("adding amount number_of_coins {}, {} for {}".format(df.number_of_coins[-1], df.number_of_coins[-(index)] * df.current_price[-(index)] * .99925 * btc_price, df.symbol[-1]))
    
    printing("display_current_coin_value: Total Amount USD of Trade Bot: {}".format(totalAmount))
    
def save_coin_list(coin_trade_data_list):

    pairs = []
        
    for coin in coin_trade_data_list:
        if(coin.symbol == "OVERFLOW"):
            continue
        pairs.append(coin.symbol)
    df = pd.DataFrame(pairs, columns=['Pair'])
     
    df.to_pickle('coinList.pkl')
    
def load_coin_list():

    pairs = []

    df = pd.read_pickle('coinList.pkl')

    for pair in df.Pair:
    
        pairs.append(pair)
        
    return pairs
    
def save_bad_coin_list(pairs):

    df = pd.DataFrame(pairs, columns=['Pair', 'Reason'])
     
    df.to_pickle('badCoinList.pkl')
    df.to_csv('badCoinList.csv')
    
def load_bad_coin_list():

    pairs = []

    df = pd.read_pickle('badCoinList.pkl')

    for pair in df.Pair:
    
        pairs.append(pair)
        
    return pairs
    
def display_bad_coins():
    pairs = load_bad_coin_list()
    
    printing("\ndisplay_bad_coins: Bad Coins(Removed from trading previously)")
    for pair in pairs:
        printing(pair)

def get_percent_correct_coin_trades():
    
    coin_trade_data_list = load_coin_trade_history()
    
    totalCorrect = 0
    total = 0
    
    for df in coin_trade_data_list:
        totalCorrectCurrent = 0
        totalCurrent = 0
        for i in range(len(df)):
            if(i > 1  and i < len(df)-1):
                if(df.current_price[i] <= df.current_price[i+1] and df.last_action[i] == 'Buy'):
                    totalCorrectCurrent += 1
                if(df.current_price[i] >= df.current_price[i+1] and df.last_action[i] == 'Sell'):
                    totalCorrectCurrent += 1
                totalCurrent += 1
        if(totalCurrent != 0):
            printing("get_percent_correct_coin_trades: Coin {} percent correct predictions = {}".format(df.symbol[0], totalCorrectCurrent/totalCurrent))
        else:
            printing("get_percent_correct_coin_trades: Coin {} zero trades".format(df.symbol[0]))
        totalCorrect += totalCorrectCurrent
        total += totalCurrent
        
    if(total != 0):
        printing("get_percent_correct_coin_trades: Total percent correct {}".format(totalCorrect/total))
    else:
        printing("get_percent_correct_coin_trades: Zero coins/trades error")
        
def sort_percent(e):
    return e['percentCorrect']
    
def sort_amount(e):
    return e['percentGained']

def sort_success(e):
    return e['percentSuccessful']
      
def get_best_coins():
    
    coin_trade_data_list = load_coin_trade_history()
    
    coins = []
    
    try:
    
        for df in coin_trade_data_list:
            totalCorrectCurrent = 0
            totalCurrent = 0
            for i in range(len(df)):
                if(i > 1  and i < len(df)-1):
                    if(df.current_price[i] <= df.current_price[i+1] and df.last_action[i] == 'Buy'):
                        totalCorrectCurrent += 1
                    if(df.current_price[i] >= df.current_price[i+1] and df.last_action[i] == 'Sell'):
                        totalCorrectCurrent += 1
                    totalCurrent += 1
            
            try:
                percentCorrect = totalCorrectCurrent/float(totalCurrent)
                
                amountGained = df.new_amount[len(df)-1]-df.new_amount[0]
                
                percentGained = amountGained/float(df.new_amount[0])
                
                priceChange = df.current_price[len(df)-1]-df.current_price[0]
                
                pricePercent = priceChange/float(df.current_price[0])
                
                number_of_successful_ordersPercent = df.number_of_successful_orders[len(df)-1]/float(len(df))
                        
                coins.append({'coin' : df.symbol[0], 'percentCorrect' : percentCorrect, 'percentSuccessful': number_of_successful_ordersPercent, 'percentGained' : percentGained, 'pricePercent' : pricePercent})
            except Exception as ex:
                printing("get_best_coins: Coin {} failed {}".format(df.symbol[0], ex)) 
            
        coins.sort(reverse=True, key=sort_percent)
        
        # i=0
        # printing("get_best_coins: percentCorrect ranked highest to lowest")
        # printing("i: coin   percentCorrect  percentSuccessful  percentGained  priceChangedPercent")
        # for coin in coins:
            # pc = coin['percentCorrect']
            # ps = coin['percentSuccessful']
            # pg = coin['percentGained']
            # pp = coin['pricePercent']
            # printing("{}: {}:    {}      {}      {}      {}".format(i, coin['coin'], f'{pc:.6f}', f'{ps:.6f}', f'{pg:.6f}', f'{pp:.6f}'))
            # i += 1
            
        df = pd.DataFrame(coins)
     
        df.to_pickle('stats/percentCorrectList.pkl')
        
        df.to_csv('stats/percentCorrectList.csv')
            
        coins.sort(reverse=True, key=sort_amount)

        # i=0
        # printing("get_best_coins: percentGained ranked highest to lowest")
        # printing("coin   percentCorrect  percentSuccessful  percentGained  priceChangedPercent")
        # for coin in coins:
            # pc = coin['percentCorrect']
            # ps = coin['percentSuccessful']
            # pg = coin['percentGained']
            # pp = coin['pricePercent']
            # printing("{}: {}:    {}      {}      {}      {}".format(i, coin['coin'], f'{pc:.6f}', f'{ps:.6f}', f'{pg:.6f}', f'{pp:.6f}'))
            # i += 1
            
        df = pd.DataFrame(coins)
     
        df.to_pickle('stats/percentGainedList.pkl')
        
        df.to_csv('stats/percentGainedList.csv')
            
        coins.sort(reverse=True, key=sort_success)

        # i=0
        # printing("get_best_coins: percentSuccessful ranked highest to lowest")
        # printing("coin   percentCorrect  percentSuccessful  percentGained  priceChangedPercent")
        # for coin in coins:
            # pc = coin['percentCorrect']
            # ps = coin['percentSuccessful']
            # pg = coin['percentGained']
            # pp = coin['pricePercent']
            # printing("{}: {}:    {}      {}      {}      {}".format(i, coin['coin'], f'{pc:.6f}', f'{ps:.6f}', f'{pg:.6f}', f'{pp:.6f}'))
            # i += 1
            
        df = pd.DataFrame(coins)
     
        df.to_pickle('stats/percentSuccessfulList.pkl')
        
        df.to_csv('stats/percentSuccessfulList.csv')
           
    except Exception as ex:
        printing("get_best_coins: Ran in to an unexpected error {}".format(ex))
        
def chop_last_line_data_df(coin_trade_data_list):
    
    for coin in coin_trade_data_list:
        coin.data_df = coin.data_df[:-1]
        
def last_5_of_coin(coin_trade_data_list, pair):
    for coin in coin_trade_data_list:
        if(coin.symbol == pair):
            printing(coin.data_df.tail())
            
def display_current_stats(coin_trade_data_list, start_price, initial_wallet_value, start_amount, number_of_hours_since, last_trade_amount, last_market_amount, last_amount, trade_symbol, bnb_initial_value_symbol):

    trade_symbol_price = get_current_coin_value('USDT/' + trade_symbol)
    
    price_per_coin = start_amount/(len(coin_trade_data_list)-1)
    market_held_value = 0
    
    current_bnb_worth_symbol = trade.get_number_of_coins('BNB/' + trade_symbol) * trade.get_coin_price('BNB/' + trade_symbol)
    
    bnb_spent_symbol = bnb_initial_value_symbol - current_bnb_worth_symbol
    
    for coin in coin_trade_data_list:
        
        if(coin.symbol == "OVERFLOW"):
            continue
    
        percentage = (coin.data_df.Close[-1]-coin.data_df.Close[-2])/coin.data_df.Close[-2]
        
        if(number_of_hours_since+1 > len(coin.data_df.Close)):
            percentage_total = (coin.data_df.Close[-1]-coin.data_df.Close[-len(coin.data_df.Close)])/coin.data_df.Close[-len(coin.data_df.Close)]
        else: 
            percentage_total = (coin.data_df.Close[-1]-coin.data_df.Close[-(number_of_hours_since+1)])/coin.data_df.Close[-(number_of_hours_since+1)]
        
        market_btc = price_per_coin*percentage_total + price_per_coin
        
        market_held_value += market_btc
    
    average_percent_total = (market_held_value-start_amount)/start_amount

    total_current_amount = 0

    for coin in coin_trade_data_list:
    
        current_value = coin.current_amount
            
        total_current_amount += current_value
        
    total_current_amount -= bnb_spent_symbol
        
    percent_change = (trade_symbol_price-start_price)/start_price
    
    total_current_percent_change = (total_current_amount-start_amount)/start_amount
    
    amount_after_initial = initial_wallet_value-start_amount
    
    printing("display_current_stats: initial_wallet_value: {}".format(initial_wallet_value)) 
    printing("display_current_stats: start_amount: {}".format(start_amount)) 
    printing("display_current_stats: amount_after_initial: {}".format(amount_after_initial)) 
    
    current_wallet_value = trade.current_wallet_worth_symbol(trade_symbol)
    printing("display_current_stats: Current {} Value of Wallet(Minus BNB): {}".format(trade_symbol, current_wallet_value)) 
    printing("display_current_stats: Current USD Value of Wallet(Minus BNB): {}".format(current_wallet_value*trade_symbol_price)) 
    
    traded_worth_in_wallet = current_wallet_value-amount_after_initial-bnb_spent_symbol
    
    traded_worth_in_wallet_percent_change = (traded_worth_in_wallet-start_amount)/start_amount
        
    printing()
    printing("display_current_stats: {} Analysis\n".format(trade_symbol))
    printing("display_current_stats: Money Spent {}                 : {} Percentage Change: {}%".format(trade_symbol, round(start_amount, 7), "0.00"))
    printing("display_current_stats: Money Traded Worth {}          : {} Percentage Change: {}%".format(trade_symbol, round(total_current_amount, 7), round(total_current_percent_change * 100, 2)))
    printing("display_current_stats: Money Wallet Traded Worth {}   : {} Percentage Change: {}%".format(trade_symbol, round(traded_worth_in_wallet, 7), round(traded_worth_in_wallet_percent_change * 100, 2)))
    printing("display_current_stats: Money Held(in coins) Worth {}  : {} Percentage Change: {}%".format(trade_symbol, round(market_held_value, 7), round(average_percent_total * 100, 2)))
    printing("display_current_stats: Money Held(in {}) Worth {}    : {} Percentage Change: {}%".format(trade_symbol, trade_symbol, round(start_amount, 7), round(0.00 * 100, 2)))
        
    
    if(trade_symbol != 'USD'):
        start_amountUSD = start_amount * get_current_coin_value(trade_symbol + '/USD')
        
        totalcurrent_amountUSD = total_current_amount * trade_symbol_price
        marketHeldUSD = market_held_value * trade_symbol_price
        btcHeldValueUSD = start_amountUSD * percent_change + start_amountUSD
        total_current_percent_changeUSD = (totalcurrent_amountUSD-start_amountUSD)/start_amountUSD
        
        traded_worth_in_walletUSD = traded_worth_in_wallet * trade_symbol_price
        traded_worth_in_wallet_percent_changeUSD = (traded_worth_in_walletUSD-start_amountUSD)/start_amountUSD
        
        printing()
        printing("display_current_stats: USD Analysis\n")
        printing("display_current_stats: Money Spent USD                 : ${} Percentage Change: {}%".format(round(start_amountUSD, 7), "0.00"))
        printing("display_current_stats: Money Traded Worth USD          : ${} Percentage Change: {}%".format(round(totalcurrent_amountUSD, 7), round(total_current_percent_changeUSD * 100, 2)))
        printing("display_current_stats: Money Wallet Traded Worth USD   : ${} Percentage Change: {}%".format(round(traded_worth_in_walletUSD, 7), round(traded_worth_in_wallet_percent_changeUSD * 100, 2)))
        printing("display_current_stats: Money Held(in coins) Worth USD  : ${} Percentage Change: {}%".format(round(marketHeldUSD, 7), round(average_percent_total * 100, 2)))
        printing("display_current_stats: Money Held(in {}) Worth USD    : ${} Percentage Change: {}%".format(trade_symbol, round(btcHeldValueUSD, 7), round(percent_change * 100, 2)))
    
    total_coin_amount = 0
    
    for coin in coin_trade_data_list:
    
        current_coin_value = coin.current_amount
            
        total_coin_amount += current_coin_value
        
    total_coin_amountUSD = total_coin_amount * trade_symbol_price
    
    printing()
    printing("display_current_stats: Coin Worth Analysis\n")
    printing("display_current_stats: Coin Worth USD : ${}".format(round(total_coin_amountUSD, 7)))
    printing("display_current_stats: Coin Worth {} :  {}".format(trade_symbol, round(total_coin_amount, 7)))
    
    trade_difference = total_current_amount-last_trade_amount
    trade_differenceUSD = trade_difference * trade_symbol_price
    trade_difference_percent = (total_current_amount-last_trade_amount)/last_trade_amount
    
    market_difference = market_held_value-last_market_amount
    market_differenceUSD = market_difference * trade_symbol_price
    market_difference_percent = (market_held_value-last_market_amount)/last_market_amount
    
    price_difference = trade_symbol_price-last_amount
    price_difference_percent = (trade_symbol_price-last_amount)/last_amount
    
    printing()
    printing("display_current_stats: Stats for Last Trade\n")
    printing("display_current_stats: Amount Change Last Trade        : {}: {} USD: ${} Percentage Change: {}%".format(trade_symbol, round(trade_difference, 7), round(trade_differenceUSD, 7), round(trade_difference_percent * 100, 2)))
    printing("display_current_stats: Market Change Last Trade        : {}: {} USD: ${} Percentage Change: {}%".format(trade_symbol, round(market_difference, 7), round(market_differenceUSD, 7), round(market_difference_percent * 100, 2)))
    printing("display_current_stats: {} Change Last Trade           : {}: {} USD: ${} Percentage Change: {}%".format(trade_symbol, trade_symbol, "0.00", round(price_difference, 7), round(price_difference_percent * 100, 2)))
    
    return total_current_amount, market_held_value, trade_symbol_price
    
def prune_bad_coins(coin_trade_data_list):
    names_pruned = []
    reasons_pruned = []
    names_and_reasons_pruned = []
    coinsToPrune = []
    overflow_coin = coin_trade_data_list[-1]
    for i in range(len(coin_trade_data_list)):
        if(coin_trade_data_list[i].prune == True):
            names_and_reasons_pruned.append((coin_trade_data_list[i].symbol, coin_trade_data_list[i].prune_reason))
            overflow_coin.current_amount += coin_trade_data_list[i].current_amount
            
            try:
                shutil.move('tradeHistoryDataCSV/' + coin_trade_data_list[i].symbol.replace("/", "-") + '.csv', 'deletedCoinsCSV/' + coin_trade_data_list[i].symbol.replace("/", "-") + '.csv')
            except Exception as ex:
                printing("prune_bad_coins: problem saving data for deleted coin {} error: {}".format(coin_trade_data_list[i].symbol, ex))
            
            try:
                trade.cancel_orders(coin_trade_data_list[i].symbol, trade.get_exchange())
                trade.sell_all_of_coin(coin_trade_data_list[i].symbol)
            except Exception as ex:
                printing("prune_bad_coins: problem selling/cancelling order of bad coin {} error: {}".format(coin_trade_data_list[i].symbol, ex))
                
            coinsToPrune.append(coin_trade_data_list[i])
    
    for coin in coinsToPrune:
        coin_trade_data_list.remove(coin)
    
    return names_and_reasons_pruned
    
    
def periodize(market_data_frames, period):

    newmarket_data_frames = []
    
    for df in market_data_frames:
    
        newDf = pd.DataFrame(columns = df.columns)
        
        print("length of df: {}, length of period: {}".format(len(df.Close), period))
       
        if(len(df.Close) > period * 2):
            for i in range(0, len(df.Close)-period, period):
            
                #printing("i: {}, date: {}".format(i, df.index[i+period-1]))
            
                new_row = pd.Series({'Date': df.index[i+period-1], 'Close': df.Close[i+period-1], 'Open': df.Close[i], 'High': max(df.High[i:i+period]), 'Low': min(df.Low[i:i+period]), 'Volume': sum(df.Volume[i:i+period])})
                
                newDf = newDf.append(new_row, ignore_index=True)
                
            newDf.set_index('Date', inplace = True)
            
            if(len(newDf.Close) > 0):
                
                newmarket_data_frames.append(newDf)
       
    return newmarket_data_frames
        
def avg_volume(e):
    return e['avgVolume']
    
def load_individual_coin(filepath, pair):
    
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df.Date, format="%Y-%m-%d %I-%p")
    df.set_index('Date', inplace=True)
    df = df[['Close', 'Open', 'High', 'Low', 'Volume USD']]
    df = df.rename(columns = {'Volume USD' : 'Volume'})
    df = df[::-1]
    
    market_data_frames= []
    pairs = []
    
    market_data_frames.append(df)
    pairs.append(pair)
    
    return market_data_frames, pairs
    

 
def test_run_trades():

    ### Setting Initial Variables ####
    
    pd.set_option("display.precision", 10)
    
    btc_price = get_current_coin_value('btcusdt')
    
    n_per_in = 8
    
    amount_to_spend_USD = 100
    
    starting_amount_per_coin_USD = 2.1
    
    trade_cost = .00075
    
    
    #####
    # One of these must be zero, 
    # the other will be used for testing
    ########
    number_of_hours_to_simulate = 0
    
    percent_of_hours_to_simulate = .2
    ########
    
    
    timeframe = '1h'
    
    exchange = trade.get_exchange()
    
    load_model = False
    
    ##################################
    
    # printing("test_run_trades: Performing Initial Data Updates")
    # update_data_intial(exchange, timeframe, apiKey)
        
    printing("test_run_trades: Grabbing pairs/data")
   

    market_data_frames, pairs = get_pair_info(n_per_in)
    
    #market_data_frames, pairs = load_individual_coin("extraData/ethbtc.csv", "ethbtc")
    
    ### Optional Periodization ###
    
    # period_length = 24 #default 1, broken into 1 hour periods
    
    # market_data_frames = periodize(market_data_frames, period_length)
    
    ##############################
    
    if(amount_to_spend_USD > 0):
        starting_amount_per_coin_USD = amount_to_spend_USD/len(market_data_frames)
        
        starting_amount_per_coin_BTC = starting_amount_per_coin_USD/btc_price
        
        amount_to_spend_BTC = amount_to_spend_USD/btc_price
        
    printing("test_run_trades: Spending {} USD {} BTC per coin for {} USD {} BTC total".format(starting_amount_per_coin_USD, round(starting_amount_per_coin_BTC, 7), amount_to_spend_USD, round(amount_to_spend_BTC, 7)))
    
    # averageVol = []
    
    # for i in range(len(pairs)):
        # totalVol = 0
        # for vol in market_data_frames[i].Volume:
            # totalVol += vol
        # averageVol.append(totalVol/len(market_data_frames[i].Volume))
    
    # pairVol = []
    
    # for i in range(len(pairs)):
    
        # pairVol.append({'pair' : pairs[i], 'avgVolume': averageVol[i]})
        
    # pairVol.sort(reverse=True, key=avg_volume)
    
    # pprint.pprinting(pairVol)
    
    # sys.exit()
    
    # if(load_model):
        # n_per_in = 8
        # n_per_out = 1
        # n_features = market_data_frames[0].shape[1]
        # modelBear = create_model(n_per_in, n_features)
        # modelBear.load_weights('modelBear')
        # modelBull = create_model(n_per_in, n_features)
        # modelBull.load_weights('modelBull')
        # modelNeutral = create_model(n_per_in, n_features)
        # modelNeutral.load_weights('modelNeutral')
        
    period_length = 1 #default 1, broken into 1 hour periods
        
    # market_data_frames_period = periodize(market_data_frames, period_length) #  market_data_frames
    
    market_data_frames_period = market_data_frames

    number_of_hours_to_simulate = int(number_of_hours_to_simulate / period_length)
        
    printing("Length of data: {}".format(sum(len(i) for i in market_data_frames_period)))
    
    n_per_in = 5
    future_price_intervals = 1 #dont fuck with this XD
    n_features = market_data_frames_period[0].shape[1]
    n_per_out = 1
    # model = create_model(n_per_in, n_features)
    # model.load_weights('ETHmodel')
    
    printing(market_data_frames_period[0][0:5])
    
    printing(market_data_frames_period[0].dtypes)
    
    #sys.exit()
    
    starting_amount_per_coin_USD = 1000
    
    # model, n_per_in, n_per_out, n_features = train_model(market_data_frames_period, pairs, number_of_hours_to_simulate, True, n_per_in, future_price_intervals, percent_of_hours_to_simulate)
    # model.save_weights('nickTests')
    
    model = create_model_erik_test(n_per_in, n_features)
    model.load_weights('Test4model-npn-5')
     
    # for i in range(len(pairs)):
            
        # enable_console()
        
        #printing("Length of data: {}".format(len(market_data_frames_period[i])))
        #printing("test_run_trades: Running model Tests n_per_in: {} future_price_intervals: {} period_length: {} pair: {}".format(n_per_in, future_price_intervals, period_length, pairs[i]))
        
        # acc, gainpercent = run_tests(model, n_per_in, n_per_out, n_features, 450, starting_amount_per_coin_USD, [market_data_frames_period[i]], [pairs[i]], True, future_price_intervals)
        
        # enable_console()
        # printing("test_run_trades: {} acc: {} gp: {}".format(pairs[i], acc, gainpercent))
        
    acc, gainpercent, gainpercentperhour = 0, 0, 0 
    
    # acc, gainpercent, gainpercentperhour = run_tests(model, n_per_in, n_per_out, n_features, number_of_hours_to_simulate, starting_amount_per_coin_USD, market_data_frames_period, pairs, True, future_price_intervals)
    # run_tests_custom(model, n_per_in, n_per_out, n_features, 30000, starting_amount_per_coin_USD, market_data_frames_period, pairs, True, future_price_intervals)
    #run_tests(model, n_per_in, n_per_out, n_features, number_of_hours_to_simulate, start_amount, market_data_frames, pairs, getEvenSamples, future_price_intervals, percent_of_hours_to_simulate)
    
    if(pairs_to_trade != None):
        new_market_data_frames =  []
        for i in range(len(pairs_to_trade)):
            new_market_data_frames.append(market_data_frames[pairs.index(pairs_to_trade[i])])
        market_data_frames = new_market_data_frames
        pairs = pairs_to_trade
        
    #print(pairs)
    
    # model, n_per_in, n_per_out, n_features = train_model(test_market_data_frames_period, test_pairs, number_of_hours_to_simulate, True, 3, future_price_intervals, percent_of_hours_to_simulate)
    
    # print("\n" * 5)
    # for n_per_in in range(3, 24):
    
        # print("\n\ntest_run_trades: testing n_per_in={}".format(n_per_in))
        
        # for i in range(0, 5):
        
            # print("\nTest {} of 5".format(i+1))
        
            # disable_console()
            # lock_console()
            # model, n_per_in, n_per_out, n_features = train_model(test_market_data_frames_period, test_pairs, number_of_hours_to_simulate, True, n_per_in, future_price_intervals, percent_of_hours_to_simulate)
            # unlock_console()
            # enable_console()
            
            # model.save_weights("Test{}model-npn-{}".format(i+1, n_per_in))
            
            # meanAcc, meanEndamount, meanTestAcc, meanTotalAcc, meanTestEndAmounts, meanTotalEndAmounts = run_simple_tests(model, n_per_in, n_per_out, n_features, 500, 1000, test_market_data_frames_period, test_pairs, True, future_price_intervals, percent_of_hours_to_simulate)
            
            # disable_console()
            # lock_console()
            # meanCustomGain, meanCustomAcc = run_tests_custom(model, n_per_in, n_per_out, n_features, 500, 1000, test_market_data_frames_period, test_pairs, True, future_price_intervals)
            # unlock_console()
            # enable_console()
            
            # print("test_run_trades: meanAcc: {}".format(meanAcc))
            # print("test_run_trades: meanEndamount: {}".format(meanEndamount))
            # print("test_run_trades: meanTestAcc: {}".format(meanTestAcc))
            # print("test_run_trades: meanTestEndAmounts: {}".format(meanTestEndAmounts))
            # print("test_run_trades: meanTotalAcc: {}".format(meanTotalAcc))
            # print("test_run_trades: meanTotalEndAmounts: {}".format(meanTotalEndAmounts))
            # print("test_run_trades: meanCustomAcc: {}".format(meanCustomAcc))
            # print("test_run_trades: meanCustomGain: {}".format(meanCustomGain))
        
    
    run_tests(model, n_per_in, n_per_out, n_features, 0, 1000000, test_market_data_frames_period, test_pairs, True, future_price_intervals, percent_of_hours_to_simulate)

    #printing("test_run_trades: {} acc: {} gp: {}".format("AVERAGE OF ALL COINS", acc, gainpercent))
    
    # sys.exit()
    
def run_trades():
    
    pd.set_option("display.precision", 10)
    
    trade_symbol = 'USD'
    
    price = get_current_coin_value('USDT/' + trade_symbol)
    
    exchange = trade.get_exchange()
    
    n_per_in = 5
    
    amount_to_spend = 400
    
    starting_amount_per_coin = 200
    
    trade_cost = 0.00
    
    timeframe = '1h'
    
    load_model = True
    
    pairs_to_trade = ['BTC/USD', 'ETH/USD']
    
    number_of_coins_to_trade = -1 # -1 indicates trade max number possible
    
    if(pairs_to_trade == None):
       number_of_coins_to_trade = 2 # -1 indicates trade max number possible
    
    printing("run_trades: Performing Initial Data Updates")
    update_data_intial(exchange, timeframe, apiKey, pairs_to_trade)
        
    printing("run_trades: Grabbing pairs/data")
    market_data_frames, pairs = get_pair_info(n_per_in)
    
    if(pairs_to_trade != None):
        new_market_data_frames =  []
        for i in range(len(pairs_to_trade)):
            new_market_data_frames.append(market_data_frames[pairs.index(pairs_to_trade[i])])
        market_data_frames = new_market_data_frames
        pairs = pairs_to_trade
    
    
    # Reduce the number of pairs being traded
    # market_data_frames = market_data_frames[0:int(len(market_data_frames)/4)]
    # pairs = pairs[0:int(len(pairs)/4)]
    
    period_length = 1 #default 1, broken into 1 hour periods
    future_price_intervals = 1 #default 1, x number of hours into the future for prediction
    number_of_hours_to_simulate = 0 #used for testing
        
    market_data_frames_period = market_data_frames # periodize(market_data_frames, period_length)
    
    # sys.exit()
    
    if(load_model):
        n_per_in = 5
        n_per_out = 1
        n_features = market_data_frames[0].shape[1]
        model = create_model_erik_test(n_per_in, n_features)
        model.load_weights('Test4model-npn-5')
    else:
        printing("run_trades: Training Model")
        model, n_per_in, n_per_out, n_features = train_model(market_data_frames_period, pairs, number_of_hours_to_simulate, True, n_per_in, future_price_intervals)
        model.save_weights('model')
        
    
    # market_data_frames = market_data_frames[-10:]
    # pairs = pairs[-10:]
    
    # printing("test_run_trades: Running Tests")
    # run_tests(model, n_per_in, n_per_out, n_features, numberOfHoursToSimulate+n_per_in, starting_amount_per_coin_BTC, market_data_frames, pairs, predictedValuesA)
    
    # sys.exit()
    
    
        
    if(amount_to_spend > 0):
        if(number_of_coins_to_trade == -1):
            starting_amount_per_coin = amount_to_spend/len(market_data_frames)
        elif(number_of_coins_to_trade == 0):
            printing("run_trades: You must trade at least 1 coin, exiting program")
            sys.exit()
        else:
            starting_amount_per_coin = amount_to_spend/number_of_coins_to_trade
        
    elif(starting_amount_per_coin > 0):
        if(number_of_coins_to_trade == -1):
            amount_to_spend = starting_amount_per_coin * len(market_data_frames)
        elif(number_of_coins_to_trade == 0):
            printing("run_trades: You must trade at least 1 coin, exiting program")
            sys.exit()
        else:
            amount_to_spend = starting_amount_per_coin * number_of_coins_to_trade
        
        
        
    printing("run_trades: Spending {} {} per coin for {} {} total".format(starting_amount_per_coin, trade_symbol, amount_to_spend, trade_symbol))
    coin_trade_data_list = []
    
    printing("run_trades: Creating Coin List")
    for i in range(len(pairs)):
    
        pair = pairs[i]
        df = market_data_frames[i]
        
        coin = CoinTradeInfo(pair, starting_amount_per_coin, df, trade_symbol)
        if(number_of_coins_to_trade > 0):
            if((i % int(len(pairs)/number_of_coins_to_trade)) == 0):
                coin_trade_data_list.append(coin)
            if(len(coin_trade_data_list) >= number_of_coins_to_trade):
                break
        else:
            coin_trade_data_list.append(coin)
            
    printing("run_trades: Trading {} Coins: {}".format(len(coin_trade_data_list), coin_trade_data_list))
    
    # sys.exit()
        
    printing("run_trades: Clearing Deleted Coin's Directory")
    
    files = os.listdir(os.curdir + "/deletedCoinsCSV")

    for file in files:
        os.remove('deletedCoinsCSV/' + file)
        
    prune_bad_coins(coin_trade_data_list)
    
    printing("run_trades: Appending Overflow Coin")
    coin = CoinTradeInfo("OVERFLOW", 0, None, trade_symbol)
    coin_trade_data_list.append(coin)
    
    # printing("run_trades: Printing last 5 for coin: {}".format("aebtc"))
    # last_5_of_coin(coin_trade_data_list, "aebtc")
    
    printing("run_trades: Saving Coin List")
    save_coin_list(coin_trade_data_list)
    
    last_trade_amount = amount_to_spend
    last_market_amount = amount_to_spend
    last_amount = price
    number_of_hours_ran = 0
    
    # printing("run_trades: Executing Initial Trade")    
    # amount_to_spend_BTC, coin_trade_data_list = execute_initial_trade(coin_trade_data_list, starting_amount_per_coin, exchange)
    
    initial_wallet_value = trade.current_wallet_worth_symbol(trade_symbol)
    bnb_initial_value_symbol = trade.get_number_of_coins('BNB/' + trade_symbol) * trade.get_coin_price('BNB/' + trade_symbol)
    printing("run_trades: Initial {} Value of Wallet(minus BNB): {}".format(trade_symbol, initial_wallet_value)) 
    
    printing("run_trades: Saving Coin Data")
    save_coin_trade_history(coin_trade_data_list)
    
    list_of_bad_coins = []
    
    # sys.exit()
    
    printing("run_trades: Running Bot")
    while(True):
        
        current_minutes = datetime.utcnow().minute
        
        if(current_minutes < 5):
        
            number_of_hours_ran += 1
             
            printing()
            printing("run_trades: Hour has changed Time: {}".format(datetime.utcnow()))
            
            printing()
            printing("run_trades: Performing Data Updates And Trading")
            try:
                update_data_and_trade(exchange, timeframe, apiKey, coin_trade_data_list, model, n_per_in, n_per_out, trade_cost, trade_symbol)
            except Exception as ex:
                printing("run_trades: Couldn't Update Data and/or Trade trying once again in 15 seconds, Error {}".format(ex))
                time.sleep(15)
                try:
                    update_data_and_trade(exchange, timeframe, apiKey, coin_trade_data_list, model, n_per_in, n_per_out, trade_cost, trade_symbol)
                except Exception as ex:
                    printing("run_trades: Couldn't Update Data and/or Trade exiting program, Error {}".format(ex))
                    sys.exit()
            
            time.sleep(15)
            
            printing()
            printing("run_trades: Saving Coin Data")
            try:
                save_coin_trade_history(coin_trade_data_list)
                save_coin_list(coin_trade_data_list)
            except Exception as ex:
                printing("run_trades: Couldn't Save Coin Data trying once again in 15 seconds, Error {}".format(ex))
                time.sleep(15)
                try:
                    save_coin_trade_history(coin_trade_data_list)
                    save_coin_list(coin_trade_data_list)
                except Exception as ex:
                    printing("run_trades: Couldn't Save Coin Data exiting program, Error {}".format(ex))
                    sys.exit()
            printing()
            
            # printing("run_trades: Pruning bad coins")
            # num_before = len(coin_trade_data_list)
            # names_and_reasons_pruned = prune_bad_coins(coin_trade_data_list)
            # printing("run_trades: {} bad coins pruned len(coin_trade_data_list) before: {} after: {}".format(len(names_and_reasons_pruned), num_before, len(coin_trade_data_list)))
            # printing("run_trades: list of bad coins: {}".format(names_and_reasons_pruned))
            
            # for name, reason in names_and_reasons_pruned:
                # list_of_bad_coins.append((str(int(time.time())), name, reason))
            
            # save_bad_coin_list(list_of_bad_coins)
            
            printing()
            printing("run_trades: Displaying Current Profit/Loss")
            
            last_trade_amount, last_market_amount, last_amount = display_current_stats(coin_trade_data_list, price, initial_wallet_value, amount_to_spend, number_of_hours_ran, last_trade_amount, last_market_amount, last_amount, trade_symbol, bnb_initial_value_symbol)
            
            while(datetime.utcnow().minute <= 5):
                time.sleep(5)



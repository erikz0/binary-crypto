import os
import sys
import functions
from datetime import timedelta
from statistics import mean

historicalData = "data/bitcoin1h.csv"
sizeOfModelData = 5000
marketData = "bitfinex"
symbolData = "btcusd"
periodData = "3600"

binaryModel = None
model = None
n_per_in = 24
n_per_out = 1
n_features = 5
modelWeights = "models/cp.ckpt"

percentTestData = 0.0
epochs = 100
batch_size = 64
earlyStop = True
saveWeights = True
displayResults = True

df = []
normalizedDf = []
close_scaler = None
train_close_scaler = None
testDf = []
trainDf = []
normalizedTrainDf = []

numberToPredict = 500
percentToPredict = .05
prevPredictions = []

predictions = []
shortDf = []
actual = []

startAmount = 1000
tradingFee = .00075

calcPredictions = []
calcShortDf = [] 

calcPredictionsNumberPredicts = 0

def start_app():

    global historicalData, sizeOfModelData, marketData, symbolData, periodData, model, n_per_in, n_per_out, n_features, modelWeights
    global modelWeights, percentTestData, epochs, batch_size, earlyStop, saveWeights, displayResults, df, close_scaler, testDf, numberToPredict
    global percentToPredict, predictions, shortDf, actual, normalizedDf, trainDf, normalizedTrainDf, startAmount, tradingFee, train_close_scaler, calcPredictionsNumberPredicts, binaryModel

    # functions.disable_console()
    
    df, testDf, trainDf, n_features = functions.get_data(periodData, marketData, symbolData, percentTestData=percentTestData, saveData=True, historical=historicalData, size=sizeOfModelData)

    model = functions.create_model(n_per_in, n_per_out, n_features)
    
    binaryModel = functions.create_model_binary(n_per_in, n_per_out, n_features)
    
    normalizedTrainDf, train_close_scaler = functions.normalize_df(trainDf)
    
    normalizedDf, close_scaler = functions.normalize_df(df)
    
    normalizedDf.to_csv("normalizedDf.csv")
    
    # functions.train_model(normalizedTrainDf, epochs, batch_size, model, n_per_in, n_per_out, loadWeights=True, pathToWeights=modelWeights, earlyStop=earlyStop, saveWeights=saveWeights, displayResults=displayResults, close_scaler=train_close_scaler)
    
    # functions.enable_console()

def print_commands():

    print()
    
    print("\thelp : show these tips again\n")
    
    print("\tloaddata : load new data for the model(preset data already loaded)")
    print("\toptions:")
    print("\t\t-file <filename> : (default data/bitcoin1h.csv)")
    print("\t\t-size <sizeOfData> : (default maxSize)")
    print("\t\t-percentTestData <percentage(float)> : percentage of data to use for testing (default .005)")
    print("\t\t-market <market> : (default bitfinex)")
    print("\t\t-symbol <symbol> : (default btcusd)")
    print("\t\t-period <period> : (default 3600)")
    print()
    
    print("\ttrainmodel : train the model(model is pretrained/loaded with weights)")
    print("\toptions:")
    print("\t\t-loadmodel : load model weights from file (default models/cp.ckpt)")
    print("\t\t-epochs <epochs> : (default 100)")
    print("\t\t-batch_size <batch size> : (default 32)")
    print("\t\t-earlystop <True/False> : (default True)")
    print("\t\t-saveWeights <True/False> : (default True)")
    print("\t\t-displayResults <True/False> : Display Loss/Accuracy Charts (default False)")
    print()
    
    print("\tpredict : calculate predictions (default None)")
    print("\toptions:")
    print("\t\t-numberToPredict <size(int)> : (default 500)")
    print("\t\t-percentToPredict <percentage(float)> : (default .05)")
    print()
    
    print("\tmarketprice : commands to get the markets current prices")
    print("\toptions:")
    print("\t\t-current : Get the current periods market price (default last whole hour)\n")
    print("\t\t-next : Get the next periods (predicted) market price (default next whole hour)\n")
    print("\t\t-last5 : Get the last 5 periods market price (default last 5 hours)\n")
    print("\t\t-last5predicted : Get the last 5 periods market price (default last 5 hours)\n")
    print()
    
    print("\tplotpredictions : Plot Predictions vs Actual Data, calculates predictions if not done already\n")
    print("\toptions:")
    print("\t\t-includeNext : include the next predicted price beyond (default next whole hour)\n")
    print("\t\t-prev : weird option for erik to diagnose his autism\n")
    print()
    
    print("\tcalcgains : run test trading on past data(default data is percentTestData)")
    print("\toptions:")
    print("\t\t-start <amount in USD>: how much money to start with in dollars worth of btc (default 1000)")
    print("\t\t-tradingFee <percentage(float)> : percent of each trade for a fee(default (.00075).075%)")
    print("\t\t-info <True/False> : show detailed information about trades(default True)")
    print("\t\t-noloss : disable trading for a loss(default off)")
    print("\t\t-numberHours <numberHours(int)> : number of hours to calc_gain for(defaults to percentTestData)")
    print("\t\t-opposite : turn on the opposite strategy")
    print("\t\t-sequence : run calcgains on hours [numberHours->0] and only display results")
    print("\t\t-conservebuy : conservebuy")

    print()        
    
    print("\ttime : time operations")
    print("\toptions:")
    print("\t\t-latestData : time of most recent data point")
    print("\t\t-lastHour : time of most recent whole hour(should line up with latestData)")
    print("\t\t-realUTC : time UTC+0")
    print()
    
    print("\tprintmodel : print model summary")
    
    print("\tsaveAll : save all data currently being used (default data/<files>)")
    
    print("\texit: exit and terminate the trading bot\n")
    
def handle_command(command):

    global historicalData, sizeOfModelData, marketData, symbolData, periodData, model, n_per_in, n_per_out, n_features, modelWeights
    global modelWeights, percentTestData, epochs, batch_size, earlyStop, saveWeights, displayResults, df, close_scaler, testDf, numberToPredict
    global percentToPredict, predictions, shortDf, actual, normalizedDf, trainDf, normalizedTrainDf, startAmount, tradingFee, train_close_scaler, calcPredictionsNumberPredicts
    global calcPredictions, calcShortDf, prevPredictions, binaryModel

    if(command == "exit"):
        #Add other exit routines
        return "exit"
        
    elif(command == "help"):
        print_commands()
        return None  
        
    elif("loaddata" in command):
    
        if("-file" in command):
            file = grabValue(command, "-file")
            historicalData = file
            
        if("-size" in command):
            size = grabValue(command, "-size")
            sizeOfModelData = size
            
        if("-percentTestData" in command):
            percentTestData = float(grabValue(command, "-percentTestData"))
        
        if("-market" in command):
            market = grabValue(command, "-market")
            marketData = market
            
        if("-symbol" in command):
            symbol = grabValue(command, "-symbol")
            symbolData = symbol
            
        if("-period" in command):
            period = grabValue(command, "-period")
            periodData = period
            
        df, testDf, trainDf = functions.get_data(periodData, marketData, symbolData, percentTestData=percentTestData, saveData=False, historical=historicalData, size=sizeOfModelData)
    
        normalizedTrainDf, train_close_scaler = functions.normalize_df(trainDf)
    
        normalizedDf, close_scaler = functions.normalize_df(df)
        
    elif("trainmodel" in command):
    
        loadWeights = False
        pathToWeights = modelWeights
        binary = False
       
        if("-evaluate" in command):
            
            testDf.to_csv("Test df xasd.csv")
            
            normalizedTestDf, test_close_scaler = functions.normalize_df(testDf)
            
            X_test, y_test = functions.split_sequence_binary(normalizedTestDf.to_numpy(), n_per_in, df, test_close_scaler)
        
            scores = binaryModel.evaluate(X_test, y_test, verbose=1)
            
            print(scores)
            return
        
        if("-loadmodel" in command):
            file = grabValue(command, "-loadmodel")
            loadWeights = True
            pathToWeights = file
            
        if("-epochs" in command):
            epochs = int(grabValue(command, "-epochs"))
            
        if("-batch_size" in command):
            batch_size = int(grabValue(command, "-batch_size"))
            
        if("-earlystop" in command):
            earlyStop = grabValue(command, "-earlystop")
            
        if("-saveWeights" in command):
            saveWeights = grabValue(command, "-saveWeights") 
            
        if("-displayResults" in command):
            displayResults = grabValue(command, "-displayResults")  
        
        
         
        if("-binary" in command):
            binary = True
            
        if(binary):
            functions.train_model_binary(normalizedTrainDf, epochs, batch_size, binaryModel, n_per_in, n_per_out, loadWeights=loadWeights, pathToWeights=pathToWeights, earlyStop=earlyStop, saveWeights=saveWeights, displayResults=displayResults, close_scaler=train_close_scaler)
        else:
            functions.train_model(normalizedTrainDf, epochs, batch_size, model, n_per_in, n_per_out, loadWeights=loadWeights, pathToWeights=pathToWeights, earlyStop=earlyStop, saveWeights=saveWeights, displayResults=displayResults, close_scaler=train_close_scaler)
        
    
    
        
    elif("plotpredictions" in command):
        
        if(len(predictions) == 0):
            predictions, shortDf, prevPredictions = functions.get_predictions(normalizedDf, n_per_in, n_per_out, n_features, close_scaler, model, False, numberPredictions=numberToPredict)
            
        actual = functions.transform_back(normalizedDf, close_scaler)
        
        if("-prev" in command):
            functions.plot_predictions_actual_prev(predictions, actual, prevPredictions)
        else:
            functions.plot_predictions_actual(predictions, actual)
        
    elif("predict" in command):
    
        timeRemaining = True
        binary = False
        
        if("-timeRemaining" in command):
            timeRemaining = grabValue(command, "-timeRemaining")
        if("-binary" in command):
            binary = True
            
        if("-numberToPredict" in command):
            numberToPredict = int(grabValue(command, "-numberToPredict"))
            
        if("-percentToPredict" in command):
            percentToPredict = float(grabValue(command, "-percentToPredict"))
            numberToPredict = (int(percentToPredict * len(df)))
            
        if(binary):
            predictions, shortDf, prevPredictions = functions.get_predictions_binary(normalizedDf, n_per_in, n_per_out, n_features, close_scaler, model, timeRemaining, numberPredictions=numberToPredict)
        else:
            predictions, shortDf, prevPredictions = functions.get_predictions(normalizedDf, n_per_in, n_per_out, n_features, close_scaler, model, timeRemaining, numberPredictions=numberToPredict)
            
    elif("marketprice" in command):     
    
        Tdf, TtestDf, TtrainDf = functions.get_data(periodData, marketData, symbolData, percentTestData=0.0, saveData=False, historical=historicalData, size=sizeOfModelData)
    
        TnormalizedTrainDf, _ = functions.normalize_df(TtrainDf)
    
        TnormalizedDf, Tclose_scaler = functions.normalize_df(Tdf)
        
        smallPredictions, smallShortDf, _ = functions.get_predictions(TnormalizedDf, n_per_in, n_per_out, n_features, Tclose_scaler, model, False, numberPredictions=n_per_in+5)
        
        if("-current" in command):
            print("Current real price as of: " + str(df.tail(1).index[0]) + " : " + str(df.tail(1).Close[0]))
            print("Current predicted price as of: " + str(smallPredictions.tail(1).index[0]) + " : " + str(smallPredictions.tail(1).Close[0]))
            
        if("-next" in command):
            nextPrice, prevPrice = functions.get_next_predicted_price(normalizedDf, n_per_in, n_features, model, close_scaler)
            print("Current predicted price as of: " + str(smallPredictions.tail(1).index[0]) + " : " + str(prevPrice))
            print("Next predicted price as of: " + str(smallPredictions.tail(1).index[0]+timedelta(hours=1)) + " : " + str(nextPrice))
            
    
    elif("calcgains" in command):
    
        showInfo = True
        noLoss = False
        numberHours = int(percentTestData * len(df))
        opposite = False
        cashout = False
        conservebuy = False
        
        if("-start" in command):
            startAmount = int(grabValue(command, "-start"))
            
        if("-tradingFee" in command):
            tradingFee = float(grabValue(command, "-tradingFee"))
         
        if("-info" in command):
            showInfo = grabValue(command, "-info")
        
        if("-noloss" in command):
            noLoss = True
            
        if("-numberHours" in command):
            numberHours = int(grabValue(command, "-numberHours")) 
        
        if("-opposite" in command):
            opposite = True
        
        if("-cashout" in command):
            cashout = True
            
        if("-conservebuy" in command):
            conservebuy = True
            
        if(numberHours+n_per_in != calcPredictionsNumberPredicts):
            
            calcPredictionsNumberPredicts = numberHours+n_per_in
                
            calcPredictions, calcShortDf, _ = functions.get_predictions(normalizedDf, n_per_in, n_per_out, n_features, close_scaler, model, True, numberPredictions=calcPredictionsNumberPredicts)
            
            actual = functions.transform_back(normalizedDf, close_scaler)
            
        if("-sequence" in command):
            avgList = []
            print("Length calcPredictions: " + str(len(calcPredictions)))
            print("Length n_per_in: " + str(n_per_in))
            print("len(calcPredictions)-n_per_in: " + str(len(calcPredictions)-n_per_in))
            for i in range(len(calcPredictions), n_per_in, -1):
                # print("yo")
                endAmount = functions.calc_gain(actual, calcPredictions.tail(i), startAmount, tradingFee, n_per_in, n_per_out, noLoss, cashout, opposite, showInfo=False)
                # print(endAmount)
                gain = endAmount-startAmount
                gainPerHour = gain/(i-n_per_in)
                print("Gain per hour iteration(" + str(i-n_per_in) + ") : " + str(gainPerHour))
                avgList.append(gainPerHour)
                
            print("\n\nFinal Sequence Statistics: \n\n")
            
            print("Starting dollar amount: " + str(startAmount) + "\n")
            
            startBTC = float(actual.tail(len(calcPredictions))["Close"][n_per_in-1])
            endBTC = float(actual.tail(1)["Close"][0])
            avgBTCChange = ((endBTC-startBTC)/(len(calcPredictions)-n_per_in))*(startAmount/startBTC)
            
            print("Avg dollar gain per hour: " + str(mean(avgList)))
            print("Avg BTC change per hour(dollars): " + str(avgBTCChange))
            print()
            
            endAmountAvg = (mean(avgList)*(len(calcPredictions)-n_per_in)) + startAmount
            endBTCHeld = ((endBTC-startBTC)/startBTC) * startAmount + startAmount
            print("Ending dollar amount traded (average): " + str(endAmountAvg))
            print("Ending dollar amount if invested/held: " + str(endBTCHeld))
            print()
            
            percentChangeBTC = (((endBTC-startBTC)/startBTC)*100)
            percentChangePort = (((endAmountAvg-startAmount)/startAmount)*100)
            print("Percent change in BTC: " + str(round(percentChangeBTC, 2)) + "%")
            print("Percent change in Portfolio: " + str(round(percentChangePort, 2)) + "%")
            print()
            
            maxGainHour = max(avgList)
            indexMax = len(calcPredictions)-avgList.index(maxGainHour)-n_per_in
            print("Highest Gain/Hour: " + str(maxGainHour))
            print("Highest Gain From Hour: " + str(indexMax))
            print()
            
        else:
            functions.calc_gain(actual, calcPredictions, startAmount, tradingFee, n_per_in, n_per_out, noLoss, cashout, conservebuy, opposite, showInfo=showInfo) 
        
    elif("printmodel" in command):      
        print()
        model.summary()
        print()
    return None

def grabValue(command, param):
    value = None
    try:
        startIndex = command.index(param)+len(param)+2
        endIndex = startIndex + command[startIndex:].index(">")
        
        value = command[startIndex:endIndex]
        
        print("Value: " + str(value))
        
    except Exception as ex:
        print("Missing " + param + " <> " + str(ex))
        
    if(value != None and value.lower() == "true"):
        value = True
    elif(value != None and value.lower() == "false"):
        value = False
        
    return value
        
        
    
import multi

def handle_command(command):

    if(command == "exit"):
        #Add other exit routines
        return "exit"
        
    # elif(command == "help"):
        # print_commands()
        # return None
        
    # elif(command == "currentStats"):
        # multi.display_current_stats()
        
    elif(command == "currentCoinValue"):
        multi.display_current_coin_value()
        
    elif(command == "tradePercents"):
        multi.get_percent_correct_coin_trades()
        
    elif(command == "compareCoinAmounts"):
        multi.compareCoinAmounts()
        
    elif(command == "getBestCoins"):
        multi.get_best_coins()
        
    elif(command == "getBadCoins"):
        multi.display_bad_coins()
        
    elif(command == "sellAllCoins"):
        multi.sell_all_coins()
        
    elif(command == "startTradeBot"):
        return "startBot"
        
    elif(command == "startTestTrades"):
        return "startTest"
    
    
    
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
    
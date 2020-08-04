import logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import functions
#import trade
import commands
#import ccxt

# exchange = ccxt.binance()

# exchange = trade.set_auth_info(exchange)
   
# trade.print_wallet(exchange.fetch_balance())

exit = False

print()

print("-" * 70)

print()

print("Welcome to the Red Sun\n")

print("-" * 70)

print()

commands.start_app()

while(exit==False):

    print("$", end =" ")
    inputText = input()
    
    flag = commands.handle_command(inputText)
    
    if(flag == "exit"):
        exit = True
        functions.disable_console()
        continue
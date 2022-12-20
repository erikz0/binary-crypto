import logging
import os
import threading
import multi
import commands
from printer import *

exit = False

printing()

printing("-" * 70)

printing()

printing("Welcome to the Red Sun 2\n")

printing("-" * 70)

printing()

test = threading.Thread(target=multi.test_run_trades, args=(), daemon=True)

tradeBot = threading.Thread(target=multi.run_trades, args=(), daemon=True)

while(exit==False):

    printing("$ ", terminator = "!n")
    inputText = input()
    
    flag = commands.handle_command(inputText)
    
    if(flag == "exit"):
        exit = True
        multi.disable_console()
        continue
    elif(flag == "startBot"):
        tradeBot.start()
    elif(flag == "startTest"):
        test.start()
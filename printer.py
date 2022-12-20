import logging
import time
import sys

red_logger = logging.getLogger()
red_logger.setLevel(logging.INFO)

output_file_handler = logging.FileHandler("redsun-" + str(time.time()) + ".log")
stdout_handler = logging.StreamHandler(sys.stdout)

red_logger.addHandler(output_file_handler)
red_logger.addHandler(stdout_handler)

logger_disabled = False

def printing(string="", one="", two="", three="", terminator = "newline"):
    global logger_disabled, logger
    if(logger_disabled == False):
        if(terminator == "!n"):
            for handler in red_logger.handlers:
                handler.terminator = ""
            string = str(string) + str(one) + str(two) + str(three)
            red_logger.info("[" + str(int(time.time())) + "] " + string)
            for handler in red_logger.handlers:
                handler.terminator = "\n"
        else:
            string = str(string) + str(one) + str(two) + str(three)
            red_logger.info("[" + str(int(time.time())) + "] " + string)
        
        
def disable_logger():
    global logger_disabled
    logger_disabled = True

def enable_logger():
    global logger_disabled
    logger_disabled = False
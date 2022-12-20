# Red Sun
Predict Cryptocurrency Price Movement
-----------------------------------------------------------------------------------------------

Nicknamed the "Red Sun" this trading bot is capable of changing how we view speculative trading entirely. 

# Background

Since I first heard of Bitcoin at 13 years old(2012), til buying Dogecoin as a joke at 14 years old(2013) and losing the harddrive, and onto becoming a massive supporter of cryptocurrency in 2017, it has been a curious itch of mine to discover the hidden patterns behind the markets and predict the future. I hope to find a way to turn lead into gold.

# Description

This project is being designed as an application with a command-line interface. 

It implements Tensorflow, sklearn, keras, ta, numpy, pandas, matplotlib...

It connects to the BinanceAPI as well as the KrakenAPI to get hourly cryptocurrency data and turn it into a machine learning ready data format.

Its learning algorithm is fairly simple, it looks at a sequence of previous hours data and makes a prediction, up or down, and trades accordingly.

As this program is designed, it concurrently trades 40+ crypto currencies and keeps track of all of their history of trades. This spread out approach to trading ensures that a generalized machine learning algorithm can be applied irrespective of individual differences in coins price action.

An Example of Automated Trading:

![image](https://user-images.githubusercontent.com/43886647/144365738-c07755c7-4e7c-4acf-8688-33f22ba2c824.png)









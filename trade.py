import ccxt
import pprint
import pandas as pd
import time
import historical_scrape
from printer import *

# print (ccxt.exchanges)

# markets = exchange.load_markets ()

# print (exchange.id)

# btcMarket = exchange.markets['BTC/USDT']

#pprint.pprinting(btcMarket)

# pprint.pprinting(dir(exchange))

# printing("Creds: " + str(exchange.checkRequiredCredentials()))

# printing("Wallet Address: " + str(exchange.myTrades))

#pprint.pprinting(exchange.fetch_balance()['info']['balances'])

authInfos = {
    'binance': { 
        'apiKey': '<insert_apiKey_for_binance>',
        'secret': '<insert_secret_for_binance>'
    },
    'binanceus': { 
        'apiKey': '<insert_apiKey_for_binanceus>',
        'secret': '<insert_secret_for_bbinanceus>'
    }
    
}

def get_exchange():
    exchange = ccxt.binanceus()
    exchange = set_auth_info(exchange, 'binanceus')
    return exchange

def set_auth_info(exchange, exchangeName):
    exchange.apiKey = authInfos[exchangeName]['apiKey']
    exchange.secret = authInfos[exchangeName]['secret']
    
    return exchange

def get_wallet(exchange):
    
    walletInfo = exchange.fetch_balance()

    balances = walletInfo['info']['balances']
    # for coin in balances:
        # if(not float(coin['free']) == 0):
            # printing(coin)
            
    return balances
            
def sell_coin(pair, amount, exchange, price=None):
    
    if(price == None):
        
        printing("\nsell_coin: marketSellOrder: Selling {} of coin {}".format(amount, pair))
    
        order = exchange.createMarketSellOrder(pair, amount)
        
        return float(order['info']['cummulativeQuoteQty'])
    
    else:
    
        printing("\nsell_coin: limitSellOrder: Attempting to Sell {} of coin {} at price {}".format(amount, pair, price))
        
        order = exchange.createLimitSellOrder(pair, amount, price)
        
        return order
    
def buy_coin(pair, amount, exchange, price=None):
    
    if(price == None):
    
        printing("\nbuy_coin: marketBuyOrder: Buying {} of coin {}".format(amount, pair))
    
        order = exchange.createMarketBuyOrder(pair, amount)
        
        return float(order['info']['cummulativeQuoteQty'])
    
    else:
        
        printing("\nbuy_coin: limitBuyOrder: Attempting to buy {} of coin {} at price {}".format(amount, pair, price))
        
        order = exchange.createLimitBuyOrder(pair, amount, price)
        
        return order
   
def sell_all_of_coin(pair):

    exchange = get_exchange()
    
    numberOfCoins = get_number_of_coins(pair, exchange)

    order = exchange.createMarketSellOrder(pair, numberOfCoins)
        
    printing("\nsell_coin: marketSellOrder: Selling {} of coin {}".format(numberOfCoins, pair))
    
def get_pair_data(pair, exchange, timeframe):
    
    df = historical_scrape.get_all_binance(pair.replace("/", "-"), timeframe, save = True)
    
    df['Close'] = df['Close'].astype(float)
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Volume'] = df['Volume'].astype(float)
    df['quote_av'] = df['quote_av'].astype(float)
    df['trades'] = df['trades'].astype(float)
    df['tb_base_av'] = df['tb_base_av'].astype(float)
    df['tb_quote_av'] = df['tb_quote_av'].astype(float)
    
    df = df[~df.index.duplicated()]
    
    return df
    
def get_binance_pair(pair):

    return pair[0:-3].upper() + "/" + pair[-3:].upper()
    
def get_coin_price(pair):
    exchange = get_exchange()
    return float(exchange.fetchTicker(pair)['close'])

def get_pairs(currency):
    if(currency.lower() == 'usd'):
        return get_usd_pairs()
    
def get_usd_pairs():
    exchange = get_exchange()
    markets = exchange.load_markets()
    
    pairs = []

    for pair in list(markets.keys()):
        if("/USD" in pair[-4:]):
            pairs.append(pair)
            
    return pairs

    
def current_wallet_btc_worth():
    exchange = get_exchange()
    balances = get_wallet(exchange)
    
    totalBTCAmount = 0
    for coin in balances:
        if(float(coin['free']) == 0 and float(coin['locked']) == 0):
            continue
        if(coin['asset'] == 'BTC'):
            totalBTCAmount += float(coin['free'])
            totalBTCAmount += float(coin['locked'])
            continue
        if(coin['asset'] == 'BNB'):
            continue
        try:
            priceCoin = exchange.fetchTicker(coin['asset'] + "/BTC")['close']
            numberOfCoins = float(coin['free'])
            numberOfCoins += float(coin['locked'])
            btcAmount = float(priceCoin) * float(numberOfCoins)
            totalBTCAmount += float(btcAmount)
        except Exception as ex:
            printing("current_wallet_btc_worth: Couldn't find amount for asset {} free {} error: {}".format(coin['asset'], coin['free'], ex))
        
    return totalBTCAmount
    
def current_wallet_worth_symbol(symbol):
    exchange = get_exchange()
    balances = get_wallet(exchange)
    
    totalAmount = 0
    for coin in balances:
        if(float(coin['free']) == 0 and float(coin['locked']) == 0):
            continue
        if(coin['asset'] == symbol):
            totalAmount += float(coin['free'])
            totalAmount += float(coin['locked'])
            continue
        if(coin['asset'] == 'BNB'):
            continue
        try:
            priceCoin = exchange.fetchTicker(coin['asset'] + "/" + symbol)['close']
            numberOfCoins = float(coin['free'])
            numberOfCoins += float(coin['locked'])
            btcAmount = float(priceCoin) * float(numberOfCoins)
            totalAmount += float(btcAmount)
        except Exception as ex:
            printing("current_wallet_worth_symbol: Couldn't find amount for asset {} free {} error: {}".format(coin['asset'], coin['free'], ex))
        
    return totalAmount
    
def query_order(orderId, pair, exchange):

    numberOfRetries = 0
    
    error = True
    
    exception = ""

    while(error == True and numberOfRetries < 10):

        try:

            order = exchange.fetchOrder(orderId, pair)
            
            error = False
        
        except Exception as ex:
            
            printing("query_order: Couldn't query order retrying in 5 seconds error: {}".format(ex))
            
            exception = ex
            
        time.sleep(5)
        
    if(error == True):
        raise Exception(exception)

    return order

def cancel_orders(pair, exchange):
    try:
        exchange.cancelAllOrders(pair)
    except Exception as ex:
        printing("Trade.cancel_orders: failed on pair {} ex: {}".format(pair, ex))
    
def get_bid_ask(pair, exchange):
    orderbook = exchange.fetch_order_book (pair)
    bid = orderbook['bids'][0][0] if len (orderbook['bids']) > 0 else None
    ask = orderbook['asks'][0][0] if len (orderbook['asks']) > 0 else None
    return bid, ask

def get_number_of_coins(pair):

    exchange = get_exchange()
    
    balances = get_wallet(exchange)
    
    numberOfCoins = 0
    
    # printing("trying to find number of coins for {}".format(pair))
    
    for coin in balances:
    
        # if(float(coin['free']) > 0 or float(coin['locked']) > 0):
            # printing(coin)
            # printing("Asset: {} Pair: {}".format(coin['asset'], pair[0:-3].upper()))
        
        if(coin['asset'] == pair[0:len(coin['asset'])].upper()):
            numberOfCoins += float(coin['free'])
            numberOfCoins += float(coin['locked'])
            # printing("returning number of coins: {}".format(numberOfCoins))
            return numberOfCoins
            
    # printing("returning number of coins: {}".format(numberOfCoins))        
    return numberOfCoins
            
# exchange = set_auth_info(exchange)

# exchange = get_exchange()

# printing(get_pair_data('XLM/USD', exchange, '1h')[0:5])

# pprint.pprinting(markets)

# walletInfo = exchange.fetch_balance()


# exchange.cancelAllOrders("AE/BTC")
# exchange.cancelAllOrders("ANKR/BTC")
# exchange.cancelAllOrders("ANT/BTC")
# exchange.cancelAllOrders("AE/BTC")
# exchange.cancelAllOrders("AE/BTC")

# balances = get_wallet(exchange)
    
# balances = walletInfo['info']['balances']
    
# numberOfCoins = 0

# pair = "yfiibtc"

# printing("trying to find number of coins for {}".format(pair))

# for coin in balances:

    # if(float(coin['free']) > 0 or float(coin['locked']) > 0):get_coin_price
        # printing(coin)
        # printing("Asset: {} Pair: {}".format(coin['asset'], pair[0:-3].upper()))
    
    # if(coin['asset'] == pair[0:-3].upper()):
        # numberOfCoins += float(coin['free'])
        # numberOfCoins += float(coin['locked'])
        # printing("returning number of coins: {}".format(numberOfCoins))
        
# printing("returning number of coins: {}".format(numberOfCoins))

# pprint.pprinting(exchange.createLimitBuyOrder('HOT/BTC', 5000, 0.00000004))

# pprint.pprinting(exchange.createMarketBuyOrder('HOT/BTC', 5000))


# pprint.pprinting(exchange.fetch_my_trades('ADA/BTC'))

# pprint.pprinting(exchange.fetchOrder(19204813, 'HOT/BTC'))

# orderbook = exchange.fetch_order_book ('HOT/BTC')
# bid = orderbook['bids'][0][0] if len (orderbook['bids']) > 0 else None
# ask = orderbook['asks'][0][0] if len (orderbook['asks']) > 0 else None
# spread = (ask - bid) if (bid and ask) else None
# print (exchange.id, 'market price', { 'bid': bid, 'ask': ask, 'spread': spread })

# pprint.pprinting(dir(exchange))

# data = exchange.fetchOHLCV('ADA/BTC', '1h')

# newdf = pd.DataFrame(data, columns=[
            # 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'
        # ])
        
# newdf['Date'] = pd.to_datetime(newdf.Date, unit='ms')       
        
# printing(newdf.tail())


# exchange.createMarketSellOrder('BTC/USDT', .001)

# print_wallet(exchange.fetch_balance())

# sell_coin('adabtc', 1.00, exchange)

# print_wallet(exchange.fetch_balance())
        
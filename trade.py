import ccxt
import pprint

# print (ccxt.exchanges)

exchange = ccxt.binance()

# markets = exchange.load_markets ()

# print (exchange.id)

# btcMarket = exchange.markets['BTC/USDT']

#pprint.pprint(btcMarket)

# pprint.pprint(dir(exchange))



# print("Creds: " + str(exchange.checkRequiredCredentials()))

# print("Wallet Address: " + str(exchange.myTrades))

#pprint.pprint(exchange.fetch_balance()['info']['balances'])

def set_auth_info(exchange):
    exchange.apiKey = 'xneps3aMqYgB5Kfklcsa58L4FKJAd5z9O6xvBXfPEi7Wo1AZgMyrHnLNoAypKo5X'
    exchange.secret = 'Y2NrEokkppTu4BsCeFjnlGHCauz2GbqvR7GtUH05d5RN9O0GE4DoaMd6yqzuETGU'
    
    return exchange

def print_wallet(walletInfo):
    balances = walletInfo['info']['balances']
    for coin in balances:
        if(not float(coin['free']) == 0):
            print(coin)
        
# print_wallet(exchange.fetch_balance())

# exchange.createMarketSellOrder('BTC/USDT', .001)
        
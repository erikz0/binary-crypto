from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopLimitOrderRequest, ClosePositionRequest, GetOrdersRequest, QueryOrderStatus
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, OrderType
from alpaca.trading.stream import TradingStream
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.enums import OrderType
from alpaca.data.live.crypto import CryptoDataStream, CryptoFeed
from alpaca.trading.models import Order as AlpacaOrder
from alpaca.common.exceptions import APIError

from live_paper_trade_crypto import API_KEY, SECRET_KEY, PAPER

trader   = TradingClient(API_KEY, SECRET_KEY, paper=PAPER)

pre_pos = trader.get_open_position('ETHUSD')
pre_qty = float(pre_pos.qty)

print(pre_qty)


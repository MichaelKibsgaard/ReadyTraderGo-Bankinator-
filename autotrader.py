# Copyright 2021 Optiver Asia Pacific Pty. Ltd.
#
# This file is part of Ready Trader Go.
#
#     Ready Trader Go is free software: you can redistribute it and/or
#     modify it under the terms of the GNU Affero General Public License
#     as published by the Free Software Foundation, either version 3 of
#     the License, or (at your option) any later version.
#
#     Ready Trader Go is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU Affero General Public License for more details.
#
#     You should have received a copy of the GNU Affero General Public
#     License along with Ready Trader Go.  If not, see
#     <https://www.gnu.org/licenses/>.

"""
We are trying to weight the stochastic and some moving averages to trade trends.
WE arnt too sure how we can hedge yet
Our code is messy we had lots of uni classes sorry !

"""

import asyncio
import itertools
import pandas as pd
import numpy as np
import scipy as sc
from scipy import stats

from typing import List

from ready_trader_go import BaseAutoTrader, Instrument, Lifespan, MAXIMUM_ASK, MINIMUM_BID, Side, timer, score_board

LOT_SIZE = 10
POSITION_LIMIT = 100
TICK_SIZE_IN_CENTS = 100


class AutoTrader(BaseAutoTrader):
    """Example Auto-trader.

    When it starts this auto-trader places ten-lot bid and ask orders at the
    current best-bid and best-ask prices respectively. Thereafter, if it has
    a long position (it has bought more lots than it has sold) it reduces its
    bid and ask prices. Conversely, if it has a short position (it has sold
    more lots than it has bought) then it increases its bid and ask prices.
    """

    def __init__(self, loop: asyncio.AbstractEventLoop, team_name: str, secret: str):
        """Initialise a new instance of the AutoTrader class."""
        super().__init__(loop, team_name, secret)
        self.order_ids = 0
        self.bids = set()
        self.asks = set()
        self.ask_id = self.ask_price = self.bid_id = self.bid_price = self.position = 0
        self.price = []
        self.bid_list = []
        self.ask_list = []
        self.time = []
        self.trend_lines = []
        self.trade_history = []
        self.stopwatch = timer.Timer(1.0, 1.0)
        self.hack = 0
        self.hack_2 = 1
        self.hack_3 = 1
        self.stopwatch_waiter = 0
        self.attempted_buys = [0]
        self.amount_sold = 0
        self.position_countdown = (0, False)
        self.last_attempted_trade = [0]

    def on_error_message(self, client_order_id: int, error_message: bytes) -> None:
        """Called when the exchange detects an error.

        If the error pertains to a particular order, then the client_order_id
        will identify that order, otherwise the client_order_id will be zero.
        """
        self.logger.warning("error with order %d: %s", client_order_id, error_message.decode())

        if client_order_id != 0:
            self.send_cancel_order(client_order_id)
            self.logger.warning(("i saved u", client_order_id))
            self.on_order_status_message(client_order_id, 0, 0, 0)
            if self.hack_3 == 0:
                self.hack_3 = 1

    def on_hedge_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when one of your hedge orders is filled, partially or fully.

        The price is the average price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.

        If the order was unsuccessful, both the price and volume will be zero.
        """
        self.logger.info("received hedge filled for order %d with average price %d and volume %d", client_order_id,
                         price, volume)

    def on_order_book_update_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                                     ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically to report the status of an order book.

        The sequence number can be used to detect missed or out-of-order
        messages. The five best available ask (i.e. sell) and bid (i.e. buy)
        prices are reported along with the volume available at each of those
        price levels.
        """
        """
        This part of the code starts our stopwatch
        """
        if self.hack == 0:
            self.stopwatch.start()
            self.hack += 1

        """
        This section of the code is for maintaing the length of certain lists 
        """

        def list_maintanace(list_2, list_3):
            "Maintains the price and volume lists to a reasonable length, which is 30 seconds in this case"
            if not list_2:
                return
            y = list_2[-1]
            a = y - 100
            while (self.time[0] < a):
                list_2.pop(0)
                list_3.pop(0)
            return

        def trend_line_maintance():
            "Maintains the length of the trendline list, "
            self.trend_lines = self.trend_lines[-200:]
            return
        """
        Creates our price and time lists and continuously fills them
        """
        price_index = ((bid_prices[0] + ask_prices[-1]) / 2)
        self.bid_list.append(bid_prices[0])
        self.ask_list.append(ask_prices[-1])
        self.price.append(price_index)
        self.time.append(self.stopwatch.advance())
        data_list = list(zip(self.time, self.price))
        """
        Maintains the lists by calling their functions
        """
        list_maintanace(self.time, self.price)
        trend_line_maintance()

        def fix_into_dataframe(d):
            """
            Creates a Data Frame
            """
            data = np.array(d, dtype=float)
            df = pd.DataFrame(data, dtype=float)
            return df
        self.df2 = fix_into_dataframe(data_list)

        def stochastic_indicator(trade_data):
            """
            Generates our Stochastic value given trade data in form (time, price)
            """
            trade_data.columns = ["time", 'price']
            if trade_data.empty:
                return 0
            else:
                first_price = trade_data.iloc[0, 1]
            min_price = trade_data["price"].min()
            max_price = trade_data["price"].max()
            K = (first_price - min_price) / (max_price - min_price)
            return K * 100

        stochastic_output = stochastic_indicator(self.df2)

        def fivetwo_moving_avg(trade_data):
            """returns (fifty seconds, two hundred seconds mean)
            Given trade value data in form (time, price)
            """
            trade_data.columns = ["time", 'price']
            if trade_data.empty:
                curr_time = 0
            else:
                curr_time = trade_data.iloc[-1, 0]
            # opening price
            fifty_sec = trade_data[trade_data["time"] > curr_time - 5]["price"]
            two_hund_sec = trade_data[trade_data["time"] >= curr_time - 30]["price"]
            x = float(fifty_sec.mean()) / 100
            y = float(two_hund_sec.mean()) / 100
            return x, y
        moving_average_output = fivetwo_moving_avg(self.df2)

        """
        Creates our trendline data in one list
        """
        self.trend_lines.append((moving_average_output, stochastic_output))

    def on_order_filled_message(self, client_order_id: int, price: int, volume: int) -> None:
        """Called when when of your orders is filled, partially or fully.

        The price is the price at which the order was (partially) filled,
        which may be better than the order's limit price. The volume is
        the number of lots filled at that price.
        """
        self.logger.info("received order filled for order %d with price %d and volume %d", client_order_id,
                         price, volume)

        if client_order_id in self.bids:
            self.position += volume
            self.logger.info("i did something useful")
            if self.hack_3 == 0:
                self.hack_3 = 1
            self.trade_history.append((self.time[-1], volume, self.bid_list[-1]))

        elif client_order_id in self.asks:
            self.logger.info("i did something useful2")
            self.position -= volume
            if self.hack_2 == 0:
                self.hack_2 = 1
            if self.hack_3 == 0:
                self.hack_3 = 1
            self.trade_history.append((self.time[-1], -volume, self.ask_list[-1]))

        if len(self.trade_history) > 100:
            self.trade_history.pop(0)

        """
                if client_order_id in self.bids:
            self.position += volume
            self.send_hedge_order(next(self.order_ids), Side.ASK, MINIMUM_BID, volume)
        elif client_order_id in self.asks:
            self.position -= volume
            self.send_hedge_order(next(self.order_ids), Side.BID,
                                  MAXIMUM_ASK//TICK_SIZE_IN_CENTS*TICK_SIZE_IN_CENTS, volume)
        """

    def on_order_status_message(self, client_order_id: int, fill_volume: int, remaining_volume: int,
                                fees: int) -> None:
        """Called when the status of one of your orders changes.

        The fill_volume is the number of lots already traded, remaining_volume
        is the number of lots yet to be traded and fees is the total fees for
        this order. Remember that you pay fees for being a market taker, but
        you receive fees for being a market maker, so fees can be negative.

        If an order is cancelled its remaining volume will be zero.
        """
        self.logger.info("received order status for order %d with fill volume %d remaining %d and fees %d",
                         client_order_id, fill_volume, remaining_volume, fees)
        if remaining_volume == 0:
            if client_order_id == self.bid_id:
                self.bid_id = 0
            elif client_order_id == self.ask_id:
                self.ask_id = 0

            # It could be either a bid or an ask
            self.bids.discard(client_order_id)
            self.asks.discard(client_order_id)

    def on_trade_ticks_message(self, instrument: int, sequence_number: int, ask_prices: List[int],
                               ask_volumes: List[int], bid_prices: List[int], bid_volumes: List[int]) -> None:
        """Called periodically when there is trading activity on the market.

        The five best ask (i.e. sell) and bid (i.e. buy) prices at which there
        has been trading activity are reported along with the aggregated volume
        traded at each of those price levels.

        If there are less than five prices on a side, then zeros will appear at
        the end of both the prices and volumes arrays.
        """

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-0.2 * x))

        def trade_strategy(metric):
            if autosell() == 1:
                return
            #take_profits()
            # Unit tests that are not really unit tests
            if np.isnan(metric[-1]):
                return "nan was found"
            if metric[1] == 0:
                return
            mean_list = [m[1] for m in self.trend_lines]
            mean_list = np.array(mean_list)
            mean_value = np.nanmean(mean_list)
            sample_standard = np.nanstd(mean_list, ddof = 1)

            normal_distribution = sc.stats.norm(mean_value, sample_standard)
            probability = 1 - normal_distribution.cdf(70)
            probability_2 = normal_distribution.cdf(30)
            diff = metric[0][0] - metric[0][1]
            diff = float(diff)

            total = int(diff * 2)
            if total == 0:
                return

            difference_of_probability = probability_2 - probability
            if self.hack_3 == 0:
                self.hack_3 = 1
                return

            self.logger.info((difference_of_probability, total))

            if difference_of_probability > 0.05 and (total < 8):
                if self.trade_history:
                    total = abs(total)
                    last_time, last_volume, price_4 = self.trade_history[-1]
                    if last_volume > 0 and (self.time[-1] - last_time < 10) and (self.last_attempted_trade[-1] - last_time < 10):
                        return
                # This executes the Buy
                self.hack_3 = 0
                buy_function(self.ask_list[-1], abs(total//2))

            elif difference_of_probability < -0.05 and (total > 0):
                if self.trade_history:
                    total = abs(total)
                    last_time, last_volume, price_4 = self.trade_history[-1]
                    if last_volume < 0 and (self.time[-1] - last_time < 4):
                        return
                        # This calls the sell side
                    self.hack_3 = 0
                    sell_function(self.bid_list[-1], abs(total * 2))

            elif total < 10000:
                if self.trade_history:
                    total = abs(total)
                    last_time, last_volume, price_4 = self.trade_history[-1]
                    if last_volume > 0 and (self.time[-1] - last_time < 10) and (self.last_attempted_trade[-1] - last_time < 10):
                        return
                if total >= 2:
                    self.hack_3 = 0
                    buy_function(self.ask_list[-1], abs(total // 2))
                # This executes the Buy

            elif total > 0:
                if self.trade_history:
                    total = abs(total)
                    last_time, last_volume, price_4 = self.trade_history[-1]
                    if last_volume < 0 and (self.time[-1] - last_time < 4):
                        return
                        # This calls the sell side
                    self.hack_3 = 0
                    sell_function(self.bid_list[-1], abs(total // 2))
            self.logger.info("no order chosen")
            return

        def take_profits():
            """
            Secure the Profits
            """

            if len(self.trade_history) < 0:
                return
            dont_over_count = self.position
            important_list = self.trade_history[-1:]
            for item in important_list[::-1]:
                last_time, last_volume, last_price = item
                if dont_over_count <= 0:
                    return
                if last_volume > 0:
                    dont_over_count -= last_volume
                    profit_percent = (self.bid_list[-1] - last_price)/last_price
                    self.logger.info(profit_percent)
                    self.logger.info((profit_percent,last_price,self.bid_list[-1]))
                    if(profit_percent > 0.009):
                        self.logger.info(('profits are secured yay', last_price, self.bid_list[-1], dont_over_count))
                        sell_function_3(self.bid_list[-1], abs(last_volume * 2))
                        return
            return

        def autosell():
            if self.position > 10:
                x, y = self.position_countdown
                if not y:
                    self.position_countdown = (self.time[-1] + 45, True)
                    return
                sell_amount = self.position - 10
                if self.time[-1] > x and (self.hack_2 == 1):
                    self.logger.info(("Autoseller BRRR", self.position_countdown, self.stopwatch.advance()))
                    self.hack_2 = 0
                    sell_function_2(self.bid_list[-1], abs(sell_amount))
                    self.position_countdown = (0, False)
                    return 1
            return

        def buy_function(price, amount):
            # dont buy more than 100
            if self.position + amount > 100:
                amount = 100 - self.position
            if amount == 0:
                return
            if price == 0:
                return
            # Buy part of the function
            self.order_ids += 1
            bid_id = self.order_ids
            self.attempted_buys.append(self.time[-1])
            self.logger.info(("WORKINGONBUYING", price, amount, self.stopwatch.advance()))
            self.send_insert_order(bid_id, Side.BUY, int((((price) // 100) * TICK_SIZE_IN_CENTS)), amount,
                               Lifespan.FILL_AND_KILL)
            self.last_attempted_trade.append(self.stopwatch.advance())
            self.bids.add(bid_id)
            return

        def sell_function(price, amount):
            # dont sell less than 0
            if self.position - amount < 0:
                amount = self.position
            if amount == 0:
                return -1
            if price == 0:
                return

            # Sell part of the function
            self.order_ids += 1
            ask_id = self.order_ids
            self.logger.info(("WORKINGONSELLING", price, amount, self.stopwatch.advance()))
            self.send_insert_order(ask_id, Side.SELL, int((((price) // 100) * TICK_SIZE_IN_CENTS)), abs(amount),
                                   Lifespan.FILL_AND_KILL)
            self.last_attempted_trade.append(self.stopwatch.advance())
            self.asks.add(ask_id)
            return

        def sell_function_2(price, amount):
            # dont sell less than 0
            if self.position - amount < 0:
                self.position = amount
            if amount == 0:
                return -1
            if price == 0:
                return

            # Sell part of the function
            self.order_ids += 1
            ask_id = self.order_ids
            self.logger.info(("WORKINGONSELLING", price, amount, self.stopwatch.advance()))
            self.send_insert_order(ask_id, Side.SELL, int((((price) // 100) * TICK_SIZE_IN_CENTS)), abs(amount),
                                   Lifespan.GOOD_FOR_DAY)
            self.last_attempted_trade.append(self.stopwatch.advance())
            self.asks.add(ask_id)
            return

        def sell_function_3(price, amount):
            # dont sell less than 0
            if self.position - amount < 0:
                self.position = amount
            if amount == 0:
                return -1
            if price == 0:
                return

            # Sell part of the function
            self.order_ids += 1
            ask_id = self.order_ids
            self.logger.info(("WORKINGONSELLING", price, amount, self.stopwatch.advance()))
            self.send_insert_order(ask_id, Side.SELL, int((((price) // 100) * TICK_SIZE_IN_CENTS)), abs(amount),
                                   Lifespan.GOOD_FOR_DAY)
            self.last_attempted_trade.append(self.stopwatch.advance())
            self.asks.add(ask_id)
            return

        x = self.trend_lines[-1]
        self.logger.info(((trade_strategy(self.trend_lines[-1]))))

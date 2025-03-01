from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TFTBacktester:
    def __init__(self, signal, index, long_cost=0.0016, short_cost=0.00326):
        """
        :param signal: DataFrame used for backtesting, containing columns: [Date, instrument, pred, real_ret].
        :param index: DataFrame of the benchmark index, containing columns: [trade_date, close].
        :param long_cost: Transaction cost for going long.
        :param short_cost: Transaction cost for going short.
        """
        self.signal = signal.copy()
        self.index = index.copy()
        self.long_cost = long_cost
        self.short_cost = short_cost

    # ============= Performance Metrics Calculation ============= #
    @staticmethod
    def calc_return_metrics(data, adj=52):
        """
        Calculate annualized return, annualized volatility, and annualized Sharpe ratio, 
        as well as Sortino ratio based on the given return data.

        :param data: DataFrame or Series of returns (e.g., daily or weekly returns).
        :param adj: Adjustment factor for annualization (e.g., 52 for weekly).
        :return: DataFrame with columns [Annual Return, Annual Vol, Annual Sharpe, Annual Sortino].
        """
        # Sharpe Ratio
        summary = dict()
        summary["Annual Return"] = (data.mean() + 1) ** adj - 1
        summary["Annual Vol"] = data.std() * np.sqrt(adj)
        summary["Annual Sharpe"] = summary["Annual Return"] / summary["Annual Vol"]

        # Sortino Ratio
        neg_std = data[data < 0].std() * np.sqrt(adj)
        sortino = summary["Annual Return"] / neg_std
        summary["Annual Sortino"] = sortino.replace([np.inf, -np.inf], np.nan)
        return pd.DataFrame(summary, index=data.columns)

    @staticmethod
    def calc_risk_metrics(data, var=0.05):
        """
        Calculate risk metrics such as Skewness, Excess Kurtosis, VaR, CVaR, and drawdown-related metrics.

        :param data: DataFrame or Series of returns (e.g., weekly returns).
        :param var: Significance level for VaR and CVaR calculation (e.g., 0.05).
        :return: DataFrame including skewness, excess kurtosis, VaR, CVaR, max drawdown, and recovery days.
        """
        summary = dict()
        summary["Skewness"] = data.skew()
        summary["Excess Kurtosis"] = data.kurtosis()
        summary[f"VaR ({var})"] = data.quantile(var, axis=0)
        summary[f"CVaR ({var})"] = data[data <= data.quantile(var, axis=0)].mean()

        # Calculate drawdown
        wealth_index = 1000 * (1 + data).cumprod()
        previous_peaks = wealth_index.cummax()
        drawdowns = (wealth_index - previous_peaks) / previous_peaks

        summary["Max Drawdown"] = drawdowns.min()

        summary["Bottom"] = drawdowns.idxmin()
        summary["Peak"] = previous_peaks.idxmax()

        recovery_date = []
        for col in wealth_index.columns:
            prev_max = previous_peaks[col][: drawdowns[col].idxmin()].max()
            recovery_wealth = pd.DataFrame([wealth_index[col][drawdowns[col].idxmin() :]]).T
            recovery_date.append(
                recovery_wealth[recovery_wealth[col] >= prev_max].index.min()
            )
        summary["Recovery"] = ["-" if pd.isnull(i) else i for i in recovery_date]

        summary["Recovery (days)"] = [
            (i - j).days if i != "-" else "-"
            for i, j in zip(summary["Recovery"], summary["Bottom"])
        ]
        summary.pop('Bottom')
        summary.pop('Peak')
        summary.pop('Recovery')

        return pd.DataFrame(summary, index=data.columns)

    def calc_performance_metrics(self, data, adj=52, var=0.05):
        """
        Aggregate method to calculate return metrics and risk metrics,
        and then compute Calmar ratio.

        :param data: DataFrame or Series of returns (e.g., daily or weekly returns).
        :param adj: Adjustment factor for annualization (e.g., 52 for weekly).
        :param var: Significance level for VaR and CVaR calculation (e.g., 0.05).
        :return: DataFrame containing performance metrics.
        """
        summary = {
            **self.calc_return_metrics(data=data, adj=adj),
            **self.calc_risk_metrics(data=data, var=var),
        }
        # Calmar Ratio
        summary["Calmar Ratio"] = summary["Annual Return"] / abs(summary["Max Drawdown"])
        return pd.DataFrame(summary, index=data.columns).T

    # ============= Backtest Trading Logic ============= #
    @staticmethod
    def mark_signal_oneday(df_in_one_day):
        """
        Given data for one day (multiple instruments),
        sort by 'pred' and label the top 10% with signal 1, bottom 10% with signal -1, and others with 0.

        :param df_in_one_day: DataFrame for a single day.
        :return: DataFrame with an additional 'signal' column.
        """
        df = df_in_one_day.copy()
        df = df.sort_values("pred")
        n = len(df)

        bot_n = int(n * 0.1)   # Number of instruments in the bottom 10%
        top_n = int(n * 0.1)   # Number of instruments in the top 10%

        df["signal"] = 0
        df.iloc[:bot_n, df.columns.get_loc("signal")] = -1
        df.iloc[-top_n:, df.columns.get_loc("signal")] = 1

        return df

    @staticmethod
    def get_trade(group):
        """
        Calculate trades (opening and closing) within each segment where signals remain unchanged.
        The total return of each segment is calculated through compound returns of real_ret.

        :param group: DataFrame grouped by a single instrument.
        :return: DataFrame of trades, including close date, instrument, return, and signal.
        """
        df = group.copy()
        # (1) Identify positions where the signal changes: if it differs from the previous row, a new segment starts
        df['prev_signal'] = df['signal'].shift(fill_value=0)
        df['segment_id'] = (df['signal'] != df['prev_signal']).cumsum()

        # (2) Group by segment_id and process each segment
        trades = []
        for _, seg_df in df.groupby('segment_id'):
            current_signal = seg_df['signal'].iloc[0]
            if current_signal == 0:
                # If the signal in this segment is 0, no trade is recorded
                continue

            # (3) Within the segment, calculate the total return by compound returns
            total_ret = (seg_df['real_ret'] + 1).prod() - 1

            # (4) Record this trade's information: use the last row's Date as close_date
            close_date = seg_df['Date'].iloc[-1]
            trades.append({
                'close_date': close_date,
                'instrument': seg_df['instrument'].iloc[0],
                'ret': total_ret,
                'signal': current_signal
            })

        return pd.DataFrame(trades)

    def process_single_day_backtest(self, j, signal):
        """
        Filter out the days where i % 5 == j, and for each of those days:
         1) Label instruments with signals.
         2) Calculate trades and their returns.
         3) Subtract transaction costs.
         4) Calculate weekly PnL, then the net value.

        :param j: Day offset (0 through 4).
        :param signal: DataFrame with columns [Date, instrument, pred, real_ret].
        :return: net_value (Series of net value by close_date), trade_dates (all close_dates of trades).
        """
        backtest_df = signal.copy()
        backtest_df["Date"] = pd.to_datetime(backtest_df["Date"])
        backtest_df = backtest_df.sort_values(["Date", "instrument"]).reset_index(drop=True)
        unique_dates = backtest_df["Date"].drop_duplicates().sort_values().tolist()

        # Only keep dates where index % 5 == j
        dateidx = {d: i for i, d in enumerate(unique_dates) if i % 5 == j}
        backtest_df = backtest_df[backtest_df.Date.isin(dateidx.keys())]

        # Label instruments each day
        backtest_df = (
            backtest_df
            .groupby("Date", group_keys=False)
            .apply(self.mark_signal_oneday)
            .sort_values(["instrument", "Date"])
        )

        # Calculate trades
        backtest_df = backtest_df.groupby('instrument').apply(self.get_trade).reset_index(drop=True)

        # Subtract transaction costs
        backtest_df["pnl"] = (
            backtest_df["signal"] * backtest_df["ret"]
            - np.where(
                backtest_df["signal"] > 0, self.long_cost,
                np.where(backtest_df["signal"] < 0, self.short_cost, 0)
            ) * backtest_df["signal"].abs()
        )

        # Daily PnL
        weekly_pnl = (
            backtest_df
            .groupby("close_date")["pnl"]
            .mean()
            .fillna(0)
            .sort_index()
        )

        # Net value curve
        net_value = (1 + weekly_pnl).cumprod()
        net_value /= net_value.iloc[0]  # Normalize to start from 1

        return net_value, backtest_df['close_date']

    # ============= Plotting ============= #
    @staticmethod
    def plot_IC(signal):
        """
        Plot the rolling IC curve.

        :param signal: DataFrame with columns [Date, instrument, pred, real_ret].
        """
        plt.figure(figsize=(13, 8))
        IC = signal.groupby('Date').apply(lambda x: x['real_ret'].corr(x['pred']))
        rolling_IC = IC.rolling(window=42).mean()
        plt.title(f'TFT IC = {IC.mean().round(4)}')
        plt.plot(rolling_IC, label='2-month rolling IC', linewidth=2)
        plt.legend()
        plt.grid()
        plt.savefig('output_backtest_result/IC.png', dpi=300, bbox_inches='tight')
        # plt.show()

    def plot_net(self, net_net, perf_df):
        """
        Plot the final net value curve alongside the CSI300 benchmark.

        :param net_net: Combined net value curve (Series or DataFrame).
        :param perf_df: DataFrame of performance metrics.
        """
        # Align benchmark index
        index_df = self.index[
            (self.index.trade_date >= net_net.index.min()) &
            (self.index.trade_date <= net_net.index.max())
        ].set_index('trade_date')

        # Calculate benchmark net value
        benchmark = (1 + index_df.close.pct_change()).cumprod().fillna(1)

        plt.figure(figsize=(13, 8))
        plt.title('TFT', fontsize=22)
        plt.plot(net_net, label='CSI300 Weekly Rebalancing Strategy', color='red', linewidth=2)
        plt.plot(benchmark, label='CSI300 benchmark', linewidth=2)
        plt.xlabel('Time', fontsize=14)
        plt.ylabel('Net Value', fontsize=14)
        plt.text(
            0.34, 0.955,
            f'Long cost = {self.long_cost * 100}%, Short cost = {self.short_cost * 100}%',
            transform=plt.gca().transAxes,
            fontsize=14,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor="lightgray")
        )

        # Add a table in the figure to show performance metrics
        cell_text = perf_df.values.round(3)
        row_labels = perf_df.index.tolist()

        the_table = plt.table(
            cellText=cell_text,
            rowLabels=row_labels,
            bbox=[0.156, 0.68, .04, .3],
        )

        for (row, col), cell in the_table.get_celld().items():
            cell.set_linewidth(0)
            cell.set_edgecolor('lightgray')
            cell.set_alpha(1)

        the_table.scale(1.0, 200)
        the_table.set_zorder(2)
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(14)

        plt.tight_layout()
        plt.legend(fontsize=14, loc='lower left')
        plt.grid()
        plt.savefig('output_backtest_result/net_value_plot.png', dpi=300, bbox_inches='tight')
        # plt.show()

    # ============= Main Execution Logic ============= #
    def run(self):
        """
        Main workflow:
        1. Calculate and plot IC.
        2. Perform multiple backtests, splitting different trade days, then merge results.
        3. Calculate and plot final performance.
        """
        self.plot_IC(self.signal)

        perf = []
        net_value_all = []
        for checkpoint in tqdm(range(5)):
            net_value_list = []
            week = []

            for j in range(5):
                net_value, trade_dates = self.process_single_day_backtest(j, self.signal)
                net_value_list.append(net_value)
                week.append(trade_dates)

            # Merge and process the final net value
            net_value_list = (
                pd.concat(net_value_list, axis=1)
                .fillna(0)
                .sum(axis=1) 
                .reset_index(name='nv')
                .set_index('close_date')
            )
            net_value_list['nv'] = net_value_list['nv'].rolling(5).mean()
            net_value_list = net_value_list[net_value_list.index.isin(week[checkpoint])].fillna(1)
            net_value_list = net_value_list['nv']

            # Calculate performance
            perf.append(self.calc_performance_metrics(net_value_list.pct_change().dropna().to_frame()))
            net_value_all.append(net_value_list)

        # Calculate final performance and plot
        perf_df = pd.concat(perf, axis=1).mean(axis=1).to_frame().rename(columns={0: 'Metrics'})
        net_net = net_value_all[-1]  

        self.plot_net(net_net, perf_df)


if __name__ == '__main__':
    long_cost = 0.0016
    short_cost = 0.00326
    signal_df = pd.read_parquet('input_data/signal.parquet')  # [Date, instrument, pred, real_ret]
    index_df = pd.read_parquet('input_data/csi300.parquet')   # [trade_date, close]

    backtester = TFTBacktester(signal_df, index_df, long_cost, short_cost)
    backtester.run()





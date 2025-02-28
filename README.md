1. **Installation and Execution** (takes about **1 minute**):

   ```bash
   cd /path/to/project
   pip install -r requirements.txt
   python backtest.py

2. **Backtest Logic** (see **backtest.py** for implementation details)

   1) Each trading day, we rank stocks by the previous day's signal value. 
      - **Top 10%**: go long  
      - **Bottom 10%**: go short  
      Positions are closed exactly 5 trading days (1 week) later.

   2) Because we open new positions every day and each position is held for 1 week, there are effectively **5 overlapping branches** of positions (one for each weekday). Each branch is closed on its corresponding day in the next week.

   3) If a position's direction (long/short) for a given asset does not change from day to day, we continue to hold it rather than re-enter. 

   4) **Transaction Costs**:
      - **Long cost**: 0.16% per trade  
        (0.03% commission for both buy & sell + 0.1% stamp duty on selling)  
      - **Short cost**: 0.326% per trade  
        (includes commission, stamp duty, and annualized securities lending rate of 8.625% divided by 52)

   5) **Important Constraint**: Since the model forecasts 5-day returns, each branch is strictly closed after exactly 1 week (no partial closings or cross-branch adjustments). In a live trading scenario, this rule might be relaxed to accommodate real-world conditions.



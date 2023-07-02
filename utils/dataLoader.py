import pandas as pd
import numpy as np
from datetime import datetime

class Dataloader():
    def __init__(self,config):
        # Sample Method
        self.T = config.T
        # self.df = None
        # self.targetName = config.targetName
        self.config = config
    def loadData(self,type,year):
        # T = "H"
        # df19 = pd.read_csv("data/2019.csv", parse_dates=['time']).set_index("time")
        # df21 = pd.read_csv("data/2021.csv", parse_dates=['time']).set_index("time")
        # df21 = pd.concat([df19,df21],axis=0)
        df21 = pd.read_csv(f"../data/{type}债{year}y-日线.csv", index_col=0, parse_dates=['DT']).set_index('DT')
        df21.ffill(inplace=True)
        # dt_object = datetime.strptime(dates[0], "%d/%m/%Y %H:%M:%S")
        if self.T =="H":
            temp1 = df21.copy().resample(self.T, label='left').ohlc().ffill()
            temp1 = temp1.loc[:, [('open', 'open'), ('high', 'high'), ('low', 'low'), ('close', 'close')]]
            temp1['volume'] = df21.resample(self.T, label='left').agg({"volume": np.sum})
            temp1.columns = ['open', 'high', 'low', 'close', 'volume']
            df = temp1[temp1.volume != 0]
        else:
            df = df21
        # for year in [2016,2017,2018,2019,2020,2021,2022]:
        year = 2010
        start = pd.to_datetime(f'{year}/01/01')
        end = pd.to_datetime(f"{year + 20}/12/31")
        # self.df = df[(df.index >= start) & (df.index <= end)]
        # self.df = df
        return df[(df.index >= start) & (df.index <= end)]
        ###############################################################
    # def TechFeatureExtra(self):
    #     df_day_trade = pd.read_csv("data/模型信号_沈博专属.csv", encoding="gbk", parse_dates=['DT']).set_index("DT")
    #     df_day_trade['牛熊震荡'] = df_day_trade['价格空间'] + df_day_trade['基本面空间']
    #     df_day_trade['牛熊震荡备忘'] = 1
    #     for idx, row in df_day_trade.reset_index().iterrows():
    #         if (idx >= 1) and (idx <= df_day_trade.shape[0] - 1):
    #             prev_row = df_day_trade.iloc[idx - 1]
    #             next_future_day = df_day_trade.reset_index().iloc[min(df_day_trade.shape[0] - 1, idx + 3)]
    #             if row['牛熊震荡'] * prev_row['牛熊震荡'] == -1:
    #                 df_day_trade.loc[row['DT']:next_future_day['DT'], '牛熊震荡备忘'] = 0
    #     df_day_trade['牛熊震荡_平滑'] = df_day_trade['牛熊震荡'] * df_day_trade['牛熊震荡备忘']
    #     df_day_trade['牛熊震荡_平滑_牛'] = df_day_trade['牛熊震荡_平滑'].map(lambda x: True if x > 0 else False)
    #     df_day_trade['牛熊震荡_平滑_熊'] = df_day_trade['牛熊震荡_平滑'].map(lambda x: True if x < 0 else False)
    #     if self.T !="D":
    #         df_day_trade_H=df_day_trade.shift(1).resample(self.T,label="left").asfreq().ffill()[['牛熊震荡_平滑_牛','牛熊震荡_平滑_熊']]
    #     else:
    #         df_day_trade_H=df_day_trade.resample(self.T,label="left").ffill()[['牛熊震荡_平滑_牛','牛熊震荡_平滑_熊']]
    #
    #     self.df=pd.concat([self.df,df_day_trade_H.loc[self.df.index,:]],axis=1)
    #     #RSI特征
    #     for ele in self.config.rsiPeriodList:
    #         for thres in self.config.res_threshold:
    #             self.df[f'close_RSI{ele}>{thres}'] = talib.RSI(self.df['close'], timeperiod=ele).map(lambda x: True if x>thres else False)
    #             self.df[f'close_RSI{ele}<{100-thres}'] = talib.RSI(self.df['close'], timeperiod=ele).map(lambda x: True if x<100-thres else False)
    #     #均线特征
    #     for ele in self.config.MAset:
    #         self.df[f'{self.config.targetName}_aboveMA{ele}'] = (self.df['close'] - self.df['close'].rolling(ele).mean()) \
    #             .map(lambda x: True if x > 0 else False)
    #         self.df[f'{self.config.targetName}_belowMA{ele}'] = (self.df['close'] - self.df['close'].rolling(ele).mean()) \
    #             .map(lambda x: True if x < 0 else False)
    #
    #     def cal_multitrend_signal(colname):
    #         self.df[f'{colname}MA9'] = self.df[f'{colname}'].rolling(9).mean()
    #         self.df[f"{colname}MA9_shift"] = self.df[f'{colname}MA9'].shift(1)
    #         self.df[f'{colname}MA10'] = (self.df[f'{colname}'] * 3 + self.df[f"{colname}MA9_shift"] * 9) / 12
    #         # df[f'{colname}MA15'] = df[f'{colname}'].rolling(15).mean()
    #         # df[f'{colname}MA20'] = df[f'{colname}'].rolling(20).mean()
    #         # df[f'{colname}MA10'] = df[f'{colname}'].rolling(10).mean()
    #         self.df[f'{colname}MA15'] = self.df[f'{colname}'].rolling(15).mean()
    #         self.df[f'{colname}MA20'] = self.df[f'{colname}'].rolling(20).mean()
    #
    #         def cal_signal_up(x):
    #             return True if x[f'{colname}MA10'] >= x[f'{colname}MA15'] >= x[f'{colname}MA20'] else False
    #         def cal_signal_down(x):
    #             return True if x[f'{colname}MA10']<= x[f'{colname}MA15'] <= x[f'{colname}MA20'] else False
    #
    #         self.df['T多头排列信号'] = self.df.apply(lambda x: cal_signal_up(x), axis=1)
    #         self.df['T空头排列信号'] = self.df.apply(lambda x: cal_signal_down(x), axis=1)
    #
    #     cal_multitrend_signal(colname='close')
    #     self.df['close_diff4'] = 2*(self.df['close'].diff(4)>0).astype(int)-1
    #     temp = (self.df['close_diff4'].rolling(9).sum().abs()>=9).astype(int)* self.df['close_diff4']
    #     self.df['神奇九转_牛反'] = temp.map(lambda x:True if x>0 else False)
    #     self.df['神奇九转_熊反'] = temp.map(lambda x:True if x<0 else False)
    #     for ele in self.config.updownMAset:
    #         self.df[f'up_cross_MA{ele}'] =  (self.df[f'{self.config.targetName}_aboveMA{20}'] - self.df[f'{self.config.targetName}_aboveMA{20}'].shift(1))>0
    #         self.df[f'down_cross_MA{ele}'] =  (self.df[f'{self.config.targetName}_aboveMA{20}'] - self.df[f'{self.config.targetName}_aboveMA{20}'].shift(1))<0
    #     upper, middle, lower = talib.BBANDS(self.df['close'],timeperiod=20,nbdevup=2,nbdevdn=2)
    #     self.df['bband_up_break'] = self.df['close']>upper
    #     self.df['bband_down_break'] = self.df['close']<lower
    #
    #     print("技术指标抽取完毕")
    #
    #
    #     #弹性特征
    #     #反转特征
    #     return self.df

    # def loadDataAll(self):
    #     # T = "H"
    #     df21 = pd.read_csv("data/T主连汇总.csv", parse_dates= {"date" : ["dataDate","barTime"]}).set_index("date")
    #     df21.ffill(inplace=True)
    #     temp1 = df21.copy().resample(self.T, label='right').ohlc().ffill()
    #     temp1 = temp1.loc[:, [('open', 'open'), ('high', 'high'), ('low', 'low'), ('close', 'close')]]
    #
    #     temp1['volume'] = df21.resample(self.T, label='right').agg({"volume": np.sum})
    #     temp1.columns = ['open', 'high', 'low', 'close', 'volume']
    #     df = temp1[temp1.volume != 0]
    #     # for year in [2016,2017,2018,2019,2020,2021,2022]:
    #     year = 2016
    #     start = pd.to_datetime(f'{year}/03/01')
    #     end = pd.to_datetime(f"{year + 6}/12/31")
    #     df_year = df[(df.index >= start) & (df.index <= end)]
    #     return df_year
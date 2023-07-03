import pandas as pd
import numpy as np
from utils.paramConfig import paramConfig
import functools
from pylab import mpl
import mplfinance as mpf
import matplotlib.pyplot as plt
import os
import talib as ta
# import turnPointClass
from utils.turnPointClass import LowPoint
from utils.dataLoader import Dataloader
import json

class FeatureExtra_cls(object):
    def __init__(self, df, config,year,type):
        self.df = df
        self.dropDelta = config.dropDelta
        self.config = config
        # self.T = T
        self.alines = []
        self.colors = []
        self.featureNameDict = {}
        self.year = year
        self.type = type
    def __createZigZagPoints(self, dfSeries, minSegSize=0.1):
        minRetrace = minSegSize
        curVal = dfSeries[0]
        curPos = dfSeries.index[0]
        curDir = 1
        dfRes = pd.DataFrame(index=dfSeries.index, columns=["Dir", "Value"])
        for ln in dfSeries.index:
            if ((dfSeries[ln] - curVal) * curDir >= 0):
                curVal = dfSeries[ln]
                curPos = ln
            else:
                retracePrc = abs((dfSeries[ln] - curVal))
                if (retracePrc >= minRetrace):
                    # dfRes.loc[curPos, 'Value'] = curVal
                    # dfRes.loc[curPos, 'Dir'] = curDir
                    # dfRes.loc[curPos, 'found'] = ln

                    dfRes.loc[ln, 'Value'] = curVal
                    dfRes.loc[ln, 'Dir'] = curDir
                    dfRes.loc[ln, 'found'] = curPos

                    curVal = dfSeries[ln]
                    curPos = ln
                    curDir = -1 * curDir
        dfRes[['Value']] = dfRes[['Value']].astype(float)
        self.df.loc[dfRes.index, ['Dir']] = dfRes['Dir']
        self.df.loc[dfRes.index, ['Value']] = dfRes['Value']
        self.df.loc[dfRes.index, ['found']] = dfRes['found']
        self.df['found'] = self.df['found'].map(lambda x: np.nan if pd.isnull(x) else pd.to_datetime(x))
        return self.df

    def __cal_multitrend_signal(self, colname):
        # self.df[f'{colname}MA9'] = self.df[f'{colname}'].rolling(9).mean()
        # self.df[f"{colname}MA9_shift"] = self.df[f'{colname}MA9'].shift(1)
        # self.df[f'{colname}MA10'] = (self.df[f'{colname}'] * 3 + self.df[f"{colname}MA9_shift"] * 9) / 12
        # df[f'{colname}MA15'] = df[f'{colname}'].rolling(15).mean()
        # df[f'{colname}MA20'] = df[f'{colname}'].rolling(20).mean()
        temp = self.df.copy()
        temp[f'{colname}MA10'] = self.df[f'{colname}'].rolling(10).mean()
        temp[f'{colname}MA15'] = self.df[f'{colname}'].rolling(15).mean()
        temp[f'{colname}MA20'] = self.df[f'{colname}'].rolling(20).mean()

        def cal_signal_up(x):
            return True if x[f'{colname}MA10'] >= x[f'{colname}MA15'] >= x[f'{colname}MA20'] else False

        def cal_signal_down(x):
            return True if x[f'{colname}MA10'] <= x[f'{colname}MA15'] <= x[f'{colname}MA20'] else False

        temp[f'{colname}多头排列信号'] = temp.apply(lambda x: cal_signal_up(x), axis=1)
        temp[f'{colname}空头排列信号'] = temp.apply(lambda x: cal_signal_down(x), axis=1)
        self.df.loc[:, [f'{colname}多头排列信号', f'{colname}空头排列信号']] = temp.loc[
            self.df.index, [f'{colname}多头排列信号', f'{colname}空头排列信号']]
        print()

    def _macroecoFeatureExtra(self):
        def cal_multitrend_signal(df, colname):
            df[f'{colname}MA9'] = df[f'{colname}'].rolling(9).mean()
            df[f"{colname}MA9_shift"] = df[f'{colname}MA9'].shift(1)
            df[f'{colname}MA10'] = (df[f'{colname}'] * 3 + df[f"{colname}MA9_shift"] * 9) / 12
            # df[f'{colname}MA10'] = df[f'{colname}'].rolling(10).mean()
            # df[f'{colname}MA15'] = df[f'{colname}'].rolling(15).mean()
            # df[f'{colname}MA20'] = df[f'{colname}'].rolling(20).mean()
            # df[f'{colname}MA10'] = df[f'{colname}'].rolling(10).mean()
            df[f'{colname}MA15'] = df[f'{colname}'].rolling(15).mean()
            df[f'{colname}MA20'] = df[f'{colname}'].rolling(20).mean()

            def cal_signal(x):
                if x[f'{colname}MA10'] >= x[f'{colname}MA15'] and x[f'{colname}MA15'] >= x[f'{colname}MA20']:
                    return 1 if colname == "T_price" else -1
                elif x[f'{colname}MA10'] <= x[f'{colname}MA15'] and x[f'{colname}MA15'] <= x[f'{colname}MA20']:
                    return -1 if colname == "T_price" else 1
                else:
                    return 0
            df[f'{colname}:信号'] = df.apply(lambda x: cal_signal(x), axis=1)
            return df
        df_day_trade = pd.read_csv("../data/模型信号_沈博专属.csv", parse_dates=['DT']).set_index("DT")
        # df_XY = df_day_trade.copy()
        # __cal_multitrend_signal()
        df_day_trade['f_RSI'] = ta.RSI(df_day_trade['国债10Y'], timeperiod=14)
        df_day_trade['反转信号'] = df_day_trade['反转信号'].map(lambda x: 0 if (x >= -4 and x <= 4) else -1 if x < -4 else 1)
        # df_day_trade['弹性信号'] = df_day_trade['弹性信号'].map(lambda x: 0 if (x >= -3 and x <= 2) else (-1 if x < -3 else 1))

        df_day_trade['f_RSI'] = df_day_trade['f_RSI'].map(lambda x: 0 if (x >= 18 and x <= 84) else (-1 if x < 20 else 1))

        df_day_trade = cal_multitrend_signal(df_day_trade, 'T_price')

        df_day_trade['流动性MA10'] = ta.MA(df_day_trade['流动性新'].values, timeperiod=10)
        df_day_trade['流动性MA15'] = ta.MA(df_day_trade['流动性新'].values, timeperiod=15)
        df_day_trade['流动性MA20'] = ta.MA(df_day_trade['流动性新'].values, timeperiod=20)

        df_day_trade['基本面MA10'] = ta.MA(df_day_trade['基本面'].values, timeperiod=10)
        df_day_trade['基本面MA15'] = ta.MA(df_day_trade['基本面'].values, timeperiod=15)
        df_day_trade['基本面MA20'] = ta.MA(df_day_trade['基本面'].values, timeperiod=20)

        df_day_trade['流动性信号'] = df_day_trade.apply(
            lambda x: 1 if x['流动性MA10'] < x['流动性MA15'] < x['流动性MA20'] else -1 if x['流动性MA10'] > x['流动性MA15'] > x[
                '流动性MA20'] else 0, axis=1)
        df_day_trade['基本面信号'] = df_day_trade.apply(
            lambda x: 1 if x['基本面MA10'] < x['基本面MA15'] < x['基本面MA20'] else -1 if x['基本面MA10'] > x['基本面MA15'] > x[
                '基本面MA20'] else 0, axis=1)
        df_day_trade['胜率自算'] = df_day_trade.apply(
            lambda x: 1 if (x['流动性信号'] + x['基本面信号']) > 0 else -1 if (x['流动性信号'] + x['基本面信号']) < 0 else 0, axis=1)

        # pd.concat([self.df,df_XY.shift(1)],axis=1)
        # self.df.loc[df_day_trade.index,['流动性信号','基本面信号','胜率自算']] = df_day_trade.loc[:,['流动性信号','基本面信号','胜率自算']]
        # temp = df_day_trade[['胜率自算', '胜率']].copy().dropna()
        # (temp['胜率自算'] == temp['胜率'] * -1).sum() / len(temp['胜率自算'])

        df_day_trade['牛熊震荡'] = df_day_trade['价格空间'] + df_day_trade['基本面空间']
        df_day_trade['牛熊震荡备忘'] = 1
        for idx, row in df_day_trade.reset_index().iterrows():
            if (idx >= 1) and (idx <= df_day_trade.shape[0] - 1):
                prev_row = df_day_trade.iloc[idx - 1]
                next_future_day = df_day_trade.reset_index().iloc[min(df_day_trade.shape[0] - 1, idx + 3)]
                if row['牛熊震荡'] * prev_row['牛熊震荡'] == -1:
                    df_day_trade.loc[row['DT']:next_future_day['DT'], '牛熊震荡备忘'] = 0
        df_day_trade['牛熊震荡_平滑'] = df_day_trade['牛熊震荡'] * df_day_trade['牛熊震荡备忘']
        df_day_trade['牛熊震荡_平滑'] = df_day_trade['牛熊震荡_平滑'].map(lambda x: 1 if x > 0 else -1 if x<0 else 0)

        # df_day_trade['牛熊震荡_平滑_牛'] = df_day_trade['牛熊震荡_平滑'].map(lambda x: True if x > 0 else False)
        # df_day_trade['牛熊震荡_平滑_熊'] = df_day_trade['牛熊震荡_平滑'].map(lambda x: True if x < 0 else False)
        if self.config.T != "D":
            df_day_trade_H = df_day_trade.shift(1).resample(self.config.T, label="left").asfreq().ffill()[
                ['牛熊震荡_平滑', '流动性信号',
                 '基本面信号', '胜率自算', 'T_price:信号', 'f_RSI', '反转信号', '弹性信号']]
        else:
            # df_day_trade_H = df_day_trade.resample(self.config.T, label="left").ffill()[['牛熊震荡_平滑','流动性信号',
            #                                                                              '基本面信号','胜率自算','T_price:信号','f_RSI','反转信号','弹性信号']]
            df_day_trade_H = df_day_trade[['牛熊震荡_平滑', '流动性信号', '基本面信号', '胜率自算', 'T_price:信号','f_RSI', '反转信号']]
        df_day_trade_H = df_day_trade_H.astype('category')
        df_day_trade_H = pd.get_dummies(df_day_trade_H).astype('bool')
        # df_day_trade_H.drop(list(df_day_trade_H.filter(regex='0$')), axis=1, inplace=True)
        self.featureNameDict['日度择时特征'] = df_day_trade_H.columns.tolist()
        if self.df.index[0]>df_day_trade_H.index[0]:
            self.df = pd.concat([self.df, df_day_trade_H.loc[self.df.index, :]], axis=1)
        else:
            self.df = pd.concat([self.df, df_day_trade_H], axis=1)

        print("宏观特征抽取完毕！")

    def _TechFeatureExtra(self,target='CLOSE'):
        # RSI特征
        df = pd.DataFrame(index=self.df.index)
        for ele in self.config.rsiPeriodList:
            for thres in self.config.res_threshold:
                df[f'close_RSI{ele}>{thres}'] = ta.RSI(self.df[target], timeperiod=ele).map(
                    lambda x: True if x > thres else False)
                df[f'close_RSI{ele}<{100 - thres}'] = ta.RSI(self.df[target], timeperiod=ele).map(
                    lambda x: True if x < 100 - thres else False)
        # 均线特征
        for ele in self.config.MAset:
            df[f'{self.config.targetName}_aboveMA{ele}'] = (
                    self.df[target] - self.df[target].rolling(ele).mean()) \
                .map(lambda x: True if x > 0 else False).astype("category")
            # df[f'{self.config.targetName}_belowMA{ele}'] = (
            #         self.df['close'] - self.df['close'].rolling(ele).mean()) \
            #     .map(lambda x: True if x < 0 else False)
        self.__cal_multitrend_signal(colname=target)
        temp0 = 2 * (self.df[target].diff(4) > 0).astype(int) - 1
        temp1 = (temp0.rolling(9).sum().abs() >= 9).astype(int) * temp0
        df['神奇九转_高9'] = temp1.map(lambda x: True if x > 0 else False)
        df['神奇九转_低9'] = temp1.map(lambda x: True if x < 0 else False)
        # for ele in self.config.updownMAset:
        #     self.df[f'up_cross_MA{ele}'] = (self.df[f'{self.config.targetName}_aboveMA{ele}'] - self.df[
        #         f'{self.config.targetName}_aboveMA{ele}'].shift(1)) > 0
        #     self.df[f'down_cross_MA{ele}'] = (self.df[f'{self.config.targetName}_aboveMA{ele}'] - self.df[
        #         f'{self.config.targetName}_aboveMA{ele}'].shift(1)) < 0
        upper, middle, lower = ta.BBANDS(self.df[target], timeperiod=20, nbdevup=2, nbdevdn=2)
        df['bband_up_break'] = self.df[target] > upper
        df['bband_down_break'] = self.df[target] < lower
        self.featureNameDict['技术特征'] = df.columns.tolist()
        self.df = pd.concat([self.df, df], axis=1)

        print("技术指标抽取完毕")

        # 弹性特征
        # 反转特征
        # return self.df

    def _breakOutFeatureExtra(self, df_results):
        lala = df_results[
            df_results[f'检测到低点后高点到低点下降{self.config.dropDelta}'] & df_results[f'{self.config.prefixstring}和前最低点靠近'] &
            df_results[f'{self.config.prefixstring}和前最低相隔>=5K线']]
        ListTemp = []
        for idx, row in lala.iterrows():
            futureLen = 20
            if not pd.isnull(row['found']):
                numerical_idx = df_results.index.get_loc(row['found'])  ##
                futureLen += 3
            else:
                numerical_idx = df_results.index.get_loc(idx)
            # 得到满足基础条件的idx，然后回溯原始数据
            df_roi = df_results.iloc[max(0, numerical_idx - self.config.lookbackT):numerical_idx + 1 + futureLen, :]
            # df_plot = df.iloc[max(0, numerical_idx - lowpoint_obj.lookbackT - 15):numerical_idx + 1 + futureLen, :]
            df_analysis = df_roi.loc[:idx, :].iloc[:-1, :]
            L_black = df_analysis[df_analysis['Dir'] == -1].set_index('found')  ##没有包含当前的低点
            LL = L_black['Value'].min();
            LL_idx = L_black['Value'].astype(float).idxmin()
            highsBetween = df_results.loc[LL_idx:row['found'], :]
            H_idx = highsBetween['close'].idxmax()
            H_value = highsBetween.loc[H_idx]['close']
            # self.W_highP =H_value
            dfFuture = df_roi.loc[idx:, :]
            if not dfFuture[(dfFuture['close'] > H_value).tolist()].empty:
                temp = dfFuture[(dfFuture['close'] > H_value).tolist()].iloc[:1, :]
                ListTemp.append(temp)
                # self.alines.append(
                #     )
                # self.colors.append()
                df_my_plot = df_roi.iloc[:-5, :]
                a = pd.Series(index=df_my_plot.index)
                a[temp.index.tolist()[0]] = df_my_plot.loc[temp.index.tolist()[0]]['high'] + 0.05
                apd = [mpf.make_addplot(a, scatter=True, markersize=30, marker='v', color='red')]

                mpf.plot(df_my_plot, type='candle', alines=dict(
                    alines=[(LL_idx, LL), (H_idx, H_value), (row['found'], row['Value']),
                            (temp.index.tolist()[0], temp['close'][0])], colors='#2b2d42'),
                         style='yahoo', addplot=apd)
                plt.show()
                print()
        doubleTestBreak = pd.concat(ListTemp).drop_duplicates().sort_index(ascending=True)
        print("复杂图形突破特征抽取完毕")
        return doubleTestBreak

    def _chartFeatureExtra(self, minSegSize,target):
        # df = pd.DataFrame(index=self.df.index)
        df = self.__createZigZagPoints(self.df[target], minSegSize=minSegSize)
        if 'open' in df.columns and 'low' in df.columns:
            df = df[['open', 'low', 'high', 'close', 'Value', 'Dir', 'found']]
        else:
            df = df[[target, 'Value', 'Dir', 'found']]
        ################################### Trend Features ###################################
        for ele in [60]:
            df[f'近{ele}bar最大下行超{self.dropDelta}'] = (df[target].rolling(ele).max() - df[target]) \
                .map(lambda x: True if x > self.dropDelta else False)

        def process_HL(x):
            lows = x[x < 0]
            if len(lows) < 2:
                return False
            return True if abs(lows[-1]) > abs(lows[-2]) else False
        def process_HH(x):
            highs = x[x > 0]
            if len(highs) < 2:
                return False
            return True if abs(highs[-1]) > abs(highs[-2]) else False

        temp = df[['Dir', 'Value']].fillna(0);
        temp = temp['Dir'] * temp['Value']
        df['低点抬高'] = temp.rolling(60).apply(process_HL)
        df['低点抬高'].fillna(False, inplace=True);
        df['低点抬高'] = df['低点抬高'].astype(bool)

        df['高点抬高'] = temp.rolling(60).apply(process_HH)
        df['高点抬高'].fillna(False, inplace=True);
        df['高点抬高'] = df['高点抬高'].astype(bool)
        if 'open' in df.columns:
            df['大阳线'] = (df[target] > df['open']) & (df[target].diff() > self.dropDelta / 5)
            df['大阴线'] = ((df['close'] < df['open']) &(df[target].diff() < -self.dropDelta / 5))
        # df[f'上涨{ self.dropDelta / 5}'] = df[target].diff() > (self.dropDelta / 5)
        # df[f'下跌{ self.dropDelta / 5}'] = df[target].diff() <- (self.dropDelta / 5)
        if 'low' in df.columns:
            df['当前K线低点>昨天低点'] = (df['low'].diff() > 0)
        # self.df['当前K线低点<昨天低点'] = (self.df['low'].diff() < 0)
        if 'high' in df.columns:
            df['当前K线高点>昨天高点'] = (df['high'].diff() > 0)
        if "open" in df.columns:
            df['当前K线开盘介于昨天开盘和收盘之间'] = ((df['open'] - df[target].shift(1)) * (
                    df['open'] - df['open'].shift(1))) < 0
        if "open" in df.columns:
            df['阳线'] = df[target] > df['open']
            df['阴线'] = df[target] < df['open']
        if ("low" in df.columns) and ("open" in df.cloumns):
            df['收盘明显高于最低'] = df.apply(lambda x: True if x[target] - x['low'] > self.dropDelta / 5 / 2 else False,
                                      axis=1)
            df['明显下引线'] = df.apply(
                lambda x: True if (x['阴线'] and (x[target] - x['low'] > self.dropDelta / 5 / 2)) else
                True if (x['阳线'] and (x['open'] - x['low'] > self.dropDelta / 5 / 2)) else False, axis=1)
            for idx, row in df.iterrows():
                numerical_idx = df.index.get_loc(idx)
                if numerical_idx < 15:
                    continue
                df_lookback = df.iloc[numerical_idx - 2:numerical_idx, :]
                if (df_lookback['阳线'][0]) and (df_lookback['阴线'][1]) and (
                        df_lookback['low'][1] > df_lookback['high'][0]):
                    df.loc[idx, '跳空后缺口不封闭'] = True
                df_lookback = df.iloc[numerical_idx - 3:numerical_idx, :]
                if df_lookback['阴线'].sum() == 3 and (df_lookback['low'].diff() < 0).sum() == 2 and (
                        df_lookback['high'].diff() < 0).sum() == 2:
                    df.loc[idx, '连续3天以上阴线且最高最低下移'] = True
                df_lookback = df.iloc[numerical_idx - 15:numerical_idx, :]
                df_hammer = df_lookback[df_lookback['明显下引线'] & df_lookback['阴线']]
                if df_hammer.shape[0] > 0:
                    if row['high'] < df_hammer['high'][-1]:
                        df.loc[idx, '当前K线最高价<长下引线阴线最高价'] = True  # 名字优化下
                    df_focus = df_lookback.loc[df_hammer.index[-1]:, :]  ##检查下引线及其后面部分到当前K线之前的中间部分
                    if row['low'] < df_focus['low'].min():
                        df.loc[idx, '当前K线最低价<长下引线阴线后所有线最低价'] = True  # 名字优化下
                    if df_focus.shape[0] > 2:
                        if df_focus['low'][1] > df_focus['low'][0]:
                            df.loc[idx, '长下引线阴线后第二根线最低价>下引线最低价'] = True
                        if ((df_focus['high'].diff() < 0)).sum() == len(df_focus['high']) - 1:
                            df.loc[idx, '长下引线阴线后所有线最高价<前一根最高价'] = True
            df['跳空后缺口不封闭'].fillna(False, inplace=True)
            df['连续3天以上阴线且最高最低下移'].fillna(False, inplace=True)
            df['长下引线阴线后所有线最高价<前一根最高价'].fillna(False, inplace=True)
            df['当前K线最高价<长下引线阴线最高价'].fillna(False, inplace=True)
            df['长下引线阴线后第二根线最低价>下引线最低价'].fillna(False, inplace=True)
            df['当前K线最低价<长下引线阴线后所有线最低价'].fillna(False, inplace=True)  # 名字优化下
        print("普通图像特征抽取完毕")
        ###################################### Patterns ####################################
        df_supres=None
        if os.path.exists(f"../res_sup_output/主力支撑_国开国债{self.year}y.csv"):
            supres = pd.read_csv(f"../res_sup_output/主力支撑_国开国债{self.year}y.csv",parse_dates=['DT']).set_index('DT')
            # supres['支撑角度_国开_10y'] = np.arctan(supres['支撑斜率_国开_10y']/0.05)/np.pi*180
            # supres['支撑角度_国债_10y'] = np.arctan(supres['支撑斜率_国债_10y']/0.05)/np.pi*180

            # supres['通道宽度_宽_国开_10y'] = (supres['阻力_国开_10y'] - supres['支撑_国开_10y']).map(lambda x: 1 if x>2 else 0 )
            # supres['通道宽度_窄_国开_10y'] = (supres['阻力_国开_10y'] - supres['支撑_国开_10y']).map(lambda x: 1 if 0<x<0.5 else 0 )

            supres['通道宽度_宽_国债_10y'] = (supres['阻力_国债_10y'] - supres['支撑_国债_10y']).map(lambda x: 1 if x>2 else 0 )
            supres['通道宽度_窄_国债_10y'] = (supres['阻力_国债_10y'] - supres['支撑_国债_10y']).map(lambda x: 1 if 0<x<0.5 else 0 )

            # supres['价格角度_国开_10y'] = np.arctan(supres['CLOSE_国开_10y'].diff(10)/(0.05*10))/np.pi*180
            supres['价格角度_国债_10y'] = np.arctan(supres['CLOSE_国债_10y'].diff(10)/(0.03*10))/np.pi*180
            supres['支撑线角度_国债_10y'] = np.arctan(supres['支撑_国债_10y'].diff(10)/(0.03*10))/np.pi*180

            # supres['价格角度30_45_国开_10y'] = (30<=supres['价格角度_国开_10y'])&(supres['价格角度_国开_10y']<=45)
            supres['价格角度30_60_国债_10y'] = (30<=supres['价格角度_国债_10y'])&(supres['价格角度_国债_10y']<=60)
            supres['支撑线角度30_60_国债_10y'] = (30<=supres['支撑线角度_国债_10y'])&(supres['支撑线角度_国债_10y']<=60)

            plt.style.use('ggplot')
            supres[['支撑_国债_10y','阻力_国债_10y','CLOSE_国债_10y']].plot(figsize=(11,6))
            df_angle = supres[supres['价格角度30_60_国债_10y']]
            df_angle2 = supres[supres['支撑线角度30_60_国债_10y']]
            plt.scatter(df_angle.index, df_angle['CLOSE_国债_10y'],color='black')
            plt.scatter(df_angle2.index, df_angle2['支撑_国债_10y'],color='red')

            # df_wide_channel = supres[supres['通道宽度_窄_国开_10y'] == 1]
            # plt.scatter(df_wide_channel.index, df_wide_channel['CLOSE_国债_10y'], color='black')
            plt.show()
            if '开' in self.type:
                df_supres= supres[['通道宽度_宽_国开_10y','通道宽度_窄_国开_10y','价格角度30_60_国开_10y','形态_国开_10y','压力线_国开_10y','支撑线_国开_10y']]
            else:
                df_supres = supres[['通道宽度_宽_国债_10y','通道宽度_窄_国债_10y','价格角度30_60_国债_10y','支撑线角度30_60_国债_10y','形态_国债_10y','压力线_国债_10y','支撑线_国债_10y']]
            print()
        else:
            print(f"主力支撑_国开国债{self.year}y.csv 不存在")

        temp = LowPoint(df, self.config).turnPointFeatureExtra()
        df = pd.concat([df, temp], axis=1).fillna(False)
        if df_supres is not None:

            df_supres = pd.get_dummies(df_supres).astype(bool)
            names_remove = df_supres.filter(regex="喇叭通道|无定义|矩形通道").columns.tolist()
            df_supres = df_supres.drop(names_remove, axis=1)
            df = pd.concat([df, df_supres], axis=1).fillna(False)
        df = df.iloc[:,4:]
        self.featureNameDict['图像特征'] = df.columns.tolist()
        # lala2 = self.df.merge(temp, left_index=True, right_index=True, how="left").fillna(False)
        self.df = pd.concat([self.df, df], axis=1)
        return df


class MyFeatureExtra_cls(FeatureExtra_cls):
    def __init__(self, df, config,year,type):
        super().__init__(df, config,year,type)

    def featureExtraction(self, tech_opt, chart_pattern_opt, macroeconomic_opt):
        if macroeconomic_opt:
            self._macroecoFeatureExtra()
        if tech_opt:
            self._TechFeatureExtra(target='CLOSE')
        if chart_pattern_opt:
            self._chartFeatureExtra(minSegSize=0.1,target='CLOSE')
        return self.df
        # if macroeconomic_opt


if __name__ == '__main__':
    type = "国"
    year = 10
    pointNum = 5
    alpha = 0.5
    # df = pd.read_csv(f"../data/{type}债{year}y-日线.csv", index_col=0, parse_dates=['DT']).set_index('DT')
    #     # # sup_res_mining(df, year, pointNum, alpha)
    #     # # df_f2 = pd.read_csv("../extractedResults/featureExtracResults.csv").set_index("time")
    #     # # df_all_close = pd.read_csv("../data/alldata.csv",parse_dates=['DT']).set_index("DT")
    #     # df_yield_rate = pd.read_csv("../data/模型信号_沈博专属.csv",parse_dates=['DT']).set_index("DT")
    #     # # df_all_close.plot()
    config_obj = paramConfig(T='D')
    # pd.read_csv("data/20192022.csv",parse_dates=['time']).set_index('time')
    df_data = Dataloader(config=config_obj).loadData(type=type,year=year)
    featureExtra_obj = MyFeatureExtra_cls(df_data, config=config_obj,year=year,type = type)
    df_features = featureExtra_obj.featureExtraction(tech_opt=True, chart_pattern_opt=True, macroeconomic_opt=False)
    fnames = []
    for k in featureExtra_obj.featureNameDict.keys():
        fnames += featureExtra_obj.featureNameDict[k]
    featureExtra_obj.df.to_csv(f"..\\特征抽取结果\\{year}y_{type}债_featureExtracResults_report.csv",encoding='utf_8_sig')
    print()
    # df_f2[fnames].values == df_features[fnames].values
    # json_object = json.dumps(featureExtra_obj.featureNameDict, indent=4)
    with open(f'..\\特征抽取结果\\{year}y_{type}债_report.json', 'w') as fp:
        json.dump(featureExtra_obj.featureNameDict, fp,sort_keys=True, indent=4,ensure_ascii=False)
    print()

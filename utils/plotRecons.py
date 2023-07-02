import mplfinance as mpf
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import talib
# import shutil
class plotPattern():
     def __init__(self,configobj,folderName):
         self.alines=[]
         self.colors=[]
         self.prefixstring = configobj.prefixstring
         self.region = configobj.hight_region
         self.dropDelta = configobj.dropDelta
         self.targetName = configobj.targetName
         self.imgCounter=0
         self.folderName=folderName
         self.T= configobj.T
         self.lookbackT = configobj.lookbackT
         self.vls=[]
         self.vls_colors=[]
         self.ma=[]
         self.configobj = configobj
         self.rsiList=[]
         self.adps=[]
         # self.total_imgs=total_imgs
     def linegenerator(self, df, idx, row, df_plot, plot,save, conditionList):
         self.adps=[]
         df_analysis = df.loc[:idx, :].iloc[:-1, :]
         datarow = df.loc[idx, :]
         L_black = df_analysis[df_analysis['Dir'] == -1].reset_index().set_index('found')  ##没有包含当前的低点
         conditionString = "".join([str(ele) for ele in conditionList])
         startLen = len(self.alines)
         if row[f'{self.prefixstring}当前点被前高支撑'] == True:
             H_black = df_analysis[df_analysis['Dir'] == 1].set_index('found')
             filtered_highs = H_black[H_black['Value'] < datarow['Value']]
             for h_idx in filtered_highs.index:
                 self.alines.append([(h_idx, filtered_highs.loc[h_idx, 'Value']), (idx, datarow['Value'])])
                 self.colors.append('y')
         if row[f'{self.prefixstring}和前最低点靠近']   or row[f'{self.prefixstring}当前点高于最低点']:
             LL = L_black['Value'].min();
             LL_idx = L_black['Value'].astype(float).idxmin()
             highsBetween = df.loc[LL_idx:datarow['found'], :]
             H_idx = highsBetween['CLOSE'].idxmax()
             H_value = highsBetween.loc[H_idx]['CLOSE']
             # self.alines.append([(LL_idx, LL), (datarow['found'], datarow['Value'])])
             # self.colors.append('#abac11')
             # dfFuture = df_plot.loc[idx:,:]

             self.alines.append(
                 [(LL_idx, LL), (H_idx, H_value), (datarow['found'], datarow['Value']), (idx, row['CLOSE'])])
             self.colors.append('#cdcf55')

         if row[f'{self.prefixstring}除去最低后还有2上相近低点'] == True or row[f'{self.prefixstring}除去最低后还有相近低点'] == True:
             LL_idx = L_black['Value'].astype(float).idxmin()
             LL_drop_lowest = L_black.drop([LL_idx], axis=0)
             LL_drop_lowest = LL_drop_lowest[(LL_drop_lowest['Value'] - datarow['Value']).abs() < self.region]
             temp = list(zip(LL_drop_lowest.index.tolist(), LL_drop_lowest['Value'].tolist()))
             # valuePart = list(zip(LL_drop_lowest['Value'].tolist()[:-1],LL_drop_lowest['Value'].tolist()[:-1]))
             temp.append((idx, datarow['Value']))
             self.alines.append(temp)
             self.colors.append('b')
         if row['低点抬高']:
             temp = df.loc[:idx, :]  ##包含了当前的低点
             temp = temp[temp['Dir'] == -1]
             self.alines.append([(temp.found[-1], temp['Value'][-1]), (temp.found[-2], temp['Value'][-2])])
             self.colors.append('b')
             print()
             # LL_idx = L_black['Value'].idxmin()
             # LL_drop_lowest = L_black.drop([LL_idx], axis=0)
             # LL_drop_lowest = LL_drop_lowest[(LL_drop_lowest['Value'] - datarow['Value']).abs() < self.region]
             # temp = list(zip(LL_drop_lowest.index.tolist(), LL_drop_lowest['Value'].tolist()))
             # # valuePart = list(zip(LL_drop_lowest['Value'].tolist()[:-1],LL_drop_lowest['Value'].tolist()[:-1]))
             # temp.append((idx, datarow['Value']))
             # self.alines.append(temp)
             # self.colors.append('b')
         if row[f'{self.prefixstring}当前低点在之前的下引线附近']:
             hammerPattern = df_analysis[df_analysis['明显下引线']]
             qualified_hammer = hammerPattern[(hammerPattern['LOW'] - datarow['Value']).abs() < self.region]
             temp = list(zip(qualified_hammer.index.tolist(), qualified_hammer['LOW'].tolist()))
             temp.append((idx, datarow['Value']))
             self.alines.append(temp)
             self.colors.append('m')
             print()
         if (row[f'{self.prefixstring}高点到低点下降{self.dropDelta}']):
             allTimeHigh = df_analysis[self.targetName].max()
             allTimeHigh_idx = df_analysis[self.targetName].idxmax()
             self.alines.append([(allTimeHigh_idx, allTimeHigh), (datarow['found'], datarow['Value'])])
             self.colors.append('r')
         if ((row[f'近{60}bar最大下行超{self.dropDelta}'])):
             allTimeHigh = df_analysis[self.targetName].max()
             allTimeHigh_idx = df_analysis[self.targetName].idxmax()
             self.alines.append([(allTimeHigh_idx, allTimeHigh), (idx, datarow['CLOSE'])])
             self.colors.append('r')

         # if row['长下引线阴线后所有线最高价<前一根最高价']:
         #     numerical_idx = df.index.get_loc(idx)
         #     df_lookback = df.iloc[numerical_idx - 15:numerical_idx, :]
         #     df_hammer = df_lookback[df_lookback['明显下引线'] & df_lookback['阴线']]
         #     if df_hammer.shape[0] > 0:
         #         # df_focus = df_lookback.loc[df_hammer.index[-1]:, :].iloc[1:, :]  ##检查中间部分
         #         self.alines.append([(df_hammer.index[-1], df_hammer.close[-1]), (idx, datarow['close'])])
         #         self.colors.append('m')

         if "MA" in conditionString:
            for ele in self.configobj.MAset:
                if "MA"+str(ele) in conditionString:
                    self.ma.append(ele)
         if "RSI" in conditionString:
             for ele in self.configobj.rsiPeriodList:
                 if "RSI" + str(ele) in conditionString:
                     df_plot['rsi'] = talib.RSI(df_plot['CLOSE'], timeperiod=ele)
                     line80 = [80] * len(df_plot['rsi'])
                     line20 = [20] * len(df_plot['rsi'])
                     ap0 = [
                         mpf.make_addplot(df_plot['rsi'], color='k', title='RSI', panel=1),
                         mpf.make_addplot(line80, panel=1, color='r', ),
                         mpf.make_addplot(line20, panel=1, color='g'),
                     ]
                     self.adps = self.adps + ap0
                     break
         endLen = len(self.alines)
         linesFramebyFrame = self.alines[startLen:endLen + 1]
         colorsFramebyFrame = self.colors[startLen:endLen + 1]
         temp = pd.Series(index=df_plot.index)
         temp[idx] = df_plot.loc[idx,'CLOSE']
         self.adps +=[mpf.make_addplot(temp,  title='Signal Triggered',scatter=True, markersize=500, marker=r'$\Downarrow$', color='#0563fa', panel=0,)]

         # vline = [(idx, df.loc[idx]['high'].min()), (idx, df['Value'].max())]
         # linesFramebyFrame.append(vline)
         # colorsFramebyFrame.append('k')
         # if save:
         #     if not os.path.exists(self.folderName):
         #         os.makedirs(self.folderName)

         # fileName = self.folderName + f'/{self.imgCounter}.png'
         # self.window['progress_1'].UpdateBar(((self.imgCounter+1)/self.total_imgs)*100)
         # df_plot['open'] = df_plot['close']
         # df_plot['low'] = df_plot['close']
         # df_plot['high'] = df_plot['close']

         # if save:
         #     mpf.plot(df_plot, type='line',savefig=fileName,tight_layout=True,addplot=self.adps)
         df_plot = df_plot.rename({"OPEN":"Open",'LOW':"Low","HIGH":"High","CLOSE":"Close"},axis=1)
         if plot:
             mpf.plot(df_plot, type='candle', alines=dict(alines=linesFramebyFrame, colors=colorsFramebyFrame),
                      style='yahoo', vlines=dict(vlines=self.vls, colors=self.vls_colors, alpha=0.1), mav=self.ma,
                      addplot=self.adps)

         # if w:
         #     plt.savefig(f'patternReconsResults/W_{self.T}/{self.imgCounter}.png')
         # else:

         self.imgCounter += 1
         plt.show()
         # print()
         # return result

     def plotbyCndition(self, df_results, filteredRows, conditionList,plot=False,save=True):

         # fileFolder = "learnedPatterns/Pattern" +str(patternNum)+"_"+ "_".join(conditionList)
         # fileFolder = "learnedPatterns"
         # fileFolder = fileFolder.replace(">=", "大于等于").replace(">", '大于').replace("<=", "小于等于").replace("<", "小于")
         # self.folderName = fileFolder + f"_{self.T}"
         # if not os.path.exists(self.folderName):
         #     os.makedirs(self.folderName)
         # else:
         #     shutil.rmtree(self.folderName)
         cols = filteredRows.columns.tolist()
         # # cols.remove(ele) for ele in conditionList
         filteredRows.loc[:, set(cols) - set(conditionList) - set(
             ['found', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'AMT', 'Dir', 'Value'])] = False
         resultList = []
         for idx, row in filteredRows.iterrows():
             futureLen = 20

             numerical_idx = df_results.index.get_loc(idx)
             # 得到满足基础条件的idx，然后回溯原始数据
             df_roi = df_results.iloc[max(0, numerical_idx - self.lookbackT):numerical_idx + 1 + futureLen, :]
             self.linegenerator(df_roi, idx, row, df_plot=df_results.iloc[max(0,
                                                                                       numerical_idx - self.lookbackT - 10):numerical_idx + 1 + futureLen,
                                                                   :], plot=plot, save=save,conditionList=conditionList)
         #     # resultList.append(result)
         # # dfPriceSta = pd.DataFrame(resultList)
         # # winrate = (dfPriceSta['delta'] > 0).sum() / dfPriceSta.shape[0]
         # # averagedWin = dfPriceSta['delta'].mean()
         # fig = mpf.figure(style='yahoo', figsize=(30, 8))
         # # ax3 = fig.add_subplot(3, 1, 3)
         # ax1 = fig.add_subplot(1, 1, 1)
         # # ax2 = fig.add_subplot(2, 1, 2)
         # if '当前低点有明显下引线' in conditionList:
         #     xx = [df_results.index.get_loc(ele) for ele in filteredRows.index.tolist()]
         #     a = [np.nan] * len(df_results)
         #     a = pd.Series(a)
         #     a[xx] = df_results.loc[filteredRows.index, 'close']
         #     test = mpf.make_addplot(a, type='scatter', markersize=20, marker='^', ax=ax1)
         #     mpf.plot(df_results, addplot=test, type='candle', alines=dict(alines=self.alines, colors=self.colors),
         #              style='yahoo', ax=ax1)
         #
         #     plt.savefig(f'{self.folderName}/全局总览.png')
         #
         # else:
         #     mpf.plot(df_results, type='candle', alines=dict(alines=self.alines, colors=self.colors), style='yahoo',
         #              ax=ax1)

         # plt.show()


         # fig = mpf.figure(style='yahoo', figsize=(30, 8))
         # ax1 = fig.add_subplot(1, 1, 1)
         # mpf.plot(df_results, type='candle', alines=dict(alines=lowpoint_obj.alines, colors=lowpoint_obj.colors),
         #          style='yahoo', ax=ax1)
         # return dfPriceSta.shape[0], winrate, averagedWin, filteredRows

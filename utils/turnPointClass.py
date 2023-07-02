import pandas as pd
import numpy as np
class LowPoint():
    # init method or constructor
    def __init__(self, df, config):
        # Sample Method
        self.df = df.copy()
        self.target = 'CLOSE'
        self.dropDelta = config.dropDelta
        self.region = config.hight_region
        self.smallgap = 2 / 100
        self.alines = []
        self.colors = []
        self.prefixstring = config.prefixstring
        self.imgCounter = 0
        self.lookbackT = config.lookbackT
        self.T = config.T
        self.folderName = None
        self.w_high = None
        self.breakout = []

    def turnPointFeatureExtra(self):
        results = []
        for idx, row in self.df.iterrows():
            if row['Dir'] == -1:
                # if idx == pd.to_datetime("2020-05-08 10:00:00"):
                #     print()
                numerical_idx = self.df.index.get_loc(row['found'])  # 从发现的低点往前看60天
                df_lookback = self.df.iloc[max(0, numerical_idx - self.lookbackT):numerical_idx, :]  ##当天的信息是不传的，靠row
                detectedData = self.detect(idx, df_lookback, row)
                results.append(detectedData)
        print("拐点相关特征抽取完毕")
        return pd.DataFrame(results).set_index('DT')

    def detect(self, idx, df, row):
        # row = df.iloc[-1,:]
        result = {}
        result['DT'] = idx
        allTimeHigh = df[self.target].max()
        allTimeHigh_idx = df[self.target].idxmax()
        L_black = df[df['Dir'] == -1].set_index('found')
        H_black = df[df['Dir'] == 1].set_index('found')
        # foundDate = df[df['Dir'] == -1]['found'].tolist()
        if L_black.shape[0] == 0:
            result[f'{self.prefixstring}高点到低点下降{self.dropDelta}'] = False
            result[f'{self.prefixstring}当前点高于最低点'] = False
            result[f'{self.prefixstring}当前点被前高支撑'] = False

            # result[f'和前最低相隔大于{self.region}'] = False
            result[f'{self.prefixstring}和前最低点靠近'] = False
            result[f'{self.prefixstring}除去最低后还有相近低点'] = False
            result[f'{self.prefixstring}除去最低后还有2上相近低点'] = False
            result[f'{self.prefixstring}当前低点在之前的下引线附近'] = False
            ##需要注意低点有下引线
            return result
        # L_black = df.loc[foundDate,:]
        # foundDate = df[df['Dir'] == 1]['found'].tolist()
        # H_black = df.loc[foundDate,:]

        # result["检测到低点"]=True

        # patternBeforeLowP = df.loc[:row['found'], :].iloc[:-1, :]
        # # hammerPattern = patternBeforeLowP[patternBeforeLowP['明显下引线']]
        # if hammerPattern.shape[0] == 0:
        #     result[f'{self.prefixstring}当前低点在之前的下引线附近'] = False
        # else:
        #     qualified_hammer = hammerPattern[(hammerPattern['low'] - row['Value']).abs() < self.region]
        #     result[f'{self.prefixstring}当前低点在之前的下引线附近'] = True if qualified_hammer.shape[0] > 0 else False

        LL = L_black['Value'].min();
        LL_idx = L_black['Value'].idxmin()

        if (allTimeHigh_idx < LL_idx) and (allTimeHigh - row['Value'] >= self.dropDelta):
            result[f'{self.prefixstring}高点到低点下降{self.dropDelta}'] = True
        else:
            result[f'{self.prefixstring}高点到低点下降{self.dropDelta}'] = False
        if (allTimeHigh_idx < LL_idx):
            result[f'{self.prefixstring}高点在之前最低点的左侧'] = True
        else:
            result[f'{self.prefixstring}高点在之前最低点的左侧'] = False

        if (LL < row['Value']):
            result[f'{self.prefixstring}当前点高于最低点'] = True
        else:
            result[f'{self.prefixstring}当前点高于最低点'] = False
        filtered_highs = H_black[H_black['Value'] < row['Value']]
        if (filtered_highs.shape[0] > 0):  ##低点被前高支撑
            result[f'{self.prefixstring}当前点被前高支撑'] = True
        else:
            result[f'{self.prefixstring}当前点被前高支撑'] = False
        # if (row['Value'] < LL and abs(row['Value'] - LL) > self.region):
        #     result[f'和前最低相隔大于{self.region}'] = True
        # else:
        #     result[f'和前最低相隔大于{self.region}'] = False
        if LL_idx in df.index.tolist():
            if df.shape[0] - df.index.get_loc(LL_idx) - 1 >= 3:
                result[f'{self.prefixstring}和前最低相隔>=3K线'] = True
                if df.shape[0] - df.index.get_loc(LL_idx) - 1 >= 5:
                    result[f'{self.prefixstring}和前最低相隔>=5K线'] = True
                else:
                    result[f'{self.prefixstring}和前最低相隔>=5K线'] = False
            else:
                result[f'{self.prefixstring}和前最低相隔>=3K线'] = False
        else:
            result[f'{self.prefixstring}和前最低相隔>=3K线'] = False

        if (-self.region <= row['Value'] - LL <= self.region):
            result[f'{self.prefixstring}和前最低点靠近'] = True
            highsBetween = df.loc[LL_idx:row['found'], :]
            H_idx = highsBetween[self.target].idxmax()
            H_value = highsBetween.loc[H_idx][self.target]
        else:
            result[f'{self.prefixstring}和前最低点靠近'] = False
        LL_drop_lowest = L_black.drop([LL_idx], axis=0)
        if LL_drop_lowest.shape[0] == 0:
            result[f'{self.prefixstring}除去最低后还有相近低点'] = False
            result[f'{self.prefixstring}除去最低后还有2上相近低点'] = False
        else:
            count = [1 if abs(ele - row['Value']) < self.region else 0 for ele in LL_drop_lowest['Value']]
            count = np.sum(count)
            if count >= 1:
                result[f'{self.prefixstring}除去最低后还有相近低点'] = True
            else:
                result[f'{self.prefixstring}除去最低后还有相近低点'] = False
            if count >= 2:
                result[f'{self.prefixstring}除去最低后还有2上相近低点'] = True
            else:
                result[f'{self.prefixstring}除去最低后还有2上相近低点'] = False

        return result

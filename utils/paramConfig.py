
class paramConfig():

    def __init__(self,T):
        self.T = T
        self.lookbackT = 60 if (T == "H" or T == "D") else 60 * 2 if T == "30T" else 0
        self.hight_region = 0.6 if T == "D" else 0.3 / 4 if (T == "H" or T == "30T") else 0.3 / 8 if T == "15T" else 0
        self.dropDelta = 1 if T == "D" else 0.5 if (T == "H" or T == "30T") else 0
        self.rsiPeriodList = [2, 14]
        self.res_threshold = [80]
        self.MAset = [40]
        self.updownMAset=self.MAset[:2]
        self.prefixstring = "检测到低点后"
        self.targetName = 'CLOSE'
        self.basic_infors = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'AMT', 'Dir', 'Value', 'found']

        ##关键趋势，我理解的不一定对就先自己暂时分了下
        self.key_trend_components = [f'近60bar最大下行超{self.dropDelta}','低点抬高', '大阳线',
                                '大阴线',
                                '长下引线阴线后所有线最高价<前一根最高价', '连续3天以上阴线且最高最低下移',
                                '跳空后缺口不封闭', f'检测到低点后高点到低点下降{self.dropDelta}', '检测到低点后当前点被前高支撑']

        ##关键形态，我理解的不一定对就先自己暂时分了下
        self.key_pattern_componenets = ['当前K线低点>昨天低点', '当前K线高点>昨天高点', '当前K线开盘介于昨天开盘和收盘之间', '阳线', '阴线', '收盘明显高于最低',
                                   '明显下引线', '当前K线最高价<长下引线阴线最高价', '当前K线最低价<长下引线阴线后所有线最低价',
                                   '长下引线阴线后第二根线最低价>下引线最低价', '检测到低点后当前点高于最低点', '检测到低点后和前最低点靠近', '检测到低点后除去最低后还有相近低点',
                                   '检测到低点后除去最低后还有2上相近低点', '检测到低点后当前低点在之前的下引线附近', '检测到低点后高点在之前最低点的左侧',
                                   '检测到低点后和前最低相隔>=3K线',
                                   '检测到低点后和前最低相隔>=5K线']
        ##关键价位
        self.key_complex_components = ['W底突破颈线']

        self.key_tech_components = [f"close_RSI{ele}>{thres}" for ele in self.rsiPeriodList for thres in self.res_threshold] \
                                   + [f"close_RSI{ele}<{100 - thres}" for ele in self.rsiPeriodList for thres in self.res_threshold]+\
                                   [f'{self.targetName}_aboveMA{ele}' for ele in self.MAset ]+[f'{self.targetName}_belowMA{ele}' for ele in self.MAset ]\
                                   +['T多头排列信号','T空头排列信号','神奇九转_牛反','神奇九转_熊反','bband_up_break','bband_down_break']

        self.key_comprehensive_components = ['牛熊震荡_平滑_牛', '牛熊震荡_平滑_熊']
        self.total_imgs =None
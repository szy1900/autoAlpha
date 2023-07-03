# 形态1
# conditionList= ['跳空后缺口不封闭','当前K线低点>昨天低点']
# #形态2
# conditionList= ['连续3天以上阴线且最高最低下移','阳线','当前K线低点>昨天低点','当前K线高点>昨天高点','当前K线开盘介于昨天开盘和收盘之间']
#
# # #形态3
# conditionList= ['长下引线阴线后所有线最高价<前一根最高价','阳线','当前K线最高价<长下引线阴线最高价','当前K线最低价<长下引线阴线后所有线最低价']
# #
# #形态4
# conditionList = ['长下引线阴线后所有线最高价<前一根最高价', '长下引线阴线后第二根线最低价>下引线最低价','阳线', '当前K线最高价<长下引线阴线最高价', '当前K线最低价<长下引线阴线后所有线最低价']
#


# conditionList=['阳线','收盘明显高于最低','检测到低点后和前最低点靠近']
# #conditionList=['近60bar最大下行超0.5','当前K线最高价<长下引线阴线最高价','当前K线最低价<长下引线阴线后所有线最低价','长下引线阴线后第二根线最低价>下引线最低价']
# conditionList=['当前K线高点>昨天高点','阳线','检测到低点后和前最低点靠近','检测到低点后和前最低相隔>=5K线']
# conditionList=['检测到低点后和前最低点靠近']
# conditionList=['大阳线', '当前K线低点>昨天低点', '当前K线高点>昨天高点', '检测到低点后当前低点在之前的下引线附近']
# conditionList=['阳线','收盘明显高于最低','检测到低点后和前最低点靠近']
# conditionList=['大阳线','检测到低点后当前点被前高支撑','收盘明显高于最低','检测到低点后当前低点在之前的下引线附近']
# conditionList=['当前价格站上40bar线' , '当前价格站上200bar线' , '阴线', '当前K线最高价<长下引线阴线最高价']
# conditionList=['近60bar最大下行超0.5','当前价格站上40bar线','大阴线']
# conditionList=['近60bar最大下行超0.5','低点抬高','长下引线阴线后所有线最高价<前一根最高价']
# conditionList=['当前价格站上40bar线','检测到低点后当前点被前高支撑','阴线']
# conditionList=['连续3天以上阴线且最高最低下移','检测到低点后高点到低点下降0.5','当前K线高点>昨天高点','收盘明显高于最低']
# #潜在类w底
T="H"
cofigObj = paramConfig(T=T)
conditionList = [f'{cofigObj.prefixstring}高点到低点下降{cofigObj.dropDelta}', f'{cofigObj.prefixstring}和前最低点靠近'
    , f'{cofigObj.prefixstring}和前最低相隔>=3K线' if T == "D" else f'{cofigObj.prefixstring}和前最低相隔>=5K线']
conditionList = ["牛熊震荡_平滑_牛", "close_RSI3<20"]
#########################
# 拷贝到这里
#####################
# #潜在类似三重底
# conditionList= [f'{cofigObj.prefixstring}高点到低点下降{delta}',f'{cofigObj.prefixstring}和前最低点靠近',f'{cofigObj.prefixstring}除去最低后还有2上相近低点']
#
# #潜在类头肩底
# conditionList= [f'{cofigObj.prefixstring}高点到低点下降{delta}',f'{cofigObj.prefixstring}和前最低点靠近',f'{cofigObj.prefixstring}当前点高于最低点',f'{cofigObj.prefixstring}除去最低后还有2上相近低点']
# conditionList = ['检测到低点后除去最低后还有相近低点']
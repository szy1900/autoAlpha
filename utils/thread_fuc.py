from utils.plotRecons import plotPattern
import pandas as pd
import numpy as np
import functools
def the_thread(cofig_obj,df_results, filteredRows, conditionList,window,folderName,total_imgs):
    cofig_obj.total_imgs = total_imgs
    pltObj = plotPattern(configobj=cofig_obj,folderName=folderName,window=window,total_imgs=total_imgs)
    pltObj.plotbyCndition(df_results, filteredRows, conditionList)
    window['total_imgs'].Update(f'图像总数:{total_imgs}')
    window['-LoadImage-'].click()


def quick_test_all_AU(updated_dataTable, window,values, cofig_obj):
    # df_Extra_results = pd.read_csv(
    #     "C:\\Users\\Administrator\\PycharmProjects\\autoAlpha\\AU\\AU_featureExtracResults.csv",
    #     parse_dates=['DT']).set_index("DT")
    df_table1 = pd.read_csv(f"AU/au_patternReco.csv", parse_dates=['DT']).set_index('DT')
    df_table1 = df_table1[['形态', '形态2']]
    df_table1 = pd.get_dummies(df_table1).astype(bool)
    selcted_fileds_xingtai = [ele for ele in df_table1.columns.tolist() if ele not in ['形态_喇叭通道', '形态_无定义']]
    df_table1 = df_table1[selcted_fileds_xingtai]
    df_table2 = pd.read_csv(f"AU/AU_featureExtracResults.csv", parse_dates=['DT']).set_index('DT')
    df_Extra_results = pd.concat([df_table1, df_table2], axis=1)
    df_Extra_results.fillna(False, inplace=True)
    # bigTable_save2 = bigTable_save.copy()
    bigTable_save2 = pd.DataFrame(index=df_Extra_results.index)
    for idx in range(len(updated_dataTable)):
        # if idx%2 ==0:
        window['progress'].update(f"{idx+1}/{len(updated_dataTable)}")
        conditionList = updated_dataTable.iloc[idx, :]['具体条件组合'].split("+")
        dfs = [df_Extra_results[ele] for ele in conditionList]
        if len(dfs) == 1:
            filteredRows = df_Extra_results[dfs[0].values]
        else:
            filteredRows = df_Extra_results[functools.reduce(lambda a, b: a & b, dfs)]
        bigTable_save2.loc[filteredRows.index,updated_dataTable.iloc[idx, :]['具体条件组合']] = True
        df_Extra_results[f"{cofig_obj.targetName}_diff10"] = df_Extra_results[f"{cofig_obj.targetName}"].diff(
            10).shift(-10)

        filteredRows[f"{cofig_obj.targetName}_diff10"] = df_Extra_results.loc[
            filteredRows.index, f"{cofig_obj.targetName}_diff10"]
        print(f"当前给定条件是:  {'+'.join(conditionList)}")
        filteredRows = filteredRows[(filteredRows.index >= pd.to_datetime(values['eval_start'])) & (
                filteredRows.index <= pd.to_datetime(values['eval_end']))]
        if filteredRows.shape[0] > 0:
            updated_dataTable.loc[idx, '次数_eval'] = filteredRows.shape[0]
            updated_dataTable.loc[idx, '胜率_eval'] = np.round(
                (filteredRows[f"{cofig_obj.targetName}_diff10"] > 0).sum() / filteredRows.shape[0], 2)
            updated_dataTable.loc[idx, '收益_eval'] = np.round(filteredRows[f"{cofig_obj.targetName}_diff10"].mean(), 2)
            # print()
            # print(
            #     f"信号次数{filteredRows.shape[0]};胜率{(filteredRows['close_diff10'] > 0).sum() / filteredRows.shape[0]};收益{filteredRows['close_diff10'].mean()}")
        else:
            updated_dataTable.loc[idx, '次数_eval'] = 0
            updated_dataTable.loc[idx, '胜率_eval'] = np.nan
            updated_dataTable.loc[idx, '收益_eval'] = np.nan
            # print(
            #     f"信号次数{0};胜率{np.nan};收益{np.nan}")
    date = values['date_input']
    bigTable_save2.to_csv(f"alpha因子信号大表_{pd.to_datetime(date).year}.csv",encoding="utf_8_sig")
    updated_dataTable.to_csv(f"挖掘因子训练评估表现总汇_{pd.to_datetime(date).year}.csv",encoding="utf_8_sig")
    window['TestAll'].click()


    # df_Extra_results.loc[filteredRows.index, 'detected'] = True
    # window['text_result'].update(
    #     f"信号次数{filteredRows.shape[0]};胜率{np.round(((filteredRows['close_diff10'] > 0).sum() / filteredRows.shape[0]) * 100, 3)};收益{np.round(filteredRows['close_diff10'].mean(), 3)}")
def quick_test_all_CDB(updated_dataTable, window,values, cofig_obj,fileName =f"CDB/cdb_featureExtracResults.csv" ):
    # df_Extra_results = pd.read_csv(
    #     "C:\\Users\\Administrator\\PycharmProjects\\autoAlpha\\AU\\AU_featureExtracResults.csv",
    #     parse_dates=['DT']).set_index("DT")

    df_table2 = pd.read_csv(fileName, parse_dates=['DT']).set_index('DT')
    df_Extra_results = df_table2
    df_Extra_results.fillna(False, inplace=True)
    # bigTable_save2 = bigTable_save.copy()
    bigTable_save2 = pd.DataFrame(index=df_Extra_results.index)
    for idx in range(len(updated_dataTable)):
        # if idx%2 ==0:
        window['progress'].update(f"{idx+1}/{len(updated_dataTable)}")
        conditionList = updated_dataTable.iloc[idx, :]['具体条件组合'].split("+")
        dfs = [df_Extra_results[ele] for ele in conditionList]
        if len(dfs) == 1:
            filteredRows = df_Extra_results[dfs[0].values]
        else:
            filteredRows = df_Extra_results[functools.reduce(lambda a, b: a & b, dfs)]
        bigTable_save2.loc[filteredRows.index,updated_dataTable.iloc[idx, :]['具体条件组合']] = True
        df_Extra_results[f"{cofig_obj.targetName}_diff10"] = df_Extra_results[f"{cofig_obj.targetName}"].diff(
            10).shift(-10)

        filteredRows[f"{cofig_obj.targetName}_diff10"] = df_Extra_results.loc[
            filteredRows.index, f"{cofig_obj.targetName}_diff10"]
        print(f"当前给定条件是:  {'+'.join(conditionList)}")
        filteredRows = filteredRows[(filteredRows.index >= pd.to_datetime(values['eval_start'])) & (
                filteredRows.index <= pd.to_datetime(values['eval_end']))]
        if filteredRows.shape[0] > 0:
            updated_dataTable.loc[idx, '次数_eval'] = filteredRows.shape[0]
            updated_dataTable.loc[idx, '胜率_eval'] = np.round(
                (filteredRows[f"{cofig_obj.targetName}_diff10"] > 0).sum() / filteredRows.shape[0], 2)
            updated_dataTable.loc[idx, '收益_eval'] = np.round(filteredRows[f"{cofig_obj.targetName}_diff10"].mean(), 2)
            # print()
            # print(
            #     f"信号次数{filteredRows.shape[0]};胜率{(filteredRows['close_diff10'] > 0).sum() / filteredRows.shape[0]};收益{filteredRows['close_diff10'].mean()}")
        else:
            updated_dataTable.loc[idx, '次数_eval'] = 0
            updated_dataTable.loc[idx, '胜率_eval'] = np.nan
            updated_dataTable.loc[idx, '收益_eval'] = np.nan
            # print(
            #     f"信号次数{0};胜率{np.nan};收益{np.nan}")
    date = values['date_input']
    # bigTable_save2.to_csv(f"alpha因子信号大表_{pd.to_datetime(date).year}.csv",encoding="utf_8_sig")
    updated_dataTable.to_csv(f"factor performance in evaluation_{pd.to_datetime(date).year}.csv",encoding="utf_8_sig")
    window['TestAll'].click()


    # df_Extra_results.loc[filteredRows.index, 'detected'] = True
    # window['text_result'].update(
    #     f"信号次数{filteredRows.shape[0]};胜率{np.round(((filteredRows['close_diff10'] > 0).sum() / filteredRows.shape[0]) * 100, 3)};收益{np.round(filteredRows['close_diff10'].mean(), 3)}")

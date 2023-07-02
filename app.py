import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os.path
from utils.patternComponents import *
# from utils.mylayout import mylayout
import matplotlib.image as mpimg
import matplotlib
import threading
from utils.thread_fuc import the_thread, quick_test_all_CDB
import shutil
from utils.geneticAlgorithm import *
from multiprocessing import Pool, freeze_support
from functools import partial
import utils.PySimpleGUI as sg

import multiprocessing

plt.style.use('ggplot')
sg.theme('DarkAmber')  # Add a touch of color

matplotlib.use('Agg')

global fitness_calculate


def fitness_calculate(solution, df_results):
    colsList = df_results.columns[:-1]
    conditionList = colsList[solution].tolist()
    if len(conditionList) == 0:
        return -100
    else:
        dfs = [df_results[ele] for ele in conditionList]
        if len(dfs) == 1:
            ldld = df_results[dfs[0].tolist()]
        else:
            ldld = df_results[functools.reduce(lambda a, b: a & b, dfs)]
        if (ldld.shape[0] >= 30) and sum(solution) <= 4:
            winrate = (ldld['delta'] > 0).sum() / ldld.shape[0]
            averagedWin = ldld['delta'].mean()
            score = winrate + averagedWin
        else:
            score = -100
        return score


# tournament selection
def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k - 1):
        # check if better (e.g. perform a tournament)
        if scores[ix] > scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1) - 2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


# mutation operator
def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            if rand() < 0.50:
                bitstring[i] = False  ##forcing the kernels to be sparser
            else:
                bitstring[i] = True if bitstring[i] == False else False


# genetic algorithm
def genetic_algorithm(window, df_results, n_bits, n_iter, n_pop, r_cross, r_mut):
    # initial population of random bitstring
    # pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
    goodList = []
    # resultRecorder = []
    pop = [np.random.choice([True, False], n_bits,p=[0.25,0.75]).tolist() for _ in range(n_pop)]
    # pop = [np.random.choice([True, False], n_bits, p=[0.5, 0.5]).tolist() for _ in range(n_pop)]

    # keep track of best solution
    best, best_eval = 0, fitness_calculate(pop[0], df_results=df_results)
    # enumerate generations
    best_eval = -100
    window['best_score'].Update("稍等学习迭代中")

    for gen in range(n_iter):
        print(f"gen is {gen}\n")
        # evaluate all candidates in the population
        with Pool(5) as p:
            # results = pool.map(partial(fitness_calculate, df_results=df_results),pop)
            scores = p.map(partial(fitness_calculate, df_results=df_results), pop)  ##多线程并行计算，快3倍左右
        # scores = [objective(c) for c in p
        # op]
        # t2 = time.time()
        # check for new best solution
        temp = pd.DataFrame({"pop": [tuple(ele) for ele in pop], 'score': scores}).drop_duplicates(
            subset=['pop']).sort_values(
            ascending=False, by='score')
        goodList.append(temp[temp['score'] > 0.8])
        for i in range(n_pop):
            if scores[i] > best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))
                window['best_score'].Update(f"best_score:{scores[i]}")
                # resultRecorder.append([gen, pop[i], scores[i]])
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            if i + 1 >= n_pop:
                continue
            p1, p2 = selected[i], selected[i + 1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
        n_pop = len(children)
        window['progress_2'].UpdateBar((gen + 1) / n_iter * 100)
    return [best, best_eval, goodList]


def pack_figure(graph, figure):
    canvas = FigureCanvasTkAgg(figure, graph.Widget)
    plot_widget = canvas.get_tk_widget()
    plot_widget.pack(side='top', fill='both', expand=1)
    return plot_widget


def GA_thread(df_results, window, featureTree):
    # print("进来了")
    T = "H"
    Train = True
    if Train:
        n_bits = len(df_results.columns) - 1
        # pop = [np.random.choice([True, False], n_bits).tolist() for _ in range(3)]
        print(f"等待挖掘因子数目:{n_bits}")
        n_iter = 5
        # define the population size
        # n_pop = 80000
        n_pop = 80000
        # crossover rate
        r_cross = 0.9
        # mutation rate
        r_mut = 0.5
        # perform the genetic algorithm search
        best, score, goodList = genetic_algorithm(window, df_results, n_bits, n_iter, n_pop, r_cross, r_mut)

        print('f(%s) = %f' % (best, score))
        print('Done!')
        findResults = pd.concat(goodList, axis=0)
        findResults['pop'] = findResults['pop'].map(lambda x: tuple(x))
        findResults.drop_duplicates(subset=['pop'], inplace=True)
        mykeys = featureTree.keys()
        for k in mykeys:
            findResults.loc[:, f'contains_{k}'] = False
        for idx, row in findResults.iterrows():
            pattern = row['pop']
            conditionList = df_results.columns[:-1][list(pattern)].tolist()
            dfs = [df_results[ele] for ele in conditionList]
            ldld = df_results[functools.reduce(lambda a, b: a & b, dfs)]
            winrate = (ldld['delta'] > 0).sum() / ldld.shape[0]
            averagedWin = ldld['delta'].mean()
            freq = ldld.shape[0]
            conditi = '+'.join(conditionList)

            findResults.loc[idx, '具体条件组合'] = conditi
            findResults.loc[idx, '条件数目'] = len(conditi.split('+'))

            findResults.loc[idx, '胜率'] = winrate
            findResults.loc[idx, '单笔收益'] = averagedWin
            findResults.loc[idx, '信号次数'] = freq
            for con in conditionList:
                for temp_key in mykeys:
                    if con in featureTree[temp_key]:
                        findResults.loc[idx, f'contains_{temp_key}'] = True
                        print()
                        break
        # for temp_key in mykeys:
        findResults.fillna(False, inplace=True)
        if not os.path.exists("data/"):
            os.makedirs("data/")

        if findResults.shape[0] > 0:
            if not os.path.exists(f'mined results/autoPatternFound_cdb.csv'):
                findResults = findResults.drop_duplicates(subset=['具体条件组合'])
                findResults.loc[:, ['次数_eval', '胜率_eval', '收益_eval']] = np.nan
                findResults.to_csv(f'mined results/autoPatternFound_cdb.csv',
                                   encoding='utf_8_sig', index=False)
            else:
                # findResults.to_csv(f'mined results/autoPatternFound_cdb.csv', mode='a', index=False,
                #                    header=False, encoding='utf-8-sig')
                ori_data = pd.read_csv(f'mined results/autoPatternFound_cdb.csv')
                # .drop_duplicates(subset=['具体条件组合'])
                tempdata = pd.concat([findResults, ori_data], axis=0)
                tempdata = tempdata.drop_duplicates(subset=['具体条件组合'])
                tempdata.to_csv(
                    f'mined results/autoPatternFound_cdb.csv', encoding='utf_8_sig', index=False)
        window['GA'].click()


def run():
    dataLength = None
    run_time = 0
    f = open("data/28bondfeatures.json")

    # f = open("dictData.json")
    featureTree = json.load(f)
    try:
        data = pd.read_csv("mined results/autoPatternFound_cdb.csv").iloc[:, 1:].sort_values(by=['条件数目', '胜率', '信号次数'],
                                                                                    ascending=[1, 0, 0]).round(3)
        updated_dataTable = data.reset_index().drop(['index'], axis=1)
        tableData = updated_dataTable[
            updated_dataTable.columns.drop(list(data.filter(regex='^contains')))].values.tolist()
        headings = updated_dataTable[
            updated_dataTable.columns.drop(list(updated_dataTable.filter(regex='^contains')))].columns.tolist()
        dataLength = updated_dataTable.shape[0]
    except:
        headings = ['score', '具体条件组合', '条件数目', '胜率', '单笔收益', '信号次数', '次数_eval', '胜率_eval', '收益_eval']
        tableData = []
        dataLength = 0

    checkbox_treedata = sg.TreeData()
    ga_checkbox = []
    for key in featureTree.keys():
        checkbox_treedata.Insert("", f"{key}", f"{key}", values=[len(featureTree[key])], icon="icons/lamp16.ico",
                                 checkbox_enabled=False)
        ga_checkbox.append(key)
        for ele in featureTree[key]:
            if type(ele) == str:
                checkbox_treedata.Insert(f"{key}", ele, ele, values=[], icon="icons/lamp16.ico", checkbox_enabled=True)

    col_checkbox_tree = [[sg.Text('File and folder browser Test'), sg.Button('重新抽取特征')],
                         [sg.Tree(data=checkbox_treedata,
                                  headings=[' 特征个数 ', ],
                                  auto_size_columns=True,
                                  num_rows=20,
                                  col0_width=40,
                                  key='-CHECKBOX_TREE-',
                                  background_color="#D8D0CE",
                                  select_mode=sg.TABLE_SELECT_MODE_BROWSE,
                                  show_expanded=False,
                                  enable_events=True),
                          ], [sg.Text("已选特征:", key='-TEXT-', auto_size_text=True, size=(50, 5))]]
    checkboxList_filter = [sg.Checkbox(f'{key}_选择', default=True, key=f'{key}_选择', enable_events=True) for key in
                           ga_checkbox]

    checkboxList = [
        sg.Checkbox(f'{key}', default=True, key=f'{key}') if key == "图像特征" else sg.Checkbox(f'{key}', default=True,
                                                                                             key=f'{key}') for key in
        ga_checkbox]
    layout = [[sg.Frame('autoAlpha', [
        checkboxList_filter + [sg.Text("训练截止日期:"), sg.InputText('2018/12/31', size=[12, 3], key="date_input")] + [
            sg.Text('评估时间:'), sg.InputText('2019/01/01', size=[12, 3], key="eval_start"), sg.Text('--'),
            sg.InputText('2022/12/31', size=[12, 3], key="eval_end")],
        [sg.Table(values=tableData, headings=headings, col_widths=len(headings) * [10], max_col_width=25,
                  auto_size_columns=False,
                  # cols_justification=('left','center','right','c', 'l', 'bad'),       # Added on GitHub only as of June 2022
                  display_row_numbers=True,
                  justification='center',
                  num_rows=20,
                  # background_color = "red",
                  alternating_row_color='lightblue',
                  key='-TABLE-',
                  # selected_row_colors='red on yellow',
                  enable_events=True,
                  # expand_x=False,
                  # expand_y=True,
                  vertical_scroll_only=False,
                  # enable_click_events=True,  # Comment out to not enable header and other clicks
                  tooltip='mined results表')], checkboxList + [
            sg.Text("              胜率:-   信号次数:-  10个时间单位后变化:-", key='text_result_quick', auto_size_text=True,
                    size=(50, 1))], [sg.Button('自动挖掘', key='start_ga'), sg.Text("best_score: 暂无", key="best_score"),
                                     sg.Button('快速评估单一组合', key='quick_test', disabled=True),
                                     sg.Button('快速评估所有', key='quick_test_all', disabled=False),
                                     sg.Text("", key="progress", size=(10, 1)), sg.Button('TestAll', visible=False),
                                     sg.Button('GA', visible=False)],
        [sg.ProgressBar(max_value=100,
                        orientation='h',
                        size=(50, 20),
                        bar_color=(
                            '#ebab34', '#D0D0D0'),
                        visible=True,
                        key='progress_2')]]),
               sg.Frame('手动勾选评测', [[sg.Column(col_checkbox_tree)], [
                   sg.Text("胜率:-   信号次数:-  10个时间单位后变化:-", key='text_result', auto_size_text=True, size=(50, 1))],
                                   [sg.Button('手动评估因子组合'),
                                    sg.Button('Cancel'),
                                    sg.Button('-LoadImage-',
                                              visible=False)], [
                                       sg.Checkbox('是否记录和显示图像',
                                                   key="image_show",
                                                   default=False)],
                                   [sg.ProgressBar(max_value=100,
                                                   orientation='h',
                                                   size=(50, 20),
                                                   bar_color=(
                                                       '#ebab34', '#D0D0D0'),
                                                   visible=False,
                                                   key='progress_1')]]),
               # sg.Frame('评测图像', [[sg.T('图像总数:0     ', key='total_imgs'), sg.T(' 当前帧号:0   ', key='curr_imgs'),
               #                    sg.InputText(size=(6, 2), key="goInput"),
               #                    sg.Button('跳转', key='go', disabled=True)],
               #                   [sg.Graph(key="-Graph-", canvas_size=(0, 0), pad=(0, 0), graph_bottom_left=(0, 0),
               #                             graph_top_right=(0, 0), background_color="lightblue")],
               #                   [sg.Button('上一帧', key='prev', disabled=True),
               #                    sg.Button('下一帧', key='next', disabled=True)]], key='-FRAME-')
               ]]
    window = sg.Window('autoAplha', layout, resizable=False, finalize=True)
    checkedList = []
    # if run_time == 1:
    fig1 = plt.figure(1)  # Create a new figure
    ax1 = plt.subplot(111)
    ax1.axis('off')
    # img = mpimg.imread('learnedPatterns_H/0.png')
    # imgplot = plt.imshow(img)
    #
    # graph1 = window['-Graph-']

    # df_results = pd.read_csv("extractedResults/featureExtracResults.csv", parse_dates=['time']).set_index("time")

    cofig_obj = paramConfig(T="D")
    imageIdx = 0
    while True:
        event, values = window.read()
        print(event, values)
        if event == "-TABLE-":
            slected_indx = values['-TABLE-'][0]
            # lala = updated_dataTable.iloc[slected_indx,:]['具体条件组合']
            # updated_dataTable.loc[:,lala.split("+")]
            window['quick_test'].update(disabled=False)
            print(slected_indx)
        if event == "quick_test_all":
            ##测试所有挖掘的因子
            threading.Thread(target=quick_test_all_CDB,
                             args=(updated_dataTable, window, values, cofig_obj, f"data/10y_bond_inputMatrix.csv"),
                             daemon=True).start()
        if event == "TestAll":
            # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
            ##测试所有挖掘的因子
            updated_dataTable = updated_dataTable.sort_values(by=['胜率_eval', '收益_eval', '次数_eval', '条件数目', ],
                                                              ascending=[0, 0, 0, 1]).round(3)
            updated_dataTable.drop_duplicates(subset=['score'], keep='first', inplace=True)
            tableData = updated_dataTable[
                updated_dataTable.columns.drop(list(updated_dataTable.filter(regex='^contains')))].values.tolist()
            updated_dataTable[
                updated_dataTable.columns.drop(list(updated_dataTable.filter(regex='^contains')))].to_csv(
                "evaluated results/因子评估结果.csv", encoding='utf_8_sig')
            window['-TABLE-'].update(values=tableData)
            # quick_test_all(updated_dataTable, window,values, cofig_obj)
        if event == "quick_test":
            df_Extra_results = pd.read_csv(
                f"data/10y_bond_inputMatrix.csv",
                        parse_dates=['DT']).set_index("DT")
            df_Extra_results.fillna(False, inplace=True)
            imageIdx = 0
            window['curr_imgs'].Update(f" 当前帧号:{imageIdx}")
            window['goInput'].Update(f"{imageIdx}")

            conditionList = updated_dataTable.iloc[slected_indx, :]['具体条件组合'].split("+")
            dfs = [df_Extra_results[ele] for ele in conditionList]
            if len(dfs) == 1:
                filteredRows = df_Extra_results[dfs[0].values]
            else:
                filteredRows = df_Extra_results[functools.reduce(lambda a, b: a & b, dfs)]
            df_Extra_results[f"{cofig_obj.targetName}_diff10"] = df_Extra_results[f"{cofig_obj.targetName}"].diff(
                10).shift(-10)

            filteredRows[f"{cofig_obj.targetName}_diff10"] = df_Extra_results.loc[
                filteredRows.index, f"{cofig_obj.targetName}_diff10"]
            print(f"当前给定条件是:  {'+'.join(conditionList)}")
            filteredRows = filteredRows[(filteredRows.index >= pd.to_datetime(values['eval_start'])) & (
                        filteredRows.index <= pd.to_datetime(values['eval_end']))]
            if filteredRows.shape[0] > 0:
                print(
                    f"信号次数{filteredRows.shape[0]};胜率{(filteredRows[f'{cofig_obj.targetName}_diff10'] > 0).sum() / filteredRows.shape[0]};收益{filteredRows[f'{cofig_obj.targetName}_diff10'].mean()}")
            else:
                print(
                    f"信号次数{0};胜率{np.nan};收益{np.nan}")
            df_Extra_results.loc[filteredRows.index, 'detected'] = True
            window['text_result_quick'].update(
                f"信号次数{filteredRows.shape[0]};胜率{np.round(((filteredRows[f'{cofig_obj.targetName}_diff10'] > 0).sum() / filteredRows.shape[0]) * 100, 3)};收益{np.round(filteredRows[f'{cofig_obj.targetName}_diff10'].mean(), 3)}")
            # sg.Text("胜率:-   信号次数:-  10个时间单位后变化:-", key='text_result', auto_size_text=True, size=(50, 1))
            if values['image_show']:
                window['progress_1'].Update(visible=True)
                fileFolder = "learnedPatterns"
                fileFolder = fileFolder.replace(">=", "大于等于").replace(">", '大于').replace("<=", "小于等于").replace("<",
                                                                                                               "小于")
                folderName = fileFolder + f"_{cofig_obj.T}"
                if not os.path.exists(folderName):
                    os.makedirs(folderName)
                else:
                    shutil.rmtree(folderName)
                    os.makedirs(folderName)
                # window['progress_1'].UpdateBar(50)

                threading.Thread(target=the_thread,
                                 args=(cofig_obj, df_Extra_results, filteredRows, conditionList, window, folderName,
                                       filteredRows.shape[0]),
                                 daemon=True).start()
        if event in ['图像特征_选择', '技术特征_选择', '日度择时特征_选择']:
            filter_combined = []
            if values['图像特征_选择']:
                filter_combined.append(updated_dataTable['contains_图像特征'].astype('bool'))
            if values['技术特征_选择']:
                filter_combined.append(updated_dataTable['contains_技术特征'].astype('bool'))
            if values['日度择时特征_选择']:
                filter_combined.append(updated_dataTable['contains_日度择时特征'].astype('bool'))
            # (updated_dataTable['contains_图像特征'] == values['图像特征_选择']) and (updated_dataTable['contains_技术特征'] == values['技术特征_选择'])
            if len(filter_combined) > 0:
                mytableData = updated_dataTable[functools.reduce(lambda a, b: a | b, filter_combined)]
                tableData = mytableData[mytableData.columns.drop(list(data.filter(regex='^contains')))].values.tolist()
            else:
                tableData = []
            # tableData = mytableData.values.tolist()
            window['-TABLE-'].update(values=tableData)
            print("Hello World")
        if event in (None, 'Cancel'):
            break
        if event in "GA":
            window['start_ga'].update(disabled=False)

            data = pd.read_csv("mined results/autoPatternFound_cdb.csv").iloc[:, 1:].sort_values(by=['条件数目', '胜率', '信号次数'],
                                                                                        ascending=[1, 0, 0]).round(3)
            updated_dataTable = data.reset_index().drop(['index'], axis=1)
            tableData = updated_dataTable[
                updated_dataTable.columns.drop(list(updated_dataTable.filter(regex='^contains')))].values.tolist()
            # tableData = data.values.tolist()
            window['-TABLE-'].update(values=tableData)

        if event in ('prev', 'next') and run_time >= 1:
            if event == 'prev':
                imageIdx = max(0, imageIdx - 1)
            if event == 'next':
                imageIdx = min(imageIdx + 1, cofig_obj.total_imgs - 1)
            fig = plt.figure(1)  # Active an existing figure
            ax = plt.gca()  # Get the current axes
            ax.cla()  # Clear the current axes
            ax.axis('off')
            img = mpimg.imread(f'learnedPatterns_{cofig_obj.T}/{imageIdx}.png')
            imgplot = plt.imshow(img)
            fig.canvas.draw()
            window['curr_imgs'].Update(f" 当前帧号:{imageIdx}")
            print(imageIdx)
        if event in "start_ga":
            window['start_ga'].update(disabled=True)
            df_table = pd.read_csv(
                "data\\10y_bond_inputMatrix.csv",
                parse_dates=['DT']).set_index("DT")
            df_table['delta'] = df_table['CLOSE'].diff(10).shift(-10)
            df_table = df_table[df_table.index <= pd.to_datetime(values['date_input'])]
            selcted_fileds = []
            for check_key in ga_checkbox:
                if values[check_key]:
                    selcted_fileds += featureTree[check_key]
            selcted_fileds += ['delta']
            df_results = df_table[selcted_fileds]
            df_results = df_results.dropna(subset=['delta'])
            df_results = df_results.fillna(False)
            threading.Thread(target=GA_thread, args=(df_results, window, featureTree), daemon=True).start()
        if event in "go":
            imageIdx = int(values["goInput"])
            imageIdx = np.clip(imageIdx, 0, cofig_obj.total_imgs - 1)
            fig = plt.figure(1)  # Active an existing figure
            ax = plt.gca()  # Get the current axes
            ax.cla()  # Clear the current axes
            ax.axis('off')
            img = mpimg.imread(f'learnedPatterns_H/{imageIdx}.png')
            imgplot = plt.imshow(img)
            fig.canvas.draw()
            window['curr_imgs'].Update(f" 当前帧号:{imageIdx}")
        # if event in "-LoadImage-":
        #     window['prev'].Update(disabled=False)
        #     window['next'].Update(disabled=False)
        #     window['go'].Update(disabled=False)
        #
        #     window['progress_1'].Update(visible=False)
        #     run_time = run_time + 1
        #     if run_time == 1:
        #         img = mpimg.imread(f'learnedPatterns_{cofig_obj.T}/{imageIdx}.png')
        #         imgplot = plt.imshow(img)
        #         pack_figure(graph1, fig1)
        #     else:
        #         fig = plt.figure(1)  # Active an existing figure
        #         ax = plt.gca()  # Get the current axes
        #         ax.cla()  # Clear the current axes
        #         ax.axis('off')
        #         img = mpimg.imread(f'learnedPatterns_{cofig_obj.T}/{imageIdx}.png')
        #         imgplot = plt.imshow(img)
        #         fig.canvas.draw()  # Rendor figure into canvas

        if event in "手动评估因子组合":
            # df_Extra_results = pd.read_csv("extractedResults/featureExtracResults.csv", parse_dates=['time']).set_index(
            #     "time")
            df_Extra_results = pd.read_csv(
                "data\\10y_bond_inputMatrix.csv",
                parse_dates=['DT']).set_index("DT")
            df_Extra_results.fillna(False, inplace=True)
            imageIdx = 0
            window['curr_imgs'].Update(f" 当前帧号:{imageIdx}")
            window['goInput'].Update(f"{imageIdx}")
            conditionList = checkedList
            dfs = [df_Extra_results[ele] for ele in conditionList]
            if len(dfs) == 1:
                filteredRows = df_Extra_results[dfs[0].values]
            else:
                filteredRows = df_Extra_results[functools.reduce(lambda a, b: a & b, dfs)]
            df_Extra_results[f"{cofig_obj.targetName}_diff10"] = df_Extra_results[f"{cofig_obj.targetName}"].diff(
                10).shift(-10)
            filteredRows[f"{cofig_obj.targetName}_diff10"] = df_Extra_results.loc[
                filteredRows.index, f"{cofig_obj.targetName}_diff10"]
            print(f"当前给定条件是:  {'+'.join(conditionList)}")
            if filteredRows.shape[0] > 0:
                print(
                    f"信号次数{filteredRows.shape[0]};胜率{(filteredRows[f'{cofig_obj.targetName}_diff10'] > 0).sum() / filteredRows.shape[0]};收益{filteredRows[f'{cofig_obj.targetName}_diff10'].mean()}")
            else:
                print(
                    f"信号次数{0};胜率{np.nan};收益{np.nan}")
            df_Extra_results.loc[filteredRows.index, 'detected'] = True
            window['text_result'].update(
                f"信号次数{filteredRows.shape[0]};胜率{np.round(((filteredRows[f'{cofig_obj.targetName}_diff10'] > 0).sum() / filteredRows.shape[0]) * 100, 3)};收益{np.round(filteredRows[f'{cofig_obj.targetName}_diff10'].mean(), 3)}")
            # sg.Text("胜率:-   信号次数:-  10个时间单位后变化:-", key='text_result', auto_size_text=True, size=(50, 1))
            if values['image_show']:
                window['progress_1'].Update(visible=True)
                fileFolder = "learnedPatterns"
                fileFolder = fileFolder.replace(">=", "大于等于").replace(">", '大于').replace("<=", "小于等于").replace("<",
                                                                                                               "小于")
                folderName = fileFolder + f"_{cofig_obj.T}"
                if not os.path.exists(folderName):
                    os.makedirs(folderName)
                else:
                    shutil.rmtree(folderName)
                    os.makedirs(folderName)
                # window['progress_1'].UpdateBar(50)

                # threading.Thread(target=the_thread,
                #                  args=(cofig_obj, df_Extra_results, filteredRows, conditionList, window, folderName,
                #                        filteredRows.shape[0]),
                #                  daemon=True).start()

            # pltObj = plotPattern(configobj=cofig_obj)
            # pltObj.plotbyCndition(df_results, filteredRows, conditionList, patternNum=1, save=True, plot=False)
            # window['-LoadImage-'].click()
            # if path.exists(f"{pltObj.folderName}"):
            #     df_results.to_csv(f"{pltObj.folderName}/allinfor.csv")
            #     with open(f'{pltObj.folderName}' + '/readme.txt', 'w') as f:
            #         f.write(
            #             f"信号次数{filteredRows.shape[0]};胜率{(filteredRows['close_diff10'] > 0).sum() / filteredRows.shape[0]};收益{filteredRows['close_diff10'].mean()}")
            # print("##############finish####################")
            print()

        if event in '-CHECKBOX_TREE-':
            # print('Key: ' + str(values['-CHECKBOX_TREE-'][0]) + '\nTree Checkbox state(before change): '
            #       + str(window['-CHECKBOX_TREE-'].CheckboxState[values['-CHECKBOX_TREE-'][0]]))
            prev_state = window['-CHECKBOX_TREE-'].CheckboxState[values['-CHECKBOX_TREE-'][0]]['checked']
            # checks or unchecks an tree node
            window['-CHECKBOX_TREE-'].change_checkbox_state(values['-CHECKBOX_TREE-'][0])
            after_state = window['-CHECKBOX_TREE-'].CheckboxState[values['-CHECKBOX_TREE-'][0]]['checked']
            if (prev_state == False) and (after_state == True):
                checkedList.append(str(values['-CHECKBOX_TREE-'][0]))
                window['-TEXT-'].update(";".join(checkedList))
            if (prev_state == True) and (after_state == False):
                checkedList.remove(str(values['-CHECKBOX_TREE-'][0]))
                window['-TEXT-'].update(";".join(checkedList))

            # print('Key: ' + str(values['-CHECKBOX_TREE-'][0]) + '\nTree Checkbox state(after change): '
            #       + str(window['-CHECKBOX_TREE-'].CheckboxState[values['-CHECKBOX_TREE-'][0]]))
            print()
        elif event == "Run" or event == 'Exit':
            pass
            # theta1 = (theta1 + 40) % 360
            # plot_figure(1, theta1)
            # theta2 = (theta2 + 40) % 260
            # plot_figure(2, theta2)
    window.close()


if __name__ == '__main__':
    freeze_support()
    updated_dataTable = None
    run()

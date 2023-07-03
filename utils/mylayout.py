import utils.PySimpleGUI as sg

def mylayout(featureTree,tableData,headings):
    # width, height = 640, 480

    checkbox_treedata = sg.TreeData()
    ga_checkbox =[]
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
    checkboxList=[sg.Checkbox(f'{key}', default=False, key=f'{key}') for key in ga_checkbox]
    layout = [[sg.Frame('autoAlpha', [checkboxList,
                                      [sg.Table(values=tableData[1:][:], headings=headings, max_col_width=25,
                                                auto_size_columns=True,
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
                                                tooltip='This is a table')], [sg.Button('自动挖掘'), sg.Text(
        "best_score: 暂无                ", key="best_score"), sg.Button('GA', visible=False)],
                                      [sg.ProgressBar(max_value=100,
                                                      orientation='h',
                                                      size=(50, 20),
                                                      bar_color=(
                                                          '#ebab34', '#D0D0D0'),
                                                      visible=True,
                                                      key='progress_2')]]),
               sg.Frame('手动勾选评测', [[sg.Column(col_checkbox_tree)], [
                   sg.Text("胜率:-   发生次数:-  10个时间单位后变化:-", key='text_result', auto_size_text=True, size=(50, 1))],
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
               sg.Frame('评测图像', [[sg.T('图像总数:0     ', key='total_imgs'), sg.T(' 当前帧号:0   ', key='curr_imgs'),
                                  sg.InputText(size=(6, 2), key="goInput"),
                                  sg.Button('跳转', key='go', disabled=True)],
                                 [sg.Graph(key="-Graph-", canvas_size=(0, 0), pad=(0, 0), graph_bottom_left=(0, 0),
                                           graph_top_right=(0, 0), background_color="lightblue")],
                                 [sg.Button('上一帧', key='prev', disabled=True),
                                  sg.Button('下一帧', key='next', disabled=True)]], key='-FRAME-')
               ]]
    return layout
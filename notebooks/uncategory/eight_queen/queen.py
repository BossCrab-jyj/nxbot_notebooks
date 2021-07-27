import copy
import random
import matplotlib.pyplot as plt
# %matplotlib inline
from ipywidgets import Button, GridBox, Layout, ButtonStyle
import itertools

# 初始化八皇后状态
def initiate(status):
    while len(status)<8:
        r=random.randint(0,7)
        if not (r in status):
            status.append(r)
    return status


# 冲突值计算
def conflict(status):
    num = 0
    conflict_chess = []
    # 计算所有皇后的冲突值
    for i in range(8):
        for j in range(i+1,8):
            
            # 1.记录同一列的情况，如status = [0,1,1,2,3,4,5,6,7]，此时status[1] == status[2]，说明当前两个皇后位置在(1,1)和(2,1),就表示这两个皇后的位置在同一列。
            if status[i]==status[j]:
                num += 1
                
                #记录冲突皇后的位置信息, [i,status[i]]或者[j,status[j]]代表一个棋子当前的位置
                location = [[i,status[i]],[j,status[j]]]
                for m in range(len(location)):
                    if location[m] not in conflict_chess:
                        # 将有冲突的皇后位置放入列表“conflict_chess”
                        conflict_chess.append(location[m])
                        
            # 2.记录皇后位置是否在斜边上。
            elif abs(status[i]-status[j])==j-i:
                num += 1
                
                #记录冲突皇后的位置信息
                location = [[i,status[i]],[j,status[j]]]
                for n in range(len(location)):
                    if location[n] not in conflict_chess:
                        conflict_chess.append(location[n])
                
    return num, conflict_chess


# 计算每个空位置的冲突值
def neighbour(status):
    
    # 创建空键值对，用来存储位置和冲突数量，如{(0,0):5},表示位置（0，0）的冲突值为5。
    close = {}
    
    # 两个循环生成8*8棋盘。
    for i in range(8):
        for j in range(8):
            
            # 当检测到当前位置有皇后时，不记录有皇后的位置
            # 如初始状态为 status = [4, 1, 5, 3, 0, 6, 2, 5]，当status[0] == 4时，不运行后面代码，直接进入下一次循环。
            if status[i] == j:
                continue
                
            # 将status的位置信息拷贝到新列表new_status中（需要将status拷贝到新的列表中，否则会改变原始的status的状态）
            new_status = copy.deepcopy(status)
            
            # 遍历棋盘上的每一个位置，执行内循环依次从0-7更新列。
            new_status[i] = j
            '''
            初始状态： [4, 1, 5, 3, 0, 6, 2, 5]
            
            [0, 1, 5, 3, 0, 6, 2, 5]
            [1, 1, 5, 3, 0, 6, 2, 5]
            [2, 1, 5, 3, 0, 6, 2, 5]
            [3, 1, 5, 3, 0, 6, 2, 5]
            [5, 1, 5, 3, 0, 6, 2, 5]
            [6, 1, 5, 3, 0, 6, 2, 5]
            [7, 1, 5, 3, 0, 6, 2, 5]
            [4, 0, 5, 3, 0, 6, 2, 5]
            
            ...（不保留当前状态皇后的位置，因此一共有56组）
            
            '''
            
            
            # 计算皇后移动到位置(i，j)时，将对应的（i，j）坐标与皇后的冲突数量添加到close字典里面。一共有56组。
            close[(i, j)] = conflict(new_status)[0]
    return close


# 画图函数
def plot(state=None):
    color = ['black', 'white', 'green', 'red']
    color_dict = {}
    for i in range(8):
        if i%2==0:
            for j in range(8):
                if j%2==0:
                    color_dict[i,j] = color[0]
                else:
                    color_dict[i,j] = color[1]
        else:
            for j in range(8):
                if j%2==0:
                    color_dict[i,j] = color[1]
                else:
                    color_dict[i,j] = color[0]
    button = {}
    
    for i in range(8):
        for j in range(8):
            button[(i,j)] = Button(layout=Layout(width='auto', height='auto'),style=ButtonStyle(button_color=str(color_dict[i,j])))
    if state!=None: 
        for i in range(8):
            button[(i,state[i])] = Button(layout=Layout(width='auto', height='auto'),style=ButtonStyle(button_color=color[2]))
            if conflict(state)[0]>0:
                queen = conflict(state)[1]
                for i in range(len(queen)):
                    row,column = queen[i]
                    button[(row,column)] = Button(layout=Layout(width='auto', height='auto'),style=ButtonStyle(button_color=color[3]))
    
    return button

def queue(it, size):
    it = iter(it)
    while True:
        p = tuple(itertools.islice(it, size))
        if not p:
            break
        yield p
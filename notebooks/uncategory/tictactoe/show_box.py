import copy
import random
import matplotlib.pyplot as plt
from ipywidgets import Button, GridBox, Layout, ButtonStyle
import itertools

# 图像布局
layout = Layout(
            width='80%',
            grid_template_columns='40px 40px 40px',
            grid_template_rows='40px 40px 40px',
            grid_gap='1px')

def plot(status):
    color = ['LightGray', 'green', 'red']
    color_dict = {}
    
    tic_format = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    for i in range(3):
        for j in range(3):
            
            if status[tic_format[i][j]]==0:
                color_dict[i,j]=color[0]
            elif status[tic_format[i][j]]==-1:
                color_dict[i,j]=color[1]
            elif status[tic_format[i][j]]==1:
                color_dict[i,j]=color[2]
    button = {}
    for m in range(3):
        for n in range(3):
            button[(m,n)] = Button(layout=Layout(width='auto', height='auto'),style=ButtonStyle(button_color=str(color_dict[m,n])))
    return button
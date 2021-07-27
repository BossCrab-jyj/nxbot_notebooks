import traitlets
import ipywidgets 
from IPython.display import display
import ipywidgets.widgets as widgets
from nxbot import Robot,event,bgr8_to_jpeg
from traitlets.config.configurable import Configurable
from uuid import uuid1
import glob
import os
import cv2
import shutil

rbt = Robot()

label_list = ['bg', '书包', '可乐', '圆规', '尺子', '果汁', '橡皮擦', '沐浴乳', 
              '洗发水', '洗洁精', '洗衣液', '爆米花', '牛奶', '笔', '茶', '薯片', '衣服', '裤子', '面包', '鞋子', '饼干']
data_dir = 'goods_datasets'

def make_dir():
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    path_dict = {}
    for name in label_list:
        label_path = os.path.join(data_dir,name)
        path_dict[name]=label_path
        if not os.path.exists(label_path):
            os.makedirs(label_path)
    return path_dict

path_dict = make_dir()

from ipywidgets import Layout, Button, Box, FloatText, Textarea, Dropdown, Label, IntSlider,FloatSlider

form_item_layout = Layout(
    display='flex',
    flex_flow='row',
    justify_content='space-between',
    )
names_items = [Box([Label(value='类别选择'),Dropdown(options=label_list)], layout=form_item_layout)]
global choose_label
choose_label = Box(names_items, layout=Layout(
        display='flex',
        flex_flow='column',
        border='solid 2px',
        align_items='stretch',
        width='50%'))


control_button_layout = widgets.Layout(width='60px', height='30px', align_self='center')
snapshot = widgets.Button(description='拍照', layout=control_button_layout)
delete_button = widgets.Button(description='删除当前文件夹', button_style='danger', layout=widgets.Layout(width='130px', height='30px', align_self='center'))
clear_button = widgets.Button(description='清空所有数据', button_style='danger', layout=widgets.Layout(width='130px', height='30px', align_self='center'))
count_button_layout = widgets.Layout(width='60px', height='30px', align_self='center')
img_count = widgets.IntText(layout=count_button_layout, value=len(os.listdir(path_dict[label_list[0]])))

collection_info = widgets.Textarea(
    placeholder='NXROBO',
    description='保存信息',
    disabled=False
)

def save_img():
    global choose_label,path_dict,img_count
    directory = path_dict[choose_label.children[0].children[1].value]
    image_path = os.path.join(directory, str(uuid1()) + '.jpg')

    with open(image_path, 'wb') as f:
        f.write(image_widget.value)
    collection_info.value = '图片已保存在'+'“'+directory+'”'+'目录下'
    
def delete_img():
    global choose_label,path_dict,img_count
    directory = path_dict[choose_label.children[0].children[1].value]
    img_list = os.listdir(directory)
    for img in img_list:
        if os.path.isdir(img):
            shutil.rmtree(os.path.join(directory,img))
        else:
            os.remove(os.path.join(directory,img))
    collection_info.value='删除成功'

def clear_datadir():
    shutil.rmtree(data_dir)
    collection_info.value='清空成功'
    make_dir()
    
snapshot.on_click(lambda x: save_img())
delete_button.on_click(lambda x: delete_img())
clear_button.on_click(lambda x: clear_datadir())

def on_new_image(evt):
    try:
        global choose_label,path_dict,img_count
        directory = path_dict[choose_label.children[0].children[1].value]
        img_count.value = len(os.listdir(directory))
    except Exception as result:
        pass
    
    image_widget.value=bgr8_to_jpeg(cv2.resize(evt.dict['data'],(320,280)))

button_layout = widgets.Layout(width='100px', height='80px', align_self='center')
stop_button = widgets.Button(description='停止', button_style='danger', layout=button_layout)
forward_button = widgets.Button(description='前进', layout=button_layout)
backward_button = widgets.Button(description='后退', layout=button_layout)
left_button = widgets.Button(description='左转', layout=button_layout)
right_button = widgets.Button(description='右转', layout=button_layout)
shiftleft_button = widgets.Button(description='左平移', layout=button_layout)
shiftright_button = widgets.Button(description='右平移', layout=button_layout)

# 将按钮拼接在一起
up_box = widgets.HBox([shiftleft_button, forward_button, shiftright_button], layout=widgets.Layout(align_self='center'))
middle_box = widgets.HBox([left_button, stop_button, right_button], layout=widgets.Layout(align_self='center'))
controls_box = widgets.VBox([up_box, middle_box, backward_button])




# 创建显示窗口
image_widget = widgets.Image(format='jpeg')
speed = 0.2
time = 1
def stop(change):
    rbt.base.stop()

def step_forward(change):
    rbt.base.forward(speed, time)

def step_backward(change):
    rbt.base.backward(speed, time)

def step_left(change):
    rbt.base.turnleft(speed, time)

def step_right(change):
    rbt.base.turnright(speed, time)
    
def shift_left(change):
    rbt.base.shiftleft(speed, time)

def shift_right(change):
    rbt.base.shiftright(speed, time)
    

    
stop_button.on_click(stop)
forward_button.on_click(step_forward)
backward_button.on_click(step_backward)
left_button.on_click(step_left)
right_button.on_click(step_right)
shiftleft_button.on_click(shift_left)
shiftright_button.on_click(shift_right)



# 创建摄像头视角滑块。
camera_x_slider = ipywidgets.FloatSlider(min=-90, max=90, step=1, value=0, description='摄像头左右')
camera_y_slider = ipywidgets.FloatSlider(min=-90, max=90, step=1, value=0, description='摄像头上下')

class Camera(Configurable):
    cx_speed = traitlets.Float(default_value=0.0)
    cy_speed = traitlets.Float(default_value=0.0)
    @traitlets.observe('cx_speed')
    def x_speed_value(self, change):
        self.cx_speed=change['new']
        rbt.base.set_ptz(x = self.cx_speed, y = self.cy_speed)

    @traitlets.observe('cy_speed')
    def a_speed_value(self, change):
        self.cy_speed=change['new']
        rbt.base.set_ptz(x = self.cx_speed, y = self.cy_speed)

camera = Camera()

camera_x_link = traitlets.dlink((camera_x_slider,'value'), (camera, 'cx_speed'), transform=lambda x: x)
camera_y_link = traitlets.dlink((camera_y_slider,'value'), (camera, 'cy_speed'), transform=lambda x: x)

print('加载成功!')
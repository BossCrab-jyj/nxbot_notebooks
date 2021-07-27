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

data_dir = 'signal_datasets'
forward_dir = os.path.join(data_dir,'forward')
stop_dir = os.path.join(data_dir,'stop')
left_dir = os.path.join(data_dir,'left')
right_dir = os.path.join(data_dir,'right')
turn_dir = os.path.join(data_dir,'turn')
red_dir = os.path.join(data_dir,'red')
yellow_dir = os.path.join(data_dir,'yellow')
red_yellow_dir = os.path.join(data_dir,'red_yellow')
green_dir = os.path.join(data_dir,'green')
bg_dir = os.path.join(data_dir,'bg')

try:
    os.makedirs(forward_dir)
    os.makedirs(stop_dir)
    os.makedirs(left_dir)
    os.makedirs(right_dir)
    os.makedirs(turn_dir)
    os.makedirs(red_dir)
    os.makedirs(yellow_dir)
    os.makedirs(red_yellow_dir)
    os.makedirs(green_dir)
    os.makedirs(bg_dir)
except FileExistsError:
    pass


    
control_button_layout = widgets.Layout(width='60px', height='30px', align_self='center')

stop_save = widgets.Button(description='停止', layout=control_button_layout)
forward_save = widgets.Button(description='前进', layout=control_button_layout)
left_save = widgets.Button(description='左转', layout=control_button_layout)
right_save = widgets.Button(description='右转', layout=control_button_layout)
turn_save = widgets.Button(description='掉头', layout=control_button_layout)
red_save = widgets.Button(description='红灯', layout=control_button_layout)
yellow_save = widgets.Button(description='黄灯', layout=control_button_layout)
red_yellow_save = widgets.Button(description='红黄灯', layout=control_button_layout)
green_save = widgets.Button(description='绿灯', layout=control_button_layout)
bg_save = widgets.Button(description='背景', layout=control_button_layout)
delete_button = widgets.Button(description='清空所有数据', button_style='danger', layout=widgets.Layout(width='130px', height='30px', align_self='center'))
make_dir_button = widgets.Button(description='重新创建文件夹', layout=widgets.Layout(width='130px', height='30px', align_self='center'))

count_button_layout = widgets.Layout(width='60px', height='30px', align_self='center')

stop_count = widgets.IntText(layout=count_button_layout, value=len(os.listdir(stop_dir)))
forward_count = widgets.IntText(layout=count_button_layout, value=len(os.listdir(forward_dir)))
left_count = widgets.IntText(layout=count_button_layout, value=len(os.listdir(left_dir)))
right_count = widgets.IntText(layout=count_button_layout, value=len(os.listdir(right_dir)))
turn_count = widgets.IntText(layout=count_button_layout, value=len(os.listdir(turn_dir)))
red_count = widgets.IntText(layout=count_button_layout, value=len(os.listdir(red_dir)))
yellow_count = widgets.IntText(layout=count_button_layout, value=len(os.listdir(yellow_dir)))
red_yellow_count = widgets.IntText(layout=count_button_layout, value=len(os.listdir(red_yellow_dir)))
green_count = widgets.IntText(layout=count_button_layout, value=len(os.listdir(green_dir)))
bg_count = widgets.IntText(layout=count_button_layout, value=len(os.listdir(bg_dir)))

def save_snapshot(directory):
    image_path = os.path.join(directory, str(uuid1()) + '.jpg')
    with open(image_path, 'wb') as f:
        f.write(image_widget.value)
        
def save_stop():
    global stop_dir, stop_count
    save_snapshot(stop_dir)
    stop_count.value = len(os.listdir(stop_dir))
    
def save_forward():
    global forward_dir, forward_count
    save_snapshot(forward_dir)
    forward_count.value = len(os.listdir(forward_dir))

def save_left():
    global left_dir, left_count
    save_snapshot(left_dir)
    left_count.value = len(os.listdir(left_dir))
    
def save_right():
    global right_dir, right_count
    save_snapshot(right_dir)
    right_count.value = len(os.listdir(right_dir))
    
def save_turn():
    global turn_dir, turn_count
    save_snapshot(turn_dir)
    turn_count.value = len(os.listdir(turn_dir))

def save_red():
    global red_dir, red_count
    save_snapshot(red_dir)
    red_count.value = len(os.listdir(red_dir))

def save_yellow():
    global yellow_dir, yellow_count
    save_snapshot(yellow_dir)
    yellow_count.value = len(os.listdir(yellow_dir))
    
def save_red_yellow():
    global red_yellow_dir, red_yellow_count
    save_snapshot(red_yellow_dir)
    red_yellow_count.value = len(os.listdir(red_yellow_dir))
    
def save_green():
    global green_dir, green_count
    save_snapshot(green_dir)
    green_count.value = len(os.listdir(green_dir))    
    
def save_bg():
    global bg_dir, bg_count
    save_snapshot(bg_dir)
    bg_count.value = len(os.listdir(bg_dir))

def delete(_):
    try:
        shutil.rmtree(data_dir)
        print('删除成功！')
    except Exception as result:
        print('该文件不存在！，请重新创建')
    
    stop_count.value = 0
    forward_count.value = 0
    left_count.value = 0
    right_count.value = 0
    turn_count.value = 0
    red_count.value = 0
    yellow_count.value = 0
    red_yellow_count.value = 0
    green_count.value = 0
    bg_count.value = 0
    

def remake_dir(_):
    try:
        os.makedirs(forward_dir)
        os.makedirs(stop_dir)
        os.makedirs(left_dir)
        os.makedirs(right_dir)
        os.makedirs(turn_dir)
        os.makedirs(red_dir)
        os.makedirs(yellow_dir)
        os.makedirs(red_yellow_dir)
        os.makedirs(green_dir)
        os.makedirs(bg_dir)
    except FileExistsError:
        print('该文件夹已被创建')
    print('创建成功')    
    

stop_save.on_click(lambda x: save_stop())
forward_save.on_click(lambda x: save_forward())
left_save.on_click(lambda x: save_left())
right_save.on_click(lambda x: save_right())
turn_save.on_click(lambda x: save_turn())
red_save.on_click(lambda x: save_red())
yellow_save.on_click(lambda x: save_yellow())
red_yellow_save.on_click(lambda x: save_red_yellow())
green_save.on_click(lambda x: save_green())
bg_save.on_click(lambda x: save_bg())
delete_button.on_click(delete)
make_dir_button.on_click(remake_dir)
# 拼接按键和数量显示窗口
collect_box = widgets.VBox([widgets.HBox([stop_save, forward_save, left_save, right_save, turn_save, red_save, yellow_save, red_yellow_save, green_save, bg_save]), 
                    widgets.HBox([stop_count, forward_count, left_count, right_count, turn_count, red_count, yellow_count, red_yellow_count, green_count,bg_count])])


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
    
def on_new_image(evt):
    image_widget.value=bgr8_to_jpeg(cv2.resize(evt.dict['data'],(320,280)))
    
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
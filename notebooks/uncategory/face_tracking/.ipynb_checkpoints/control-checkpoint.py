from nxbot import Robot,ObjectDetector,bgr8_to_jpeg,event,pid
from IPython.display import display
import ipywidgets.widgets as widgets
from traitlets.config.configurable import Configurable
import ipywidgets
import traitlets

rbt = Robot()

button_layout = widgets.Layout(width='100px', height='80px', align_self='center')

#创建控制按钮。
stop_button = widgets.Button(description='停止', button_style='danger', layout=button_layout)
forward_button = widgets.Button(description='前进', layout=button_layout)
backward_button = widgets.Button(description='后退', layout=button_layout)
left_button = widgets.Button(description='左转', layout=button_layout)
right_button = widgets.Button(description='右转', layout=button_layout)
shiftleft_button = widgets.Button(description='左平移', layout=button_layout)
shiftright_button = widgets.Button(description='右平移', layout=button_layout)

# 默认运行速度和时间。
speed = 0.3
times = 2

#定义所有运动模式。
def stop(change):
    rbt.base.stop()

def step_forward(change):
    rbt.base.forward(speed, times)

def step_backward(change):
    rbt.base.backward(speed, times)

def step_left(change):
    rbt.base.turnleft(speed*2, times)

def step_right(change):
    rbt.base.turnright(speed*2, times)

def shift_left(change):
    rbt.base.shiftleft(speed, times)

def shift_right(change):
    rbt.base.shiftright(speed, times)

# 通过“on_click”方法来触发小车进行运动。
stop_button.on_click(stop)
forward_button.on_click(step_forward)
backward_button.on_click(step_backward)
left_button.on_click(step_left)
right_button.on_click(step_right)
shiftleft_button.on_click(shift_left)
shiftright_button.on_click(shift_right)

# 把按键拼接在一起。
if rbt.name=='dachbot':
    up_box = widgets.HBox([shiftleft_button, forward_button, shiftright_button], layout=widgets.Layout(align_self='center'))
elif rbt.name=='dbot':
    up_box = widgets.HBox([forward_button], layout=widgets.Layout(align_self='center'))
    
middle_box = widgets.HBox([left_button, stop_button, right_button], layout=widgets.Layout(align_self='center'))
controls_box = widgets.VBox([up_box, middle_box, backward_button])

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
camera_slider = ipywidgets.VBox([camera_x_slider, camera_y_slider])

def unlink_control():
    camera_x_link.unlink()
    camera_y_link.unlink()
    
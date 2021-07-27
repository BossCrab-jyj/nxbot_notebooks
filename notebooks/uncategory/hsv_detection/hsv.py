import ipywidgets.widgets as widgets
from traitlets.config.configurable import Configurable
from IPython.display import display
from nxbot import Robot,ObjectDetector,bgr8_to_jpeg,event,pid
import cv2
import numpy as np
from ipywidgets import Layout, Box, Dropdown, Label, Widget
import ipywidgets
import threading
import traitlets
import time

# 设置颜色下拉选项布局
form_item_layout = Layout(
    display='flex',
    flex_flow='row',
    justify_content='space-between',
    )
# 设置颜色选择布局方式
layout = Layout(
        display='flex',
        flex_flow='column',
        border='solid 2px',
        align_items='stretch',
        width='50%')

# 颜色选项
color_list=['红色','黄色','蓝色','绿色','紫色','粉红色']

# 颜色选项下拉菜单
list_options =[Box([Label(value='颜色选择'),Dropdown(options=color_list)], layout=form_item_layout)]

# 颜色选择部件
color_widget = Box(list_options, layout=layout)

image_widget = widgets.Image(format='jpeg')
mask_widget = widgets.Image(format='jpeg')

H_MIN_slider = ipywidgets.IntSlider(min=0, max=180, step=1, value=0, description='H_MIN')
H_MAX_slider = ipywidgets.IntSlider(min=0, max=180, step=1, value=180, description='H_MAX')

S_MIN_slider = ipywidgets.IntSlider(min=0, max=255, step=1, value=0, description='S_MIN')
S_MAX_slider = ipywidgets.IntSlider(min=0, max=255, step=1, value=255, description='S_MAX')

V_MIN_slider = ipywidgets.IntSlider(min=0, max=255, step=1, value=0, description='V_MIN')
V_MAX_slider = ipywidgets.IntSlider(min=0, max=255, step=1, value=255, description='V_MAX')

HSV_BOX = widgets.VBox([H_MIN_slider,H_MAX_slider,S_MIN_slider,S_MAX_slider,V_MIN_slider,V_MAX_slider])

def choose_color(name,last_name):
    
    if last_name!=name:
        if name == '红色':
            H_MIN_slider.value,S_MIN_slider.value,V_MIN_slider.value = 157,43,43
            H_MAX_slider.value,S_MAX_slider.value,V_MAX_slider.value = 180, 255, 255
        
        elif name == '黄色':
            H_MIN_slider.value,S_MIN_slider.value,V_MIN_slider.value = 14, 43, 43
            H_MAX_slider.value,S_MAX_slider.value,V_MAX_slider.value = 32, 255, 255

        elif name == '蓝色':
            H_MIN_slider.value,S_MIN_slider.value,V_MIN_slider.value = 86,46,39
            H_MAX_slider.value,S_MAX_slider.value,V_MAX_slider.value = 119, 255, 255

        elif name == '绿色':
            H_MIN_slider.value,S_MIN_slider.value,V_MIN_slider.value = 35, 43, 43
            H_MAX_slider.value,S_MAX_slider.value,V_MAX_slider.value = 90, 255, 255

        elif name == '紫色':
            H_MIN_slider.value,S_MIN_slider.value,V_MIN_slider.value = 115,43,43
            H_MAX_slider.value,S_MAX_slider.value,V_MAX_slider.value = 131, 255, 255

        elif name == '粉红色':
            H_MIN_slider.value,S_MIN_slider.value,V_MIN_slider.value = 163, 18, 39
            H_MAX_slider.value,S_MAX_slider.value,V_MAX_slider.value = 180, 255, 255
            
        last_name = name
        
    h_min = H_MIN_slider.value
    s_min = S_MIN_slider.value
    v_min = V_MIN_slider.value
    
    h_max = H_MAX_slider.value
    s_max = S_MAX_slider.value
    v_max = V_MAX_slider.value
    
    color_lower = np.array([h_min,s_min,v_min])
    color_upper = np.array([h_max,s_max,v_max])
    
    return color_lower,color_upper,last_name


global detect_flag
detect_flag = True

def prediction():
    global detect_flag
    detect_flag = True
    # 创建显示窗口
    
    kernel = np.ones((3,3),np.uint8)#3x3的卷积核
    last_name = ''
    
    while detect_flag:
        
        image = rbt.camera.read()
        if image is not None:
            # 将图像转换为HSV格式
            hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV) 
            # 获取手动选择的颜色
            color_name = color_widget.children[0].children[1].value
            # HSV颜色范围
            color_lower, color_upper,last_name = choose_color(color_name,last_name)
            # 固定HSV颜色范围
            mask=cv2.inRange(hsv,color_lower,color_upper)
            # 图像腐蚀
            mask=cv2.erode(mask,kernel,iterations=1)
            # 图像膨胀
            mask=cv2.dilate(mask,kernel,iterations=1)
            # 图像滤波，卷积核5×5，标准差为0
            mask=cv2.GaussianBlur(mask,(5,5),0)
            # 显示二值图
            mask_widget.value = bgr8_to_jpeg(cv2.resize(mask,(400, 280)))
            # 找出滤波后的图像轮廓
            cnts=cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2] 
            # 如果有轮廓
            if len(cnts)>0:
                # 找出轮廓最大的那个区域
                cnt = max (cnts,key=cv2.contourArea)
                x1, y1, w, h = cv2.boundingRect(cnt)
                cx, cy, x2, y2 = w / 2 + x1, h / 2 + y1, x1 + w, y1 + h
                cv2.rectangle(image, (x1, y1), (x2, y2), (0,0,255), 1)
            image_widget.value = bgr8_to_jpeg(cv2.resize(image,(400, 280)))


# 创建摄像头视角滑块。
camera_x_slider = ipywidgets.FloatSlider(min=-90, max=90, step=1, value=0, description='摄像头左右')
camera_y_slider = ipywidgets.FloatSlider(min=-40, max=30, step=1, value=0, description='摄像头上下')

class Camera(Configurable):
    cx_speed = traitlets.Float(default_value=0.0)
    cy_speed = traitlets.Float(default_value=0.0)
    @traitlets.observe('cx_speed')
    def x_speed_value(self, change):
        time.sleep(0.1)
        self.cx_speed=change['new']
        rbt.base.set_ptz(x = self.cx_speed, y = self.cy_speed)

    @traitlets.observe('cy_speed')
    def a_speed_value(self, change):
        time.sleep(0.1)
        self.cy_speed=change['new']
        rbt.base.set_ptz(x = self.cx_speed, y = self.cy_speed)

camera = Camera()

camera_x_link = traitlets.dlink((camera_x_slider,'value'), (camera, 'cx_speed'), transform=lambda x: x)
camera_y_link = traitlets.dlink((camera_y_slider,'value'), (camera, 'cy_speed'), transform=lambda x: x)            
slider_box = ipywidgets.VBox([camera_x_slider, camera_y_slider])                
            

rbt = Robot()
rbt.connect()
rbt.camera.start()
rbt.base.set_ptz(0)

def start():
    # 创建线程
    process1 = threading.Thread(target=prediction,)
    # 启动线程
    process1.start()

    display(widgets.HBox([image_widget,mask_widget]))
    display(color_widget)
    display(widgets.HBox([HSV_BOX,slider_box]))

def close():
    global detect_flag
    detect_flag = False
    rbt.disconnect()
    Widget.close_all()
    camera_x_link.unlink()
    camera_y_link.unlink()
    
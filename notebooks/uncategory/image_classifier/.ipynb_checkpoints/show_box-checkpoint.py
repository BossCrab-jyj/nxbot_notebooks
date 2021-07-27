import io
import os
import IPython
import ctypes
from PIL import ImageDraw,ImageFont
import PIL.Image
from uuid import uuid1
import numpy as np
from scipy.misc import imread
import ipywidgets.widgets as widgets
from ipywidgets import Image, VBox, HBox, Widget, Button, HTML
from IPython.display import display
# from ipywebrtc import CameraStream, ImageRecorder
from ipywidgets import Layout, Button, Box, FloatText, Textarea, Dropdown, Label, IntSlider,FloatSlider, interact, interactive,SelectionSlider

import os
import shutil
import inspect
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F

global image
image = None


global thread_sign
thread_flag = False
global my_tread_1
my_tread_1 = None

VIDEO_WIDTH = 300 # 窗口宽度，按需调整
VIDEO_HEIGHT = 300 # 窗口高度，按需调整


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FLAG_STOP = False	# 终止标记
class_count=0
global data_root_dir
data_root_dir = 'data/dataset'
class_name  = 'class_'

form_item_layout = Layout(
    display='flex',
    flex_flow='row',
    justify_content='space-between',
    )

layout = Layout(
        display='flex',
        flex_flow='column',
        border='solid 2px',
        align_items='stretch',
        width='50%')
'''
输入类别名称
'''
input_label = widgets.Text(
    value='',
    placeholder='默认创建“class_n”',
    description='创建类别',
    disabled=False
)

'''
创建新类别
'''
add_label = Button(description="创建新类别",
                  button_style='primary')

def add(_):
    name = input_label.value
    # 如果不输入类别，添加默认类别，格式为：class_n，n表示数字
    if name=='':
        global classes_dir,class_count
        classes_dir = '{}/{}{}'.format(data_root_dir,class_name, class_count)
        try:
            os.makedirs(classes_dir)
            print('成功创建类别{}{}'.format(class_name, class_count))
        except FileExistsError:
            print('类别{}{}已创建'.format(class_name, class_count))
        class_count+=1
    else:
        classes_dir = '{}/{}'.format(data_root_dir,name)
        try:
            os.makedirs(classes_dir)
            print('成功创建类别{}'.format(name))
        except FileExistsError:
            print('类别{}已创建'.format(name))
    
add_label.on_click(add)


'''
外部接口-添加类别
'''
def add_classes():
    display(HBox([input_label,add_label]))

'''
外部接口-抓取图片
'''
import threading
import time
image_widget = widgets.Image(format='jpeg', width=300, height=300)

def bgr8_to_jpeg(value, quality=50):
    return bytes(cv2.imencode('.jpg', value)[1])

def get_camera_data(classes, flag):
    global thread_flag
    thread_flag = True
    global cap
    cap = cv2.VideoCapture(0)
    cap.set(3,300)
    cap.set(4,300)
    if flag=='collect':
        while True:
            ret, origin_img = cap.read()
            global image
            image = PIL.Image.fromarray(cv2.cvtColor(origin_img,cv2.COLOR_BGR2RGB))

            global image_widget
            image_widget.value = bgr8_to_jpeg(origin_img)

            global FLAG_STOP
            if FLAG_STOP==True:
                break
    elif flag=='predict':
        trained_state_dict,model_path = get_trained_model(len(classes))
        trained_model = torch.load(model_path)
        trained_model.to(device)
        
        mean = 255.0 * np.array([0.471, 0.448, 0.408])
        stdev = 255.0 * np.array([0.234, 0.239, 0.242])
        normalize = torchvision.transforms.Normalize(mean, stdev)


        # 数据预处理
        def preprocess(camera_value):
            x = camera_value
            x = cv2.resize(x,(224,224),interpolation=cv2.INTER_CUBIC)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            x = x.transpose((2, 0, 1))
            x = np.ascontiguousarray(x, dtype=np.float32)
            x = normalize(torch.from_numpy(x)).unsqueeze(0).to(device)
            return x

        
        while True:
            ret, frame = cap.read()
            image = preprocess(frame)
            predection = trained_model(image)
            output = F.softmax(predection, dim=1)
            prob, predict = torch.max(output, 1)
            label = classes[predict]
            pil_img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            pilimg = PIL.Image.fromarray(pil_img)
            draw = ImageDraw.Draw(pilimg)
            font = ImageFont.truetype("simhei.ttf",20,encoding="utf-8")
            # 将类别与概率显示在图像上。
            draw.text((0,0),label+'  概率：'+ ' %.2f' % (prob*100)+'%',(255,0,0),font=font)
            new_frame = cv2.cvtColor(np.array(pilimg),cv2.COLOR_RGB2BGR)
            
#             global image_widget
            image_widget.value = bgr8_to_jpeg(new_frame)
            
#             global FLAG_STOP
            if FLAG_STOP==True:
#                 cap.release()
                break

    
            
'''
camera图片保存
'''
save_Button = Button(description="保存图片",
                  button_style='primary')
def save_cam(_):
    global image
    global select_Widget
    
    son_dir = select_Widget.children[0].children[1].value
    if image!=None:
        if son_dir!=None:
            classes_dir = os.path.join(data_root_dir, son_dir)
            img_count = len(os.listdir(classes_dir))
            image.save('{}/{}.png'.format(classes_dir, img_count))
            print('保存成功！')
        else:
            print('保存失败，没有找到类别文件夹。')

save_Button.on_click(save_cam) # 注册单击事件        


def snapshot(classes=None,flag='collect'):
    
    FLAG_STOP = False
    global thread_flag
    if thread_flag==False:
        global my_tread_1
        my_tread_1 = threading.Thread(target=get_camera_data, args=(classes,flag,))
        my_tread_1.start()
    
    '''
    关闭摄像头
    '''
    btn_stop = Button(description="关闭摄像头",
                      button_style='danger')
    def close_cam(_):
        global my_tread_1
        
        if my_tread_1!=None:
            close_thread(my_tread_1)
            my_tread_1=None
        FLAG_STOP = True
        global thread_flag
        thread_flag = False
        global cap
        cap.release()
        
    btn_stop.on_click(close_cam) # 注册单击事件
    
    
    '''
    删除类别
    '''
    btn_delete = Button(description="删除当前类别",
                      button_style='warning')

    def delete(_):
        img_path = select_Widget.children[0].children[1].value
        img_dir = os.path.join(data_root_dir, img_path)
        shutil.rmtree(img_dir)
        print('成功删除{}文件夹'.format(img_path))
    btn_delete.on_click(delete)
    
    
    '''
    选择类别
    '''                                                                                                                                                                     
    global data_root_dir
    list_dir = os.listdir(data_root_dir)
    if '.ipynb_checkpoints' in list_dir:
        list_dir.remove('.ipynb_checkpoints')
    list_options =[Box([Label(value='类别选择'),Dropdown(options=list_dir)], layout=form_item_layout)]
    global select_Widget
    select_Widget = Box(list_options, layout=layout)
    
    
    # 从摄像头中捕获图片
    if flag == 'collect':
        display(image_widget)
        display(select_Widget)
        display(save_Button)
        display(VBox([btn_delete, btn_stop]))
    
    
    # 进行预测
    elif flag == 'predict':
        display(image_widget)
        display(btn_stop)


'''
关闭线程
'''
def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")
        
def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)
    

def close_thread(my_tread):
    stop_thread(my_tread)



    
'''
上传图片
'''

saveImage_Button = Button(description="保存",button_style='primary')
def save_imgs(_):
    save_imgdata()
    upload_widgets.close()
    
    
saveImage_Button.on_click(save_imgs)


def select_image(imgs_flag=None):
    if imgs_flag==None:
        global upload_widgets
        upload_widgets = widgets.FileUpload(
        accept='image/*',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
        multiple=True  # True to accept multiple files upload else False
        )
        '''
        选择保存上传图片的类别
        '''
        global data_root_dir
        list_dir = os.listdir(data_root_dir)
        if '.ipynb_checkpoints' in list_dir:
            list_dir.remove('.ipynb_checkpoints')
        list_options =[Box([Label(value='类别选择'),Dropdown(options=list_dir)], layout=form_item_layout)]
        global upload_dir_Widget
        upload_dir_Widget = Box(list_options, layout=layout)

        '''
        删除类别
        '''
        global btn_del
        btn_del = Button(description="删除当前类别",
                          button_style='warning')

        def delete(_):
            img_dir = os.path.join(data_root_dir, upload_dir_Widget.children[0].children[1].value)
            shutil.rmtree(img_dir)
            print('类别{}已被删除'.format(img_dir))
        btn_del.on_click(delete)

        display(upload_dir_Widget)
        display(upload_widgets)
        display(saveImage_Button)
        display(btn_del)
    else:
        print('图片上传成功！')
        data_root_dir = imgs_flag
    
'''
保存图片
'''
def save_imgdata():
    uploader = upload_widgets.value
    name_list = list(uploader.keys())
    son_dir = upload_dir_Widget.children[0].children[1].value
    if son_dir!=None:
        upload_dir = os.path.join(data_root_dir,son_dir)
        for i in range(len(name_list)):
                # 不保存其它格式文件
            if uploader[name_list[i]]['metadata']['type'] not in ['image/jpeg','image/png']:
                continue
            im_in = uploader[name_list[i]]['content']
            value = io.BytesIO(im_in)
            roiImg = PIL.Image.open(value)
            roiImg.save(os.path.join(upload_dir,name_list[i]))
        print('成功将图片保存在“{}”目录下！'.format(upload_dir))
    else:
        print('保存失败，没有找到类别文件夹。')

'''
上传视频并保存为图片
'''        

    
'''
上传图片
'''


up_video_Button = Button(description="保存视频",button_style='primary')

def save_video(_):
    upload_video()
    
up_video_Button.on_click(save_video)



'''
选择视频文件
'''
def select_video(video_dir=None):
    global video_dir_flag
    video_dir_flag = video_dir
    global select_video_widgets
    select_video_widgets = widgets.FileUpload(
    accept='*',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
    multiple=False  # True to accept multiple files upload else False
    )
    if video_dir_flag==None:
        display(select_video_widgets)
    display(up_video_Button)


def upload_video():
#     try:
    global video_dir_flag
    if video_dir_flag==None:
        uploader = select_video_widgets.value
        video_name = list(uploader.keys())
        first_name = video_name[0]
        with open("data/video_data/"+first_name, "wb") as fp:
            fp.write(uploader[first_name]['content'])
        name_0 = os.path.splitext(first_name)[0]
        video_path = os.path.join('data/video_data',first_name)
        mk_path = os.path.join('data/dataset',name_0)
        if not os.path.exists(mk_path):
            os.mkdir(mk_path)

        capt = cv2.VideoCapture()
        capt.open(video_path)
        save_count=0
        fps = capt.get(cv2.CAP_PROP_FPS)

        frames = capt.get(cv2.CAP_PROP_FRAME_COUNT)
        print("视频帧率为：", int(fps))
        print("视频帧数为：", int(frames))
        print('开始保存，最多只能保存300张图片！')
        for i in range(int(frames)):
            if i>300:
                continue
            ret, frame = capt.read()
            frame = cv2.resize(frame,(300,300),interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(mk_path+'/'+str(save_count) + '.jpg',frame)
            save_count+=1
        capt.release()
        
        if save_count>0:
            print('保存成功！')
        
    else:
        V_path = 'data/video_data'
        name_list = os.listdir(V_path)
        for name in name_list:
            if name.endswith('.avi') or name.endswith('.mp4'):
                video_path = os.path.join(V_path,name)
                name_0 = os.path.splitext(name)[0]
                mk_path = os.path.join('data/dataset',name_0)
                if not os.path.exists(mk_path):
                    os.mkdir(mk_path)
                
                capt = cv2.VideoCapture()
                print(video_path)
                capt.open(video_path)
                save_count=0
                fps = capt.get(cv2.CAP_PROP_FPS)
                frames = capt.get(cv2.CAP_PROP_FRAME_COUNT)
                
                print("视频帧率为：", int(fps))
                print("视频帧数为：", int(frames))
                print('开始保存!')
                for i in range(int(frames)):
                    if i>300:
                        continue
                    ret, frame = capt.read()
                    frame = cv2.resize(frame,(300,300),interpolation=cv2.INTER_CUBIC)
                    save_dir = mk_path +'/'+str(save_count) + '.jpg'
                    cv2.imwrite(save_dir,frame)
                    save_count+=1
                    
                capt.release()
                
                if save_count>0:
                    print('保存成功！')

                
                    
#     except:
#         print('上传失败！')
    
        
        
        
'''
选择自己上传图片或者使用摄像头收集图片
''' 
# upload_list = ['上传图片','上传视频']
upload_list = ['上传图片','上传视频','调用摄像头']
def upload_way():
    list_options =[Box([Label(value='选择收集图片方式'),Dropdown(options=upload_list)], layout=form_item_layout)]
    global uploadWay_Widget
    uploadWay_Widget = Box(list_options, layout=layout) 
    display(uploadWay_Widget)
    
def get_uploadWay():
    method = uploadWay_Widget.children[0].children[1].value
    return method


'''
设置参数
'''
def set_param():
    learning_rate = [i / 10000+0.0001 for i in range(1000)]
    list_params =[Box([Label(value='valid_percente'),FloatSlider(min=0.1, max=0.5,step=0.01,value=0.15)], layout=form_item_layout),
                Box([Label(value='epochs'),IntSlider(min=1, max=500,value=10)], layout=form_item_layout),
                Box([Label(value='batch_size'),IntSlider(min=4, max=128,step=4)], layout=form_item_layout),
                Box([Label(value='learning_rate'),SelectionSlider(options=[("%.4f"%i,i) for i in learning_rate] ,value=0.001)], layout=form_item_layout),
                ]
    global param_Widget
    param_Widget = Box(list_params, layout=layout)
    display(param_Widget)

'''
获取参数
'''
def get_param():
    global param_Widget
    valid_percente = param_Widget.children[0].children[1].value
    epochs = param_Widget.children[1].children[1].value
    batch_size = param_Widget.children[2].children[1].value
    learning_rate = param_Widget.children[3].children[1].value
    return valid_percente, epochs, batch_size, learning_rate


'''
选择模型
'''

models_list = ['alexnet','resnet18','mobilenet_v2', 'vgg11','squeezenet1_1']
def choose_model():
    list_models =[Box([Label(value='模型选择'),Dropdown(options=models_list)], layout=form_item_layout)]
    global models_Widget
    models_Widget = Box(list_models, layout=layout)
    display(models_Widget)

    
def frozen_layer(pretrained, model):
    if pretrained == True:
        for parma in model.parameters():
            parma.requires_grad = False    
            
    # 加载预训练模型与神经网络框架
def models_bag(model_name,num_of_classes,pretrained):
    global data_root_dir
    list_dir = os.listdir(data_root_dir)
    if '.ipynb_checkpoints' in list_dir:
        list_dir.remove('.ipynb_checkpoints')
    
    if models_list[0] in model_name:
        model = models.alexnet(pretrained=pretrained)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_of_classes)
    elif models_list[1] in model_name:
        model = models.resnet18(pretrained=pretrained)
        model.fc = torch.nn.Linear(model.fc.in_features, num_of_classes)
    elif models_list[2] in model_name:
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_of_classes)
    elif models_list[3] in model_name:
        model = models.vgg11(pretrained=pretrained)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_of_classes)
    elif models_list[4] in model_name:
        model = models.squeezenet1_1(pretrained=pretrained)
        model.classifier[1]=torch.nn.Conv2d(512, num_of_classes, kernel_size=(1, 1), stride=(1, 1))
    return model

def model_name():
    name = models_Widget.children[0].children[1].value
    return name

def get_model(num_classes,pretrained=True):
    name = model_name()
    model_ = models_bag(name,num_classes,pretrained)
    return model_



'''
选择优化器
'''

op_list = ['SGD','SGD_Momentum','SGD_Momentum_L2', 'RMSprop','Adam']
def set_optimizer():
    list_op =[Box([Label(value='优化器'),Dropdown(options=op_list)], layout=form_item_layout)]
    global opt_Widget
    opt_Widget = Box(list_op, layout=layout)
    display(opt_Widget)

    
def optimizer(name,model):
    learning_rate = get_param()[3]
    if name == op_list[0]:
        # SGD 随机梯度下降
        opt = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif name == op_list[1]:
        # momentum 动量加速,在SGD函数里指定momentum的值
        opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    elif name == op_list[2]:
        # SGD 随机梯度下降加上 Momentum 动量加速，再加上L2正则防止过拟合
        opt = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.001)
    elif name == op_list[3]:
        # RMSprop 指定参数alpha
        opt = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9)
    elif name == op_list[4]:
        # Adam 自适应优化器 参数betas=(0.9, 0.99)
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    return opt

'''
获取优化器
'''
def get_optimizer(model):
    name = opt_Widget.children[0].children[1].value
    opt = optimizer(name, model)
    return opt

'''
删除缓存文件夹'.ipynb_checkpoints'
'''
def clean_data(data_dir):
    dele_dir = '.ipynb_checkpoints'
    img_dirs =  os.listdir(data_dir)
    if dele_dir in img_dirs:
        shutil.rmtree(os.path.join(data_dir,dele_dir))
    for img_dir in img_dirs:
        img_path = os.path.join(data_dir,img_dir)
        img_path_list = os.listdir(img_path)
        if dele_dir in img_path_list:
            shutil.rmtree(dele_dir)
    
    return data_root_dir


'''
选择已训练模型
'''
models_dir = 'model'

def choose_trained_model():
    trained_models_list = os.listdir(models_dir)
    if '.ipynb_checkpoints' in trained_models_list:
        trained_models_list.remove('.ipynb_checkpoints')
    trained_model =[Box([Label(value='选择已保存模型'),Dropdown(options=trained_models_list)], layout=form_item_layout)]
    global trained_model_Widget
    trained_model_Widget = Box(trained_model, layout=layout)
    display(trained_model_Widget)
'''
获取已训练模型
'''
def get_trained_model(num_of_classes,pretrained=False):
    global trained_model_Widget
    trained_model_name = trained_model_Widget.children[0].children[1].value
    trained_model = models_bag(trained_model_name,num_of_classes,pretrained)
    model_path = os.path.join(models_dir, trained_model_name)
    return trained_model, model_path


'''
选择测试图片
'''
def choose_test_img(test_dir=None):
    global test_flag
    test_flag = test_dir
    if test_flag==None:
        global choose_testImg_widgets
        choose_testImg_widgets = widgets.FileUpload(
        accept='image/*',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
        multiple=True  # True to accept multiple files upload else False
        )
        display(choose_testImg_widgets)

def get_test_img():
    global test_flag
    
    img_list=[]
    if test_flag == None:
        img_types = ['image/png','image/jpg','image/jpeg']
        uploader = choose_testImg_widgets.value
        name_list = list(uploader.keys())
        for i in range(len(name_list)):
                        # 不保存其它格式文件
            img_type = uploader[name_list[i]]['metadata']['type']
            if img_type not in img_types:
                continue
            im_in = uploader[name_list[i]]['content']
            value = io.BytesIO(im_in)
            roiImg = PIL.Image.open(value)
            if roiImg.mode=='L' or roiImg.mode=='RGBA':
                roiImg = roiImg.convert("RGB")

            img_list.append(roiImg)
    else:
        test_dir = test_flag
        name_list = os.listdir(test_dir)
        for name in name_list:
            roiImg = PIL.Image.open(os.path.join(test_dir,name))
            if roiImg.mode=='L' or roiImg.mode=='RGBA':
                roiImg = roiImg.convert("RGB")
            img_list.append(roiImg)    
                
    return img_list
    


def show_predict(labels,probs):
    image_margin = '0 0 0 0'
    boxes = []
    hbox_layout = Layout()
    hbox_layout.width = '100%'
    hbox_layout.justify_content = 'flex-start'

    green_box_layout = Layout()
    green_box_layout.width = '200px'
    green_box_layout.height = '200px'
    green_box_layout.border = '2px solid green'
    caption_size = 'h3'

    def make_box_for_grid(show_img, fit):
        """
        Make a VBox to hold caption/image for demonstrating
        option_fit values.
        """
        # Make the caption
        if fit is not None:
            fit_str = "'{}'".format(fit)
        else:
            fit_str = str(fit)

        h = HTML(value='' + str(fit_str) + '')

        # Make the green box with the image widget inside it
        boxb = Box()
        boxb.layout = green_box_layout
        boxb.children = [show_img]

        # Compose into a vertical box
        vb = VBox()
        vb.layout.align_items = 'center'
        vb.children = [h, boxb]
        return vb
    
    global test_flag
    if test_flag==None:
        img_list=list(choose_testImg_widgets.value.keys())

        for i in range(len(labels)):
            uploaded_file = choose_testImg_widgets.value[img_list[i]]['content']
            show_img = widgets.Image(value=uploaded_file, width=200,height=200)
            show_img.layout.margin = image_margin
        #     show_img.layout.
            boxes.append(make_box_for_grid(show_img, '类别为：'+ labels[i]+'，   概率：'+probs[i]))
    else:
        test_dir = test_flag
        name_list = os.listdir(test_dir)
        for i in range(len(name_list)):
            img_dir = os.path.join(test_dir,name_list[i])
            roiImg = PIL.Image.open(img_dir)
            imgByteArr = io.BytesIO()
            roiImg.save(imgByteArr, format='PNG')
            imgByteArr = imgByteArr.getvalue()
            show_img = widgets.Image(value=imgByteArr, width=200,height=200)
            show_img.layout.margin = image_margin
        #     show_img.layout.
            boxes.append(make_box_for_grid(show_img, '类别为：'+ labels[i]+'，   概率：'+probs[i]))
        
    vb = VBox()
    h = HTML(value='<{size}>预测结果为：</{size}>'.format(size=caption_size))
    vb.layout.align_items = 'stretch'
    hb = HBox()
    hb.layout = hbox_layout
    hb.children = boxes

    vb.children = [h, hb]
    display(vb)
            

        
'''
计算目录下多个文件夹里的图片均值与方差
'''

def mean_std(filepath):
    print('正在计算图片的均值与方差，请耐心等待!')
    index = os.listdir(filepath)
    num_dir = len(index)
    R_mean = G_mean = B_mean = R_var = G_var = B_var = 0.0
    all_imgs = 0
    for i in(index):
        filepaths = os.path.join(filepath, i)
        pathDir = os.listdir(filepaths)
        all_imgs += len(pathDir)

    R_channel = 0
    G_channel = 0
    B_channel = 0
    pixls = 0
    for i in(index):
        filepaths = os.path.join(filepath, i)
        pathDir = os.listdir(filepaths)
        for idx in range(len(pathDir)):
            filename = pathDir[idx]
            img_path = os.path.join(filepaths, filename)
            img = imread((img_path), mode='RGB') / 255.0
            pixls += img.shape[0]*img.shape[1]
            R_channel += np.sum(img[:, :, 0])
            G_channel += np.sum(img[:, :, 1])
            B_channel += np.sum(img[:, :, 2])

    R_mean = R_channel / pixls
    G_mean = G_channel / pixls
    B_mean = B_channel / pixls

    R_channel = 0
    G_channel = 0
    B_channel = 0
    for i in(index):
        filepaths = os.path.join(filepath, i)
        pathDir = os.listdir(filepaths)
        for idx in range(len(pathDir)):
            filename = pathDir[idx]
            img_path = os.path.join(filepaths, filename)
            img = imread((img_path), mode='RGB') / 255.0
            R_channel += np.sum((img[:, :, 0] - R_mean) ** 2)
            G_channel += np.sum((img[:, :, 1] - G_mean) ** 2)
            B_channel += np.sum((img[:, :, 2] - B_mean) ** 2)

    R_var = np.sqrt(R_channel / pixls)
    G_var = np.sqrt(G_channel / pixls)
    B_var = np.sqrt(B_channel / pixls)
    mean = [R_mean, G_mean, B_mean]
    std = [R_var, G_var, B_var]
    print('计算完成！')
    return  mean, std



# def snapshot(flag='collect'):
    
    
#     camera = CameraStream(constraints=
#                       {'facing_mode': 'user',	
#                        'audio': False,	
#                        'video': { 'width': VIDEO_WIDTH, 'height': VIDEO_HEIGHT}
#                        })	# 另一种CameraStream创建方式，参考下文组件介绍部分
#     image_recorder = ImageRecorder(stream=camera)
#     predict_Widget = Image(width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
#     '''
#     关闭摄像头
#     '''
#     btn_stop = Button(description="关闭摄像头",
#                       button_style='danger')
#     def close_cam(_):
#         FLAG_STOP = True
        
#     #   Widget.close_all() #关闭所有组件，不建议使用，如果想继续使用必须重启内核。
#         camera.close()
#         image_recorder.close()
#         select_Widget.close()
#         btn_stop.close()
#         input_label.close()
#         add_label.close()
#         btn_delete.close()
#     btn_stop.on_click(close_cam) # 注册单击事件
    
    
#     '''
#     删除类别
#     '''
#     btn_delete = Button(description="删除当前类别",
#                       button_style='warning')

#     def delete(_):
#         img_path = select_Widget.children[0].children[1].value
#         img_dir = os.path.join(data_root_dir, img_path)
#         shutil.rmtree(img_dir)
#         print('成功删除{}文件夹'.format(img_path))
#     btn_delete.on_click(delete)
    

#     '''
#     抓取图片
#     '''
#     def cap_image(_):	# 处理ImageRecord抓取到的图片的过程
#         if FLAG_STOP:
#             return	# 停止处理

#         im_in = PIL.Image.open(io.BytesIO(image_recorder.image.value))
#         im_array = np.array(im_in)[..., :3]
#         im_out = PIL.Image.fromarray(im_array)
#         son_dir = select_Widget.children[0].children[1].value
#         if son_dir!=None:
#             classes_dir = os.path.join(data_root_dir, son_dir)
#             img_count = len(os.listdir(classes_dir))
#             im_out.save('{}/{}.png'.format(classes_dir, img_count))
#             print('保存成功！')
#         else:
#             print('保存失败，没有找到类别文件夹。')

#     def pridict_camera(_):	# 处理ImageRecord抓取到的图片的过程
#         if FLAG_STOP:
#             return	# 停止处理
        
#         im_in = Image.open(io.BytesIO(image_recorder.image.value))
#         im_array = np.array(im_in)[..., :3]
        
# #         # cv2转效率较低
#         predict_Widget.value = bytes(cv2.imencode('.jpg', im_array[..., ::-1])[1])
# #         im_out = Image.fromarray(im_array)
# #         f = io.BytesIO()
# #         im_out.save(f, format='Jpeg') # BMP 17ms，PNG 100多ms，Jpeg 25ms
# #         predict_Widget.value = f.getvalue()
#         image_recorder.recording = True	# 重新设置属性，使ImageRecorder继续抓取
        
#     '''
#     选择类别
#     '''                                                                                                                                                                     
#     list_dir = os.listdir(data_root_dir)
#     if '.ipynb_checkpoints' in list_dir:
#         list_dir.remove('.ipynb_checkpoints')
#     list_options =[Box([Label(value='类别选择'),Dropdown(options=list_dir)], layout=form_item_layout)]
#     select_Widget = Box(list_options, layout=layout)
    
    
#     # 从摄像头中捕获图片
#     if flag == 'collect':
#         image_recorder.image.observe(cap_image, names=['value'])
#         display(HBox([camera, image_recorder]))
#         display(select_Widget)
    

    
#     # 进行预测
#     elif flag == 'predict':
#         image_recorder.image.observe(pridict_camera, names=['value'])
#         display(image_recorder)
#         display(predict_Widget)

#     display(VBox([btn_delete, btn_stop]))
    
from ipywidgets import Layout, Button, Box, FloatText, Textarea, Dropdown, Label, IntSlider,FloatSlider

form_item_layout = Layout(
    display='flex',
    flex_flow='row',
    justify_content='space-between',
    )

def goods_edit(goods_info):
    box_list = []
    classes_list = []
    clsses_dict = {}
    name_list = []
    name_dict = {}

    last_class = ''
    for goods_name in goods_info:
        classes = goods_info[goods_name][0]
        price = goods_info[goods_name][1]
        box = Box([Label(value='商品名称：'+goods_name+'，单价（元）：'),Textarea(str(price))])

        classes_list.append(classes)

        if classes != last_class:
            name_list=[]
            box_list=[]
            name_list.append(goods_name)
            box_list.append(box)
        else:
            name_list.append(goods_name)
            box_list.append(box)

        last_class = classes
        clsses_dict[classes]=name_list
        name_dict[classes] = box_list



    classes_list = list(set(classes_list))
    names_items = [Box([Label(value='类别选择'),Dropdown(options=classes_list)], layout=form_item_layout)]

    classes = Box(names_items, layout=Layout(
            display='flex',
            flex_flow='column',
            border='solid 2px',
            align_items='stretch',
            width='50%'))
    return classes,name_dict

def edit_price(goods_class):
    price_tag = Box(goods_class, layout=Layout(
                    display='flex',
                    flex_flow='column',
                    border='solid 2px',
                    align_items='stretch',
                    width='60%'))
    return price_tag
from ipywidgets import Layout, Button, Box, FloatText, Textarea, Dropdown, Label, IntSlider,FloatSlider, interact, interactive,SelectionSlider

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

filename = 'modules/data/custom/classes.names'
names_list=[]

with open(filename, "r+",encoding='utf-8') as dataSet:
    names = dataSet.readlines()
    for name in names:
        names_list.append(name)

list_options =[Box([Label(value='选择跟随的类别'),Dropdown(options=names_list)], layout=form_item_layout)]

label_widget = Box(list_options, layout=layout)

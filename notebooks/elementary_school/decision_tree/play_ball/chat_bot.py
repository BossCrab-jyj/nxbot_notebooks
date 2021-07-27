from nxbot import Robot,event
from random import choice
import time
import re

rbt = Robot()
rbt.speech.start()


features = ['天气','温度','湿度','刮风']
contents = ['今天天气怎么样？', '温度情况如何？','湿度情况如何？','是否有刮风？']
list_again = ['对不起，刚刚走神了，没听清，请再说一遍吧！', '你的声音真好听，再说一遍行不行？', '你沉默的样子真好看！', '你说错了哦！再说一遍吧！']
list_stop = ['我等得你好辛苦，不和你玩了', '再见了同学们', '后会有期！','拜拜！']
weathers = ['晴','阴','雨']
temperatures =  ['高','中','低']
humiditys = ['高','中']
winds = ['刮风','不刮风']

# 回答
def reply(content, features):
    num_times = 0
    result = None
    rbt.speech.play_text(content,True)
    dicts = '刮风'
    while result == None:
        value = rbt.speech.asr()
        if value:
            if dicts in features:
                if '不' not in value:
                    is_wind = len(re.findall('刮风', value))
                    if is_wind>0:
                        result = features[0]
                else:
                    is_wind = len(re.findall('刮风', value))
                    if is_wind>0:
                        result = features[1]

            for feature in value:
                if feature in features:
                    result = feature
            if num_times<3:
                if result == None:
                    print('请再说一遍！')
                    rbt.speech.play_text(choice(list_again),True)
                    rbt.speech.play_text(content,True)
                    
        if num_times == 3:
            answer = choice(list_stop)
            rbt.speech.play_text(answer,True)
            print("已退出，请重新运行！")
            raise SystemExit
            
        num_times += 1
    if result != None:
        print('语音识别特征为：', result)
    return result


def classify(tree,feats,featValue):
    
    # 在构建好的决策树中找出父节点。
    firstFeat = list(tree.keys())[0]
    
    # 找出当前父节点特征下的所有子节点。
    secondDict = tree[firstFeat]
    
    # 父节点在给定特征列表中的位置索引。
    featIndex = feats.index(firstFeat)
    
    # 不断的在决策树中做判断，判断当前样本属性是否与决策中的属性相等，直到找到叶节点输出类别。
    for key in secondDict.keys():
        if featValue[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key],feats,featValue)
            else:
                classLabel = secondDict[key]
    return classLabel


def chat(tree):
    weather = reply(contents[0], weathers)
    temperature = reply(contents[1], temperatures)
    humidity = reply(contents[2], humiditys)
    wind = reply(contents[3], winds)
    datasets = [weather, temperature, humidity, wind]
    print('特征列表：', datasets)
    classLabel = classify(tree,features,datasets)
    if classLabel=='打球':
        rbt.speech.play_text('今天天气不错，可以去打球',True)
    elif classLabel=='不打球':
        rbt.speech.play_text('今天天气有点糟糕，不适合出去打球',True)
    
    
    
    
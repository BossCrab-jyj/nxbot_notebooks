import jieba
import re
from random import choice
from nxbot import Robot,event
import time
import sys



common_used_numerals_tmp = {'零': 0, '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
                            '十': 10, '百': 100, '千': 1000, '万': 10000, '亿': 100000000}
common_used_numerals = {}
for key in common_used_numerals_tmp:
    common_used_numerals[key] = common_used_numerals_tmp[key]
 
 
def chinese2digits(uchars_chinese):
    total = 0
    r = 1  # 表示单位：个十百千...
    for i in range(len(uchars_chinese) - 1, -1, -1):
        val = common_used_numerals.get(uchars_chinese[i])
        if val >= 10 and i == 0:  # 应对 十三 十四 十*之类
            if val > r:
                r = val
                total = total + val
            else:
                r = r * val
                # total =total + r * x
        elif val >= 10:
            if val > r:
                r = val
            else:
                r = r * val
        else:
            total = total + r * val
    return total
 
num_str_start_symbol = ['一', '二', '两', '三', '四', '五', '六', '七', '八', '九','十']
more_num_str_symbol = ['零', '一', '二', '两', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿']
 
def changeChineseNumToArab(oriStr):
    lenStr = len(oriStr);
    aProStr = ''
    if lenStr == 0:
        return aProStr;
 
    hasNumStart = False;
    numberStr = ''
    for idx in range(lenStr):
        if oriStr[idx] in num_str_start_symbol:
            if not hasNumStart:
                hasNumStart = True;
 
            numberStr += oriStr[idx]
        else:
            if hasNumStart:
                if oriStr[idx] in more_num_str_symbol:
                    numberStr += oriStr[idx]
                    continue
                else:
                    numResult = str(chinese2digits(numberStr))
                    numberStr = ''
                    hasNumStart = False;
                    aProStr += numResult
 
            aProStr += oriStr[idx]
            pass
 
    if len(numberStr) > 0:
        resultNum = chinese2digits(numberStr)
        aProStr += str(resultNum)
 
    return aProStr



rbt = Robot()

list_other = ['你想抢哪几颗糖果？', "输入错误，请从第{}颗开始选择", "输入错误，您最多只能取 {} 颗糖果,请重新选择糖果数量", '请按照顺序抢糖果']
list_again = ['对不起，刚刚走神了，没听清，请再说一遍吧！', '你的声音真好听，再说一遍行不行？', '你沉默的样子真好看！']
list_stop = ['我等得你好辛苦，不和你玩啦', '再见了同学们', '没人敢向我挑战吗，看来我是找不到对手了','无敌是多么，多么寂寞，后会有期！']


# 语音识别返回识别数字
def asr():
    value = rbt.speech.asr()
    if value:
        value = changeChineseNumToArab(value)
        print('识别结果：',value)
        time.sleep(3)
        seg_list = jieba.cut(value, cut_all=False)
        source = ''.join(seg_list)
        number = re.findall(r"\d+\.?\d*",source)
        number = [ int(x) for x in number ]
    else:
        number=[]
        
    return number



# 是否按照顺序报数
def order(count, number):
    for i in range(count-1):
        if number[i]+1 != number[i+1]:
            return False
# 退出
def quit():
    answer = choice(list_stop)
    rbt.speech.play_text(answer,True)
    print("已退出，请重新运行！")
    raise SystemExit


# 开始识别，输出学生选择的第一个糖果索引，最后一个糖果索引和糖果的总数。
def analyse():
    count = 0
    number = []
    num_times_1 = 0
    #判断有没有检测到数字
    
    while count == 0:
        number = asr()
        count = len(number)
        judge = order(count, number)
        num_times_2 = 0

        # 判断是否按照顺序报数
        while judge==False:
            rbt.speech.play_text(list_other[3], True)
            rbt.speech.play_text(list_other[0], True)
            number = asr()
            count = len(number)
            judge = order(count, number)
            num_times_2 += 1
            if num_times_2 == 3:
                quit()
            
            num_times_1 += 1
        if num_times_1 == 3:
            if count == 0:
                quit()
                
        if number==[]: 
            rbt.speech.play_text(choice(list_again),True)
            
    first = int(number[0])
    last = int(number[-1])
    return first, last, count

# 输入dachbot选择的最后一个数，糖果数量限制
# 输出同学说出的最后糖果数量
def detect(new_last, limit_value):
    
    # 提前加载jieba模型
    seg_list = jieba.cut('开始', cut_all=False)
    source = ''.join(seg_list)
    
    rbt.speech.play_text(list_other[0], True)
    first, pre_last, count = analyse()
    
    # 判断是否是按照上一次继续报数
    num_times_3 = 0
    while (first == new_last + 1) == False:
        voice = list_other[1].format(new_last + 1)
        rbt.speech.play_text(voice, True)
        first, pre_last, count = analyse()
        num_times_3 += 1
        if num_times_3 == 3:
            quit()
            
    # 判断有没有报超出
    limit = limit_value
    num_times_4 = 0
    while (count in range(1,limit+1)) == False:
        voice = list_other[2].format(limit)
        rbt.speech.play_text(voice, True)
        first, pre_last, count = analyse()
        num_times_4 += 1
        if num_times_4 == 3:
            quit()
            
    last = pre_last

    return last
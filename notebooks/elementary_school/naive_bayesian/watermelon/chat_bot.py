from nxbot import Robot,event
rbt = Robot()
rbt.speech.start()

reply = ['坏瓜，坏瓜不能吃','此乃上等好瓜，可以尽情享受美味']
ques = '再说一遍'

def chat(pred):
    if pred == 0:
        rbt.speech.play_text(reply[0],True)
    else:
        rbt.speech.play_text(reply[1],True)
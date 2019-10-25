#coding=utf-8
from __future__ import unicode_literals
from threading import Timer
from wxpy import *
import requests

# 导入模块
from wxpy import *
# 初始化机器人，扫码登陆
bot = Bot()
# 搜索名称含有 "xxx"这里是自己的微信昵称
my_friend = bot.friends().search('朱子权', sex=MALE, city="秦皇岛")[0]

bot = None
def get_news():
    #获取一个连接中的内容
    url = "http://open.iciba.com/dsapi/"
    r = requests.get(url)
    print(r.json())
    contents = r.json()['content']
    translation = r.json()['translation']
    return contents,translation

def login_wechat():
    global bot
    bot = Bot()
    # bot = Bot(console_qr=2,cache_path="botoo.pkl")#linux环境上使用
def send_news():
    if bot == None:
        login_wechat()

    try:
        my_friend = bot.friends().search(u'Ryan')[0] #xxx表示微信昵称
        #my_friend.send(get_news()[0])
        #my_friend.send(get_news()[1][5:])
        my_friend.send(u" 睿哥你是最棒的")
        t = Timer(7200, send_news) #360是秒数
        t.start()
    except:
        print(u"失败！！")


if __name__ == "__main__":
    send_news()
    print(get_news()[0])

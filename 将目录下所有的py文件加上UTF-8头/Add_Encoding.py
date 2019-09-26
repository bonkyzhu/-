#coding=utf-8
import os
import re

pre = './'

lists = []

def gci(filepath):
#遍历filepath下所有文件，包括子目录
  files = os.listdir(filepath)
  for fi in files:
    fi_d = os.path.join(filepath,fi)            
    if os.path.isdir(fi_d):
      gci(fi_d)                  
    elif fi_d.endswith('.py') and fi != '__init__.py':
      lists.append(fi_d)
      print(fi_d)

gci(pre)

def add_all(address_list):
  for address in address_list:
    if address == './Add_Encoding.py':
      continue
    file = open(address)
    content = file.read()
    if not re.findall('^#coding=utf-8', content):
      content = '#coding=utf-8\n' + content
      with open(address, 'w') as f:
        f.write(content)
      print("文件%s 已处理"%address)

add_all(lists)

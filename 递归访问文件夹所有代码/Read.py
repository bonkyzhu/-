#!/usr/bin/env python

import os

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

with open('code.md', 'w') as file:
  for location in lists:
    if location != './Read.py':
      content = location + '\n```python\n' +\
        open(location).read() + '\n```\n'
      file.write(content)

#coding=utf-8
import os
import re
from translate import translate
from threadpool import ThreadPool, makeRequests
import chardet

pre = './'
line = '-'*80 + '\n'
lists = []
special = ['.git', 'Read.py', '.trans']

def judge(fi):
  if os.path.exists(fi+'.trans'):
    return False
  for s in special:
    if fi.find(s) != -1:
      return False
  return True

def gci(filepath):
#遍历filepath下所有文件，包括子目录
  files = os.listdir(filepath)
  for fi in files:
    fi_d = os.path.join(filepath,fi)            
    if os.path.isdir(fi_d):
      gci(fi_d)                  
    elif judge(fi_d):
      lists.append(fi_d)
      print(fi_d)

def Translate(source):
  source = source[0]
  target = translate(source)
  return target

def translate_file(file):
  content = open(file).read()
  if file.endswith('.md'):
    pattern = re.compile('(```(.+\n)+```|`\w+`)')
    split_content = re.split(pattern, content)
    for s_c in split_content:
      if s_c[0][0] != '`':
        s_c = Translate(s_c[0])
    trans_content = split_content

  elif file.endswith('.txt'):
    trans_content = Translate(content)

  elif file.endswith('.py') or file.endswith('.sh') or re.findall('/\.', file): 
    pattern = re.compile('#.+\n')
    trans_content = re.sub(pattern, Translate, content)

  if trans_content != content:
    with open(file + '.trans', 'w') as f:
      f.write(trans_content)

gci(pre)

# 多线程运行
poolsize = 8
pool = ThreadPool(poolsize)
requests = makeRequests(translate_file, lists)
trans = [pool.putRequest(req) for req in requests] 
pool.wait()
